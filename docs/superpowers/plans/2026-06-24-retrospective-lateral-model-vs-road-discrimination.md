# Retrospective Lateral — Offline Model/Path-vs-Road Discrimination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an analysis-only module that classifies each top lateral-weave / low-speed-swing episode as **model/path artifact (A)**, **real road/corridor feature the model faithfully follows (B)**, or **fixed-period loop limit-cycle (C)**, using only the existing route NPZ caches — so the next correction decision is evidence-led instead of a guess.

**Architecture:** Add one new module `retrospective_lateral/code/discrimination.py` plus its test file. It reuses the existing `signal_utils` (band filtering, masked RMS, GPS cells, spectral peak, contiguous regions) and `config` constants, and reads the already-built `results/cache/route_*.npz` files and `results/reports/symptom_catalog.csv`. It puts the vision-model path, the perceived lane-center/road-edges, the planner `desired_curvature`, the controller `cp_final_command`, and three **model-independent** realized-path references (CAN yaw, calibrated yaw, GPS-heading rate) onto a **common curvature axis** (lateral offset at lookahead `L` → curvature `2·y/L²`), then discriminates A/B/C with three tests: (1) within-window coherence/residual between planned path, perceived lane, and independent realized motion; (2) cross-pass spatial reproducibility over shared GPS cells; (3) spectral-peak-vs-speed slope. No vehicle-control code, `opendbc_repo/`, `panda/`, or comma device is touched. All outputs land under the already-gitignored `results/`.

**Tech Stack:** Python 3.11 in `.venv311`, NumPy, SciPy, pandas, pytest. CSV/Markdown artifacts.

---

## Scope Check

This is one coherent analysis subsystem (episode-level source discrimination) layered on top of the existing `retrospective_lateral` package. It produces one testable workflow and stays one plan. It deliberately does **not** implement any vehicle-control change, model swap, or controlled-drive tooling — those are downstream of this plan's verdict.

## Why this is analysis-only and separate from driving code

The review (`docs/superpowers/reports/2026-06-24-retrospective-lateral-fresh-full-analysis-root-cause-review.md`) established with high confidence that the oscillation originates **upstream of the controller**, but could not separate (A) a model/path-planner artifact from (B) the model faithfully following a genuinely wandering road, from (C) a fixed-period loop limit-cycle. The correction *layer* depends entirely on which of A/B/C is true. This module answers that question from logs before any `opendbc_repo/opendbc/car/ford/` change is even planned. It writes only to `retrospective_lateral/results/` (gitignored) and imports nothing from the car stack.

## File Structure

Create:
- `retrospective_lateral/code/discrimination.py` — all discrimination math, the episode classifier, and the CLI runner. One module because every function shares the same per-window curvature-bundle representation and the functions are small; splitting would scatter one analysis across files.
- `retrospective_lateral/tests/test_discrimination.py` — synthetic-signal unit tests for every public function.

Modify:
- `retrospective_lateral/code/config.py` — append discrimination constants (schema version, lookahead, thresholds, unit conversion). No existing constant is changed.

Do **not** modify `opendbc_repo/`, `panda/`, `extract.py`, `metrics.py`, `drilldown.py`, or any vehicle-control file.

Generated outputs (all under the existing `retrospective_lateral/results/.gitignore` glob):
- `results/reports/discrimination_window_records.csv`
- `results/reports/discrimination_repeat_pass.csv`
- `results/reports/discrimination_frequency_speed.csv`
- `results/reports/discrimination_episode_classification.csv`
- `results/reports/discrimination_summary.csv`

---

### Task 1: Config constants + curvature/heading primitives

**Files:**
- Modify: `retrospective_lateral/code/config.py`
- Create: `retrospective_lateral/code/discrimination.py`
- Test: `retrospective_lateral/tests/test_discrimination.py`

- [ ] **Step 1: Append discrimination constants to `config.py`**

Append these lines to the end of `retrospective_lateral/code/config.py` (do not edit existing lines):

```python

# --- Offline model/path-vs-road discrimination (analysis-only) ---
DISCRIM_SCHEMA_VERSION = "discrim-v1"
DISCRIM_LOOKAHEAD_M = 20.0            # primary lookahead for offset->curvature conversion
DISCRIM_MIN_SPEED_MPS = 1.5          # below this, yawRate/v curvature is unreliable
DISCRIM_MAX_LAG_S = 2.0              # cross-correlation search half-window
DISCRIM_ROAD_COHERENCE_MIN = 0.6     # |corr| threshold for "moving together"
DISCRIM_ARTIFACT_RESIDUAL_RATIO = 0.5  # (model-minus-lane RMS)/(lane RMS) >= this => model adds motion
DISCRIM_REPRO_FRACTION_ROAD = 0.5    # cross-pass profile corr >= this => road-reproducible
DISCRIM_REPRO_MIN_SHARED_CELLS = 4   # min shared GPS cells to compare two passes
DISCRIM_FREQ_FLAT_HZ_PER_MPH = 0.002 # |d(peakHz)/d(mph)| below this => speed-independent (loop-like)
DISCRIM_TOP_N_PER_SYMPTOM = 40       # how many worst episodes per symptom to discriminate
MPS_TO_MPH = 2.2369362920544
```

- [ ] **Step 2: Write the failing test for the two primitives**

Create `retrospective_lateral/tests/test_discrimination.py`:

```python
import math

import numpy as np

from retrospective_lateral.code import discrimination as D


def test_offset_to_curvature_parabolic_relation():
    # A lateral offset of 0.5 m at 20 m lookahead implies curvature 2*y/L^2.
    assert math.isclose(D.offset_to_curvature(0.5, 20.0), 2 * 0.5 / (20.0 ** 2), rel_tol=1e-9)
    arr = D.offset_to_curvature(np.array([0.0, 0.2, np.nan]), 10.0)
    assert arr[0] == 0.0
    assert math.isclose(arr[1], 2 * 0.2 / 100.0, rel_tol=1e-9)
    assert np.isnan(arr[2])


def test_offset_to_curvature_rejects_nonpositive_lookahead():
    try:
        D.offset_to_curvature(1.0, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for lookahead <= 0")


def test_gps_course_deg_cardinal_directions():
    # Heading north: lat increasing, lon constant -> ~0 deg. Heading east -> ~90 deg.
    n = 50
    lat_north = 34.0 + np.arange(n) * 1e-4
    lon_const = np.full(n, -84.0)
    course_n = D.gps_course_deg(lat_north, lon_const)
    assert abs(((np.nanmedian(course_n) + 180) % 360) - 180) < 5.0  # ~0 deg

    lat_const = np.full(n, 34.0)
    lon_east = -84.0 + np.arange(n) * 1e-4
    course_e = D.gps_course_deg(lat_const, lon_east)
    assert abs(np.nanmedian(course_e) - 90.0) < 5.0
```

- [ ] **Step 3: Run the test to verify it fails**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'retrospective_lateral.code.discrimination'`.

- [ ] **Step 4: Create `discrimination.py` with the primitives**

Create `retrospective_lateral/code/discrimination.py`:

```python
"""Offline model/path-vs-road discrimination for the Explorer ST lateral symptoms.

Analysis-only. Reads existing route NPZ caches and the symptom catalog; writes only
to retrospective_lateral/results/. Imports nothing from the vehicle-control stack.
"""

from __future__ import annotations

import math

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.signal_utils import (
    contiguous_regions,
    filter_continuous,
    gps_cells,
    heading_bin_deg,
    rms_masked,
    spectral_peak_hz,
)


def offset_to_curvature(offset_m, lookahead_m: float):
    """Convert a lateral offset at a forward lookahead to an implied path curvature.

    For a small-curvature arc through the origin, y(L) ~= kappa * L^2 / 2, so
    kappa ~= 2*y / L^2. Lets model path / lane center / road edge offsets be compared
    on the same 1/m axis as desired_curvature and yawRate/v.
    """
    L = float(lookahead_m)
    if L <= 0:
        raise ValueError("lookahead_m must be > 0")
    return 2.0 * np.asarray(offset_m, dtype=float) / (L * L)


def gps_course_deg(lat_deg, lon_deg):
    """Course over ground in degrees [0, 360), 0 = North, clockwise, from GPS displacement.

    Independent of the vision model and of EPAS. NaN where it cannot be computed.
    Approximate: treats finite samples as evenly spaced; intended as a coarse,
    model-independent heading reference, not a precision signal.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    out = np.full(lat.shape, np.nan)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return out
    lat0 = float(np.nanmedian(lat[ok]))
    east_m = lon * C.M_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat0))
    north_m = lat * C.M_PER_DEG_LAT
    dx = np.gradient(east_m)
    dy = np.gradient(north_m)
    course = np.degrees(np.arctan2(dx, dy)) % 360.0
    course[~ok] = np.nan
    return course
```

- [ ] **Step 5: Run the test to verify it passes**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add retrospective_lateral/code/config.py retrospective_lateral/code/discrimination.py retrospective_lateral/tests/test_discrimination.py
git commit -m "analysis: discrimination config + curvature/heading primitives"
```

---

### Task 2: Independent path-curvature references + cross-correlation lag

**Files:**
- Modify: `retrospective_lateral/code/discrimination.py`
- Test: `retrospective_lateral/tests/test_discrimination.py`

- [ ] **Step 1: Add failing tests for the references and lag**

Append to `retrospective_lateral/tests/test_discrimination.py`:

```python
def test_path_curvature_from_rate_guards_low_speed():
    rate = np.array([0.1, 0.1, 0.1])
    v = np.array([10.0, 1.0, np.nan])  # 0.1/10 = 0.01; v=1 < 1.5 guard -> nan; nan -> nan
    out = D.path_curvature_from_rate(rate, v, min_speed_mps=1.5)
    assert math.isclose(out[0], 0.01, rel_tol=1e-9)
    assert np.isnan(out[1])
    assert np.isnan(out[2])


def test_xcorr_best_recovers_known_lag_and_sign():
    fs = 20.0
    t = np.arange(0, 30, 1 / fs)
    f = 0.2
    a = np.sin(2 * np.pi * f * t)
    d = 6  # samples; b is a delayed by d samples (a leads b)
    b = np.concatenate([np.full(d, np.nan), a[:-d]])
    corr, lag_s = D.xcorr_best(a, b, fs, max_lag_s=2.0)
    assert corr > 0.95
    assert abs(lag_s - d / fs) < 1.5 / fs  # positive => a leads b


def test_xcorr_best_returns_nan_on_flat_signal():
    fs = 20.0
    a = np.ones(200)
    b = np.zeros(200)
    corr, lag_s = D.xcorr_best(a, b, fs, max_lag_s=2.0)
    assert math.isnan(corr) and math.isnan(lag_s)
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: FAIL with `AttributeError: module ... has no attribute 'path_curvature_from_rate'`.

- [ ] **Step 3: Implement the references and lag**

Append to `retrospective_lateral/code/discrimination.py`:

```python
def path_curvature_from_rate(rate, v_mps, min_speed_mps: float = C.DISCRIM_MIN_SPEED_MPS):
    """Realized path curvature = angular rate / speed (1/m). NaN below the speed guard.

    Use with CAN yawRate or calibrated yawRate to get a vision-model-independent path.
    """
    rate = np.asarray(rate, dtype=float)
    v = np.asarray(v_mps, dtype=float)
    out = np.full(rate.shape, np.nan)
    ok = np.isfinite(rate) & np.isfinite(v) & (v >= float(min_speed_mps))
    out[ok] = rate[ok] / v[ok]
    return out


def gps_path_curvature(lat_deg, lon_deg, v_mps, fs_hz: float = C.FS_HZ,
                       min_speed_mps: float = C.DISCRIM_MIN_SPEED_MPS):
    """Heading-rate curvature from GPS course (1/m). Fully model- and EPAS-independent.

    Coarse/noisy; used only as a third independent corroborator, not a primary metric.
    """
    course = gps_course_deg(lat_deg, lon_deg)
    rate = np.full(course.shape, np.nan)
    ok = np.isfinite(course)
    if ok.sum() >= 3:
        unwrapped = np.unwrap(np.radians(course[ok]))
        rate[ok] = np.gradient(unwrapped) * float(fs_hz)
    return path_curvature_from_rate(rate, v_mps, min_speed_mps=min_speed_mps)


def xcorr_best(a, b, fs_hz: float, max_lag_s: float = C.DISCRIM_MAX_LAG_S):
    """Best Pearson correlation over integer lags on the largest shared finite run.

    Returns (corr, lag_s). lag_s > 0 means `a` leads `b`. (nan, nan) if undecidable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    min_len = max(12, int(fs_hz * 3))
    regions = contiguous_regions(np.isfinite(a) & np.isfinite(b), min_len=min_len)
    if not regions:
        return (math.nan, math.nan)
    s, e = max(regions, key=lambda r: r[1] - r[0])
    aa = a[s:e]
    bb = b[s:e]
    n = len(aa)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return (math.nan, math.nan)
    max_lag = int(round(max_lag_s * fs_hz))
    best_corr = -2.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = aa[-lag:], bb[:n + lag]
        elif lag > 0:
            x, y = aa[:n - lag], bb[lag:]
        else:
            x, y = aa, bb
        if len(x) < min_len or np.std(x) == 0 or np.std(y) == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best_corr:
            best_corr = c
            best_lag = lag
    if best_corr < -1.5:
        return (math.nan, math.nan)
    return (best_corr, best_lag / fs_hz)
```

- [ ] **Step 4: Run to verify pass**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/discrimination.py retrospective_lateral/tests/test_discrimination.py
git commit -m "analysis: independent path-curvature references + xcorr lag"
```

---

### Task 3: Per-window discrimination record + tentative label

**Files:**
- Modify: `retrospective_lateral/code/discrimination.py`
- Test: `retrospective_lateral/tests/test_discrimination.py`

- [ ] **Step 1: Add a failing test using two synthetic windows (artifact-like vs road-like)**

Append to `retrospective_lateral/tests/test_discrimination.py`:

```python
def _synthetic_arrays(*, model_extra_wobble: bool):
    """Build a 30 s, 20 Hz route dict with a 0.2 Hz weave.

    Road case: lane center, model path, and realized yaw all carry the same 0.2 Hz wobble.
    Artifact case: the model path carries an EXTRA wobble not present in the lane center
    or the realized yaw (i.e. the planner invents motion).
    """
    fs = 20.0
    t = np.arange(0, 30, 1 / fs)
    L = 20.0
    f = 0.2
    v = np.full_like(t, 13.0)  # ~29 mph
    base_offset = 0.30 * np.sin(2 * np.pi * f * t)          # lane-center lateral offset (m)
    lane_curv = 2 * base_offset / (L ** 2)
    model_offset = base_offset.copy()
    if model_extra_wobble:
        model_offset = base_offset + 0.30 * np.sin(2 * np.pi * 0.27 * t + 1.0)
    yaw_rate = (2 * base_offset / (L ** 2)) * v              # realized curvature * v
    n = len(t)
    a = {
        "t": t.astype(np.float32),
        "v_ego": v.astype(np.float32),
        "yaw_rate": yaw_rate.astype(np.float32),
        "yaw_rate_calibrated": yaw_rate.astype(np.float32),
        "lat": (34.0 + np.cumsum(np.full(n, 1e-6))).astype(np.float32),
        "lon": np.full(n, -84.0, dtype=np.float32),
        "desired_curvature": lane_curv.astype(np.float32),
        "cp_final_command": (0.85 * lane_curv).astype(np.float32),
        "steering_angle_deg": (10.0 * base_offset).astype(np.float32),
        "lat_active": np.ones(n, dtype=np.float32),
        "steering_pressed": np.zeros(n, dtype=np.float32),
        "blinker": np.zeros(n, dtype=np.float32),
        "lane_change_state": np.zeros(n, dtype=np.float32),
        "lead_time_headway_s": np.full(n, 5.0, dtype=np.float32),
        "lane_prob_left": np.full(n, 0.9, dtype=np.float32),
        "lane_prob_right": np.full(n, 0.9, dtype=np.float32),
    }
    for key, off in (("model", model_offset), ("lane_center", base_offset)):
        a[f"{key}_y20"] = off.astype(np.float32)
    a["road_edge_left_y20"] = (base_offset - 1.8).astype(np.float32)
    a["road_edge_right_y20"] = (base_offset + 1.8).astype(np.float32)
    return a


def test_discriminate_window_labels_road_like_when_all_move_together():
    a = _synthetic_arrays(model_extra_wobble=False)
    rec = D.discriminate_window(a, "weave_10_70", "route_syn", 2.0, 28.0, 15.0)
    assert rec.tentative_label == "road_like"
    assert abs(rec.model_vs_lane_corr) >= 0.6
    assert abs(rec.lane_vs_independent_corr) >= 0.6


def test_discriminate_window_labels_artifact_like_when_model_adds_motion():
    a = _synthetic_arrays(model_extra_wobble=True)
    rec = D.discriminate_window(a, "weave_10_70", "route_syn", 2.0, 28.0, 15.0)
    assert rec.tentative_label == "artifact_like"
    assert rec.model_residual_over_lane >= 0.5
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: FAIL with `AttributeError: module ... has no attribute 'discriminate_window'`.

- [ ] **Step 3: Implement the record dataclass and `discriminate_window`**

Append to `retrospective_lateral/code/discrimination.py` (add `from dataclasses import dataclass` and `from retrospective_lateral.code.signal_utils import dilate_flags, erode_true` to the existing imports at the top of the file):

```python
@dataclass(frozen=True)
class DiscriminationWindow:
    symptom: str
    route_id: str
    start_s: float
    end_s: float
    peak_s: float
    speed_mph_median: float
    lookahead_m: float
    gps_cell: float
    heading_bin: int
    speed_bin: int
    road_curv_level_1pm: float
    model_curv_rms_1pm: float
    lane_curv_rms_1pm: float
    roadedge_curv_rms_1pm: float
    desired_rms_1pm: float
    cp_final_rms_1pm: float
    can_yaw_curv_rms_1pm: float
    cal_yaw_curv_rms_1pm: float
    gps_curv_rms_1pm: float
    steering_band_rms_deg: float
    model_vs_lane_corr: float
    model_vs_lane_lag_s: float
    lane_vs_independent_corr: float
    lane_vs_independent_lag_s: float
    model_minus_lane_residual_rms_1pm: float
    model_residual_over_lane: float
    spectral_peak_hz: float
    clean_fraction: float
    straight_clean: int
    tentative_label: str


def _band_for(symptom: str):
    return C.LOW_SPEED_INSPECT_BAND_HZ if symptom == "low_speed_wheel_swing" else C.DEFAULT_WEAVE_BAND_HZ


def _clean_mask(arrays) -> np.ndarray:
    n = len(arrays["t"])
    fs = C.FS_HZ

    def chan(name, default):
        return np.asarray(arrays.get(name, np.full(n, default)), dtype=float)

    lat_active = chan("lat_active", 0.0) > 0.5
    pressed = chan("steering_pressed", 0.0) > 0.5
    blink = chan("blinker", 0.0) > 0.5
    lane_change = chan("lane_change_state", 0.0) > 0.5
    headway = chan("lead_time_headway_s", np.nan)
    near_lead = np.isfinite(headway) & (headway < C.LEAD_HEADWAY_S)
    bad = dilate_flags(pressed | blink | lane_change | near_lead, int(round(C.OVERRIDE_BUFFER_S * fs)))
    clean = lat_active & ~bad
    return erode_true(clean, int(round(C.ENGAGE_ERODE_S * fs)))


def _roadedge_center(arrays, lk: str):
    left = arrays.get(f"road_edge_left_{lk}")
    right = arrays.get(f"road_edge_right_{lk}")
    if left is None or right is None:
        return np.full(len(arrays["t"]), np.nan)
    return 0.5 * (np.asarray(left, dtype=float) + np.asarray(right, dtype=float))


def _rms_band(sig, win_clean, fs, band):
    return rms_masked(filter_continuous(sig, fs, band=band), win_clean)


def discriminate_window(arrays, symptom: str, route_id: str, start_s: float, end_s: float,
                        peak_s: float, *, lookahead_m: float = C.DISCRIM_LOOKAHEAD_M,
                        band=None) -> DiscriminationWindow:
    fs = C.FS_HZ
    t = np.asarray(arrays["t"], dtype=float)
    lk = f"y{int(lookahead_m)}"
    if band is None:
        band = _band_for(symptom)

    win = (t >= start_s) & (t <= end_s)
    clean = _clean_mask(arrays)
    win_clean = win & clean
    clean_fraction = float(np.mean(clean[win])) if win.any() else math.nan

    v = np.asarray(arrays["v_ego"], dtype=float)
    speed_mph = float(np.nanmedian(v[win]) * C.MPS_TO_MPH) if win.any() else math.nan

    model_curv = offset_to_curvature(arrays[f"model_{lk}"], lookahead_m)
    lane_curv = offset_to_curvature(arrays[f"lane_center_{lk}"], lookahead_m)
    roadedge_curv = offset_to_curvature(_roadedge_center(arrays, lk), lookahead_m)
    desired = np.asarray(arrays["desired_curvature"], dtype=float)
    cp_final = np.asarray(arrays.get("cp_final_command", np.full(len(t), np.nan)), dtype=float)
    can_yaw_curv = path_curvature_from_rate(arrays["yaw_rate"], v)
    cal_yaw_curv = path_curvature_from_rate(arrays.get("yaw_rate_calibrated", np.full(len(t), np.nan)), v)
    gps_curv = gps_path_curvature(arrays["lat"], arrays["lon"], v)
    steering = np.asarray(arrays["steering_angle_deg"], dtype=float)

    model_b = filter_continuous(model_curv, fs, band=band)
    lane_b = filter_continuous(lane_curv, fs, band=band)
    indep_src = cal_yaw_curv if np.isfinite(cal_yaw_curv[win]).sum() >= int(fs * 3) else can_yaw_curv
    indep_b = filter_continuous(indep_src, fs, band=band)

    mvl_corr, mvl_lag = xcorr_best(np.where(win_clean, model_b, np.nan), np.where(win_clean, lane_b, np.nan), fs)
    lvi_corr, lvi_lag = xcorr_best(np.where(win_clean, lane_b, np.nan), np.where(win_clean, indep_b, np.nan), fs)

    lane_rms = _rms_band(lane_curv, win_clean, fs, band)
    model_rms = _rms_band(model_curv, win_clean, fs, band)
    residual = model_b - lane_b
    residual_rms = rms_masked(residual, win_clean)
    residual_ratio = residual_rms / lane_rms if (np.isfinite(lane_rms) and lane_rms > 0) else math.nan

    road_lp = filter_continuous(can_yaw_curv, fs, lowpass_hz=C.ROAD_LP_HZ)
    road_vals = np.abs(road_lp[win_clean])
    road_vals = road_vals[np.isfinite(road_vals)]
    road_level = float(np.median(road_vals)) if len(road_vals) else math.nan

    lane_prob_l = np.asarray(arrays.get("lane_prob_left", np.full(len(t), np.nan)), dtype=float)
    lane_prob_r = np.asarray(arrays.get("lane_prob_right", np.full(len(t), np.nan)), dtype=float)
    lane_ok = (np.nanmedian(lane_prob_l[win]) >= 0.5) and (np.nanmedian(lane_prob_r[win]) >= 0.5)
    straight_clean = int(
        np.isfinite(road_level) and road_level < C.ROAD_CURV_ABS_MAX_1PM
        and bool(lane_ok) and np.isfinite(clean_fraction) and clean_fraction >= 0.8
    )

    coh = C.DISCRIM_ROAD_COHERENCE_MIN
    if not (np.isfinite(lane_rms) and np.isfinite(model_rms)):
        label = "insufficient"
    elif np.isfinite(mvl_corr) and abs(mvl_corr) >= coh and np.isfinite(lvi_corr) and abs(lvi_corr) >= coh:
        label = "road_like"
    elif np.isfinite(residual_ratio) and residual_ratio >= C.DISCRIM_ARTIFACT_RESIDUAL_RATIO:
        label = "artifact_like"
    else:
        label = "ambiguous"

    course = gps_course_deg(arrays["lat"], arrays["lon"])
    heading_bin = int(np.nanmedian(heading_bin_deg(course[win], C.HEADING_BIN_DEG))) if win.any() else -1
    cell_vals = gps_cells(arrays["lat"], arrays["lon"], C.GPS_CELL_M)[win]
    cell_vals = cell_vals[np.isfinite(cell_vals)]
    gps_cell = float(np.median(cell_vals)) if len(cell_vals) else math.nan
    speed_bin = int(speed_mph // C.SPEED_BIN_MPH) if np.isfinite(speed_mph) else -1

    return DiscriminationWindow(
        symptom=symptom, route_id=route_id, start_s=float(start_s), end_s=float(end_s),
        peak_s=float(peak_s), speed_mph_median=speed_mph, lookahead_m=float(lookahead_m),
        gps_cell=gps_cell, heading_bin=heading_bin, speed_bin=speed_bin,
        road_curv_level_1pm=road_level,
        model_curv_rms_1pm=model_rms, lane_curv_rms_1pm=lane_rms,
        roadedge_curv_rms_1pm=_rms_band(roadedge_curv, win_clean, fs, band),
        desired_rms_1pm=_rms_band(desired, win_clean, fs, band),
        cp_final_rms_1pm=_rms_band(cp_final, win_clean, fs, band),
        can_yaw_curv_rms_1pm=_rms_band(can_yaw_curv, win_clean, fs, band),
        cal_yaw_curv_rms_1pm=_rms_band(cal_yaw_curv, win_clean, fs, band),
        gps_curv_rms_1pm=_rms_band(gps_curv, win_clean, fs, band),
        steering_band_rms_deg=_rms_band(steering, win_clean, fs, band),
        model_vs_lane_corr=mvl_corr, model_vs_lane_lag_s=mvl_lag,
        lane_vs_independent_corr=lvi_corr, lane_vs_independent_lag_s=lvi_lag,
        model_minus_lane_residual_rms_1pm=residual_rms, model_residual_over_lane=residual_ratio,
        spectral_peak_hz=spectral_peak_hz(np.where(win_clean, lane_b, np.nan), fs, band),
        clean_fraction=clean_fraction, straight_clean=straight_clean, tentative_label=label,
    )
```

- [ ] **Step 4: Run to verify pass**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/discrimination.py retrospective_lateral/tests/test_discrimination.py
git commit -m "analysis: per-window model-vs-road discrimination record"
```

---

### Task 4: Spatial profile + cross-pass reproducibility (the road test)

**Files:**
- Modify: `retrospective_lateral/code/discrimination.py`
- Test: `retrospective_lateral/tests/test_discrimination.py`

- [ ] **Step 1: Add failing tests for the spatial profile and reproducibility**

Append to `retrospective_lateral/tests/test_discrimination.py`:

```python
def test_spatial_curvature_profile_bins_by_gps_cell():
    a = _synthetic_arrays(model_extra_wobble=False)
    # Spread the route across multiple GPS cells by moving north steadily.
    n = len(a["t"])
    a["lat"] = (34.0 + np.arange(n) * 5e-5).astype(np.float32)
    prof = D.spatial_curvature_profile(a, 2.0, 28.0, D._band_for("weave_10_70"))
    assert len(prof) >= 2
    any_cell = next(iter(prof.values()))
    assert "lane" in any_cell and "model" in any_cell and "n" in any_cell


def test_cross_pass_reproducibility_high_for_identical_profiles():
    cells = [float(i) for i in range(10)]
    prof_a = {c: {"lane": math.sin(c), "model": 0.0, "n": 5} for c in cells}
    prof_b = {c: {"lane": math.sin(c), "model": 0.0, "n": 5} for c in cells}
    res = D.cross_pass_reproducibility([prof_a, prof_b], key="lane")
    assert res["n_shared_cells"] >= 4
    assert res["median_pairwise_corr"] > 0.95
    assert res["reproducible"] is True


def test_cross_pass_reproducibility_low_for_independent_noise():
    cells = [float(i) for i in range(12)]
    prof_a = {c: {"lane": math.sin(c), "model": 0.0, "n": 5} for c in cells}
    prof_b = {c: {"lane": math.cos(3 * c + 1.7), "model": 0.0, "n": 5} for c in cells}
    res = D.cross_pass_reproducibility([prof_a, prof_b], key="lane")
    assert res["reproducible"] is False
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: FAIL with `AttributeError: ... 'spatial_curvature_profile'`.

- [ ] **Step 3: Implement the spatial profile and reproducibility**

Append to `retrospective_lateral/code/discrimination.py` (add `from itertools import combinations` to the top-of-file imports):

```python
def spatial_curvature_profile(arrays, start_s: float, end_s: float, band,
                              *, lookahead_m: float = C.DISCRIM_LOOKAHEAD_M,
                              cell_m: float = C.GPS_CELL_M) -> dict:
    """Mean band-filtered lane/model curvature per GPS cell over a window.

    Keying the wobble to physical GPS cells (not time) lets two passes over the same
    road be compared. A wobble that reproduces by cell across passes is a road feature.
    """
    fs = C.FS_HZ
    t = np.asarray(arrays["t"], dtype=float)
    lk = f"y{int(lookahead_m)}"
    win = (t >= start_s) & (t <= end_s)
    lane_b = filter_continuous(offset_to_curvature(arrays[f"lane_center_{lk}"], lookahead_m), fs, band=band)
    model_b = filter_continuous(offset_to_curvature(arrays[f"model_{lk}"], lookahead_m), fs, band=band)
    cells = gps_cells(arrays["lat"], arrays["lon"], cell_m)
    out: dict[float, dict] = {}
    idx = np.flatnonzero(win)
    for i in idx:
        c = cells[i]
        if not np.isfinite(c):
            continue
        bucket = out.setdefault(float(c), {"_lane": [], "_model": []})
        if np.isfinite(lane_b[i]):
            bucket["_lane"].append(float(lane_b[i]))
        if np.isfinite(model_b[i]):
            bucket["_model"].append(float(model_b[i]))
    profile: dict[float, dict] = {}
    for c, bucket in out.items():
        if not bucket["_lane"]:
            continue
        profile[c] = {
            "lane": float(np.mean(bucket["_lane"])),
            "model": float(np.mean(bucket["_model"])) if bucket["_model"] else math.nan,
            "n": len(bucket["_lane"]),
        }
    return profile


def cross_pass_reproducibility(profiles: list, key: str = "lane",
                               min_shared_cells: int = C.DISCRIM_REPRO_MIN_SHARED_CELLS) -> dict:
    """Median pairwise Pearson correlation of per-cell profiles across passes.

    profiles: list of {gps_cell: {"lane":..., "model":..., "n":...}}, one per pass.
    """
    corrs = []
    shared_counts = []
    for pa, pb in combinations(profiles, 2):
        shared = sorted(set(pa) & set(pb))
        shared = [c for c in shared if np.isfinite(pa[c].get(key, np.nan)) and np.isfinite(pb[c].get(key, np.nan))]
        if len(shared) < min_shared_cells:
            continue
        va = np.array([pa[c][key] for c in shared])
        vb = np.array([pb[c][key] for c in shared])
        if np.std(va) == 0 or np.std(vb) == 0:
            continue
        corrs.append(float(np.corrcoef(va, vb)[0, 1]))
        shared_counts.append(len(shared))
    if not corrs:
        return {"n_passes": len(profiles), "n_pairs": 0, "n_shared_cells": 0,
                "median_pairwise_corr": math.nan, "reproducible": False}
    median_corr = float(np.median(corrs))
    return {
        "n_passes": len(profiles), "n_pairs": len(corrs),
        "n_shared_cells": int(np.median(shared_counts)),
        "median_pairwise_corr": median_corr,
        "reproducible": bool(median_corr >= C.DISCRIM_REPRO_FRACTION_ROAD),
    }
```

- [ ] **Step 4: Run to verify pass**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/discrimination.py retrospective_lateral/tests/test_discrimination.py
git commit -m "analysis: cross-pass spatial reproducibility (road test)"
```

---

### Task 5: Frequency-vs-speed slope + episode classifier (A/B/C)

**Files:**
- Modify: `retrospective_lateral/code/discrimination.py`
- Test: `retrospective_lateral/tests/test_discrimination.py`

- [ ] **Step 1: Add failing tests for the slope and the classifier**

Append to `retrospective_lateral/tests/test_discrimination.py`:

```python
def test_frequency_speed_slope_flat_for_fixed_frequency():
    # Fixed ~0.20 Hz peak across a range of speeds -> near-zero slope -> flat.
    records = [{"speed_mph_median": s, "spectral_peak_hz": 0.20} for s in range(15, 65, 5)]
    res = D.frequency_speed_slope(records)
    assert res["flat"] is True
    assert abs(res["slope_hz_per_mph"]) < D.C.DISCRIM_FREQ_FLAT_HZ_PER_MPH


def test_frequency_speed_slope_not_flat_when_frequency_tracks_speed():
    records = [{"speed_mph_median": s, "spectral_peak_hz": 0.10 + 0.01 * s} for s in range(15, 65, 5)]
    res = D.frequency_speed_slope(records)
    assert res["flat"] is False


def test_classify_source_picks_road_artifact_loop_ambiguous():
    base = dict(model_residual_over_lane=0.1, model_vs_lane_corr=0.9,
                lane_vs_independent_corr=0.9, straight_clean=0, lane_curv_rms_1pm=1.0,
                model_curv_rms_1pm=1.0)
    # Road: reproducible by location
    assert D.classify_source(base, repro_fraction=0.8, freq_flat=False)[0] == "road_feature_B"
    # Artifact: model adds residual, not reproducible
    art = {**base, "model_residual_over_lane": 0.9, "model_vs_lane_corr": 0.3, "lane_vs_independent_corr": 0.3}
    assert D.classify_source(art, repro_fraction=float("nan"), freq_flat=False)[0] == "model_artifact_A"
    # Loop: fixed frequency on a straight clean road, not reproducible, no residual
    loop = {**base, "straight_clean": 1}
    assert D.classify_source(loop, repro_fraction=float("nan"), freq_flat=True)[0] == "loop_limit_cycle_C"
    # Ambiguous: nothing decisive
    amb = {**base, "model_vs_lane_corr": 0.4, "lane_vs_independent_corr": 0.4}
    assert D.classify_source(amb, repro_fraction=float("nan"), freq_flat=False)[0] == "ambiguous"
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: FAIL with `AttributeError: ... 'frequency_speed_slope'`.

- [ ] **Step 3: Implement the slope and the classifier**

Append to `retrospective_lateral/code/discrimination.py`:

```python
def frequency_speed_slope(records: list) -> dict:
    """OLS slope of spectral_peak_hz on speed_mph_median across windows.

    |slope| below the flat threshold => the weave frequency is speed-independent,
    which is consistent with a fixed time-constant loop limit-cycle, not a road wavelength.
    """
    pts = [(float(r["speed_mph_median"]), float(r["spectral_peak_hz"])) for r in records
           if np.isfinite(r.get("speed_mph_median", np.nan)) and np.isfinite(r.get("spectral_peak_hz", np.nan))]
    if len(pts) < 5:
        return {"n": len(pts), "slope_hz_per_mph": math.nan, "flat": False}
    speeds = np.array([p[0] for p in pts])
    peaks = np.array([p[1] for p in pts])
    if np.std(speeds) == 0:
        return {"n": len(pts), "slope_hz_per_mph": math.nan, "flat": False}
    slope = float(np.polyfit(speeds, peaks, 1)[0])
    return {"n": len(pts), "slope_hz_per_mph": slope,
            "flat": bool(abs(slope) < C.DISCRIM_FREQ_FLAT_HZ_PER_MPH)}


def classify_source(record: dict, repro_fraction: float, freq_flat: bool) -> tuple:
    """Combine the three tests into a single A/B/C/ambiguous source label.

    Priority: insufficient -> road (reproducible) -> artifact (model residual) ->
    loop (fixed-frequency on a straight clean road) -> ambiguous.
    """
    lane_rms = record.get("lane_curv_rms_1pm", math.nan)
    model_rms = record.get("model_curv_rms_1pm", math.nan)
    if not (np.isfinite(lane_rms) and np.isfinite(model_rms)):
        return ("insufficient_evidence", "lane/model curvature unavailable")

    residual_ratio = record.get("model_residual_over_lane", math.nan)
    mvl = abs(record.get("model_vs_lane_corr", 0.0) or 0.0)
    lvi = abs(record.get("lane_vs_independent_corr", 0.0) or 0.0)
    coh = C.DISCRIM_ROAD_COHERENCE_MIN
    moving_together = mvl >= coh and lvi >= coh

    if np.isfinite(repro_fraction) and repro_fraction >= C.DISCRIM_REPRO_FRACTION_ROAD and moving_together:
        return ("road_feature_B", f"reproducible by location (corr={repro_fraction:.2f}); planned/perceived/realized coherent")
    if np.isfinite(residual_ratio) and residual_ratio >= C.DISCRIM_ARTIFACT_RESIDUAL_RATIO and not moving_together:
        return ("model_artifact_A", f"model adds motion beyond lane/road (residual/lane={residual_ratio:.2f})")
    if freq_flat and int(record.get("straight_clean", 0)) == 1:
        return ("loop_limit_cycle_C", "fixed-frequency oscillation on a straight, clean, lead-free road")
    return ("ambiguous", "no single test decisive")
```

- [ ] **Step 4: Run to verify pass**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/discrimination.py retrospective_lateral/tests/test_discrimination.py
git commit -m "analysis: frequency-vs-speed slope + A/B/C episode classifier"
```

---

### Task 6: CLI runner over the real corpus

**Files:**
- Modify: `retrospective_lateral/code/discrimination.py`
- Test: `retrospective_lateral/tests/test_discrimination.py`

- [ ] **Step 1: Add a failing test for episode selection**

Append to `retrospective_lateral/tests/test_discrimination.py`:

```python
import pandas as pd


def test_select_top_episodes_ranks_and_caps_per_symptom():
    df = pd.DataFrame({
        "route_id": [f"route_{i}" for i in range(6)],
        "symptom": ["weave_10_70"] * 3 + ["low_speed_wheel_swing"] * 3,
        "status": ["ok"] * 6,
        "start_s": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "end_s": [40.0, 50.0, 60.0, 50.0, 60.0, 70.0],
        "peak_s": [25.0, 35.0, 45.0, 45.0, 55.0, 65.0],
        "speed_mph_median": [30.0, 40.0, 50.0, 3.0, 4.0, 5.0],
        "steering_peak_to_peak_deg": [np.nan, np.nan, np.nan, 12.0, 20.0, 8.0],
        "path_curvature_band_rms_1e4": [5.0, 9.0, 2.0, np.nan, np.nan, np.nan],
    })
    sel = D.select_top_episodes(df, top_n=2)
    weave = [e for e in sel if e["symptom"] == "weave_10_70"]
    low = [e for e in sel if e["symptom"] == "low_speed_wheel_swing"]
    assert len(weave) == 2 and len(low) == 2
    assert weave[0]["route_id"] == "route_1"   # highest path RMS (9.0)
    assert low[0]["route_id"] == "route_4"     # highest steering p2p (20.0)
```

- [ ] **Step 2: Run to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: FAIL with `AttributeError: ... 'select_top_episodes'`.

- [ ] **Step 3: Implement selection, the build runner, and the CLI**

Append to `retrospective_lateral/code/discrimination.py` (add `import argparse`, `import json`, `from dataclasses import asdict`, `from pathlib import Path`, and `import pandas as pd` to the top-of-file imports):

```python
def select_top_episodes(catalog: "pd.DataFrame", top_n: int = C.DISCRIM_TOP_N_PER_SYMPTOM) -> list:
    ok = catalog[catalog.get("status", "ok").astype(str) == "ok"] if "status" in catalog else catalog
    episodes: list[dict] = []
    for symptom, rank_col in (("weave_10_70", "path_curvature_band_rms_1e4"),
                              ("low_speed_wheel_swing", "steering_peak_to_peak_deg")):
        sub = ok[ok["symptom"] == symptom].copy()
        if rank_col in sub:
            sub = sub.sort_values(rank_col, ascending=False, na_position="last")
        for _, row in sub.head(top_n).iterrows():
            episodes.append({
                "symptom": symptom, "route_id": str(row["route_id"]),
                "start_s": float(row["start_s"]), "end_s": float(row["end_s"]),
                "peak_s": float(row.get("peak_s", row["start_s"])),
            })
    return episodes


def _load_cache(cache_root: Path, route_id: str):
    npz = cache_root / f"{route_id}.npz"
    if not npz.exists():
        return None
    with np.load(npz) as data:
        return {k: data[k] for k in data.files}


def build_discrimination_outputs(report_root: Path = C.DEFAULT_REPORT_ROOT,
                                 cache_root: Path = C.DEFAULT_CACHE_ROOT,
                                 top_n: int = C.DISCRIM_TOP_N_PER_SYMPTOM) -> dict:
    report_root.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(report_root / "symptom_catalog.csv")
    episodes = select_top_episodes(catalog, top_n=top_n)

    records: list[dict] = []
    profiles: dict[str, dict] = {}   # episode_label -> spatial profile
    arrays_cache: dict[str, dict] = {}
    for ep in episodes:
        rid = ep["route_id"]
        arrays = arrays_cache.get(rid) or _load_cache(cache_root, rid)
        if arrays is None:
            continue
        arrays_cache[rid] = arrays
        rec = discriminate_window(arrays, ep["symptom"], rid, ep["start_s"], ep["end_s"], ep["peak_s"])
        label = f"{rid}@{ep['peak_s']:.1f}"
        row = asdict(rec)
        row["episode_label"] = label
        records.append(row)
        profiles[label] = spatial_curvature_profile(arrays, ep["start_s"], ep["end_s"], _band_for(ep["symptom"]))

    records_df = pd.DataFrame(records)
    records_df.to_csv(report_root / "discrimination_window_records.csv", index=False)

    # Cross-pass reproducibility: every pair of episodes (different routes) sharing GPS cells.
    repro_rows: list[dict] = []
    best_repro: dict[str, float] = {}
    labels = list(profiles)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            if la.split("@")[0] == lb.split("@")[0]:
                continue  # same route is not an independent pass
            res = cross_pass_reproducibility([profiles[la], profiles[lb]], key="lane")
            if res["n_pairs"] == 0:
                continue
            repro_rows.append({"episode_a": la, "episode_b": lb, **res})
            for lab in (la, lb):
                prev = best_repro.get(lab, float("-inf"))
                if np.isfinite(res["median_pairwise_corr"]) and res["median_pairwise_corr"] > prev:
                    best_repro[lab] = res["median_pairwise_corr"]
    pd.DataFrame(repro_rows).to_csv(report_root / "discrimination_repeat_pass.csv", index=False)

    # Frequency-vs-speed per symptom.
    freq_rows = []
    for symptom in ("weave_10_70", "low_speed_wheel_swing"):
        sub = [r for r in records if r["symptom"] == symptom]
        freq_rows.append({"symptom": symptom, **frequency_speed_slope(sub)})
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(report_root / "discrimination_frequency_speed.csv", index=False)
    freq_flat_by_symptom = {r["symptom"]: bool(r["flat"]) for r in freq_rows}

    # Final per-episode classification.
    class_rows = []
    for r in records:
        repro = best_repro.get(r["episode_label"], float("nan"))
        label, why = classify_source(r, repro, freq_flat_by_symptom.get(r["symptom"], False))
        class_rows.append({"episode_label": r["episode_label"], "symptom": r["symptom"],
                           "route_id": r["route_id"], "speed_mph_median": r["speed_mph_median"],
                           "tentative_label": r["tentative_label"], "best_repro_corr": repro,
                           "source_label": label, "reason": why})
    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(report_root / "discrimination_episode_classification.csv", index=False)

    summary = (class_df.groupby(["symptom", "source_label"]).size()
               .reset_index(name="episodes") if len(class_df) else pd.DataFrame())
    summary.to_csv(report_root / "discrimination_summary.csv", index=False)

    return {"episodes": len(records), "repeat_pass_pairs": len(repro_rows),
            "classified": len(class_rows)}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline model/path-vs-road discrimination (analysis-only)")
    parser.add_argument("--report-root", type=Path, default=C.DEFAULT_REPORT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
    parser.add_argument("--top-n", type=int, default=C.DISCRIM_TOP_N_PER_SYMPTOM)
    args = parser.parse_args(argv)
    result = build_discrimination_outputs(args.report_root, args.cache_root, args.top_n)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run to verify the unit test passes**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_discrimination.py -q
```

Expected: PASS (16 tests).

- [ ] **Step 5: Smoke-run the CLI on the real corpus**

Run:

```bash
.venv311/bin/python -m retrospective_lateral.code.discrimination --top-n 40
```

Expected: a JSON line like `{"classified": <N>, "episodes": <N>, "repeat_pass_pairs": <M>}` with `episodes > 0`, and five new `discrimination_*.csv` files under `retrospective_lateral/results/reports/`.

- [ ] **Step 6: Commit**

```bash
git add retrospective_lateral/code/discrimination.py retrospective_lateral/tests/test_discrimination.py
git commit -m "analysis: discrimination CLI runner over the corpus"
```

---

### Task 7: QA verification + focused report

**Files:**
- Create: `docs/superpowers/reports/2026-06-24-retrospective-lateral-model-vs-road-discrimination.md`
- (No code changes; verification + write-up only.)

- [ ] **Step 1: Run the full retrospective test suite (no regressions)**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests -q
```

Expected: all tests pass, including the 16 new `test_discrimination.py` tests.

- [ ] **Step 2: Verify outputs, row counts, and that nothing forbidden changed**

Run:

```bash
.venv311/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
base = Path("retrospective_lateral/results/reports")
for f in ["discrimination_window_records.csv", "discrimination_repeat_pass.csv",
          "discrimination_frequency_speed.csv", "discrimination_episode_classification.csv",
          "discrimination_summary.csv"]:
    df = pd.read_csv(base / f)
    print(f, "rows=", len(df), "cols=", len(df.columns))
cls = pd.read_csv(base / "discrimination_episode_classification.csv")
print(cls.groupby(["symptom", "source_label"]).size())
PY

git check-ignore retrospective_lateral/results/reports/discrimination_summary.csv
git status --short opendbc_repo panda
git status --short retrospective_lateral/results
```

Expected: each CSV has `rows > 0`; the classification breakdown prints A/B/C/ambiguous counts; `git check-ignore` echoes the path (ignored); `git status --short opendbc_repo panda` is empty; `git status --short retrospective_lateral/results` is empty (results stay ignored/unstaged).

- [ ] **Step 3: Write the focused findings report**

Create `docs/superpowers/reports/2026-06-24-retrospective-lateral-model-vs-road-discrimination.md` with these sections, filled in from the Step 2 output (replace each bracket with the actual number/finding — do not leave brackets):

```markdown
# 2026-06-24 Model/Path-vs-Road Discrimination Findings

## Executive Summary
Classified the top [N] weave and [N] low-speed episodes as model-artifact (A),
road-feature (B), loop-limit-cycle (C), or ambiguous, using existing caches only.
Result counts per symptom: [paste discrimination_summary.csv].
Recommendation: [do not change driving code yet | ready for a model/path change plan
| ready for a loop-timing investigation], per the decision rule below.

## Method
- Common curvature axis via offset_to_curvature(y, L=20 m).
- Independent realized-path references: CAN yaw, calibrated yaw, GPS-heading rate.
- Three tests: within-window coherence/residual; cross-pass spatial reproducibility;
  spectral-peak-vs-speed slope.

## Results
- Window records: [count] ; repeat-pass pairs: [count].
- Frequency-vs-speed slope: weave [slope] Hz/mph (flat=[bool]); low-speed [slope].
- Per-episode classification table: [summarize A/B/C/ambiguous, name notable episodes].

## Decision (gate from the review report)
- If a clear majority classify B (road feature) and reproduce by location ->
  correction layer is lane-centering target/filtering; plan a model/path change A/B.
- If a clear majority classify A (model artifact) -> correction layer is model
  selection / model-side path smoothing; plan an offline replay of a candidate model.
- If C (loop limit-cycle) dominates on straight clean roads -> investigate the
  fixed-period loop element (delay/filter/update cadence); still upstream of PI gains.
- If ambiguous dominates -> the smallest on-road controlled drive (PI fixed at golden,
  vary one upstream factor) is justified; follow AGENTS.md offroad-safe workflow.

## Confounders / limits
- GPS-heading curvature is coarse; it corroborates, it does not arbitrate alone.
- offset_to_curvature is a small-angle approximation.
- Repeat-pass needs >= [DISCRIM_REPRO_MIN_SHARED_CELLS] shared GPS cells; single-pass
  corridors get NaN reproducibility and cannot be classified B.
- route_67/route_68 have no NPZ and are absent from the corpus.
- No vehicle-control code, opendbc_repo/, or panda/ was touched; all outputs are gitignored.

## Commands Run
[paste the exact commands from Steps 1-2 and their key results]
```

- [ ] **Step 4: Commit the report**

```bash
git add docs/superpowers/reports/2026-06-24-retrospective-lateral-model-vs-road-discrimination.md
git commit -m "docs: model-vs-road discrimination findings report"
```

---

## Self-Review

**1. Spec coverage** (against the four offline steps in §7.1 of the review report):
- Model/lane/path overlay discrimination (model-artifact vs faithful-follow) → Task 3 (`discriminate_window` coherence + residual) + Task 5 classifier. ✓
- Same-corridor repeat-pass decomposition → Task 4 (`spatial_curvature_profile`, `cross_pass_reproducibility`) + Task 6 pairing. ✓
- Independent road-truth references (CAN yaw / calibrated yaw / GPS heading) → Task 2. ✓
- Fixed-frequency / loop probe → Task 5 (`frequency_speed_slope`) + classifier C branch. ✓
- Low-speed short-lookahead refinement: the lookahead is a parameter (`DISCRIM_LOOKAHEAD_M`, default 20 m); per-packet horizon extension is **out of scope** for this plan and is left to the existing `model_horizon` drilldown — noted here so it is not mistaken for a gap.

**2. Placeholder scan:** every code step contains complete, runnable code; the report template (Task 7 Step 3) explicitly instructs the executor to replace each bracket with a real value and forbids leaving brackets. No "TODO"/"handle edge cases"/"similar to Task N" remain.

**3. Type consistency:** `DiscriminationWindow` field names defined in Task 3 are the exact keys consumed by `classify_source` (Task 5: `lane_curv_rms_1pm`, `model_curv_rms_1pm`, `model_residual_over_lane`, `model_vs_lane_corr`, `lane_vs_independent_corr`, `straight_clean`) and serialized by `asdict` in Task 6. `cross_pass_reproducibility` returns `median_pairwise_corr`/`n_pairs`/`reproducible`, all read back in Task 6. `frequency_speed_slope` returns `flat`/`slope_hz_per_mph`, read in Task 6 and tested in Task 5. `select_top_episodes` emits dicts with `symptom/route_id/start_s/end_s/peak_s`, consumed by the Task 6 loop. Channel keys (`model_y20`, `lane_center_y20`, `road_edge_left_y20`, `road_edge_right_y20`, `yaw_rate`, `yaw_rate_calibrated`, `lat`, `lon`, `desired_curvature`, `cp_final_command`, `lead_time_headway_s`, `lane_prob_left/right`) all exist in the verified 137-channel cache schema. Config constants referenced (`DISCRIM_*`, `MPS_TO_MPH`, `LOW_SPEED_INSPECT_BAND_HZ`, `DEFAULT_WEAVE_BAND_HZ`, `ROAD_CURV_ABS_MAX_1PM`, `ROAD_LP_HZ`, `GPS_CELL_M`, `HEADING_BIN_DEG`, `SPEED_BIN_MPH`, `LEAD_HEADWAY_S`, `OVERRIDE_BUFFER_S`, `ENGAGE_ERODE_S`) are all defined (Task 1 additions or pre-existing).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-24-retrospective-lateral-model-vs-road-discrimination.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
