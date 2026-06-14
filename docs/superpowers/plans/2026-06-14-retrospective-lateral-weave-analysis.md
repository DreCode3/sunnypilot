# Retrospective Lateral Weave Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible existing-log analysis pipeline that detects, localizes, compares, and reports the 2021 Ford Explorer ST straight/gentle-section weave symptoms.

**Architecture:** Create a new `retrospective_lateral/` analysis package rather than extending older one-off scripts. The package has focused modules for route discovery, signal utilities, telemetry parsing, extraction/cache writing, symptom metrics, matched comparisons, reporting, and QA checks. Outputs are cached per route and summarized through CLI commands so the full corpus can be reprocessed incrementally.

**Tech Stack:** Python 3.11 in `.venv311`, NumPy, SciPy, pandas, zstandard/Cap'n Proto through openpilot `LogReader`, pytest, CSV/JSON/NPZ artifacts, multiprocessing via `concurrent.futures.ProcessPoolExecutor`.

---

## Scope Check

The approved spec covers one coherent subsystem: a retrospective lateral-analysis pipeline. It contains multiple modules, but they produce one testable workflow and should remain one implementation plan. Do not implement vehicle-control changes in this plan.

## File Structure

Create a new root-level analysis package:

- `retrospective_lateral/README.md`: how to run extraction, analysis, reports, and QA.
- `retrospective_lateral/code/__init__.py`: package marker and version.
- `retrospective_lateral/code/config.py`: constants, default paths, bands, thresholds, schema versions.
- `retrospective_lateral/code/routes.py`: local route discovery across old/new layouts.
- `retrospective_lateral/code/signal_utils.py`: filtering, masked metrics, flag erosion/dilation, GPS cell helpers, contiguous-window helpers.
- `retrospective_lateral/code/telemetry.py`: `CP:`, `CX1:`, `LC:` parsing and config-confidence recovery.
- `retrospective_lateral/code/extract.py`: rlog reading, signal collection, resampling, cache writing, extraction CLI.
- `retrospective_lateral/code/metrics.py`: low-speed wheel-swing and 10-70 mph weave window/episode metrics plus stage localization.
- `retrospective_lateral/code/compare.py`: speed/location/evidence-tier matching and historical scorecard comparisons.
- `retrospective_lateral/code/report.py`: CSV/JSON/Markdown report writer and next-experiment summary.
- `retrospective_lateral/code/qa.py`: built-in audit checks that support cooperative/adversarial QA.
- `retrospective_lateral/tests/`: pytest unit tests with synthetic data.
- `retrospective_lateral/results/`: generated artifacts; keep this directory gitignored by adding a local `.gitignore` inside it.
- `retrospective_lateral/qa/`: tracked cooperative/adversarial QA notes and reconciliation.

Modify:

- `requirements-analysis.txt`: add `pytest==9.0.2` for reproducible TDD in the analysis venv.

Do not modify `opendbc_repo/`, `panda/`, or any vehicle-control file in this plan.

---

### Task 1: Scaffold Package And Reproducible Test Dependency

**Files:**
- Modify: `requirements-analysis.txt`
- Create: `retrospective_lateral/README.md`
- Create: `retrospective_lateral/results/.gitignore`
- Create: `retrospective_lateral/code/__init__.py`
- Create: `retrospective_lateral/code/config.py`
- Test: `retrospective_lateral/tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

Create `retrospective_lateral/tests/test_config.py`:

```python
from pathlib import Path

from retrospective_lateral.code import config as C


def test_default_paths_are_repo_relative():
  assert C.REPO_ROOT.name == "sunnypilot"
  assert C.DEFAULT_LOG_ROOT == C.REPO_ROOT / "explorer_st_logs"
  assert C.DEFAULT_CACHE_ROOT == C.REPO_ROOT / "retrospective_lateral" / "results" / "cache"
  assert C.DEFAULT_REPORT_ROOT == C.REPO_ROOT / "retrospective_lateral" / "results" / "reports"


def test_schema_and_metric_constants_are_explicit():
  assert isinstance(C.CACHE_SCHEMA_VERSION, str)
  assert C.CACHE_SCHEMA_VERSION.startswith("retrolat-v")
  assert C.FS_HZ == 20.0
  assert C.LOW_SPEED_MPH == (1.0, 10.0)
  assert C.WEAVE_SPEED_MPH == (10.0, 70.0)
  assert C.DEFAULT_WEAVE_BAND_HZ == (0.10, 0.35)
  assert C.HUNT_GUARD_BAND_HZ == (0.50, 1.50)
  assert C.MAX_INTERP_GAP_S == 0.5
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'retrospective_lateral'`.

- [ ] **Step 3: Add pytest to the analysis requirements**

Append this exact line to `requirements-analysis.txt` after `tqdm==4.67.3`:

```text
pytest==9.0.2
```

- [ ] **Step 4: Create the scaffold files**

Create `retrospective_lateral/code/__init__.py`:

```python
"""Retrospective lateral weave analysis package for the Explorer ST logs."""

PACKAGE_VERSION = "0.1.0"
```

Create `retrospective_lateral/code/config.py`:

```python
from __future__ import annotations

from pathlib import Path

CACHE_SCHEMA_VERSION = "retrolat-v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "explorer_st_logs"
DEFAULT_CACHE_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "cache"
DEFAULT_REPORT_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "reports"

FS_HZ = 20.0
MAX_INTERP_GAP_S = 0.5

LOW_SPEED_MPH = (1.0, 10.0)
WEAVE_SPEED_MPH = (10.0, 70.0)
DEFAULT_WEAVE_BAND_HZ = (0.10, 0.35)
LOW_SPEED_INSPECT_BAND_HZ = (0.08, 0.80)
HUNT_GUARD_BAND_HZ = (0.50, 1.50)
ROAD_LP_HZ = 0.035
ROAD_CURV_ABS_MAX_1PM = 0.0015

ENGAGE_ERODE_S = 2.0
OVERRIDE_BUFFER_S = 1.0
BLINKER_BUFFER_S = 1.0
LANE_CHANGE_BUFFER_S = 1.0
LEAD_BUFFER_S = 2.0
LEAD_HEADWAY_S = 1.6

MIN_LOW_SPEED_WINDOW_S = 6.0
LOW_SPEED_WINDOW_S = 10.0
WEAVE_WINDOW_S = 30.0
MIN_WEAVE_ELIGIBLE_S = 20.0

GPS_CELL_M = 80.0
SPEED_BIN_MPH = 2.5
HEADING_BIN_DEG = 45.0

M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON_AT_EQUATOR = 111_320.0
```

Create `retrospective_lateral/results/.gitignore`:

```gitignore
*
!.gitignore
```

Create `retrospective_lateral/README.md`:

```markdown
# Retrospective Lateral Weave Analysis

This package analyzes existing Explorer ST openpilot/sunnypilot logs for two symptoms:

- Low-speed steering-wheel swing at 1-10 mph.
- Straight/gentle-section path and wheel weave at 10-70 mph.

Run commands from the repository root with `.venv311/bin/python`.

The package is analysis-only. It must not modify vehicle-control code.
```

- [ ] **Step 5: Run the test to verify it passes**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements-analysis.txt retrospective_lateral/README.md retrospective_lateral/results/.gitignore retrospective_lateral/code/__init__.py retrospective_lateral/code/config.py retrospective_lateral/tests/test_config.py
git commit -m "analysis: scaffold retrospective lateral package"
```

---

### Task 2: Route Discovery Across Local Log Layouts

**Files:**
- Create: `retrospective_lateral/code/routes.py`
- Test: `retrospective_lateral/tests/test_routes.py`

- [ ] **Step 1: Write route-discovery tests**

Create `retrospective_lateral/tests/test_routes.py`:

```python
from pathlib import Path

from retrospective_lateral.code.routes import discover_routes, segment_index_from_path


def touch(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"")


def test_segment_index_from_modern_segment_dir():
  path = Path("explorer_st_logs/route_b8/000000b8--56af909567--31/rlog.zst")
  assert segment_index_from_path(path) == 31


def test_segment_index_from_flat_segment_file():
  path = Path("explorer_st_logs/route_49/rlog_3.zst")
  assert segment_index_from_path(path) == 3


def test_discover_routes_finds_modern_and_flat_layouts(tmp_path):
  touch(tmp_path / "route_b8" / "000000b8--56af909567--0" / "rlog.zst")
  touch(tmp_path / "route_b8" / "000000b8--56af909567--1" / "rlog.zst")
  touch(tmp_path / "route_49" / "rlog_0.zst")
  touch(tmp_path / "route_49" / "rlog_1.zst")
  routes = discover_routes(tmp_path)
  by_id = {r.route_id: r for r in routes}
  assert sorted(by_id) == ["route_49", "route_b8"]
  assert [s.segment_index for s in by_id["route_b8"].segments] == [0, 1]
  assert [s.segment_index for s in by_id["route_49"].segments] == [0, 1]
  assert by_id["route_b8"].layout == "modern"
  assert by_id["route_49"].layout == "flat"
```

- [ ] **Step 2: Run the tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_routes.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `retrospective_lateral.code.routes`.

- [ ] **Step 3: Implement route discovery**

Create `retrospective_lateral/code/routes.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentRef:
  route_id: str
  segment_index: int
  rlog_path: Path
  qlog_path: Path | None = None


@dataclass(frozen=True)
class RouteRef:
  route_id: str
  route_dir: Path
  layout: str
  segments: tuple[SegmentRef, ...]


def segment_index_from_path(path: Path) -> int:
  text = str(path)
  modern = re.search(r"--(\d+)/rlog\.zst$", text)
  if modern:
    return int(modern.group(1))
  flat = re.search(r"rlog_(\d+)\.zst$", text)
  if flat:
    return int(flat.group(1))
  plain = re.search(r"/(\d+)/rlog\.zst$", text)
  if plain:
    return int(plain.group(1))
  return 0


def _route_id_from_dir(route_dir: Path) -> str:
  return route_dir.name


def _modern_segments(route_dir: Path, route_id: str) -> list[SegmentRef]:
  out: list[SegmentRef] = []
  for rlog in route_dir.glob("000000*--*/rlog.zst"):
    out.append(SegmentRef(route_id=route_id, segment_index=segment_index_from_path(rlog), rlog_path=rlog))
  return sorted(out, key=lambda s: s.segment_index)


def _flat_segments(route_dir: Path, route_id: str) -> list[SegmentRef]:
  out: list[SegmentRef] = []
  for rlog in route_dir.glob("rlog_*.zst"):
    out.append(SegmentRef(route_id=route_id, segment_index=segment_index_from_path(rlog), rlog_path=rlog))
  if (route_dir / "rlog.zst").exists():
    out.append(SegmentRef(route_id=route_id, segment_index=0, rlog_path=route_dir / "rlog.zst"))
  return sorted(out, key=lambda s: s.segment_index)


def discover_routes(root: Path) -> list[RouteRef]:
  routes: list[RouteRef] = []
  for route_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("route_")):
    route_id = _route_id_from_dir(route_dir)
    modern = _modern_segments(route_dir, route_id)
    flat = _flat_segments(route_dir, route_id)
    if modern:
      routes.append(RouteRef(route_id=route_id, route_dir=route_dir, layout="modern", segments=tuple(modern)))
    elif flat:
      routes.append(RouteRef(route_id=route_id, route_dir=route_dir, layout="flat", segments=tuple(flat)))
  return routes
```

- [ ] **Step 4: Run route tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_routes.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/routes.py retrospective_lateral/tests/test_routes.py
git commit -m "analysis: discover retrospective route layouts"
```

---

### Task 3: Signal Utilities With Synthetic Tests

**Files:**
- Create: `retrospective_lateral/code/signal_utils.py`
- Test: `retrospective_lateral/tests/test_signal_utils.py`

- [ ] **Step 1: Write signal utility tests**

Create `retrospective_lateral/tests/test_signal_utils.py`:

```python
import numpy as np

from retrospective_lateral.code import signal_utils as S


def test_fill_guarded_keeps_long_gaps_nan():
  x = np.array([0.0, 1.0, np.nan, np.nan, np.nan, 5.0])
  y = S.fill_guarded(x, max_gap_samples=1)
  assert np.isfinite(y[2])
  assert np.isnan(y[3])
  assert np.isfinite(y[4])


def test_rms_and_peak_to_peak_ignore_nan():
  x = np.array([1.0, -1.0, np.nan, 1.0, -1.0])
  mask = np.array([True, True, True, False, False])
  assert S.rms_masked(x, mask) == 1.0
  assert S.peak_to_peak_masked(x, mask) == 2.0


def test_dilate_and_erode_boolean_flags():
  flags = np.array([False, False, True, False, False])
  assert S.dilate_flags(flags, radius=1).tolist() == [False, True, True, True, False]
  assert S.erode_true(np.ones(5, dtype=bool), radius=1).tolist() == [False, True, True, True, False]


def test_contiguous_regions_filters_short_runs():
  mask = np.array([False, True, True, False, True, True, True])
  assert S.contiguous_regions(mask, min_len=3) == [(4, 7)]


def test_gps_cell_and_heading_bin_are_stable():
  lat = np.array([34.0, 34.0001])
  lon = np.array([-84.0, -84.0001])
  cells = S.gps_cells(lat, lon, cell_m=80.0)
  assert cells.shape == (2,)
  assert np.isfinite(cells).all()
  assert S.heading_bin_deg(np.array([1.0, 44.0, 46.0]), bin_deg=45.0).tolist() == [0, 0, 1]
```

- [ ] **Step 2: Run the tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_signal_utils.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `signal_utils`.

- [ ] **Step 3: Implement signal utilities**

Create `retrospective_lateral/code/signal_utils.py`:

```python
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch

from retrospective_lateral.code import config as C


def fill_guarded(x: np.ndarray, max_gap_samples: int) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  ok = np.isfinite(x)
  if ok.sum() < 2:
    return np.full_like(x, np.nan, dtype=float)
  idx = np.arange(len(x))
  out = np.interp(idx, idx[ok], x[ok])
  left = np.searchsorted(idx[ok], idx, side="right") - 1
  right = left + 1
  left = np.clip(left, 0, ok.sum() - 1)
  right = np.clip(right, 0, ok.sum() - 1)
  nearest_dist = np.minimum(np.abs(idx - idx[ok][left]), np.abs(idx - idx[ok][right]))
  out[nearest_dist > max_gap_samples] = np.nan
  return out


def _sos_band(lo_hz: float, hi_hz: float, fs_hz: float) -> np.ndarray:
  return butter(3, [lo_hz, hi_hz], btype="band", fs=fs_hz, output="sos")


def _sos_low(cutoff_hz: float, fs_hz: float) -> np.ndarray:
  return butter(2, cutoff_hz, btype="low", fs=fs_hz, output="sos")


def filter_continuous(x: np.ndarray, fs_hz: float, *, band: tuple[float, float] | None = None,
                      lowpass_hz: float | None = None, max_gap_s: float = C.MAX_INTERP_GAP_S) -> np.ndarray:
  max_gap_samples = max(1, int(round(max_gap_s * fs_hz)))
  filled = fill_guarded(np.asarray(x, dtype=float), max_gap_samples=max_gap_samples)
  good = np.isfinite(filled)
  if good.sum() < max(12, int(fs_hz * 3)):
    return np.full(len(filled), np.nan)
  src = np.where(good, filled, 0.0)
  if band is not None:
    y = sosfiltfilt(_sos_band(band[0], band[1], fs_hz), src)
  elif lowpass_hz is not None:
    y = sosfiltfilt(_sos_low(lowpass_hz, fs_hz), src)
  else:
    raise ValueError("band or lowpass_hz is required")
  out = np.full(len(filled), np.nan)
  out[good] = y[good]
  return out


def rms_masked(x: np.ndarray, mask: np.ndarray) -> float:
  vals = np.asarray(x, dtype=float)[np.asarray(mask, dtype=bool)]
  vals = vals[np.isfinite(vals)]
  if len(vals) == 0:
    return math.nan
  return float(np.sqrt(np.mean(vals ** 2)))


def peak_to_peak_masked(x: np.ndarray, mask: np.ndarray) -> float:
  vals = np.asarray(x, dtype=float)[np.asarray(mask, dtype=bool)]
  vals = vals[np.isfinite(vals)]
  if len(vals) == 0:
    return math.nan
  return float(np.nanmax(vals) - np.nanmin(vals))


def dilate_flags(flags: np.ndarray, radius: int) -> np.ndarray:
  flags = np.asarray(flags, dtype=bool)
  if radius <= 0:
    return flags.copy()
  return uniform_filter1d(flags.astype(float), 2 * radius + 1, mode="constant", cval=0.0) > 0


def erode_true(flags: np.ndarray, radius: int) -> np.ndarray:
  flags = np.asarray(flags, dtype=bool)
  if radius <= 0:
    return flags.copy()
  return uniform_filter1d(flags.astype(float), 2 * radius + 1, mode="constant", cval=0.0) >= 1.0


def contiguous_regions(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
  mask = np.asarray(mask, dtype=bool)
  padded = np.r_[False, mask, False]
  changes = np.flatnonzero(padded[1:] != padded[:-1])
  regions = [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]
  return [(a, b) for a, b in regions if b - a >= min_len]


def gps_cells(lat_deg: np.ndarray, lon_deg: np.ndarray, cell_m: float) -> np.ndarray:
  lat = np.asarray(lat_deg, dtype=float)
  lon = np.asarray(lon_deg, dtype=float)
  out = np.full(lat.shape, np.nan)
  ok = np.isfinite(lat) & np.isfinite(lon)
  if ok.sum() == 0:
    return out
  lat0 = float(np.nanmedian(lat[ok]))
  x = lon * C.M_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat0))
  y = lat * C.M_PER_DEG_LAT
  gx = np.floor(x / cell_m)
  gy = np.floor(y / cell_m)
  out[ok] = gx[ok] * 10_000_000 + gy[ok]
  return out


def heading_bin_deg(heading_deg: np.ndarray, bin_deg: float) -> np.ndarray:
  heading = np.asarray(heading_deg, dtype=float)
  out = np.full(heading.shape, -1, dtype=int)
  ok = np.isfinite(heading)
  out[ok] = np.floor((heading[ok] % 360.0) / bin_deg).astype(int)
  return out


def spectral_peak_hz(x: np.ndarray, fs_hz: float, band: tuple[float, float]) -> float:
  vals = np.asarray(x, dtype=float)
  vals = vals[np.isfinite(vals)]
  if len(vals) < int(fs_hz * 10):
    return math.nan
  freqs, power = welch(vals - np.mean(vals), fs=fs_hz, nperseg=min(2048, len(vals)))
  band_mask = (freqs >= band[0]) & (freqs <= band[1])
  if not band_mask.any() or np.nanmax(power[band_mask]) <= 0:
    return math.nan
  return float(freqs[band_mask][np.argmax(power[band_mask])])
```

- [ ] **Step 4: Run signal utility tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_signal_utils.py -q
```

Expected: PASS.

- [ ] **Step 5: Run all current retrospective tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add retrospective_lateral/code/signal_utils.py retrospective_lateral/tests/test_signal_utils.py
git commit -m "analysis: add retrospective signal utilities"
```

---

### Task 4: Telemetry Parsers And Config Confidence

**Files:**
- Create: `retrospective_lateral/code/telemetry.py`
- Test: `retrospective_lateral/tests/test_telemetry.py`

- [ ] **Step 1: Write telemetry parser tests**

Create `retrospective_lateral/tests/test_telemetry.py`:

```python
import math

from retrospective_lateral.code.telemetry import (
  ConfigEvidence,
  parse_cp_line,
  parse_cx1_line,
  parse_lc_line,
  recover_pi_config,
)


def test_parse_lc_line_extracts_controller_fields():
  line = "LC: off=0.120 ll=0.100 pos=0.0200 scl=1.00 conf=0.90 wid=3.60 int=0.4000 P=0.000060 I=0.000080 curv=0.001234 spd=24.0"
  row = parse_lc_line(line, t=123.0)
  assert row is not None
  assert row.t == 123.0
  assert row.offset_m == 0.120
  assert row.integral == 0.4000
  assert row.p_term == 0.000060
  assert row.i_term == 0.000080
  assert row.speed_mps == 24.0


def test_parse_cp_line_extracts_pipeline_fields():
  line = "CP: des=0.001000 pred=0.002000 ema=0.001200 preRL=0.001250 RL=0.001240 send=0.001240 meas=0.001100 | ovr=0 rst=0 ramp=0 rlClip=0 aw=0 | ang=1.0 tq=0.02"
  row = parse_cp_line(line, t=3.0)
  assert row is not None
  assert row.desired_curvature == 0.001
  assert row.predicted_curvature == 0.002
  assert row.final_command == 0.00124
  assert row.override == 0


def test_parse_cx1_line_extracts_schema_v1_fields():
  line = "CX1: 100 12.00 +0.01000 +0.120 +0.0010000 +0.0001000 +0.001100 +0.000900 +0.001200 +0.001000 +0.001050 +0.001000 950 4090 +3.00 +1.00 +0.20 0 0 0.500 0.300 1.000 +0.020 +0.3000 +0.000300 0 0 0.0400 0"
  row = parse_cx1_line(line)
  assert row is not None
  assert row.frame == 100
  assert row.speed_mps == 12.0
  assert row.command_curvature == 0.001
  assert row.path4_enabled == 0


def test_recover_pi_config_from_lc_terms():
  evidence = recover_pi_config(
    offsets=[0.10, -0.20, 0.12],
    p_terms=[0.00005, -0.00010, 0.00006],
    integrals=[0.50, -0.30, 0.40],
    i_terms=[0.00010, -0.00006, 0.00008],
  )
  assert isinstance(evidence, ConfigEvidence)
  assert evidence.pi_set == "golden"
  assert evidence.confidence == "proven"
  assert math.isclose(evidence.lc_kp, 0.0005, rel_tol=1e-6)
  assert math.isclose(evidence.lc_ki, 0.0002, rel_tol=1e-6)
```

- [ ] **Step 2: Run telemetry tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_telemetry.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `telemetry`.

- [ ] **Step 3: Implement telemetry parsing**

Create `retrospective_lateral/code/telemetry.py` with these dataclasses and parser functions:

```python
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class LCTelemetry:
  t: float
  offset_m: float
  lane_line_offset_m: float
  path_position_m: float
  lane_scale: float
  confidence: float
  lane_width_m: float
  integral: float
  p_term: float
  i_term: float
  curvature: float
  speed_mps: float


@dataclass(frozen=True)
class CPTelemetry:
  t: float
  desired_curvature: float
  predicted_curvature: float
  ema_curvature: float
  pre_rate_limit: float
  rate_limited: float
  final_command: float
  measured_curvature: float
  override: int
  reset: int
  ramp: int
  rate_limited_flag: int
  anti_windup: int
  steering_angle_deg: float
  steering_torque: float


@dataclass(frozen=True)
class CX1Telemetry:
  frame: int
  speed_mps: float
  yaw_rate: float
  lateral_accel: float
  command_curvature: float
  command_rate: float
  measured_curvature: float
  desired_curvature: float
  predicted_curvature: float
  ema_curvature: float
  pre_rate_limit: float
  rate_limited: float
  command_int: int
  rate_int: int
  steering_angle_deg: float
  steering_rate_deg_s: float
  steering_torque: float
  override: int
  lane_change: int
  lookup_time_s: float
  blend: float
  curvature_factor: float
  lane_offset_m: float
  integral: float
  pred_minus_des: float
  burst: int
  path4_release: int
  smooth_tau_s: float
  path4_enabled: int


@dataclass(frozen=True)
class ConfigEvidence:
  pi_set: str
  confidence: str
  lc_kp: float | None
  lc_ki: float | None
  n_samples: int
  reason: str


LC_RE = re.compile(
  r"LC: off=(?P<off>[-+\d.]+) ll=(?P<ll>[-+\d.]+) pos=(?P<pos>[-+\d.]+) "
  r"scl=(?P<scl>[-+\d.]+) conf=(?P<conf>[-+\d.]+) wid=(?P<wid>[-+\d.]+) "
  r"int=(?P<int>[-+\d.]+) P=(?P<P>[-+\deE.]+) I=(?P<I>[-+\deE.]+) "
  r"curv=(?P<curv>[-+\d.]+) spd=(?P<spd>[-+\d.]+)"
)

CP_RE = re.compile(
  r"CP: des=(?P<des>[-+\d.]+) pred=(?P<pred>[-+\d.]+) ema=(?P<ema>[-+\d.]+) "
  r"preRL=(?P<pre>[-+\d.]+) RL=(?P<rl>[-+\d.]+) send=(?P<send>[-+\d.]+) meas=(?P<meas>[-+\d.]+) "
  r"\| ovr=(?P<ovr>\d+) rst=(?P<rst>\d+) ramp=(?P<ramp>\d+) rlClip=(?P<clip>\d+) aw=(?P<aw>\d+) "
  r"\| ang=(?P<ang>[-+\d.]+) tq=(?P<tq>[-+\d.]+)"
)


def _float_group(match: re.Match[str], name: str) -> float:
  return float(match.group(name))


def parse_lc_line(text: str, t: float) -> LCTelemetry | None:
  match = LC_RE.search(text)
  if match is None:
    return None
  return LCTelemetry(
    t=t,
    offset_m=_float_group(match, "off"),
    lane_line_offset_m=_float_group(match, "ll"),
    path_position_m=_float_group(match, "pos"),
    lane_scale=_float_group(match, "scl"),
    confidence=_float_group(match, "conf"),
    lane_width_m=_float_group(match, "wid"),
    integral=_float_group(match, "int"),
    p_term=_float_group(match, "P"),
    i_term=_float_group(match, "I"),
    curvature=_float_group(match, "curv"),
    speed_mps=_float_group(match, "spd"),
  )


def parse_cp_line(text: str, t: float) -> CPTelemetry | None:
  match = CP_RE.search(text)
  if match is None:
    return None
  return CPTelemetry(
    t=t,
    desired_curvature=_float_group(match, "des"),
    predicted_curvature=_float_group(match, "pred"),
    ema_curvature=_float_group(match, "ema"),
    pre_rate_limit=_float_group(match, "pre"),
    rate_limited=_float_group(match, "rl"),
    final_command=_float_group(match, "send"),
    measured_curvature=_float_group(match, "meas"),
    override=int(match.group("ovr")),
    reset=int(match.group("rst")),
    ramp=int(match.group("ramp")),
    rate_limited_flag=int(match.group("clip")),
    anti_windup=int(match.group("aw")),
    steering_angle_deg=_float_group(match, "ang"),
    steering_torque=_float_group(match, "tq"),
  )


def parse_cx1_line(text: str) -> CX1Telemetry | None:
  if not text.startswith("CX1: ") or "SCHEMA=" in text:
    return None
  parts = text.split()[1:]
  if len(parts) != 29:
    return None
  return CX1Telemetry(
    frame=int(parts[0]),
    speed_mps=float(parts[1]),
    yaw_rate=float(parts[2]),
    lateral_accel=float(parts[3]),
    command_curvature=float(parts[4]),
    command_rate=float(parts[5]),
    measured_curvature=float(parts[6]),
    desired_curvature=float(parts[7]),
    predicted_curvature=float(parts[8]),
    ema_curvature=float(parts[9]),
    pre_rate_limit=float(parts[10]),
    rate_limited=float(parts[11]),
    command_int=int(parts[12]),
    rate_int=int(parts[13]),
    steering_angle_deg=float(parts[14]),
    steering_rate_deg_s=float(parts[15]),
    steering_torque=float(parts[16]),
    override=int(parts[17]),
    lane_change=int(parts[18]),
    lookup_time_s=float(parts[19]),
    blend=float(parts[20]),
    curvature_factor=float(parts[21]),
    lane_offset_m=float(parts[22]),
    integral=float(parts[23]),
    pred_minus_des=float(parts[24]),
    burst=int(parts[25]),
    path4_release=int(parts[26]),
    smooth_tau_s=float(parts[27]),
    path4_enabled=int(parts[28]),
  )


def _ratios(num: Sequence[float], den: Sequence[float], min_abs_den: float) -> list[float]:
  out: list[float] = []
  for n, d in zip(num, den, strict=False):
    if math.isfinite(n) and math.isfinite(d) and abs(d) >= min_abs_den:
      out.append(float(n) / float(d))
  return out


def recover_pi_config(offsets: Sequence[float], p_terms: Sequence[float],
                      integrals: Sequence[float], i_terms: Sequence[float]) -> ConfigEvidence:
  kp_ratios = _ratios(p_terms, offsets, min_abs_den=0.02)
  ki_ratios = _ratios(i_terms, integrals, min_abs_den=0.02)
  lc_kp = median(kp_ratios) if len(kp_ratios) >= 3 else None
  lc_ki = median(ki_ratios) if len(ki_ratios) >= 3 else None
  n = min(len(kp_ratios), len(ki_ratios))
  if lc_kp is None:
    return ConfigEvidence("unknown", "unknown", None, lc_ki, n, "insufficient LC P/off samples")
  if lc_kp >= 0.0003:
    return ConfigEvidence("golden", "proven", float(lc_kp), float(lc_ki) if lc_ki is not None else None, n, "LC P/off median indicates strong Kp")
  return ConfigEvidence("weak", "proven", float(lc_kp), float(lc_ki) if lc_ki is not None else None, n, "LC P/off median indicates weak Kp")
```

- [ ] **Step 4: Run telemetry tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_telemetry.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/telemetry.py retrospective_lateral/tests/test_telemetry.py
git commit -m "analysis: parse Ford lateral telemetry"
```

---

### Task 5: Extraction Cache Core And Synthetic Resampling Tests

**Files:**
- Create: `retrospective_lateral/code/extract.py`
- Test: `retrospective_lateral/tests/test_extract.py`

- [ ] **Step 1: Write extraction core tests using synthetic sparse channels**

Create `retrospective_lateral/tests/test_extract.py`:

```python
import json
import numpy as np

from retrospective_lateral.code.extract import RawChannels, resample_channels, route_cache_metadata


def test_resample_channels_outputs_fixed_grid_and_masks_large_gaps():
  raw = RawChannels()
  raw.add("carState", 0.0, {"v_ego": 10.0, "steering_angle_deg": 0.0, "yaw_rate": 0.0})
  raw.add("carState", 0.05, {"v_ego": 10.0, "steering_angle_deg": 1.0, "yaw_rate": 0.01})
  raw.add("carState", 1.50, {"v_ego": 10.0, "steering_angle_deg": 2.0, "yaw_rate": 0.02})
  raw.add("carControl", 0.0, {"lat_active": 1.0, "act_curvature": 0.001})
  raw.add("carControl", 1.50, {"lat_active": 1.0, "act_curvature": 0.002})
  out = resample_channels(raw, fs_hz=20.0)
  assert out["t"].shape[0] == 31
  assert np.isfinite(out["v_ego"][0])
  assert np.isnan(out["v_ego"][16])
  assert out["lat_active"][0] == 1.0


def test_route_cache_metadata_is_json_serializable(tmp_path):
  meta = route_cache_metadata(
    route_id="route_b8",
    schema_version="retrolat-v1",
    segments=3,
    config_confidence="proven",
    notes=["LC telemetry recovered"],
  )
  encoded = json.dumps(meta, sort_keys=True)
  assert "route_b8" in encoded
  assert meta["segments"] == 3
```

- [ ] **Step 2: Run extraction tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `extract`.

- [ ] **Step 3: Implement raw channel storage and resampling**

Create `retrospective_lateral/code/extract.py` with these initial components:

```python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "opendbc_repo")

from openpilot.tools.lib.logreader import LogReader

from retrospective_lateral.code import config as C
from retrospective_lateral.code.routes import RouteRef, discover_routes
from retrospective_lateral.code.signal_utils import fill_guarded
from retrospective_lateral.code.telemetry import parse_cp_line, parse_cx1_line, parse_lc_line, recover_pi_config


@dataclass
class RawChannels:
  times: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
  values: dict[str, dict[str, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
  log_messages: list[tuple[float, str]] = field(default_factory=list)
  init: dict[str, Any] = field(default_factory=dict)
  car_params: dict[str, Any] = field(default_factory=dict)

  def add(self, family: str, t: float, row: dict[str, float]) -> None:
    self.times[family].append(float(t))
    for key, value in row.items():
      self.values[family][key].append(float(value) if value is not None else np.nan)


def _interp_numeric(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray, fs_hz: float) -> np.ndarray:
  if len(t_src) < 2:
    return np.full_like(t_grid, np.nan, dtype=np.float32)
  order = np.argsort(t_src)
  t = t_src[order]
  y = y_src[order]
  out = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
  sample_gap = np.full_like(t_grid, np.nan, dtype=float)
  nearest = np.clip(np.searchsorted(t, t_grid), 0, len(t) - 1)
  prev = np.clip(nearest - 1, 0, len(t) - 1)
  sample_gap = np.minimum(np.abs(t_grid - t[nearest]), np.abs(t_grid - t[prev]))
  out[sample_gap > C.MAX_INTERP_GAP_S] = np.nan
  return out.astype(np.float32)


def _nearest_flag(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
  if len(t_src) < 1:
    return np.zeros_like(t_grid, dtype=np.float32)
  order = np.argsort(t_src)
  t = t_src[order]
  y = y_src[order]
  idx = np.clip(np.searchsorted(t, t_grid), 0, len(t) - 1)
  return (y[idx] > 0.5).astype(np.float32)


def resample_channels(raw: RawChannels, fs_hz: float = C.FS_HZ) -> dict[str, np.ndarray]:
  all_times: list[float] = []
  for ts in raw.times.values():
    all_times.extend(ts)
  if len(all_times) < 2:
    return {}
  t0 = float(np.nanmin(all_times))
  t1 = float(np.nanmax(all_times))
  t_grid = np.arange(t0, t1 + 0.5 / fs_hz, 1.0 / fs_hz, dtype=float)
  out: dict[str, np.ndarray] = {"t": (t_grid - t0).astype(np.float32), "mono_time": t_grid.astype(np.float64)}

  mapping = {
    "carState": ["v_ego", "v_ego_raw", "a_ego", "steering_angle_deg", "steering_rate_deg", "steering_torque", "yaw_rate"],
    "carControl": ["act_curvature", "current_curvature"],
    "controlsState": ["desired_curvature", "controls_curvature"],
    "modelV2": ["model_y0", "model_y20", "lane_center_y0", "lane_center_y20", "lane_width_y0", "lane_width_y20", "lane_prob_left", "lane_prob_right", "orientation_rate_z0"],
    "liveLocationKalman": ["lat", "lon", "yaw_rate_calibrated", "roll", "pitch"],
    "liveCalibration": ["cal_roll", "cal_pitch", "cal_yaw"],
  }
  flag_mapping = {
    "carState": ["steering_pressed", "left_blinker", "right_blinker", "can_valid"],
    "carControl": ["lat_active", "long_active"],
    "modelV2": ["lane_change_state"],
  }
  for family, keys in mapping.items():
    t = np.asarray(raw.times.get(family, []), dtype=float)
    for key in keys:
      y = np.asarray(raw.values.get(family, {}).get(key, []), dtype=float)
      out[key] = _interp_numeric(t, y, t_grid, fs_hz) if len(y) else np.full_like(t_grid, np.nan, dtype=np.float32)
  for family, keys in flag_mapping.items():
    t = np.asarray(raw.times.get(family, []), dtype=float)
    for key in keys:
      y = np.asarray(raw.values.get(family, {}).get(key, []), dtype=float)
      out[key] = _nearest_flag(t, y, t_grid) if len(y) else np.zeros_like(t_grid, dtype=np.float32)
  out["blinker"] = ((out.get("left_blinker", 0) > 0.5) | (out.get("right_blinker", 0) > 0.5)).astype(np.float32)
  return out


def route_cache_metadata(route_id: str, schema_version: str, segments: int,
                         config_confidence: str, notes: list[str]) -> dict[str, Any]:
  return {
    "route_id": route_id,
    "schema_version": schema_version,
    "segments": int(segments),
    "config_confidence": config_confidence,
    "notes": list(notes),
  }
```

- [ ] **Step 4: Run extraction core tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/extract.py retrospective_lateral/tests/test_extract.py
git commit -m "analysis: add retrospective extraction core"
```

---

### Task 6: Rlog Extraction CLI And Route Cache Manifest

**Files:**
- Modify: `retrospective_lateral/code/extract.py`
- Test: `retrospective_lateral/tests/test_extract_cli.py`

- [ ] **Step 1: Write a CLI smoke test that runs route discovery without real logs**

Create `retrospective_lateral/tests/test_extract_cli.py`:

```python
import json
import subprocess
from pathlib import Path


def test_extract_cli_list_routes(tmp_path):
  route = tmp_path / "route_x1" / "000000x1--abc--0"
  route.mkdir(parents=True)
  (route / "rlog.zst").write_bytes(b"not-a-real-log")
  cmd = [
    ".venv311/bin/python",
    "-m",
    "retrospective_lateral.code.extract",
    "--log-root",
    str(tmp_path),
    "--list-routes",
  ]
  proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=True)
  rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
  assert rows == [{"route_id": "route_x1", "segments": 1, "layout": "modern"}]
```

- [ ] **Step 2: Run the CLI test to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract_cli.py -q
```

Expected: FAIL because `--list-routes` is not implemented.

- [ ] **Step 3: Add message extraction functions and CLI**

Modify `retrospective_lateral/code/extract.py` by adding:

```python
def _as_log_text(log_message: Any) -> str:
  text = str(log_message)
  if text.startswith("{"):
    try:
      decoded = json.loads(text)
      return str(decoded.get("msg", text))
    except Exception:
      return text
  return text


def extract_route_raw(route: RouteRef) -> tuple[RawChannels, list[str]]:
  raw = RawChannels()
  notes: list[str] = []
  for segment in route.segments:
    try:
      for msg in LogReader(str(segment.rlog_path)):
        which = msg.which()
        t = msg.logMonoTime * 1e-9
        if which == "initData" and not raw.init:
          init = msg.initData
          raw.init = {"commit": str(init.gitCommit), "branch": str(init.gitBranch), "dirty": bool(init.dirty)}
        elif which == "carParams" and not raw.car_params:
          cp = msg.carParams
          raw.car_params = {
            "carFingerprint": str(cp.carFingerprint),
            "wheelbase": float(cp.wheelbase),
            "steerRatio": float(cp.steerRatio),
            "steerActuatorDelay": float(cp.steerActuatorDelay),
          }
        elif which == "carState":
          cs = msg.carState
          raw.add("carState", t, {
            "v_ego": cs.vEgo,
            "v_ego_raw": cs.vEgoRaw,
            "a_ego": cs.aEgo,
            "steering_angle_deg": cs.steeringAngleDeg,
            "steering_rate_deg": cs.steeringRateDeg,
            "steering_torque": cs.steeringTorque,
            "yaw_rate": cs.yawRate,
            "steering_pressed": 1.0 if cs.steeringPressed else 0.0,
            "left_blinker": 1.0 if cs.leftBlinker else 0.0,
            "right_blinker": 1.0 if cs.rightBlinker else 0.0,
            "can_valid": 1.0 if cs.canValid else 0.0,
          })
        elif which == "carControl":
          cc = msg.carControl
          raw.add("carControl", t, {
            "lat_active": 1.0 if cc.latActive else 0.0,
            "long_active": 1.0 if cc.longActive else 0.0,
            "act_curvature": float(cc.actuators.curvature),
            "current_curvature": float(cc.currentCurvature),
          })
        elif which == "controlsState":
          st = msg.controlsState
          raw.add("controlsState", t, {
            "desired_curvature": float(st.desiredCurvature),
            "controls_curvature": float(st.curvature),
          })
        elif which == "modelV2":
          m = msg.modelV2
          lane_change = 0 if str(m.meta.laneChangeState) == "off" else 1
          raw.add("modelV2", t, {
            "model_y0": _interp_model_xy(m.position.x, m.position.y, 0.0),
            "model_y20": _interp_model_xy(m.position.x, m.position.y, 20.0),
            "lane_center_y0": _lane_center(m, 0.0)[0],
            "lane_center_y20": _lane_center(m, 20.0)[0],
            "lane_width_y0": _lane_center(m, 0.0)[1],
            "lane_width_y20": _lane_center(m, 20.0)[1],
            "lane_prob_left": float(m.laneLineProbs[1]) if len(m.laneLineProbs) > 2 else np.nan,
            "lane_prob_right": float(m.laneLineProbs[2]) if len(m.laneLineProbs) > 2 else np.nan,
            "orientation_rate_z0": float(m.orientationRate.z[0]) if len(m.orientationRate.z) else np.nan,
            "lane_change_state": float(lane_change),
          })
        elif which == "liveLocationKalman":
          loc = msg.liveLocationKalman
          lat = lon = yaw_cal = roll = pitch = np.nan
          if loc.positionGeodetic.valid and len(loc.positionGeodetic.value) >= 2:
            lat = float(loc.positionGeodetic.value[0])
            lon = float(loc.positionGeodetic.value[1])
          if loc.angularVelocityCalibrated.valid and len(loc.angularVelocityCalibrated.value) >= 3:
            yaw_cal = float(loc.angularVelocityCalibrated.value[2])
          if loc.orientationNED.valid and len(loc.orientationNED.value) >= 2:
            roll = float(loc.orientationNED.value[0])
            pitch = float(loc.orientationNED.value[1])
          raw.add("liveLocationKalman", t, {"lat": lat, "lon": lon, "yaw_rate_calibrated": yaw_cal, "roll": roll, "pitch": pitch})
        elif which == "liveCalibration":
          rpy = list(msg.liveCalibration.rpyCalib)
          raw.add("liveCalibration", t, {
            "cal_roll": rpy[0] if len(rpy) > 0 else np.nan,
            "cal_pitch": rpy[1] if len(rpy) > 1 else np.nan,
            "cal_yaw": rpy[2] if len(rpy) > 2 else np.nan,
          })
        elif which == "logMessage":
          text = _as_log_text(msg.logMessage)
          raw.log_messages.append((t, text))
    except Exception as exc:
      notes.append(f"{segment.rlog_path}: {type(exc).__name__}: {exc}")
  return raw, notes


def _interp_model_xy(xs: Any, ys: Any, xq: float) -> float:
  x = np.asarray(list(xs), dtype=float)
  y = np.asarray(list(ys), dtype=float)
  ok = np.isfinite(x) & np.isfinite(y)
  if ok.sum() < 2 or xq < np.nanmin(x[ok]) or xq > np.nanmax(x[ok]):
    return np.nan
  return float(np.interp(xq, x[ok], y[ok]))


def _lane_center(model: Any, xq: float) -> tuple[float, float]:
  if len(model.laneLines) <= 2:
    return np.nan, np.nan
  left = _interp_model_xy(model.laneLines[1].x, model.laneLines[1].y, xq)
  right = _interp_model_xy(model.laneLines[2].x, model.laneLines[2].y, xq)
  if not np.isfinite(left) or not np.isfinite(right):
    return np.nan, np.nan
  return float(0.5 * (left + right)), float(abs(right - left))


def _write_route_cache(route: RouteRef, cache_root: Path, force: bool = False) -> dict[str, Any]:
  cache_root.mkdir(parents=True, exist_ok=True)
  out_npz = cache_root / f"{route.route_id}.npz"
  out_json = cache_root / f"{route.route_id}.json"
  if out_npz.exists() and out_json.exists() and not force:
    return json.loads(out_json.read_text())
  raw, notes = extract_route_raw(route)
  arrays = resample_channels(raw, fs_hz=C.FS_HZ)
  lc_rows = [parse_lc_line(text, t) for t, text in raw.log_messages]
  lc_rows = [row for row in lc_rows if row is not None]
  evidence = recover_pi_config(
    offsets=[row.offset_m for row in lc_rows],
    p_terms=[row.p_term for row in lc_rows],
    integrals=[row.integral for row in lc_rows],
    i_terms=[row.i_term for row in lc_rows],
  )
  if arrays:
    np.savez_compressed(out_npz, **arrays)
  meta = route_cache_metadata(route.route_id, C.CACHE_SCHEMA_VERSION, len(route.segments), evidence.confidence, notes)
  meta.update({"pi_set": evidence.pi_set, "lc_kp": evidence.lc_kp, "lc_ki": evidence.lc_ki, "init": raw.init, "car_params": raw.car_params})
  out_json.write_text(json.dumps(meta, indent=2, sort_keys=True))
  return meta


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-root", type=Path, default=C.DEFAULT_LOG_ROOT)
  parser.add_argument("--cache-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
  parser.add_argument("--route", action="append", default=[])
  parser.add_argument("--list-routes", action="store_true")
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args(argv)
  routes = discover_routes(args.log_root)
  if args.route:
    wanted = set(args.route)
    routes = [r for r in routes if r.route_id in wanted]
  if args.list_routes:
    for route in routes:
      print(json.dumps({"route_id": route.route_id, "segments": len(route.segments), "layout": route.layout}, sort_keys=True))
    return 0
  rows = []
  for route in routes:
    rows.append(_write_route_cache(route, args.cache_root, force=args.force))
    print(json.dumps(rows[-1], sort_keys=True))
  args.cache_root.mkdir(parents=True, exist_ok=True)
  (args.cache_root / "manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Smoke-run route listing on the real corpus**

Run:

```bash
.venv311/bin/python -m retrospective_lateral.code.extract --list-routes | head -5
```

Expected: JSON lines containing real route ids such as `route_0c`.

- [ ] **Step 6: Commit**

```bash
git add retrospective_lateral/code/extract.py retrospective_lateral/tests/test_extract_cli.py
git commit -m "analysis: add retrospective extraction CLI"
```

---

### Task 7: Low-Speed Wheel-Swing Metrics

**Files:**
- Create: `retrospective_lateral/code/metrics.py`
- Test: `retrospective_lateral/tests/test_metrics_low_speed.py`

- [ ] **Step 1: Write low-speed detector tests**

Create `retrospective_lateral/tests/test_metrics_low_speed.py`:

```python
import numpy as np

from retrospective_lateral.code.metrics import detect_low_speed_wheel_swing


def synthetic_low_speed_route():
  fs = 20.0
  t = np.arange(0.0, 20.0, 1.0 / fs)
  steer = 6.0 * np.sin(2 * np.pi * 0.25 * t)
  return {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 3.0, dtype=np.float32),
    "steering_angle_deg": steer.astype(np.float32),
    "steering_rate_deg": np.gradient(steer, 1.0 / fs).astype(np.float32),
    "steering_pressed": np.zeros_like(t, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
    "blinker": np.zeros_like(t, dtype=np.float32),
    "lane_change_state": np.zeros_like(t, dtype=np.float32),
    "act_curvature": (0.001 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32),
  }


def test_detect_low_speed_wheel_swing_finds_large_angle_episode():
  episodes = detect_low_speed_wheel_swing("route_test", synthetic_low_speed_route())
  assert len(episodes) >= 1
  worst = episodes[0]
  assert worst.symptom == "low_speed_wheel_swing"
  assert worst.route_id == "route_test"
  assert worst.speed_mph_median < 10.0
  assert worst.steering_peak_to_peak_deg >= 11.0
  assert worst.command_peak_to_peak_curvature > 0.0015
```

- [ ] **Step 2: Run low-speed metric tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_low_speed.py -q
```

Expected: FAIL because `metrics.py` does not exist.

- [ ] **Step 3: Implement low-speed episode detection**

Create `retrospective_lateral/code/metrics.py` with:

```python
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.signal_utils import (
  contiguous_regions,
  dilate_flags,
  erode_true,
  filter_continuous,
  peak_to_peak_masked,
  rms_masked,
  spectral_peak_hz,
)


@dataclass(frozen=True)
class Episode:
  symptom: str
  route_id: str
  start_s: float
  end_s: float
  peak_s: float
  speed_mph_median: float
  steering_peak_to_peak_deg: float
  steering_rate_rms_deg_s: float
  steering_band_rms_deg: float
  command_peak_to_peak_curvature: float
  path_curvature_band_rms_1e4: float
  stage_first_growth: str
  confidence: str

  def to_row(self) -> dict[str, object]:
    return asdict(self)


def _flag_radius(seconds: float) -> int:
  return int(round(seconds * C.FS_HZ))


def _base_clean_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
  n = len(arrays["t"])
  lat_active = erode_true(arrays.get("lat_active", np.zeros(n)) > 0.5, _flag_radius(C.ENGAGE_ERODE_S))
  no_override = ~dilate_flags(arrays.get("steering_pressed", np.zeros(n)) > 0.5, _flag_radius(C.OVERRIDE_BUFFER_S))
  no_blinker = ~dilate_flags(arrays.get("blinker", np.zeros(n)) > 0.5, _flag_radius(C.BLINKER_BUFFER_S))
  no_lane_change = ~dilate_flags(arrays.get("lane_change_state", np.zeros(n)) > 0.5, _flag_radius(C.LANE_CHANGE_BUFFER_S))
  return lat_active & no_override & no_blinker & no_lane_change


def _path_curvature(arrays: dict[str, np.ndarray]) -> np.ndarray:
  v = arrays.get("v_ego", np.array([], dtype=float)).astype(float)
  yaw = arrays.get("yaw_rate_calibrated", arrays.get("yaw_rate", np.full_like(v, np.nan))).astype(float)
  return np.divide(yaw, v, out=np.full_like(v, np.nan), where=(v > 1.0) & np.isfinite(yaw))


def detect_low_speed_wheel_swing(route_id: str, arrays: dict[str, np.ndarray]) -> list[Episode]:
  t = arrays["t"].astype(float)
  speed_mph = arrays["v_ego"].astype(float) * 2.23694
  clean = _base_clean_mask(arrays)
  speed_gate = (speed_mph >= C.LOW_SPEED_MPH[0]) & (speed_mph <= C.LOW_SPEED_MPH[1])
  eligible = clean & speed_gate & np.isfinite(arrays["steering_angle_deg"])
  min_len = int(round(C.MIN_LOW_SPEED_WINDOW_S * C.FS_HZ))
  steer = arrays["steering_angle_deg"].astype(float)
  steer_rate = arrays.get("steering_rate_deg", np.gradient(steer, 1.0 / C.FS_HZ)).astype(float)
  command = arrays.get("act_curvature", np.full_like(steer, np.nan)).astype(float)
  path_curv = _path_curvature(arrays)
  steer_band = filter_continuous(steer, C.FS_HZ, band=C.LOW_SPEED_INSPECT_BAND_HZ)
  path_band = filter_continuous(path_curv, C.FS_HZ, band=C.LOW_SPEED_INSPECT_BAND_HZ)

  episodes: list[Episode] = []
  for start, end in contiguous_regions(eligible, min_len=min_len):
    mask = np.zeros(len(t), dtype=bool)
    mask[start:end] = True
    steer_ptp = peak_to_peak_masked(steer, mask)
    if not np.isfinite(steer_ptp) or steer_ptp < 6.0:
      continue
    local_abs = np.abs(np.where(mask, steer_band, np.nan))
    peak_idx = int(np.nanargmax(local_abs)) if np.isfinite(local_abs).any() else start
    cmd_ptp = peak_to_peak_masked(command, mask)
    path_rms = rms_masked(path_band, mask) * 1e4
    if np.isfinite(cmd_ptp) and cmd_ptp > 0.0005:
      stage = "final_command_or_before"
    elif np.isfinite(path_rms) and path_rms > 0.5:
      stage = "actual_path_or_plant"
    else:
      stage = "steering_wheel_only"
    episodes.append(Episode(
      symptom="low_speed_wheel_swing",
      route_id=route_id,
      start_s=float(t[start]),
      end_s=float(t[end - 1]),
      peak_s=float(t[peak_idx]),
      speed_mph_median=float(np.nanmedian(speed_mph[mask])),
      steering_peak_to_peak_deg=float(steer_ptp),
      steering_rate_rms_deg_s=rms_masked(steer_rate, mask),
      steering_band_rms_deg=rms_masked(steer_band, mask),
      command_peak_to_peak_curvature=float(cmd_ptp) if np.isfinite(cmd_ptp) else math.nan,
      path_curvature_band_rms_1e4=float(path_rms) if np.isfinite(path_rms) else math.nan,
      stage_first_growth=stage,
      confidence="supported",
    ))
  return sorted(episodes, key=lambda e: e.steering_peak_to_peak_deg, reverse=True)
```

- [ ] **Step 4: Run low-speed tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_low_speed.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/metrics.py retrospective_lateral/tests/test_metrics_low_speed.py
git commit -m "analysis: detect low-speed wheel swing episodes"
```

---

### Task 8: 10-70 mph Weave Metrics And Stage Localization

**Files:**
- Modify: `retrospective_lateral/code/metrics.py`
- Test: `retrospective_lateral/tests/test_metrics_weave.py`

- [ ] **Step 1: Write weave detector tests**

Create `retrospective_lateral/tests/test_metrics_weave.py`:

```python
import numpy as np

from retrospective_lateral.code.metrics import detect_weave_windows


def synthetic_weave_route(stage: str):
  fs = 20.0
  t = np.arange(0.0, 60.0, 1.0 / fs)
  wave = np.sin(2 * np.pi * 0.18 * t)
  des = 0.00025 * wave if stage == "desired" else np.zeros_like(t)
  cmd = 0.00030 * wave if stage in ("desired", "command") else np.zeros_like(t)
  yaw = 18.0 * (0.00035 * wave)
  return {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 18.0, dtype=np.float32),
    "steering_angle_deg": (0.7 * wave).astype(np.float32),
    "steering_rate_deg": np.gradient(0.7 * wave, 1.0 / fs).astype(np.float32),
    "yaw_rate": yaw.astype(np.float32),
    "steering_pressed": np.zeros_like(t, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
    "blinker": np.zeros_like(t, dtype=np.float32),
    "lane_change_state": np.zeros_like(t, dtype=np.float32),
    "act_curvature": cmd.astype(np.float32),
    "desired_curvature": des.astype(np.float32),
    "model_y20": np.zeros_like(t, dtype=np.float32),
    "lane_center_y20": np.zeros_like(t, dtype=np.float32),
  }


def test_detect_weave_windows_finds_path_weave():
  windows = detect_weave_windows("route_test", synthetic_weave_route("command"))
  assert len(windows) >= 1
  worst = windows[0]
  assert worst.symptom == "weave_10_70"
  assert worst.path_curvature_band_rms_1e4 > 2.0
  assert worst.steering_band_rms_deg > 0.3
  assert worst.stage_first_growth == "controller_or_command"


def test_detect_weave_windows_identifies_desired_stage():
  windows = detect_weave_windows("route_test", synthetic_weave_route("desired"))
  assert windows[0].stage_first_growth == "model_or_desired"
```

- [ ] **Step 2: Run weave tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_weave.py -q
```

Expected: FAIL because `detect_weave_windows` is not implemented.

- [ ] **Step 3: Add weave window dataclass and detector**

Modify `retrospective_lateral/code/metrics.py` by adding:

```python
@dataclass(frozen=True)
class WeaveWindow:
  symptom: str
  route_id: str
  start_s: float
  end_s: float
  speed_mph_median: float
  path_curvature_band_rms_1e4: float
  steering_band_rms_deg: float
  command_band_rms_1e4: float
  desired_band_rms_1e4: float
  model_y20_band_rms_m: float
  steer_per_path: float
  spectral_peak_hz: float
  stage_first_growth: str
  evidence_note: str

  def to_row(self) -> dict[str, object]:
    return asdict(self)


def _stage_label(path_rms: float, steer_rms: float, cmd_rms: float, des_rms: float, model_rms: float) -> tuple[str, str]:
  if np.isfinite(model_rms) and model_rms > 0.02:
    return "model_or_desired", "model path y20 contains slow-band motion"
  if np.isfinite(des_rms) and des_rms >= 0.5 * max(path_rms, 1e-9):
    return "model_or_desired", "desiredCurvature contains comparable slow-band motion"
  if np.isfinite(cmd_rms) and cmd_rms >= 0.5 * max(path_rms, 1e-9):
    return "controller_or_command", "final command contains comparable slow-band motion"
  if np.isfinite(path_rms) and path_rms > 0.5:
    return "actual_path_or_plant", "actual path contains slow-band motion not obvious in command"
  if np.isfinite(steer_rms) and steer_rms > 0.2:
    return "steering_wheel_only", "steering motion exceeds path motion"
  return "unknown", "slow-band energy below stage thresholds"


def detect_weave_windows(route_id: str, arrays: dict[str, np.ndarray]) -> list[WeaveWindow]:
  t = arrays["t"].astype(float)
  speed_mph = arrays["v_ego"].astype(float) * 2.23694
  clean = _base_clean_mask(arrays)
  speed_gate = (speed_mph >= C.WEAVE_SPEED_MPH[0]) & (speed_mph <= C.WEAVE_SPEED_MPH[1])
  path_curv = _path_curvature(arrays)
  road_lp = filter_continuous(path_curv, C.FS_HZ, lowpass_hz=C.ROAD_LP_HZ)
  gentle = np.abs(road_lp) <= C.ROAD_CURV_ABS_MAX_1PM
  eligible = clean & speed_gate & gentle & np.isfinite(path_curv)
  window_len = int(round(C.WEAVE_WINDOW_S * C.FS_HZ))
  min_eligible = int(round(C.MIN_WEAVE_ELIGIBLE_S * C.FS_HZ))

  path_band = filter_continuous(path_curv, C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  steer_band = filter_continuous(arrays["steering_angle_deg"].astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  cmd_band = filter_continuous(arrays.get("act_curvature", np.full_like(path_curv, np.nan)).astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  des_band = filter_continuous(arrays.get("desired_curvature", np.full_like(path_curv, np.nan)).astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  model_band = filter_continuous(arrays.get("model_y20", np.full_like(path_curv, np.nan)).astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)

  windows: list[WeaveWindow] = []
  for start in range(0, max(0, len(t) - window_len + 1), window_len):
    end = start + window_len
    mask = np.zeros(len(t), dtype=bool)
    mask[start:end] = eligible[start:end]
    if mask.sum() < min_eligible:
      continue
    path_rms = rms_masked(path_band, mask) * 1e4
    steer_rms = rms_masked(steer_band, mask)
    if not np.isfinite(path_rms) or path_rms < 0.2:
      continue
    cmd_rms = rms_masked(cmd_band, mask) * 1e4
    des_rms = rms_masked(des_band, mask) * 1e4
    model_rms = rms_masked(model_band, mask)
    stage, note = _stage_label(path_rms, steer_rms, cmd_rms, des_rms, model_rms)
    peak_hz = spectral_peak_hz(np.where(mask, path_band, np.nan), C.FS_HZ, C.DEFAULT_WEAVE_BAND_HZ)
    windows.append(WeaveWindow(
      symptom="weave_10_70",
      route_id=route_id,
      start_s=float(t[start]),
      end_s=float(t[end - 1]),
      speed_mph_median=float(np.nanmedian(speed_mph[mask])),
      path_curvature_band_rms_1e4=float(path_rms),
      steering_band_rms_deg=float(steer_rms) if np.isfinite(steer_rms) else math.nan,
      command_band_rms_1e4=float(cmd_rms) if np.isfinite(cmd_rms) else math.nan,
      desired_band_rms_1e4=float(des_rms) if np.isfinite(des_rms) else math.nan,
      model_y20_band_rms_m=float(model_rms) if np.isfinite(model_rms) else math.nan,
      steer_per_path=float(steer_rms / path_rms) if np.isfinite(steer_rms) and path_rms > 0 else math.nan,
      spectral_peak_hz=float(peak_hz) if np.isfinite(peak_hz) else math.nan,
      stage_first_growth=stage,
      evidence_note=note,
    ))
  return sorted(windows, key=lambda w: w.path_curvature_band_rms_1e4, reverse=True)
```

- [ ] **Step 4: Run weave tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_weave.py -q
```

Expected: PASS.

- [ ] **Step 5: Run all metric tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_low_speed.py retrospective_lateral/tests/test_metrics_weave.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add retrospective_lateral/code/metrics.py retrospective_lateral/tests/test_metrics_weave.py
git commit -m "analysis: detect 10-70 mph weave windows"
```

---

### Task 9: Metric CLI And Episode Catalog Writer

**Files:**
- Modify: `retrospective_lateral/code/metrics.py`
- Test: `retrospective_lateral/tests/test_metrics_cli.py`

- [ ] **Step 1: Write a metrics CLI test using synthetic NPZ cache**

Create `retrospective_lateral/tests/test_metrics_cli.py`:

```python
import csv
import subprocess
from pathlib import Path

import numpy as np


def test_metrics_cli_writes_episode_catalog(tmp_path):
  cache = tmp_path / "cache"
  out = tmp_path / "out"
  cache.mkdir()
  fs = 20.0
  t = np.arange(0.0, 20.0, 1.0 / fs)
  steer = 6.0 * np.sin(2 * np.pi * 0.25 * t)
  np.savez_compressed(
    cache / "route_synth.npz",
    t=t.astype(np.float32),
    v_ego=np.full_like(t, 3.0, dtype=np.float32),
    steering_angle_deg=steer.astype(np.float32),
    steering_rate_deg=np.gradient(steer, 1.0 / fs).astype(np.float32),
    steering_pressed=np.zeros_like(t, dtype=np.float32),
    lat_active=np.ones_like(t, dtype=np.float32),
    blinker=np.zeros_like(t, dtype=np.float32),
    lane_change_state=np.zeros_like(t, dtype=np.float32),
    act_curvature=(0.001 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32),
    yaw_rate=np.zeros_like(t, dtype=np.float32),
  )
  cmd = [
    ".venv311/bin/python",
    "-m",
    "retrospective_lateral.code.metrics",
    "--cache-root",
    str(cache),
    "--out",
    str(out),
  ]
  subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True)
  catalog = out / "symptom_catalog.csv"
  assert catalog.exists()
  rows = list(csv.DictReader(catalog.open()))
  assert rows[0]["symptom"] == "low_speed_wheel_swing"
```

- [ ] **Step 2: Run the CLI test to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_cli.py -q
```

Expected: FAIL because `metrics.py` has no CLI.

- [ ] **Step 3: Add CSV writers and CLI to `metrics.py`**

Append to `retrospective_lateral/code/metrics.py`:

```python
def _load_npz(path: str) -> dict[str, np.ndarray]:
  data = np.load(path)
  return {key: data[key] for key in data.files}


def write_symptom_catalog(cache_root: str, out_dir: str) -> list[dict[str, object]]:
  import csv
  from pathlib import Path

  cache = Path(cache_root)
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, object]] = []
  for npz in sorted(cache.glob("route_*.npz")):
    route_id = npz.stem
    arrays = _load_npz(str(npz))
    rows.extend(ep.to_row() for ep in detect_low_speed_wheel_swing(route_id, arrays))
    rows.extend(win.to_row() for win in detect_weave_windows(route_id, arrays))
  fieldnames = sorted({key for row in rows for key in row})
  catalog = out / "symptom_catalog.csv"
  with catalog.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  return rows


def main(argv: list[str] | None = None) -> int:
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--cache-root", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args(argv)
  rows = write_symptom_catalog(args.cache_root, args.out)
  print(f"wrote {len(rows)} symptom rows to {args.out}/symptom_catalog.csv")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
```

- [ ] **Step 4: Run metrics CLI test**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_cli.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/metrics.py retrospective_lateral/tests/test_metrics_cli.py
git commit -m "analysis: write retrospective symptom catalog"
```

---

### Task 10: Matched Historical Comparisons

**Files:**
- Create: `retrospective_lateral/code/compare.py`
- Test: `retrospective_lateral/tests/test_compare.py`

- [ ] **Step 1: Write comparison tests**

Create `retrospective_lateral/tests/test_compare.py`:

```python
import pandas as pd

from retrospective_lateral.code.compare import evidence_tier, location_speed_scorecard


def test_evidence_tier_prefers_location_matched():
  assert evidence_tier(has_location=True, has_speed=True, same_corridor=False) == "location_matched"
  assert evidence_tier(has_location=True, has_speed=True, same_corridor=True) == "same_corridor_transition"
  assert evidence_tier(has_location=False, has_speed=True, same_corridor=False) == "speed_matched"


def test_location_speed_scorecard_uses_matched_cells():
  df = pd.DataFrame([
    {"route_id": "r1", "config": "A", "cell": 1, "heading_bin": 2, "speed_bin": 20, "path_curvature_band_rms_1e4": 2.0, "steering_band_rms_deg": 0.6},
    {"route_id": "r2", "config": "B", "cell": 1, "heading_bin": 2, "speed_bin": 20, "path_curvature_band_rms_1e4": 1.0, "steering_band_rms_deg": 0.3},
    {"route_id": "r3", "config": "A", "cell": 2, "heading_bin": 2, "speed_bin": 20, "path_curvature_band_rms_1e4": 9.0, "steering_band_rms_deg": 9.0},
  ])
  rows = location_speed_scorecard(df, config_a="A", config_b="B")
  assert len(rows) == 1
  row = rows[0]
  assert row["evidence_tier"] == "location_matched"
  assert row["path_effect_pct"] == -50.0
  assert row["steer_effect_pct"] == -50.0
```

- [ ] **Step 2: Run comparison tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_compare.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `compare`.

- [ ] **Step 3: Implement comparison helpers**

Create `retrospective_lateral/code/compare.py`:

```python
from __future__ import annotations

import math

import pandas as pd


def evidence_tier(*, has_location: bool, has_speed: bool, same_corridor: bool) -> str:
  if same_corridor and has_location and has_speed:
    return "same_corridor_transition"
  if has_location and has_speed:
    return "location_matched"
  if has_speed:
    return "speed_matched"
  return "descriptive"


def _pct(new: float, base: float) -> float:
  if not math.isfinite(new) or not math.isfinite(base) or base == 0:
    return math.nan
  return float(100.0 * (new - base) / base)


def location_speed_scorecard(df: pd.DataFrame, *, config_a: str, config_b: str) -> list[dict[str, object]]:
  needed = {"config", "cell", "heading_bin", "speed_bin", "path_curvature_band_rms_1e4", "steering_band_rms_deg"}
  missing = needed - set(df.columns)
  if missing:
    raise ValueError(f"missing columns: {sorted(missing)}")
  rows: list[dict[str, object]] = []
  strata = ["cell", "heading_bin", "speed_bin"]
  for key, group in df.groupby(strata, dropna=True):
    a = group[group["config"] == config_a]
    b = group[group["config"] == config_b]
    if a.empty or b.empty:
      continue
    a_path = float(a["path_curvature_band_rms_1e4"].median())
    b_path = float(b["path_curvature_band_rms_1e4"].median())
    a_steer = float(a["steering_band_rms_deg"].median())
    b_steer = float(b["steering_band_rms_deg"].median())
    rows.append({
      "comparison": f"{config_b}_minus_{config_a}",
      "stratum": "|".join(str(x) for x in key),
      "config_a": config_a,
      "config_b": config_b,
      "n_a": int(len(a)),
      "n_b": int(len(b)),
      "evidence_tier": evidence_tier(has_location=True, has_speed=True, same_corridor=False),
      "path_a": a_path,
      "path_b": b_path,
      "path_effect_pct": _pct(b_path, a_path),
      "steer_a": a_steer,
      "steer_b": b_steer,
      "steer_effect_pct": _pct(b_steer, a_steer),
    })
  return rows
```

- [ ] **Step 4: Run comparison tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_compare.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/compare.py retrospective_lateral/tests/test_compare.py
git commit -m "analysis: compare matched retrospective strata"
```

---

### Task 11: Report Writer And Next-Experiment Summary

**Files:**
- Create: `retrospective_lateral/code/report.py`
- Test: `retrospective_lateral/tests/test_report.py`

- [ ] **Step 1: Write report tests**

Create `retrospective_lateral/tests/test_report.py`:

```python
import pandas as pd

from retrospective_lateral.code.report import write_markdown_report


def test_write_markdown_report_includes_evidence_language(tmp_path):
  symptom = pd.DataFrame([
    {"symptom": "weave_10_70", "route_id": "route_x", "path_curvature_band_rms_1e4": 3.0, "stage_first_growth": "controller_or_command"},
  ])
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": "location_matched", "path_effect_pct": -20.0, "steer_effect_pct": -10.0},
  ])
  path = write_markdown_report(tmp_path, symptom, scorecard)
  text = path.read_text()
  assert "# Retrospective Lateral Weave Analysis Report" in text
  assert "weave_10_70" in text
  assert "location_matched" in text
  assert "controlled_drive_needed" in text
```

- [ ] **Step 2: Run report tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_report.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `report`.

- [ ] **Step 3: Implement Markdown report writer**

Create `retrospective_lateral/code/report.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _top_rows(df: pd.DataFrame, n: int = 10) -> str:
  if df.empty:
    return "_No rows._"
  view = df.head(n).fillna("")
  columns = list(view.columns)
  header = "| " + " | ".join(columns) + " |"
  sep = "| " + " | ".join(["---"] * len(columns)) + " |"
  body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in view.iterrows()]
  return "\n".join([header, sep] + body)


def write_markdown_report(out_dir: str | Path, symptom_catalog: pd.DataFrame, scorecard: pd.DataFrame) -> Path:
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  report = out / "retrospective_lateral_report.md"
  text = "\n".join([
    "# Retrospective Lateral Weave Analysis Report",
    "",
    "## Evidence Rules",
    "",
    "Claims use evidence tiers: descriptive, speed_matched, location_matched, same_corridor_transition, and controlled_drive_needed.",
    "No root-cause claim should be promoted without cooperative and adversarial QA.",
    "",
    "## Top Symptom Episodes",
    "",
    _top_rows(symptom_catalog, 10),
    "",
    "## Historical Scorecard",
    "",
    _top_rows(scorecard, 20),
    "",
    "## Next Experiment Guidance",
    "",
    "Rows that remain confounded after location and speed matching should be labeled controlled_drive_needed.",
    "",
  ])
  report.write_text(text)
  return report
```

- [ ] **Step 4: Run report tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/report.py retrospective_lateral/tests/test_report.py
git commit -m "analysis: report retrospective weave results"
```

---

### Task 12: Built-In QA Audit Checks

**Files:**
- Create: `retrospective_lateral/code/qa.py`
- Test: `retrospective_lateral/tests/test_qa.py`

- [ ] **Step 1: Write QA audit tests**

Create `retrospective_lateral/tests/test_qa.py`:

```python
import pandas as pd

from retrospective_lateral.code.qa import audit_scorecard_claims


def test_audit_rejects_single_drive_supported_claim():
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": "location_matched", "n_a": 1, "n_b": 1, "claim": "supported"},
  ])
  findings = audit_scorecard_claims(scorecard)
  assert findings[0]["severity"] == "blocker"
  assert "single-drive" in findings[0]["message"]


def test_audit_accepts_conservative_claim():
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": "location_matched", "n_a": 3, "n_b": 3, "claim": "suggestive"},
  ])
  assert audit_scorecard_claims(scorecard) == []
```

- [ ] **Step 2: Run QA tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_qa.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `qa`.

- [ ] **Step 3: Implement QA audit checks**

Create `retrospective_lateral/code/qa.py`:

```python
from __future__ import annotations

import pandas as pd


def audit_scorecard_claims(scorecard: pd.DataFrame) -> list[dict[str, str]]:
  findings: list[dict[str, str]] = []
  if scorecard.empty:
    return findings
  for idx, row in scorecard.iterrows():
    claim = str(row.get("claim", "")).lower()
    if claim in {"supported", "refuted", "root-cause likely", "root_cause_likely"}:
      n_a = int(row.get("n_a", 0))
      n_b = int(row.get("n_b", 0))
      tier = str(row.get("evidence_tier", ""))
      if n_a < 2 or n_b < 2:
        findings.append({
          "severity": "blocker",
          "row": str(idx),
          "message": f"single-drive claim is not allowed for {row.get('comparison', '')}",
        })
      if tier not in {"location_matched", "same_corridor_transition"}:
        findings.append({
          "severity": "blocker",
          "row": str(idx),
          "message": f"strong claim requires location-matched evidence for {row.get('comparison', '')}",
        })
  return findings
```

- [ ] **Step 4: Run QA tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_qa.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add retrospective_lateral/code/qa.py retrospective_lateral/tests/test_qa.py
git commit -m "analysis: add retrospective QA audits"
```

---

### Task 13: End-To-End Command Wrapper

**Files:**
- Create: `retrospective_lateral/code/run_all.py`
- Test: `retrospective_lateral/tests/test_run_all.py`

- [ ] **Step 1: Write command-construction test**

Create `retrospective_lateral/tests/test_run_all.py`:

```python
from retrospective_lateral.code.run_all import build_pipeline_steps


def test_build_pipeline_steps_uses_cache_and_report_dirs():
  steps = build_pipeline_steps(log_root="logs", cache_root="cache", report_root="reports", routes=["route_b8"])
  assert steps[0][:4] == [".venv311/bin/python", "-m", "retrospective_lateral.code.extract", "--log-root"]
  assert "--route" in steps[0]
  assert steps[1][:3] == [".venv311/bin/python", "-m", "retrospective_lateral.code.metrics"]
```

- [ ] **Step 2: Run run-all tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_run_all.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `run_all`.

- [ ] **Step 3: Implement run-all wrapper**

Create `retrospective_lateral/code/run_all.py`:

```python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from retrospective_lateral.code import config as C


def build_pipeline_steps(log_root: str, cache_root: str, report_root: str, routes: list[str]) -> list[list[str]]:
  extract = [".venv311/bin/python", "-m", "retrospective_lateral.code.extract", "--log-root", log_root, "--cache-root", cache_root]
  for route in routes:
    extract.extend(["--route", route])
  metrics = [".venv311/bin/python", "-m", "retrospective_lateral.code.metrics", "--cache-root", cache_root, "--out", report_root]
  return [extract, metrics]


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-root", default=str(C.DEFAULT_LOG_ROOT))
  parser.add_argument("--cache-root", default=str(C.DEFAULT_CACHE_ROOT))
  parser.add_argument("--report-root", default=str(C.DEFAULT_REPORT_ROOT))
  parser.add_argument("--route", action="append", default=[])
  args = parser.parse_args(argv)
  Path(args.cache_root).mkdir(parents=True, exist_ok=True)
  Path(args.report_root).mkdir(parents=True, exist_ok=True)
  for step in build_pipeline_steps(args.log_root, args.cache_root, args.report_root, args.route):
    print("+ " + " ".join(step))
    subprocess.run(step, check=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
```

- [ ] **Step 4: Run run-all tests**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_run_all.py -q
```

Expected: PASS.

- [ ] **Step 5: Smoke-run one known route**

Run:

```bash
.venv311/bin/python -m retrospective_lateral.code.run_all --route route_b8
```

Expected: creates `retrospective_lateral/results/cache/route_b8.npz`, `route_b8.json`, and `retrospective_lateral/results/reports/symptom_catalog.csv`.

- [ ] **Step 6: Commit**

```bash
git add retrospective_lateral/code/run_all.py retrospective_lateral/tests/test_run_all.py
git commit -m "analysis: add retrospective pipeline runner"
```

---

### Task 14: Cooperative QA Pass

**Files:**
- Create: `retrospective_lateral/qa/qa_cooperative.md`
- Modify code/tests only if QA findings require fixes.

- [ ] **Step 1: Dispatch cooperative QA**

Use `superpowers:subagent-driven-development` if executing with subagents. Ask the cooperative QA worker to review:

```text
Review the retrospective_lateral implementation against docs/superpowers/specs/2026-06-14-retrospective-lateral-weave-analysis-design.md.

Focus on whether the implementation matches the design, whether tests cover the intended behavior, whether outputs are reproducible, and whether any assumptions are unclear.

Run:
.venv311/bin/python -m pytest retrospective_lateral/tests -q
.venv311/bin/python -m retrospective_lateral.code.run_all --route route_b8

Write findings with file/line references and classify each as blocker, important, or nit.
```

- [ ] **Step 2: Record cooperative QA findings**

Create `retrospective_lateral/qa/qa_cooperative.md` with this structure:

```markdown
# Cooperative QA Findings

## Commands Run

- `.venv311/bin/python -m pytest retrospective_lateral/tests -q`
- `.venv311/bin/python -m retrospective_lateral.code.run_all --route route_b8`

## Findings

## Reconciliation
```

Fill `Findings` and `Reconciliation` with the actual QA output and the main agent's response to each finding.

- [ ] **Step 3: Fix accepted cooperative QA blockers**

For each accepted blocker, write a focused failing test first, implement the minimal fix, and rerun:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests -q
```

Expected: PASS.

- [ ] **Step 4: Commit cooperative QA artifact and fixes**

```bash
git add retrospective_lateral
git commit -m "analysis: address cooperative QA for retrospective pipeline"
```

---

### Task 15: Adversarial QA Pass

**Files:**
- Create: `retrospective_lateral/qa/qa_adversarial.md`
- Modify code/tests only if QA findings require fixes.

- [ ] **Step 1: Dispatch adversarial QA**

Use a different worker from cooperative QA. If available and practical, use Claude Code as an additional independent reviewer with the same prompt and the spec file.

Prompt:

```text
Adversarially review the retrospective_lateral implementation and the route_b8 smoke output.

Try to break the analysis. Look for sign-convention mistakes, speed leakage, route/date/config mislabeling, sample-independence violations, filtering leakage, missing lead/override/lane-change gates, overclaiming, untested corrupt-log behavior, and mismatch with docs/superpowers/specs/2026-06-14-retrospective-lateral-weave-analysis-design.md.

Run the tests and inspect generated outputs. Findings must include file/line references and a severity.
```

- [ ] **Step 2: Record adversarial QA findings**

Create `retrospective_lateral/qa/qa_adversarial.md` with this structure:

```markdown
# Adversarial QA Findings

## Commands Run

- `.venv311/bin/python -m pytest retrospective_lateral/tests -q`
- `.venv311/bin/python -m retrospective_lateral.code.run_all --route route_b8`

## Findings

## Reconciliation
```

Fill `Findings` and `Reconciliation` with the actual adversarial output and main-agent disposition.

- [ ] **Step 3: Fix accepted adversarial QA blockers**

For each accepted blocker, write or update a test first, implement the minimal fix, then rerun:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests -q
.venv311/bin/python -m retrospective_lateral.code.run_all --route route_b8
```

Expected: PASS and successful route smoke output.

- [ ] **Step 4: Commit adversarial QA artifact and fixes**

```bash
git add retrospective_lateral
git commit -m "analysis: address adversarial QA for retrospective pipeline"
```

---

### Task 16: Full-Corpus Extraction And Initial Report

**Files:**
- Generated: `retrospective_lateral/results/cache/*`
- Generated: `retrospective_lateral/results/reports/symptom_catalog.csv`
- Generated: `retrospective_lateral/results/reports/retrospective_lateral_report.md`
- Create or modify: `retrospective_lateral/README.md`

- [ ] **Step 1: Run the full retrospective pipeline**

Run:

```bash
.venv311/bin/python -m retrospective_lateral.code.run_all
```

Expected: per-route cache files under `retrospective_lateral/results/cache/` and symptom catalog under `retrospective_lateral/results/reports/`.

- [ ] **Step 2: Record runtime and route counts**

Run:

```bash
find retrospective_lateral/results/cache -name 'route_*.npz' | wc -l
wc -l retrospective_lateral/results/reports/symptom_catalog.csv
```

Expected: nonzero route-cache count and symptom catalog line count.

- [ ] **Step 3: Update README with observed run command and outputs**

Modify `retrospective_lateral/README.md` to include:

```markdown
## Current Run

Full-corpus command:

```bash
.venv311/bin/python -m retrospective_lateral.code.run_all
```

Primary outputs:

- `retrospective_lateral/results/cache/`
- `retrospective_lateral/results/reports/symptom_catalog.csv`
- `retrospective_lateral/results/reports/retrospective_lateral_report.md`

Generated result files under `retrospective_lateral/results/` are intentionally gitignored.
```

- [ ] **Step 4: Run the full test suite**

Run:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests -q
```

Expected: PASS.

- [ ] **Step 5: Commit README and any code fixes**

Do not commit generated `retrospective_lateral/results/` data.

```bash
git add retrospective_lateral/README.md retrospective_lateral/code retrospective_lateral/tests
git commit -m "analysis: document retrospective full-corpus run"
```

---

### Task 17: Completion Verification

**Files:**
- No new files unless verification reveals a needed fix.

- [ ] **Step 1: Run final tests**

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests -q
```

Expected: PASS.

- [ ] **Step 2: Verify no generated results are staged**

```bash
git status --short
```

Expected: no staged generated files under `retrospective_lateral/results/`.

- [ ] **Step 3: Verify the latest report exists locally**

```bash
test -f retrospective_lateral/results/reports/symptom_catalog.csv
```

Expected: command exits 0.

- [ ] **Step 4: Final cooperative/adversarial QA evidence check**

Confirm both files exist and include reconciliation text:

```bash
test -f retrospective_lateral/qa/qa_cooperative.md
test -f retrospective_lateral/qa/qa_adversarial.md
rg -n "Reconciliation" retrospective_lateral/qa/qa_cooperative.md retrospective_lateral/qa/qa_adversarial.md
```

Expected: both files found and both contain `Reconciliation`.

- [ ] **Step 5: Summarize outcome for the user**

Report:

- Commit range created during implementation.
- Test commands and results.
- Number of routes cached.
- Number of symptom rows.
- Top 3 low-speed wheel-swing episodes.
- Top 3 10-70 mph weave episodes.
- Any QA blockers that remain as documented residual risk.
