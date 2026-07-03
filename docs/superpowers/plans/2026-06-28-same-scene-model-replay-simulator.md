# Same-Scene Model-Replay Simulator — Implementation Plan

> ## ✅ STATUS: DATA-SUPPORTED — viable (corrected twice on 2026-06-28; the "data wall" was an eligibility bug)
> An adversarial-QA "fatal data wall" (~50 s of eligible data) was **investigated and fully overturned** after two rounds of user pushback. It (and my first correction) inherited the **engaged** weave-detector's eligibility into **disengaged** scene selection, where three of its filters are wrong: (a) **near-lead exclusion** — pointless here, both models see the identical lead so it cancels; (b) **2 s erosion** — deletes this driver's many brief takeovers; (c) **`~steering_pressed` (override) exclusion — SELF-CONTRADICTORY on disengaged data: when disengaged the human IS steering, so `steering_pressed` is ~always true and `~pressed` deletes nearly everything** (this was the dominant killer); plus a 70 mph speed cap that excludes long highway-cruise straights (only 4% of disengaged data but the longest runs). With **disengaged-appropriate gating** (drop override + near-lead + erosion; speed 10–85; straight/gentle `|curv|<0.005`) the data is abundant: **disengaged = 28 h (48.5%) of the corpus; disengaged-straight/gentle ≈ 9 h, with ~460–520 runs ≥ 20 s, 100–139 runs ≥ 60 s, longest 6–11 min** (frame-bearing-now: **68–77 min, 59–62 runs ≥ 20 s, 15–22 ≥ 60 s, longest 3–4 min**). Hundreds of long windows → a paired same-scene comparison is well-powered. **No frame pull is even strictly required** (though it adds power). Lesson (logged to memory): *never inherit one analysis's eligibility into a different one without re-deriving it* — over-strict filtering is this project's recurring failure mode; `~steering_pressed` and disengaged are contradictory.
>
> **What this changes vs the tasks below:**
> 1. **Eligibility for the sim metric (Task 4):** for DISENGAGED scene selection, **drop `~steering_pressed`, near-lead, AND erosion**; keep disengaged + straight/gentle (`|curv|<0.005`) + speed 10–85 + (optionally) not-blinker/lane-change. The Task-4 `disengaged_eligible_mask` code below is corrected accordingly. Require contiguous runs ≥ 10–20 s; report per-window cycle count.
> 2. **Frame pull is optional** (not required): 68–77 min usable locally now; pull more `fcamera.hevc` (read-only) only to add power.
> 3. **Add Task 0 (data-sufficiency gate, PASSES):** the corrected scan; run it first to record the window count.
>
> **Structural fixes still required (adversarial QA, all verified):** (1) the Phase-0 GO/NO-GO must replay the **actual logged bundle** for a route, not the stock ONNX against a route that ran a different/unknown model; demote tier-b to a non-gating diagnostic (gate on tier-a self-test + end-to-end run + same-bundle self-consistency, per scope §8). (2) The **compiled warp PIXEL pipeline** (`sunnypilot/modeld_v2/warp.py` `make_frame_prepare`/`Tensor.from_blob`/`stride_pad`/12-channel temporal pack) is a first-class task — `build_warp`'s 3×3 alone produces no model input. (3) The recurrent temporal core + `get_action_from_model` post-step (Task 9b) is a multi-day effort, not a one-task "seam"; a wrong buffer stride would pass the low-amplitude straight-window anchor while biasing the weave — design tests that catch it. (4) **`rpyCalib` is passed DIRECTLY to `get_warp_matrix`** (`modeld.py:294`); no conversion (corrected below). (5) `TinygradRunner` needs bundle/metadata loading (`_load_models→get_active_bundle`), not just a Params stub. (6) Fail-fast ordering: Task 0 data scan + submodule-init + stock-ONNX CPU compile on day 1.
>
> The architecture (direct tinygrad-CPU inference, reuse parse/warp/FrameReader, identical-Mac pairing) is sound and the data supports it. Address the structural fixes during execution; the project is **unblocked**.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Mac-native, analysis-only simulator that re-runs driving-model bundles (CD210 / OPM7 / Nevada) on identical **disengaged** recorded camera frames and compares their slow-weave (predicted-path) behavior with every confound but the model weights removed.

**Architecture:** Direct tinygrad-**CPU** inference (the macOS `process_replay`/modeld-process path is blocked — `msgq` is unbuildable here). Reuse openpilot's msgq-free pieces (`parse_model_outputs`, `get_warp_matrix`, `FrameReader`) behind a `Params` stub; compile each bundle's **ONNX → Mac pkl** via tinygrad's `compile3.py`. A two-tier faithfulness anchor (tinygrad-vs-onnxruntime correctness + a *looser, non-load-bearing* replay-vs-log realism check) gates the build; the cross-model verdict rests on **identical-Mac-pipeline pairing** on the same disengaged frames.

**Tech Stack:** Python 3.11 (`.venv311`), vendored `tinygrad_repo` (Metal/CPU backends + `examples/openpilot/compile3.py`), `onnxruntime` (self-test only), NumPy/SciPy/pandas, `openpilot.tools.lib.framereader.FrameReader`, pytest. Outputs are CSV/parquet under the gitignored `retrospective_lateral/results/model_replay/`.

**Source of truth:** `docs/superpowers/specs/2026-06-28-same-scene-model-replay-simulator-scope.md` (twice-QA'd). Read it before starting.

> **Revision note (post-collaborative-QA, verified against the real repo):** (1) **Frame-bearing routes are `route_8d`/`route_7*`/`route_8*` — NOT `route_c*`** (which have NPZ but no hevc); all segment globs use `route_8d`. (2) **`FrameReader` takes `pix_fmt` at construction** (default `rgb24`) — pass `pix_fmt="nv12"` for model YUV; frame access is `fr.get(fidx)` (single index, **no `count` kwarg**), pinned against the real API in Task 6. (3) **The logged `desired_curvature` (NPZ) is `controlsState`'s = post-`LAT_SMOOTH`/`get_action_from_model`** — so the replay must apply that same post-step (`get_curvature_from_output` + `LAT_SMOOTH_SECONDS` + `mlsim`/generation gating, `modeld.py:163-178`) to anchor against it; **Task 9 is split into 9a (warp + VISION-ONLY `lane_lines`/`road_edges` anchor — NOT `model_y20`, which is policy-derived via `plan[Plan.POSITION]`) and 9b (recurrent policy loop + `desiredCurvature` post-step anchor)** to de-risk the scope §12-#2 primary risk. (4) `get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame)` is called with **`rpyCalib` passed directly** as `device_from_calib_euler` (`modeld.py:294` — NO conversion; corrected per adversarial QA). (5) `params_stub` hardened (seed dict + `put` coerces non-str via `str(val).encode()`); the lifted inference path must **bypass `get_active_bundle`** (it does `ModelBundle(**params.get(...))`). (6) `TemporalState` (Task 8) is a throwaway scaffold — the real buffer is `FULL_HISTORY_BUFFER_LEN=99` (~5s) + `temporal_idxs_map` strides + `hidden_state` feedback, built for real in 9b; `WARMUP_S` ties to that, not a 2.0s literal. (7) `onnxruntime` left unpinned (confirm a macOS-arm64/py3.11 wheel at install).

**Hard constraints (do not violate):** no vehicle-control / `opendbc_repo/` / `panda/` / `selfdrive/` *driving-code* edits (import read-only); Mac only, never the Comma device for compute; pulling frames/logs from the device is read-only (tar-over-ssh); all generated files stay under `retrospective_lateral/results/` (gitignored).

---

## File Structure

New package `model_replay_sim/` (sibling of `retrospective_lateral/`, analysis-only):

- `model_replay_sim/__init__.py` — package marker/version.
- `model_replay_sim/config.py` — paths, weave band, tolerances, schema version, bundle registry, model-input constants.
- `model_replay_sim/params_stub.py` — minimal in-memory `Params` replacement so the lifted runner/transform code runs off-device.
- `model_replay_sim/compile_bundle.py` — wrap `compile3.py`: ONNX → Mac (`DEV=CPU`) pkl + provenance (tinygrad SHA, onnx sha) + tinygrad-vs-onnxruntime self-test (anchor tier a).
- `model_replay_sim/frames.py` — `FrameReader` wrapper: hevc → model-input YUV; segment + warmup handling.
- `model_replay_sim/calib.py` — the 3×3 `model_transform`: `get_warp_matrix(rpyCalib, intrinsics, bigmodel)` (rpyCalib passed directly) + the sunnypilot `camera_offset` shear with `0.9/0.1` EMA.
- `model_replay_sim/warp.py` — **the warp PIXEL pipeline** (the model input is NOT the 3×3 alone): apply `model_transform` to the NV12 buffer + pack the 12-channel current+prev tensor, mirroring `sunnypilot/modeld_v2/warp.py` (`make_frame_prepare`/`frames_to_tensor`; `FrameReader` nv12 is a contiguous `(h*w*3//2,)` array with no stride pad — reuse a numpy/cv2 `warpPerspective` equivalent, validated by the Phase-0 vision sanity check).
- `model_replay_sim/infer.py` — load compiled pkl(s) **+ a minimal bundle/metadata loader** (the real `TinygradRunner` pulls `models`/`input_shapes`/`is_20hz` from a bundle + `*_metadata.pkl` via `get_active_bundle`; a Params stub alone is insufficient — load metadata directly); recurrent vision→policy loop + temporal-state core; 2-model and OPM7 3-model (`plan+planplus`) paths.
- `model_replay_sim/parse.py` — wrap `parse_model_outputs.Parser` → `desiredCurvature` / `model_y20` / `orientation_rate` / `lane_lines`.
- `model_replay_sim/metrics.py` — weave-band RMS + **disengaged** eligibility gate (reuse `retrospective_lateral.code.signal_utils`).
- `model_replay_sim/anchor.py` — tier-a (self-test) + tier-b (replay-vs-log) fidelity report.
- `model_replay_sim/bundles.py` — fetch/verify each bundle's ONNX (ref-commit + git-lfs) + metadata-driven 2-vs-3-model + `big_img` need.
- `model_replay_sim/run.py` — orchestrator: parallel decode/preprocess/metrics, serial inference, decode-once/infer-N×, paired comparison + decision.
- `model_replay_sim/tests/` — pytest (synthetic + small real-segment regression).
- `retrospective_lateral/results/model_replay/` — generated artifacts (gitignored via the existing `results/.gitignore`).

**Build order is feasibility-gated:** Phase 0 (Tasks 1–9, stock-model end-to-end = the real GO/NO-GO) → Phase 1 (Tasks 10–12, bundles + throughput) → Phase 2 (Task 13, full run) → Phase 3 (Task 14, QA + report). **Do not pull any CD210/OPM7/Nevada hevc until Phase 0 passes.**

---

### Task 0: Day-1 feasibility gate (data + compile) — do this BEFORE writing any package code

**Files:** none created — this is a GO/NO-GO spike. Record results in the commit message / a scratch note.

The two cheapest kill signals are front-loaded here so we never build 8 tasks before learning the approach is dead.

- [ ] **Step 1: Confirm eligible-data sufficiency (the corrected scan)**

Run:
```bash
.venv311/bin/python - <<'PY'
import numpy as np, glob, os, sys; sys.path.insert(0,".")
from retrospective_lateral.code.signal_utils import filter_continuous
FS=20.0; allnpz=sorted(glob.glob("retrospective_lateral/results/cache/route_*.npz"))
fb=set(p.split("/")[1] for p in glob.glob("explorer_st_logs/route_*/*/fcamera.hevc"))
def runs(m):
    out=[];i=0;n=len(m)
    while i<n:
        if m[i]:
            j=i
            while j<n and m[j]: j+=1
            out.append((j-i)/FS); i=j
        else: i+=1
    return out
def scan(L):
    allr=[]
    for npz in L:
        z=dict(np.load(npz)); n=len(z['t']); g=lambda k,d:np.asarray(z.get(k,np.full(n,d)),float)
        dis=g('lat_active',0)<0.5; v=g('v_ego',np.nan); mph=v*2.23694
        yr=g('yaw_rate',np.nan); pc=np.full(n,np.nan); ok=np.isfinite(yr)&np.isfinite(v)&(v>3); pc[ok]=yr[ok]/v[ok]
        road=filter_continuous(pc,FS,lowpass_hz=0.035)
        m=dis&(mph>=10)&(mph<=85)&np.isfinite(road)&(np.abs(road)<0.005)&~(g('blinker',0)>0.5)&~(g('lane_change_state',0)>0.5)
        allr+=runs(m)
    allr.sort(reverse=True); return allr
for L,lab in [(allnpz,"FULL CORPUS"),([n for n in allnpz if os.path.basename(n)[:-4] in fb],"FRAME-BEARING NOW")]:
    a=scan(L); print(f"{lab}: total {sum(a)/60:.0f}min | >=20s:{sum(1 for x in a if x>=20)} >=60s:{sum(1 for x in a if x>=60)} | longest {a[0]:.0f}s")
PY
```
Expected: FRAME-BEARING NOW shows ~50+ runs ≥20 s (≈68–77 min). **GO** if ≥ ~30 runs ≥20 s with local frames; else pull more hevc (Task 10 mechanism) first.

- [ ] **Step 2: Init tinygrad + compile the in-tree stock ONNX on `DEV=CPU` (the true first feasibility signal)**

Run:
```bash
git submodule update --init tinygrad_repo
.venv311/bin/pip install onnxruntime   # confirm a macOS-arm64/py3.11 wheel resolves
DEV=CPU PYTHONPATH=tinygrad_repo:. .venv311/bin/python tinygrad_repo/examples/openpilot/compile3.py \
    selfdrive/modeld/models/driving_vision.onnx /tmp/driving_vision_cpu.pkl
```
(Read `compile3.py`'s real argv/env first — pin the exact form.) **GO** if it compiles and (with `SELFTEST=1`) tinygrad-CPU matches onnxruntime. **NO-GO** if `DEV=CPU` can't compile this ONNX (unsupported op) — then the whole Mac approach needs a different backend; STOP and report before building.

- [ ] **Step 3: Record the GO/NO-GO** (window counts + compile/selftest result) and proceed to Task 1 only on GO.

---

### Task 1: Environment + package scaffold + config

**Files:**
- Create: `model_replay_sim/__init__.py`, `model_replay_sim/config.py`, `model_replay_sim/tests/__init__.py`
- Modify: `requirements-analysis.txt` (add `onnxruntime`)
- Test: `model_replay_sim/tests/test_config.py`

- [ ] **Step 1: Initialize the tinygrad submodule and the onnxruntime dep**

Run:
```bash
git submodule update --init tinygrad_repo
.venv311/bin/python -c "import sys; sys.path.insert(0,'tinygrad_repo'); import tinygrad; print('tinygrad', tinygrad.__file__)"
```
Expected: tinygrad imports from `tinygrad_repo/tinygrad/`. (If it fails, STOP — the build cannot proceed; report the submodule/init error.)

Append to `requirements-analysis.txt` after the last line:
```text
onnxruntime==1.20.1
```
Then: `.venv311/bin/pip install onnxruntime==1.20.1` (used only by the Phase-0 self-test).

- [ ] **Step 2: Write the failing config test**

Create `model_replay_sim/tests/test_config.py`:
```python
from pathlib import Path
from model_replay_sim import config as C


def test_paths_and_constants():
    assert C.REPO_ROOT.name == "sunnypilot"
    assert C.RESULTS_ROOT == C.REPO_ROOT / "retrospective_lateral" / "results" / "model_replay"
    assert C.FS_HZ == 20.0
    assert C.WEAVE_BAND_HZ == (0.10, 0.35)
    assert C.MEDMODEL_INPUT_SIZE == (512, 256)
    assert C.TINYGRAD_PATH == C.REPO_ROOT / "tinygrad_repo"
    # anchor tolerances: tier-a tight, tier-b loose (device->Mac numerics differ)
    assert C.SELFTEST_ATOL < 1e-2
    assert C.ANCHOR_CORR_MIN == 0.95
    assert C.ANCHOR_BAND_RATIO == (0.85, 1.15)


def test_bundle_registry_has_refs():
    # ref commits verified during scope adversarial QA
    assert C.BUNDLES["CD210"]["ref"] == "55f66e22"
    assert C.BUNDLES["CD210"]["repo"] == "commaai/openpilot"
    assert C.BUNDLES["Nevada"]["ref"] == "3193eac5"
    assert C.BUNDLES["OPM7"]["repo"] == "sunnypilot/sunnypilot"
    assert C.BUNDLES["OPM7"]["split"] is True
    assert C.BUNDLES["CD210"]["split"] is False
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_config.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_replay_sim'`.

- [ ] **Step 4: Create the scaffold**

Create `model_replay_sim/__init__.py`:
```python
"""Same-scene model-replay simulator (Mac-native, analysis-only)."""
PACKAGE_VERSION = "0.1.0"
```

Create `model_replay_sim/tests/__init__.py` (empty file).

Create `model_replay_sim/config.py`:
```python
from __future__ import annotations
from pathlib import Path

SCHEMA_VERSION = "modelsim-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
TINYGRAD_PATH = REPO_ROOT / "tinygrad_repo"
RESULTS_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "model_replay"
CACHE_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "cache"  # reuse extracted rlog NPZ for eligibility/logged refs
LOG_ROOT = REPO_ROOT / "explorer_st_logs"

FS_HZ = 20.0
WEAVE_BAND_HZ = (0.10, 0.35)
LOW_FREQ_GUARD_HZ = 0.10  # curve content lives below this
MEDMODEL_INPUT_SIZE = (512, 256)

# Anchor tolerances (scope §8). Tier a = correctness (tight); tier b = realism (loose, NOT load-bearing).
SELFTEST_ATOL = 1e-3
SELFTEST_RTOL = 1e-2
ANCHOR_CORR_MIN = 0.95
ANCHOR_BAND_RATIO = (0.85, 1.15)

# Eligibility (disengaged, straight/gentle) — mirror retrospective_lateral conventions.
SPEED_BIN_MPH = 5.0
ROAD_CURV_ABS_MAX_1PM = 0.0015        # "straight" (tight)
GENTLE_CURV_ABS_MAX_1PM = 0.005       # "gentle" — used for disengaged scene selection
ROAD_LP_HZ = 0.035
WARMUP_S = 2.0  # placeholder; set from FULL_HISTORY_BUFFER_LEN/FS in Phase 1 (real value ~5 s)

BUNDLES = {
    "CD210":  {"ref": "55f66e22", "repo": "commaai/openpilot",     "split": False},
    "Nevada": {"ref": "3193eac5", "repo": "commaai/openpilot",     "split": False},
    "OPM7":   {"ref": "052692b2", "repo": "sunnypilot/sunnypilot", "split": True},
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_config.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add model_replay_sim/__init__.py model_replay_sim/config.py model_replay_sim/tests/__init__.py model_replay_sim/tests/test_config.py requirements-analysis.txt
git commit -m "model-sim: scaffold package + config"
```

---

### Task 2: Params stub (unblock the lifted runner/transform code)

**Files:**
- Create: `model_replay_sim/params_stub.py`
- Test: `model_replay_sim/tests/test_params_stub.py`

Rationale: `TinygradRunner`, `fill_model_msg`, and `get_active_bundle` import `openpilot.common.params` → `params_pyx` (uncompiled Cython). We never run the on-device config; an in-memory `Params` with the read/write surface those call sites use lets us reuse the inference code without building Cython.

- [ ] **Step 1: Write the failing test**

Create `model_replay_sim/tests/test_params_stub.py`:
```python
from model_replay_sim.params_stub import Params


def test_get_put_roundtrip():
    p = Params()
    assert p.get("CameraOffset") is None
    p.put("CameraOffset", b"-0.04")
    assert p.get("CameraOffset") == b"-0.04"


def test_typed_helpers():
    p = Params()
    p.put_bool("WideCameraOnly", True)
    assert p.get_bool("WideCameraOnly") is True
    assert p.get_bool("MissingKey") is False
    p.put("CameraOffset", b"-0.04")
    assert abs(p.get_float("CameraOffset") - (-0.04)) < 1e-9
    assert p.get_float("MissingKey", 1.22) == 1.22


def test_int_put_coerces_to_ascii_not_zero_buffer():
    p = Params({"CameraOffset": b"-0.04"})           # seed dict supported
    assert p.get("CameraOffset") == b"-0.04"
    p.put("ModelRunnerTypeCache", 54)                # runner writes an int here
    assert p.get("ModelRunnerTypeCache") == b"54"    # NOT bytes(54) (54 zero bytes)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_params_stub.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `model_replay_sim/params_stub.py`:
```python
from __future__ import annotations


class Params:
    """In-memory stand-in for openpilot.common.params.Params (off-device, no Cython).

    NOTE: the lifted inference path must NOT call get_active_bundle (it does
    `ModelManagerSP.ModelBundle(**params.get("ModelManager_ActiveBundle"))`, and
    `get` returns bytes/None, never a dict). We construct ModelReplayer from the
    fetched bundle metadata directly (Task 10), so this stub only needs to satisfy
    incidental reads/writes (e.g. ModelRunnerTypeCache) the runner makes en route.
    """

    def __init__(self, seed: dict | None = None):
        self._store: dict[str, bytes] = {}
        for k, v in (seed or {}).items():
            self.put(k, v)

    def _coerce(self, val) -> bytes:
        if isinstance(val, bytes):
            return val
        if isinstance(val, str):
            return val.encode()
        return str(val).encode()  # ints etc -> b"54", NOT bytes(54) (zero buffer)

    def get(self, key: str, block: bool = False, encoding=None):
        v = self._store.get(key)
        if v is not None and encoding is not None:
            return v.decode(encoding)
        return v

    def put(self, key: str, val):
        self._store[key] = self._coerce(val)

    def put_nonblocking(self, key: str, val):
        self.put(key, val)

    def get_bool(self, key: str) -> bool:
        return self._store.get(key) == b"1"

    def put_bool(self, key: str, val: bool):
        self._store[key] = b"1" if val else b"0"

    def get_float(self, key: str, default: float | None = None):
        v = self._store.get(key)
        return float(v) if v is not None else default

    def remove(self, key: str):
        self._store.pop(key, None)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_params_stub.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/params_stub.py model_replay_sim/tests/test_params_stub.py
git commit -m "model-sim: in-memory Params stub"
```

---

### Task 3: Calibration warp + camera_offset shear (pure math, TDD)

**Files:**
- Create: `model_replay_sim/calib.py`
- Test: `model_replay_sim/tests/test_calib.py`

The shear is the exact sunnypilot logic (`sunnypilot/modeld_v2/camera_offset_helper.py:apply_camera_offset` + the `0.9*old + 0.1*new` EMA in `update`). We replicate it offline driven by logged `liveCalibration` + the device `camera_offset` param (the −3.05° mount note in memory ties here).

- [ ] **Step 1: Write the failing test**

Create `model_replay_sim/tests/test_calib.py`:
```python
import numpy as np
from model_replay_sim.calib import apply_camera_offset, camera_offset_ema


def test_apply_camera_offset_matches_sunnypilot_shear():
    M = np.eye(3, dtype=np.float32)
    intr = np.array([[900., 0., 256.], [0., 900., 128.], [0., 0., 1.]], dtype=np.float32)
    height, offset = 1.22, -0.04
    out = apply_camera_offset(M, intr, height, offset)
    # shear[0,1] = offset/height ; shear[0,2] = -offset/height * cy
    assert np.isclose(out[0, 1], offset / height)
    assert np.isclose(out[0, 2], -offset / height * intr[1, 2])
    # non-identity M: exercise the actual shear @ M composition (catch transpose/sign bugs)
    M2 = np.array([[1., 0., 5.], [0., 1., 3.], [0., 0., 1.]], dtype=np.float32)
    shear = np.eye(3, dtype=np.float32)
    shear[0, 1] = offset / height
    shear[0, 2] = -offset / height * intr[1, 2]
    assert np.allclose(apply_camera_offset(M2, intr, height, offset), shear @ M2)


def test_camera_offset_ema_converges():
    val = 0.0
    for _ in range(200):
        val = camera_offset_ema(val, -0.04)
    assert abs(val - (-0.04)) < 1e-3  # 0.9/0.1 EMA converges to the target
    # one step
    assert np.isclose(camera_offset_ema(0.0, -0.04), 0.1 * -0.04)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_calib.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `model_replay_sim/calib.py`:
```python
from __future__ import annotations
import numpy as np


def apply_camera_offset(model_transform, intrinsics, height, offset_param):
    """Exact replica of sunnypilot CameraOffsetHelper.apply_camera_offset."""
    cy = intrinsics[1, 2]
    shear = np.eye(3, dtype=np.float32)
    shear[0, 1] = offset_param / height
    shear[0, 2] = -offset_param / height * cy
    return (shear @ model_transform).astype(np.float32)


def camera_offset_ema(prev: float, target: float) -> float:
    """The 0.9*old + 0.1*new smoothing used on actual_camera_offset."""
    return 0.9 * prev + 0.1 * target


# build_warp(liveCalibration_rpy, camera_offset, intrinsics, height) -> model_transform
# is integration code finalized in Phase 0 Task 9, where the exact get_warp_matrix
# signature (common/transformations/model.py) is read against a real segment and the
# result validated by the tier-b anchor. The pure shear/EMA above are unit-tested here.
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_calib.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/calib.py model_replay_sim/tests/test_calib.py
git commit -m "model-sim: camera_offset shear + EMA (pure, tested)"
```

---

### Task 4: Weave metric + disengaged eligibility (pure, TDD)

**Files:**
- Create: `model_replay_sim/metrics.py`
- Test: `model_replay_sim/tests/test_metrics.py`

Reuse `retrospective_lateral.code.signal_utils` (`filter_continuous`, `rms_masked`) for the band filter + masked RMS — do not re-implement. Metric channels (scope §9): `desiredCurvature`, `model_y20`, `orientation_rate`, `model_minus_lane`. Eligibility is **disengaged** (`latActive==0`), straight/gentle, no blinker/lane-change/near-lead.

- [ ] **Step 1: Write the failing test**

Create `model_replay_sim/tests/test_metrics.py`:
```python
import numpy as np
from model_replay_sim.metrics import weave_band_rms, disengaged_eligible_mask


def test_weave_band_rms_recovers_known_amplitude():
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    sig = 0.3 * np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz, amp 0.3 -> RMS ~0.212
    mask = np.ones_like(t, dtype=bool)
    rms = weave_band_rms(sig, mask, fs=fs, band=(0.10, 0.35))
    assert abs(rms - 0.3 / np.sqrt(2)) < 0.02


def test_disengaged_mask_keeps_driving_despite_steering_pressed_and_lead():
    # Regression for the false "data wall": disengaged driving must stay eligible even
    # though the human is steering (steering_pressed=1) and a lead is near.
    n = 400
    arrays = {
        "t": np.arange(n) / 20.0,
        "lat_active": np.r_[np.ones(200), np.zeros(200)].astype(np.float32),  # engaged then disengaged
        "v_ego": np.full(n, 20.0, np.float32),         # ~45 mph
        "yaw_rate": np.zeros(n, np.float32),           # straight
        "steering_pressed": np.ones(n, np.float32),    # human steering throughout (realistic when disengaged)
        "blinker": np.zeros(n, np.float32),
        "lane_change_state": np.zeros(n, np.float32),
        "lead_time_headway_s": np.full(n, 0.8, np.float32),  # near lead present
    }
    m = disengaged_eligible_mask(arrays)
    assert m[:200].sum() == 0       # engaged half excluded (lat_active=1)
    assert m[200:].sum() == 200     # ALL disengaged-straight kept despite pressed=1 + near lead (the fix)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_metrics.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Create `model_replay_sim/metrics.py`:
```python
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")
from retrospective_lateral.code import config as RC
from retrospective_lateral.code.signal_utils import filter_continuous, rms_masked
from model_replay_sim import config as C

MPS_TO_MPH = 2.2369362920544


def weave_band_rms(sig, mask, fs=C.FS_HZ, band=C.WEAVE_BAND_HZ) -> float:
    return rms_masked(filter_continuous(np.asarray(sig, float), fs, band=band), np.asarray(mask, bool))


def disengaged_eligible_mask(arrays) -> np.ndarray:
    """DISENGAGED scene selection for the same-scene model comparison.

    DO NOT inherit the ENGAGED weave-detector's eligibility — three of its filters are
    wrong here and created a false "data wall":
      * `~steering_pressed` is SELF-CONTRADICTORY: disengaged => the human IS steering =>
        steering_pressed is ~always true => `~pressed` deletes nearly all disengaged data.
      * near-lead exclusion is pointless: both models see the identical lead, so it cancels
        in the paired difference.
      * 2 s erosion deletes this driver's many brief takeovers.
    Keep ONLY: disengaged + straight/gentle + in-speed (10-85) + (optional) not blinker/lane-change.
    """
    n = len(arrays["t"])
    g = lambda k, d: np.asarray(arrays.get(k, np.full(n, d)), float)
    fs = C.FS_HZ
    disengaged = g("lat_active", 0.0) < 0.5           # human steering -> neutral scene
    blink = g("blinker", 0.0) > 0.5
    lc = g("lane_change_state", 0.0) > 0.5
    v = g("v_ego", np.nan)
    mph = v * MPS_TO_MPH
    in_speed = (mph >= 10) & (mph <= 85)              # NOT capped at 70: long highway straights are 70-80
    yr = g("yaw_rate", np.nan)
    pc = np.full(n, np.nan)
    ok = np.isfinite(yr) & np.isfinite(v) & (v > 3)
    pc[ok] = yr[ok] / v[ok]
    road = filter_continuous(pc, fs, lowpass_hz=C.ROAD_LP_HZ)
    gentle = np.isfinite(road) & (np.abs(road) < C.GENTLE_CURV_ABS_MAX_1PM)  # 0.005 (NOT the 0.0015 straight gate)
    return disengaged & ~blink & ~lc & in_speed & gentle   # no override mask, no near-lead, no erosion
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_metrics.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/metrics.py model_replay_sim/tests/test_metrics.py
git commit -m "model-sim: weave-band metric + disengaged eligibility"
```

---

### Task 5: Output parsing wrapper (TDD with synthetic tensors)

**Files:**
- Create: `model_replay_sim/parse.py`
- Test: `model_replay_sim/tests/test_parse.py`

Wrap `openpilot.selfdrive.modeld.parse_model_outputs.Parser` (imports msgq-free). Produce the four metric channels per frame: `desiredCurvature` (via `get_curvature_from_plan` / the plan ORIENTATION_RATE slice), `model_y20` (`position.y` interpolated at 20 m), `orientation_rate`, and `lane_center_y20` (for `model_minus_lane`). The exact slice indices come from `openpilot.selfdrive.modeld.constants.ModelConstants` / `Plan` — import them, do not hardcode.

- [ ] **Step 1: Write the failing test**

Create `model_replay_sim/tests/test_parse.py`:
```python
import numpy as np
from model_replay_sim.parse import interp_y_at_x


def test_interp_y_at_x_linear():
    xs = np.array([0., 10., 20., 30.])
    ys = np.array([0., 1., 2., 3.])
    assert np.isclose(interp_y_at_x(xs, ys, 20.0), 2.0)
    assert np.isclose(interp_y_at_x(xs, ys, 15.0), 1.5)
    assert np.isnan(interp_y_at_x(xs, ys, 40.0))  # beyond horizon -> nan
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_parse.py -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement the pure helper now; wire Parser in Phase 0**

Create `model_replay_sim/parse.py`:
```python
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, ".")
from openpilot.selfdrive.modeld.parse_model_outputs import Parser  # msgq-free (verified)


def interp_y_at_x(xs, ys, xq: float) -> float:
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 2 or xq < np.nanmin(xs[ok]) or xq > np.nanmax(xs[ok]):
        return float("nan")
    return float(np.interp(xq, xs[ok], ys[ok]))


def parse_frame_outputs(raw_outputs: dict, split: bool) -> dict:
    """raw_outputs: dict[str, np.ndarray] straight from the model run.
    For OPM7 (split=True) the caller has already merged plan = plan + planplus
    (see infer.py). Returns the per-frame metric channels.

    NOTE: the exact Parser call (parse_vision_outputs / parse_policy_outputs) and the
    Plan.ORIENTATION_RATE / position slices are PINNED in Phase 0 Task 9 against a real
    stock-model run and validated by the tier-b anchor. This wrapper is the seam; the
    pure interp helper above is unit-tested.
    """
    raise NotImplementedError("wired in Phase 0 Task 9 once Parser slices are confirmed on real output")
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_parse.py -q`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/parse.py model_replay_sim/tests/test_parse.py
git commit -m "model-sim: output-parse seam + interp helper"
```

---

### Task 6: Frame reader wrapper

**Files:**
- Create: `model_replay_sim/frames.py`
- Test: `model_replay_sim/tests/test_frames.py`

Wrap `openpilot.tools.lib.framereader.FrameReader` (imports clean). Decode `fcamera.hevc` (and `ecamera.hevc` if a bundle needs `big_img`) to the model-input YUV. The exact `FrameReader` API (`.get(idx, count, pix_fmt=...)`, frame count) is confirmed against a real local segment in this task.

- [ ] **Step 1: Write a real-segment smoke test (skips if no local hevc)**

Create `model_replay_sim/tests/test_frames.py`:
```python
import glob
import pytest
from model_replay_sim.frames import open_road_frames, frame_count, read_frame


def _a_local_segment():
    # route_8d (and route_7*/8*) have fcamera.hevc; route_c* do NOT (rlog only).
    hits = sorted(glob.glob("explorer_st_logs/route_8d/0000008d--*--0/fcamera.hevc"))
    return hits[0] if hits else None


@pytest.mark.skipif(_a_local_segment() is None, reason="no local fcamera.hevc")
def test_decode_one_nv12_frame():
    path = _a_local_segment()
    fr = open_road_frames(path)           # opens with pix_fmt='nv12'
    assert frame_count(fr) > 100
    frame = read_frame(fr, 50)            # single index; real accessor pinned in Step 1
    # nv12: a comma road camera decodes to a non-empty array
    assert frame is not None and getattr(frame, "size", 0) > 0
```

- [ ] **Step 2: Run (it FAILS to import, or skips if no hevc)**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_frames.py -q`
Expected: FAIL — module not found (or SKIP after implement if no local hevc).

- [ ] **Step 3: Implement**

First read the real accessor (Task 6 Step 1): `grep -nE 'def get|def __getitem__|yield from' openpilot/tools/lib/framereader.py`. `pix_fmt` is a **constructor** arg (default `rgb24`); `frame_count` is an instance attr. Then create `model_replay_sim/frames.py`:
```python
from __future__ import annotations
import sys
sys.path.insert(0, ".")
from openpilot.tools.lib.framereader import FrameReader


def open_road_frames(hevc_path: str) -> FrameReader:
    return FrameReader(hevc_path, pix_fmt="nv12")   # model-input YUV, NOT default rgb24


def frame_count(fr: FrameReader) -> int:
    return int(fr.frame_count)


def read_frame(fr: FrameReader, fidx: int):
    """Single-frame read. Pin the exact accessor from Step 1 (e.g. fr.get(fidx)
    or next(iter(fr[fidx:fidx+1]))); NO `count=` kwarg exists."""
    return fr.get(fidx)
```

- [ ] **Step 4: Run to verify pass/skip**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_frames.py -q`
Expected: PASS or SKIP.

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/frames.py model_replay_sim/tests/test_frames.py
git commit -m "model-sim: FrameReader wrapper + smoke test"
```

---

### Task 7: Compile wrapper (ONNX → Mac CPU pkl) + tier-a self-test

**Files:**
- Create: `model_replay_sim/compile_bundle.py`
- Test: `model_replay_sim/tests/test_compile_bundle.py`

Integration: wrap tinygrad's `examples/openpilot/compile3.py` (read its exact CLI/entry now that the submodule is initialized — it takes an ONNX path, runs with `DEV=CPU`, captures via `TinyJit`, pickles, and has a `SELFTEST` onnxruntime comparison). Record provenance: onnx sha256, **tinygrad submodule SHA**, output pkl path.

- [ ] **Step 1: Read the real compile3 entry point**

Run:
```bash
sed -n '1,80p' tinygrad_repo/examples/openpilot/compile3.py
```
Confirm: the input ONNX arg, the output pkl arg/env, `DEV`/`SELFTEST` env vars. Use these exact names below.

- [ ] **Step 2: Write the failing integration test (compiles the in-tree STOCK onnx)**

Create `model_replay_sim/tests/test_compile_bundle.py`:
```python
from pathlib import Path
from model_replay_sim.compile_bundle import compile_onnx_to_mac_pkl
from model_replay_sim import config as C


def test_compile_stock_vision_onnx(tmp_path):
    onnx = C.REPO_ROOT / "selfdrive/modeld/models/driving_vision.onnx"
    out = tmp_path / "driving_vision_cpu.pkl"
    rec = compile_onnx_to_mac_pkl(onnx, out, selftest=True)
    assert out.exists() and out.stat().st_size > 0
    assert rec["selftest_passed"] is True
    assert len(rec["tinygrad_sha"]) >= 7
    assert rec["onnx_sha256"]
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_compile_bundle.py -q`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement (fill the exact compile3 invocation from Step 1)**

Create `model_replay_sim/compile_bundle.py`:
```python
from __future__ import annotations
import hashlib, os, subprocess, sys
from pathlib import Path
from model_replay_sim import config as C


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def _tinygrad_sha() -> str:
    out = subprocess.run(["git", "-C", str(C.TINYGRAD_PATH), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def compile_onnx_to_mac_pkl(onnx_path: Path, out_pkl: Path, selftest: bool = True) -> dict:
    """Compile an ONNX to a Mac (DEV=CPU) tinygrad pkl via examples/openpilot/compile3.py.
    Fill the exact argv/env from Task 7 Step 1. Pattern (confirm against the file):
        DEV=CPU [SELFTEST=1] python tinygrad_repo/examples/openpilot/compile3.py <onnx> <out_pkl>
    """
    env = dict(os.environ, DEV="CPU", PYTHONPATH=f"{C.TINYGRAD_PATH}:{C.REPO_ROOT}")
    if selftest:
        env["SELFTEST"] = "1"
    cmd = [sys.executable, str(C.TINYGRAD_PATH / "examples/openpilot/compile3.py"),
           str(onnx_path), str(out_pkl)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    passed = proc.returncode == 0 and out_pkl.exists()
    return {
        "onnx": str(onnx_path), "out_pkl": str(out_pkl),
        "onnx_sha256": _sha256(onnx_path), "tinygrad_sha": _tinygrad_sha(),
        "selftest_passed": bool(passed and selftest),
        "returncode": proc.returncode, "stderr_tail": proc.stderr[-2000:],
    }
```

- [ ] **Step 5: Run to verify pass (this is the first real GO/NO-GO signal — tinygrad CPU compile + onnxruntime match on the stock model)**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_compile_bundle.py -q -s`
Expected: PASS. If the compile fails (unsupported op on `DEV=CPU`) or selftest mismatches, STOP and report — Metal/alternate-op work would be needed and the feasibility call changes.

- [ ] **Step 6: Commit**

```bash
git add model_replay_sim/compile_bundle.py model_replay_sim/tests/test_compile_bundle.py
git commit -m "model-sim: ONNX->Mac-CPU pkl compile + tier-a self-test"
```

---

### Task 8: Inference loop (stock 2-model path first)

**Files:**
- Create: `model_replay_sim/infer.py`
- Test: `model_replay_sim/tests/test_infer.py`

Lift the inference path from `sunnypilot/models/runners/tinygrad/tinygrad_runner.py` (the pkl load + `run_model`) behind the `Params` stub, and the recurrent vision→policy loop from `sunnypilot/modeld_v2/modeld.py` `ModelState` (temporal `features_buffer`, `prev_desired_curv` feedback, desire/traffic inputs). **Read those two files now and reproduce only the inference math** — drop the `VisionIpcClient`/`SubMaster`/msgq glue.

- [ ] **Step 1: Read the runner + ModelState loop**

Run:
```bash
sed -n '1,140p' sunnypilot/models/runners/tinygrad/tinygrad_runner.py
sed -n '1,200p' sunnypilot/modeld_v2/modeld.py
```
Note the exact: pkl load call (`TinyJit`/pickle), input tensor names (`input_imgs`/`big_input_imgs`/`desire`/`traffic_convention`/`features_buffer`/`prev_desired_curv`), `features_buffer` stride/length per `temporal_idxs`, and how outputs come back.

- [ ] **Step 2: Write a structural test (state buffer mechanics, no GPU)**

Create `model_replay_sim/tests/test_infer.py`:
```python
import numpy as np
from model_replay_sim.infer import TemporalState


def test_temporal_state_rolls_and_warms_up():
    ts = TemporalState(buffer_len=25, feature_width=8)
    assert not ts.is_warm
    for i in range(25):
        ts.push(np.full(8, float(i), np.float32))
    assert ts.is_warm
    buf = ts.buffer()
    assert buf.shape == (25, 8)
    assert np.isclose(buf[-1, 0], 24.0)   # most recent last
    assert np.isclose(buf[0, 0], 0.0)
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_infer.py -q`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement the testable state core + the inference seam**

Create `model_replay_sim/infer.py`:
```python
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, ".")
# Inject the Params stub BEFORE importing the runner so its `from openpilot.common.params import Params` resolves.
import model_replay_sim.params_stub as _stub
sys.modules.setdefault("openpilot.common.params", _stub)


class TemporalState:
    """Recurrent feature buffer with explicit warmup, mirroring modeld's features_buffer."""
    def __init__(self, buffer_len: int, feature_width: int):
        self.buffer_len = buffer_len
        self._buf = np.zeros((buffer_len, feature_width), np.float32)
        self._n = 0

    def push(self, feat: np.ndarray):
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = feat
        self._n += 1

    @property
    def is_warm(self) -> bool:
        return self._n >= self.buffer_len

    def buffer(self) -> np.ndarray:
        return self._buf.copy()


class ModelReplayer:
    """Loads a compiled bundle and runs the recurrent vision->policy loop on file frames.
    Implemented in Phase 0 Task 9 from the runner/ModelState read (Step 1): 2-model path
    for CD210/Nevada/stock; OPM7 3-model split (plan = plan + planplus) added in Task 11.
    Returns, per frame, the raw output dict consumed by parse.parse_frame_outputs.
    """
    def __init__(self, pkl_paths: dict, metadata: dict, split: bool = False):
        self.pkl_paths = pkl_paths
        self.metadata = metadata
        self.split = split

    def run_segment(self, warped_yuv_iter, desire=None, traffic_convention=None):
        raise NotImplementedError("wired in Phase 0 Task 9 against the real runner/ModelState")
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_infer.py -q`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add model_replay_sim/infer.py model_replay_sim/tests/test_infer.py
git commit -m "model-sim: temporal-state core + inference seam"
```

---

### Task 9a: Phase-0 part 1 — warp PIXEL pipeline + vision pass (cross-model sanity, not strict fidelity)

**Files:**
- Modify: `model_replay_sim/calib.py` (add `build_warp`), `model_replay_sim/infer.py` (add `run_vision`)
- Create: `model_replay_sim/warp.py` (the pixel pipeline), `model_replay_sim/anchor.py`
- Test: `model_replay_sim/tests/test_anchor_vision.py`

Build the **warp pixel pipeline + vision pass** (no recurrent policy). Two corrections from adversarial QA:

1. **The 3×3 `model_transform` is not the model input** — `warp.py` must apply it to the NV12 buffer and pack the model's expected tensor (mirror `sunnypilot/modeld_v2/warp.py` `make_frame_prepare`/`frames_to_tensor`; a numpy/`cv2.warpPerspective` equivalent is fine). This is the biggest piece of Phase 0.
2. **This is a cross-model SANITY check, not strict fidelity.** The in-tree **stock** ONNX never ran on any Explorer route, and `route_8d` ran a *different/unknown* model — so we cannot strictly match its log here. We anchor on **vision-only** `lane_lines`/`road_edges` (from `parse_vision_outputs`, `parse_model_outputs.py:99-100` → `modelV2.laneLines`/`roadEdges`; **not** `model_y20`, which is policy-derived via `plan[Plan.POSITION]`) at a **soft bar** — on clear lanes, supercombo-family models predict similar lane geometry, so a decent correlation confirms the **warp/frame/vision pipeline is sane**. The **strict same-model fidelity anchor is Task 11** (replay the *actual logged* bundle, e.g. Nevada on a Nevada route). The lane/road-edge channels are also 100% finite on eligible windows (verified) and already cached.

- [ ] **Step 1: Pick a disengaged segment that has BOTH a cached NPZ and local hevc**

Run:
```bash
.venv311/bin/python - <<'PY'
import numpy as np, glob
z = dict(np.load("retrospective_lateral/results/cache/route_8d.npz"))
dis = (np.asarray(z['lat_active'])<0.5)
mph = np.asarray(z['v_ego'],float)*2.23694
elig = dis & (mph>=10) & (mph<=70)
print("route_8d disengaged:", int(dis.sum()), "elig(10-70mph):", int(elig.sum()))
for k in ['lane_left_y20','lane_right_y20','road_edge_left_y20','road_edge_right_y20']:
    print(f"  {k} finite_on_elig:", float(np.isfinite(np.asarray(z[k],float)[elig]).mean()))
print("hevc segs:", sorted(glob.glob("explorer_st_logs/route_8d/0000008d--*--*/fcamera.hevc"))[:3])
PY
```
Record `route_8d`, a segment index with a disengaged + 10–70 mph stretch, and the time window. (route_c* has no hevc; do not use it.)

- [ ] **Step 2: Implement `calib.build_warp`**

Add to `model_replay_sim/calib.py`. `get_warp_matrix(device_from_calib_euler, intrinsics, bigmodel_frame)` takes a **device-from-calib euler rotation** (convert from the logged `liveCalibration.rpyCalib`), not raw rpy:
```python
import sys; sys.path.insert(0, ".")
from openpilot.common.transformations.model import get_warp_matrix

def build_warp(rpy_calib, camera_offset_ema_value, intrinsics, height, bigmodel_frame=False):
    # rpyCalib is passed DIRECTLY as device_from_calib_euler (modeld.py:294) — no conversion.
    # NOTE: this 3x3 alone is NOT the model input — the compiled warp pixel pipeline
    # (sunnypilot/modeld_v2/warp.py make_frame_prepare/Tensor.from_blob/stride_pad) applies it
    # to the NV12 buffer + packs the 12-channel current+prev tensor. That stage is its own task.
    M = get_warp_matrix(rpy_calib, intrinsics, bigmodel_frame)
    return apply_camera_offset(M, intrinsics, height, camera_offset_ema_value)
```

- [ ] **Step 3: Vision-only anchor test (`model_y20`)**

Create `model_replay_sim/tests/test_anchor_vision.py`:
```python
import pytest
from model_replay_sim.anchor import replay_vision_vs_logged
from model_replay_sim import config as C

SEG = None  # ("route_8d", seg_idx, t0, t1) from Step 1; None -> skip


@pytest.mark.skipif(SEG is None, reason="set SEG to the chosen disengaged route_8d segment")
def test_vision_lane_lines_are_sane():
    # CROSS-MODEL sanity (stock ONNX vs route_8d's different/unknown model): a decent
    # correlation on clear-lane geometry confirms warp/frames/vision are sane. NOT strict
    # fidelity (that is Task 11, same-model). Soft bar 0.80, not the 0.95 fidelity bar.
    rep = replay_vision_vs_logged(*SEG)
    SANITY = 0.80
    for ch in ("lane_left_y20", "lane_right_y20", "road_edge_left_y20", "road_edge_right_y20"):
        assert rep[f"corr_{ch}"] >= SANITY
```

- [ ] **Step 4: Implement `anchor.replay_vision_vs_logged` + `infer.run_vision`**

`run_vision` loads the compiled **vision** pkl, feeds warped nv12 frames, and parses `Parser().parse_vision_outputs(...)` → `lane_lines`/`road_edges` (no recurrent state, no policy). `anchor` interpolates the replayed left/right lane-line and road-edge `y@20m` and compares to the NPZ `lane_left_y20`/`lane_right_y20`/`road_edge_left_y20`/`road_edge_right_y20` over the eligible window (Pearson per channel). A pass validates warp+calib+frame-decode+camera_offset with zero policy-loop confound.

- [ ] **Step 5: Run the vision anchor; commit**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_anchor_vision.py -q -s`
Expected: PASS (validates warp+frames+vision). If it fails, the warp/camera_offset/frame-format is wrong — fix here before 9b.
```bash
git add model_replay_sim/calib.py model_replay_sim/infer.py model_replay_sim/anchor.py model_replay_sim/tests/test_anchor_vision.py
git commit -m "model-sim: Phase-0a warp + vision model_y20 anchor"
```

---

### Task 9b: Phase-0 GO/NO-GO — recurrent policy loop + post-step; gate on PIPELINE VALIDITY (not a cross-model log match)

**Files:**
- Modify: `model_replay_sim/infer.py` (recurrent loop), `model_replay_sim/parse.py` (fill `parse_frame_outputs`), `model_replay_sim/anchor.py`
- Test: `model_replay_sim/tests/test_pipeline_go_nogo.py`

This builds the recurrent core (the scope §12-#2 primary risk) and applies the `desiredCurvature` post-step. **The GO/NO-GO gate is pipeline VALIDITY, not fidelity to `route_8d`'s log** (route_8d ran a different/unknown model — a strict match is neither expected nor required; per scope §8 the verdict rests on identical-Mac pairing, and the strict same-model fidelity anchor is Task 11). Gate on: (a) the end-to-end pipeline **runs** on the chosen segment without error; (b) **same-bundle determinism** — replay the stock ONNX twice → identical within float tolerance; (c) **physical sanity** — `desiredCurvature` is finite, bounded (`|k| < 0.02`), non-constant, and reflects the `get_action_from_model` post-step (`get_curvature_from_output` + `LAT_SMOOTH_SECONDS` + `mlsim`/generation gating, `modeld.py:163-178`); (d) the weave-band RMS is a plausible non-degenerate value. A wrong temporal stride that produces *plausible-but-wrong* output is caught later by the Task-11 same-model anchor — note this residual risk.

- [ ] **Step 1: Read the real recurrent loop + post-step**

Run: `sed -n '80,180p' sunnypilot/modeld_v2/modeld.py` — note `FULL_HISTORY_BUFFER_LEN`, `temporal_idxs_map`, `features_buffer` roll from `hidden_state`, `prev_desired_curv` feedback, `desire` pulse, and `get_action_from_model`/`get_curvature_from_output`/`LAT_SMOOTH_SECONDS`. Reproduce only this math (drop the msgq glue). Set `config.WARMUP_S` from `FULL_HISTORY_BUFFER_LEN/FS_HZ`.

- [ ] **Step 2: Implement the recurrent loop + the post-step**

In `infer.ModelReplayer.run_segment` (2-model): per frame run vision→policy with the real `features_buffer`/`hidden_state`/`prev_desired_curv` feedback; in `parse.parse_frame_outputs` apply `get_curvature_from_output` + `LAT_SMOOTH` to produce `desired_curvature` matching `controlsState`, plus raw `model_y20`/`orientation_rate`/`lane_center_y20`.

- [ ] **Step 3: GO/NO-GO pipeline-validity test (determinism + sanity)**

Create `model_replay_sim/tests/test_pipeline_go_nogo.py`:
```python
import numpy as np
import pytest
from model_replay_sim.anchor import replay_pipeline_check

SEG = None  # same disengaged route_8d segment as 9a; None -> skip


@pytest.mark.skipif(SEG is None, reason="set SEG to the chosen disengaged route_8d segment")
def test_pipeline_runs_deterministic_and_sane():
    a = replay_pipeline_check(*SEG)   # runs the stock ONNX end-to-end (warp->vision->policy->post-step)
    b = replay_pipeline_check(*SEG)   # again -> determinism
    dc = np.asarray(a["desired_curvature"])
    assert np.isfinite(dc).mean() > 0.95          # produces output
    assert np.nanmax(np.abs(dc)) < 0.02           # bounded (ford curvature ceiling)
    assert np.nanstd(dc) > 1e-5                    # non-constant (reflects the scene)
    assert np.allclose(dc, np.asarray(b["desired_curvature"]),
                       atol=1e-4, rtol=0, equal_nan=True)   # same-bundle determinism
    assert 1e-6 < a["weave_band_rms"] < 1e-2       # plausible, non-degenerate weave amplitude
    assert a["post_step_applied"] is True          # get_action_from_model / LAT_SMOOTH was applied
```

- [ ] **Step 4: Run the GO/NO-GO**

Run: `.venv311/bin/python -m pytest model_replay_sim/tests/test_pipeline_go_nogo.py -q -s`
Expected: PASS. **This is the decision point for the whole approach.** If the pipeline runs deterministically and produces sane, bounded, scene-varying `desiredCurvature` with the post-step applied → **GO** to Phase 1 (fetch real bundles + the same-model fidelity anchor). If it can't run end-to-end, or output is degenerate/non-deterministic, fix the warp/temporal-loop/post-step; if unresolvable, STOP and report (fall back to on-road A/B). Do **not** gate on matching route_8d's log — that's a different model.

- [ ] **Step 5: Full suite + commit**

```bash
.venv311/bin/python -m pytest model_replay_sim/tests -q
git add model_replay_sim/infer.py model_replay_sim/parse.py model_replay_sim/anchor.py model_replay_sim/tests/test_pipeline_go_nogo.py model_replay_sim/config.py
git commit -m "model-sim: Phase-0b recurrent policy + post-step; pipeline-validity GO/NO-GO"
```

---

### Task 10: Bundle fetch + metadata (Phase 1, only after Task 9 passes)

**Files:**
- Create: `model_replay_sim/bundles.py`
- Test: `model_replay_sim/tests/test_bundles.py`

- [ ] **Step 1: Fetch the three ONNX from their ref commits (git-lfs)**

For each bundle in `C.BUNDLES`, in a throwaway worktree/clone of the `repo` at `ref`, `git lfs pull` `selfdrive/modeld/models/driving_*.onnx`, and copy into `retrospective_lateral/results/model_replay/onnx/<bundle>/`. Read each `*_metadata.pkl` (`input_shapes`/`output_slices`) to record: 2-vs-3-model, and whether `big_img` (wide camera) is consumed.
Run a one-off fetch script; record the resolved onnx sha256 per bundle.

- [ ] **Step 2: Write the metadata test**

Create `model_replay_sim/tests/test_bundles.py`:
```python
import pytest
from model_replay_sim.bundles import bundle_spec
from model_replay_sim import config as C
import os


@pytest.mark.skipif(not (C.RESULTS_ROOT / "onnx" / "OPM7").exists(), reason="bundles not fetched")
def test_opm7_is_three_model_split():
    spec = bundle_spec("OPM7")
    assert spec["split"] is True
    assert "off_policy" in spec["models"] and "on_policy" in spec["models"]


@pytest.mark.skipif(not (C.RESULTS_ROOT / "onnx" / "CD210").exists(), reason="bundles not fetched")
def test_cd210_is_two_model():
    spec = bundle_spec("CD210")
    assert spec["split"] is False
    assert "big_img" in spec  # bool: whether wide camera needed
```

- [ ] **Step 3-5:** Implement `bundles.py` (`bundle_spec(name)` reads the fetched onnx + metadata, returns model list / split flag / big_img flag / sha), compile each via Task 7, run; commit.

```bash
git add model_replay_sim/bundles.py model_replay_sim/tests/test_bundles.py
git commit -m "model-sim: bundle fetch + metadata-driven spec"
```

---

### Task 11: OPM7 3-model split + the STRICT same-model fidelity anchor (the real tier-b)

**Files:**
- Modify: `model_replay_sim/infer.py`, `model_replay_sim/parse.py`, `model_replay_sim/anchor.py`
- Test: `model_replay_sim/tests/test_split.py`, `model_replay_sim/tests/test_fidelity_anchor.py`

Two things land here: the OPM7 split path, and the **strict same-model fidelity anchor that Phase 0 deferred** (Phase 0 only proved the pipeline *runs*; this proves it's *faithful*).

- [ ] **Step 1: OPM7 split path.** Add `TinygradSplitRunner`: run vision + onPolicy + offPolicy, merge `plan = plan + planplus` (and the `plan.pop` when onPolicy present), per `sunnypilot/models/runners/tinygrad/tinygrad_runner.py:107-129`. Drive shapes/slices from each bundle's metadata (Task 10), never hardcode. Structural test on synthetic dicts that the merge sums `plan+planplus` and pops correctly.

- [ ] **Step 2: Strict fidelity anchor — replay each bundle on a route that RAN it, vs its OWN log.** This is the real tier-b (corr ≥ `C.ANCHOR_CORR_MIN` = 0.95, weave-band ratio in `C.ANCHOR_BAND_RATIO`), because the model matches the log:
  - **Nevada** → route `c4`/`c5`/`c6`/`c7` (ran Nevada; pull their `fcamera.hevc`, read-only — only rlog was pulled before).
  - **CD210** → a CD210 route (`b8`/`c1`/`c2`/`c3`; pull one route's `fcamera.hevc`).
  - **OPM7** → `route_7f` (OPM7; check for local frames first, else pull).
  Apply the `get_action_from_model` post-step so the replayed `desiredCurvature` matches `controlsState.desiredCurvature`. Per-bundle-family fidelity is the §8 safeguard that the 2-model vs 3-model paths don't bias the metric.

- [ ] **Step 3: Fidelity test.**
```python
import pytest
from model_replay_sim.anchor import replay_vs_logged
from model_replay_sim import config as C

# (bundle, route, seg, t0, t1) where the route actually ran that bundle; None -> skip
NEVADA_SEG = None; CD210_SEG = None; OPM7_SEG = None


@pytest.mark.parametrize("seg", [s for s in (NEVADA_SEG, CD210_SEG, OPM7_SEG) if s])
def test_same_model_fidelity(seg):
    rep = replay_vs_logged(*seg)   # bundle == the model that drove -> strict match expected
    assert rep["corr_desired_curv"] >= C.ANCHOR_CORR_MIN
    lo, hi = C.ANCHOR_BAND_RATIO
    assert lo <= rep["weave_rms_ratio"] <= hi
```
If a bundle family fails fidelity here (not in Phase 0), debug its warp/temporal/post-step/metadata before trusting its weave numbers. **A bundle that can't pass its own-model anchor cannot be in the comparison.**

- [ ] **Step 4: Run + commit**
```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_split.py model_replay_sim/tests/test_fidelity_anchor.py -q -s
git add model_replay_sim/infer.py model_replay_sim/parse.py model_replay_sim/anchor.py model_replay_sim/tests/test_split.py model_replay_sim/tests/test_fidelity_anchor.py
git commit -m "model-sim: OPM7 split + strict same-model fidelity anchor"
```

---

### Task 12: Throughput benchmark + lock pre-registration

**Files:**
- Create: `model_replay_sim/run.py` (benchmark entry), `retrospective_lateral/results/model_replay/PREREGISTRATION.md`
- Test: `model_replay_sim/tests/test_run_select.py`

- [ ] **Step 1:** Write `select_segments(catalog)` test (picks disengaged straight/gentle segments spanning both eras, deterministic ordering, capped N). Implement it.
- [ ] **Step 2: Benchmark** — time N frames × 1 bundle on `DEV=CPU` with W∈{1,4,8,12} `ProcessPoolExecutor` workers (decode-once cache), and once on `DEV=METAL` if it imports; record frames/sec and projected wall-clock for the candidate set. Write results to the benchmark CSV.
- [ ] **Step 3: Lock `PREREGISTRATION.md`** — fixed band, eligibility (disengaged), window length, warmup (from Task 10 buffer length), the exact segment set sized from Step 2, primary metric (paired Wilcoxon on per-window model-vs-model `desiredCurvature`/`model_y20`/`orientation_rate`/`model_minus_lane`), and the decision rule. Commit it before any cross-model number is computed.

```bash
git add model_replay_sim/run.py model_replay_sim/tests/test_run_select.py retrospective_lateral/results/model_replay/PREREGISTRATION.md
git commit -m "model-sim: throughput benchmark + locked pre-registration"
```

(Note: `PREREGISTRATION.md` lands under the gitignored `results/` — to keep a committed copy of the locked rule, also write it to `docs/superpowers/` if a tracked record is wanted.)

---

### Task 13: Full registered run + paired comparison (Phase 2)

**Files:**
- Modify: `model_replay_sim/run.py`
- Test: `model_replay_sim/tests/test_pairing.py`

- [ ] **Step 1:** Write a test for the paired comparator: given two per-window metric tables (same windows), it returns per-channel ratio + paired Wilcoxon p + sign split, and refuses to compare non-identical window sets (asserts pairing). Implement it.
- [ ] **Step 2:** Orchestrate: for each registered segment — decode+warp once (parallel CPU pool), run each bundle (serial inference, decode-once/infer-N×), parse, compute per-window weave metrics; write `weave_by_window.csv` and per-bundle `modelv2.parquet`.
- [ ] **Step 3:** Paired comparison across bundles on identical windows → `cross_model_summary.csv` with the locked decision rule; emit the verdict (incl. whether Nevada returns to the lever list, and the CD210-vs-OPM7 result).
- [ ] **Step 4: Self-checks** — rerun-stability (float tol), the same-bundle null (≈0 diff), and band/warmup sensitivity.
- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/run.py model_replay_sim/tests/test_pairing.py
git commit -m "model-sim: full registered run + paired cross-model comparison"
```

---

### Task 14: QA + findings report (Phase 3)

**Files:**
- Create: `docs/superpowers/reports/2026-06-DD-same-scene-model-replay-findings.md`

- [ ] **Step 1:** Run the full suite (`.venv311/bin/python -m pytest model_replay_sim/tests -q`) and confirm green.
- [ ] **Step 2:** Verify constraints: `git status --short opendbc_repo panda` empty; results gitignored; no driving code touched.
- [ ] **Step 3:** Collaborative then adversarial QA on the *results* (the project standard): independently re-derive the headline paired numbers; attack the anchor fidelity, the disengaged-scene assumption, and the OPM7-split comparability; verify the verdict survives.
- [ ] **Step 4:** Write the dated findings report: per-bundle weave, the paired CD210-vs-OPM7 and Nevada-vs-CD210 results, anchor fidelity, confounders/limits (open-loop caveat), and the recommendation — including whether the on-road interleaved A/B is still needed for felt-weave confirmation.
- [ ] **Step 5: Commit** the report.

```bash
git add docs/superpowers/reports/2026-06-DD-same-scene-model-replay-findings.md
git commit -m "docs: same-scene model-replay findings"
```

---

## Self-Review

**Spec coverage (scope §-by-§):** §1 objective → Tasks 9a/9b/13; §3 env (submodule init day-1, CPU compile, params stub, onnxruntime) → **Task 0** + Tasks 1/2; §4 architecture (compile/reuse/Params stub + bundle/metadata load/re-implement loop) → Tasks 2/7/8/9b; §5 components → one task each (`config`/`params_stub`/`compile_bundle`/`frames`/`calib`/**`warp`**/`infer`/`parse`/`metrics`/`anchor`/`bundles`/`run`); §6 open-loop caveat → Task 14 report; §7 ONNX fetch + **disengaged scenes (corrected eligibility, Task 4)** + big_img-from-metadata → Tasks 10/12; §8 anchor: tier-a self-test → Task 0/7, **pipeline-validity GO/NO-GO → Task 9b**, **strict same-model fidelity → Task 11**, identical-Mac pairing → Task 13; §9 metric/pre-registration/paired → Tasks 4/12/13; §10 parallelization/throughput → Task 12 benchmark + Task 13; §11 outputs → Tasks 9-13; §12 risks (data-sufficiency → Task 0, throughput → Task 12, OPM7-split + warp-pixel → Tasks 11/9a, ONNX routine fetch → Task 10) ; §13 phasing → Task-0-first ordering; §14 open questions → resolved.

**Adversarial-QA structural fixes folded into tasks:** data-wall overturned (corrected eligibility, Task 4 + Task 0 gate); warp PIXEL pipeline now first-class (`warp.py`, Task 9a); GO/NO-GO no longer matches a different-model log (Task 9b = pipeline validity; Task 11 = strict same-model fidelity on a route that ran that bundle); `rpyCalib` passed directly (no conversion); bundle/metadata loading noted (runner not Params-stub alone); fail-fast ordering (Task 0 day-1 data scan + stock compile).

**Placeholder note (intentional, not failures):** Tasks 5/8/9 defer the *exact* `Parser` slice names, the tinygrad pkl-load call, the `compile3.py` argv, and the warp-pixel reproduction to execution, because `tinygrad_repo` is uninitialized at plan-time and these must be read against the real (initialized) source + validated by the Phase-0 pipeline check and the Task-11 same-model anchor — each read is a numbered step (0-Step-2, 7-Step-1, 8-Step-1, 9a/9b/11). The pure, fully-knowable helpers (config, Params, shear/EMA, weave RMS, corrected disengaged eligibility, interp) have complete TDD code now.

**Type consistency:** `parse_frame_outputs(raw, split)` keys (`desired_curvature`, `model_y20`, `orientation_rate`, `lane_center_y20`) are produced in Task 9 and consumed by `metrics.weave_band_rms`/`disengaged_eligible_mask` (Task 4) and the comparator (Task 13). `ModelReplayer(pkl_paths, metadata, split)` signature (Task 8) is reused in Tasks 9/11/13. `compile_onnx_to_mac_pkl` provenance dict (Task 7) feeds `bundles.py` (Task 10). `TemporalState` (Task 8) is internal to `infer.py`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md`.

**Gates, in order:** **Task 0** (day-1: data-sufficiency scan + tinygrad init + stock-ONNX CPU compile self-test) is the cheapest GO/NO-GO; then **Task 9b** (Phase-0 pipeline runs deterministically + sane). Tasks 0–9b need no device pull (stock ONNX + already-cached frames/logs); only **Task 10/11** pull bundle ONNX + the known-model anchor routes' hevc, and only after Phase 0 passes.

Two execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks.
2. **Inline Execution** — execute in-session with checkpoints.
