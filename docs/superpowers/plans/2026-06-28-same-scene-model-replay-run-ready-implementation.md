# Same-Scene Model-Replay Simulator — Run-Ready Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the Mac-native same-scene model-replay simulator to the **run-ready gate** — stock + CD210 (`route_b5`) → compile → align an eligible window to the *correct* HEVC frames → run vision→policy with faithful inputs → produce a weave metric → **pass the CD210 same-model fidelity anchor** — with all six QA fixes baked in.

**Architecture:** This is the single authoritative build (FIX 3). It adopts the hardening plan's Phase-0 modules (`env`, `compile_bundle`, `alignment`, `context`, `bundles`, `assets`, `pytest_helpers`) plus the replay core (`warp`, `infer`, `parse`, `metrics`, `anchor`, `run`). It **supersedes** `…-simulator.md` and `…-simulator-hardening.md`; where it says "see original Task N," that means reuse that task's already-QA'd approach on these reconciled modules. Direct tinygrad-CPU inference (no `msgq`); import-only of production code; outputs under `retrospective_lateral/results/model_replay/`.

**Tech Stack:** Python 3.11 (`.venv311`), pytest, NumPy/SciPy/pandas + pyarrow, `openpilot.tools.lib.{logreader,framereader}`, tinygrad `compile3.py`, onnx/onnxruntime, git/git-lfs.

**Source-of-truth docs (read first):** the fix scope `docs/superpowers/specs/2026-06-28-same-scene-model-replay-run-ready-fix-scope.md` and the QA verdict `docs/superpowers/reports/2026-06-28-same-scene-model-replay-hardening-plan-qa-verdict.md`. The original/hardening plans are reference for the replay core only.

**Hard constraints:** no vehicle-control / `opendbc_repo/` / `panda/` / `selfdrive/` driving-code edits (import read-only); Mac only; pulling logs/frames is read-only (tar-over-ssh); all generated files under `retrospective_lateral/results/model_replay/` (gitignored).

**Every command that imports tinygrad or `FrameReader`** runs under the sanitized env (Task 1 `replay_env`: `DEBUG=0 DEV=CPU IMAGE=0 THREADS=0`, `PYTHONPATH` incl. `tinygrad_repo` + repo root). The bare relative path trap: `compile3.py` / tinygrad `fetch()` only accept a path starting with `/` or `.` — always pass an absolute path.

---

## File Structure

New package `model_replay_sim/` (analysis-only). One config, one module set (FIX 3):

- `config.py` — paths, env, **full-SHA `BUNDLES`** + metric/tolerance constants (`WEAVE_BAND_HZ`, `ANCHOR_CORR_MIN`, `ANCHOR_BAND_RATIO`, `GENTLE_CURV_ABS_MAX_1PM`, `SELFTEST_*`).
- `env.py` — sanitized subprocess env + `replay_python`.
- `pytest_helpers.py` — `require_real_asset` (FIX 5: unit-mode **skips**).
- `compile_bundle.py` — path-safe ONNX→Mac-CPU pkl, **single** `SELFTEST=1` run (FIX 6), provenance.
- `alignment.py` — **roadEncodeIdx-based** segment catalog + frame timeline + `map_window_to_frames` (FIX 1).
- `context.py` — route context + **per-bundle model constants sourced from the pinned commit** (FIX 2).
- `bundles.py` — ref/SHA resolve, LFS materialize (`--include=`, FIX 6), ONNX under `results/`, metadata, provenance.
- `assets.py` — anchor candidates on the fixed alignment; CD210 strict anchor on `route_b5`'s real window (FIX 4); non-skipping gates.
- `warp.py`, `infer.py`, `parse.py`, `metrics.py`, `anchor.py`, `run.py` — replay core (original plan Tasks 7–13 approach).
- `tests/` — pytest.

Generated: `results/model_replay/{catalog/*.parquet, compiled/<bundle>/, onnx/<bundle>/, anchor_assets.csv, …}`.

**Build order (fix-scope sequencing):** Task 0 (gate) → 1–2 (foundation) → **3 (FIX 1 alignment, linchpin)** → 4–6 (context/bundles/assets) → 7–9 (replay core) → **10 (CD210 fidelity anchor = run-ready gate)**.

---

### Task 0: Day-1 gate (data + compile) — reuse, unchanged

Reuse the original plan's **Task 0** verbatim (data-sufficiency scan with the corrected eligibility + `git submodule update --init tinygrad_repo` + stock-ONNX `DEV=CPU` compile self-check). Already QA'd. GO criteria: ≥ ~30 local ≥20 s windows AND the stock compile produces a pkl. (Note: the built-in `SELFTEST` will *fail* numeric parity at 1e-4 — that is expected and is not the gate; the gate is "compiles + produces a pkl," per FIX/Area-2.)

---

### Task 1: Foundation — package, config, env, asset gate

**Files:** Create `model_replay_sim/{__init__,config,env,pytest_helpers}.py`, `tests/{__init__,test_env,test_pytest_helpers,test_config}.py`; Modify `requirements-analysis.txt`.

- [ ] **Step 1: Deps + submodule.** Append to `requirements-analysis.txt` if absent: `onnx==1.22.0`, `onnxruntime==1.27.0`, `pyarrow==24.0.0`. Run `.venv311/bin/pip install -r requirements-analysis.txt` and `git submodule update --init tinygrad_repo`. (All three pins verified to have arm64/py3.11 wheels.)

- [ ] **Step 2: Failing tests.** Create `tests/test_config.py`:
```python
from model_replay_sim import config as C
def test_merged_config_shape():
    assert C.REPO_ROOT.name == "sunnypilot"
    assert C.RESULTS_ROOT == C.REPO_ROOT/"retrospective_lateral"/"results"/"model_replay"
    assert C.WEAVE_BAND_HZ == (0.10, 0.35)
    assert C.GENTLE_CURV_ABS_MAX_1PM == 0.005
    assert C.ANCHOR_CORR_MIN == 0.95
    assert C.ANCHOR_BAND_RATIO == (0.85, 1.15)
    # full-SHA bundles (FIX 3 reconciliation)
    assert C.BUNDLES["CD210"]["full_sha"] == "55f66e2246359c6593605399a0199d94d13ad90d"
    assert C.BUNDLES["CD210"]["repo"] == "commaai/openpilot" and C.BUNDLES["CD210"]["split"] is False
    assert C.BUNDLES["Nevada"]["full_sha"] == "3193eac5e385aa010694a8ac192ff38ffe000193"
    assert C.BUNDLES["OPM7"]["full_sha"] == "052692b25d63c5ddda276b5c2271383b6aff129f" and C.BUNDLES["OPM7"]["split"] is True
```
Create `tests/test_env.py` and `tests/test_pytest_helpers.py` per the hardening plan's Task 1 snippets, **except** the helper test asserts skip semantics:
```python
import pytest
from model_replay_sim.pytest_helpers import require_real_asset
def test_require_real_asset_fails_by_default():
    with pytest.raises(AssertionError, match="missing required real asset"):
        require_real_asset(False, "route_c4 fcamera")
def test_require_real_asset_skips_in_unit_mode(monkeypatch):
    monkeypatch.setenv("MODEL_REPLAY_ALLOW_MISSING_ASSETS", "1")
    with pytest.raises(pytest.skip.Exception):     # FIX 5: SKIP, not silent return
        require_real_asset(False, "route_c4 fcamera")
```
Run all three → FAIL (module missing).

- [ ] **Step 3: Implement.** `__init__.py` (`PACKAGE_VERSION="0.2.0"`). `config.py`:
```python
from __future__ import annotations
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
TINYGRAD_PATH = REPO_ROOT/"tinygrad_repo"
RESULTS_ROOT = REPO_ROOT/"retrospective_lateral"/"results"/"model_replay"
CACHE_ROOT = REPO_ROOT/"retrospective_lateral"/"results"/"cache"
LOG_ROOT = REPO_ROOT/"explorer_st_logs"
FS_HZ = 20.0
WEAVE_BAND_HZ = (0.10, 0.35)
GENTLE_CURV_ABS_MAX_1PM = 0.005
ROAD_LP_HZ = 0.035
SELFTEST_ATOL = 1e-3; SELFTEST_RTOL = 1e-2
ANCHOR_CORR_MIN = 0.95; ANCHOR_BAND_RATIO = (0.85, 1.15)
MAX_FRAME_DELTA_S = 0.03
TINYGRAD_ENV = {"DEBUG":"0","DEV":"CPU","IMAGE":"0","THREADS":"0"}
BUNDLES = {
  "CD210":  {"full_sha":"55f66e2246359c6593605399a0199d94d13ad90d","repo":"commaai/openpilot","split":False,"internal_names":{"C210M","CD210"}},
  "Nevada": {"full_sha":"3193eac5e385aa010694a8ac192ff38ffe000193","repo":"commaai/openpilot","split":False,"internal_names":{"NM","Nevada"}},
  "OPM7":   {"full_sha":"052692b25d63c5ddda276b5c2271383b6aff129f","repo":"sunnypilot/sunnypilot","split":True,"internal_names":{"OPM7"}},
}
```
`env.py` (per hardening Task 1: `replay_env()` merges `TINYGRAD_ENV` + `PYTHONPATH=f"{TINYGRAD_PATH}:{REPO_ROOT}"`; `replay_python(args)` returns `[sys.executable, *args]`). `pytest_helpers.py`:
```python
from __future__ import annotations
import os, pytest
def require_real_asset(present: bool, what: str):
    if present: return
    if os.environ.get("MODEL_REPLAY_ALLOW_MISSING_ASSETS") == "1":
        pytest.skip(f"unit mode: missing {what}")   # FIX 5: skip (yellow), never run the body
    raise AssertionError(f"missing required real asset: {what}")
```

- [ ] **Step 4: Pass + commit.** `.venv311/bin/python -m pytest model_replay_sim/tests -q` → green. `git add model_replay_sim/{__init__,config,env,pytest_helpers}.py model_replay_sim/tests requirements-analysis.txt && git commit -m "model-sim: foundation (merged config + env + skip-gate)"`.

---

### Task 2: Compile wrapper — single SELFTEST run (FIX 6)

**Files:** Create `model_replay_sim/compile_bundle.py`, `tests/test_compile_bundle.py`.

- [ ] **Step 1:** Read `tinygrad_repo/examples/openpilot/compile3.py` (CLI = `compile3.py <abs_onnx> <out_pkl>`; `SELFTEST=1` runs the 1e-4 parity; `fetch()` needs an abs path). 

- [ ] **Step 2: Failing test** (compiles the in-tree stock policy ONNX once, SELFTEST captured as a *fact* not a gate):
```python
from pathlib import Path
from model_replay_sim.compile_bundle import compile_onnx_to_pkl
from model_replay_sim import config as C
def test_compile_stock_policy(tmp_path):
    onnx = C.REPO_ROOT/"selfdrive/modeld/models/driving_policy.onnx"
    rec = compile_onnx_to_pkl(onnx, tmp_path/"policy.pkl", compare_onnxruntime=True)
    assert (tmp_path/"policy.pkl").exists()
    assert rec["compile_ok"] is True            # compile + JIT determinism succeeded
    assert "onnxruntime_strict_1e4_passed" in rec  # captured as a FACT (False is fine; not a gate)
    assert rec["tinygrad_sha"] and rec["onnx_sha256"]
```

- [ ] **Step 3: Implement.** `compile_onnx_to_pkl(onnx_path, out_pkl, compare_onnxruntime)` runs `compile3.py` **once** with `SELFTEST=1` under `replay_env()`, passing `str(Path(onnx_path).resolve())` (abs path) and `str(out_pkl)`; `compile_ok = (out_pkl.exists() and "pkl size is" in stdout)`; `onnxruntime_strict_1e4_passed = (returncode == 0)` (the SELFTEST assert raising → False); record `tinygrad_sha` (`git -C tinygrad_repo rev-parse HEAD`) + `onnx_sha256`. Do **not** treat `onnxruntime_strict_1e4_passed=False` as failure (Area-2 / verdict §4 #3: it fails on backend drift even when compile is correct).

- [ ] **Step 4: Pass (real compile, ~1-2 min) + commit.** `git commit -m "model-sim: single-pass compile wrapper + SELFTEST-as-fact"`.

---

### Task 3: **FIX 1 — Frame alignment on `roadEncodeIdx`** (the linchpin)

**Files:** Create `model_replay_sim/alignment.py`, `tests/test_alignment.py`.

Mechanics (grounded on real rlogs): `EncodeIndex` (`cereal/log.capnp:1086`) gives per road frame `frameId` (global, monotonic), `segmentNum` (which 60 s segment / HEVC file), **`segmentId` = HEVC frame index within the segment** (0..N-1, presentation order; `segmentId 0 ↔ frameId 1202 ↔ HEVC 0`), `timestampEof` (frame mono time = NPZ `mono_time` clock). `roadEncodeIdx`→`fcamera.hevc`, `wideRoadEncodeIdx`→`ecamera.hevc`.

- [ ] **Step 1: Failing tests** (`route_8d` has 44 frame segments + NPZ):
```python
import pytest
from model_replay_sim.alignment import build_frame_timeline, map_window_to_frames

def _route():
    import glob
    return "route_8d" if glob.glob("explorer_st_logs/route_8d/0000008d--*--1/fcamera.hevc") else None

@pytest.mark.skipif(_route() is None, reason="no route_8d frames")
def test_timeline_uses_segmentId_not_roadcamerastate_count():
    tl = build_frame_timeline("route_8d")
    seg1 = [r for r in tl if r.segment_num == 1]
    seg1.sort(key=lambda r: r.segment_id)
    assert seg1[0].segment_id == 0 and seg1[0].frame_id == 1202     # FIX: off-by-one regression
    # the old roadCameraState-count would have started at frameId 1203
    assert seg1[0].frame_id != 1203
    assert all(r.segment_id < r.hevc_frame_count for r in seg1)     # truncation guard

@pytest.mark.skipif(_route() is None, reason="no route_8d frames")
def test_map_window_resolves_and_rejects_unmappable():
    tl = build_frame_timeline("route_8d")
    seg1 = sorted([r for r in tl if r.segment_num == 1], key=lambda r: r.segment_id)
    monos = [seg1[10].timestamp_eof_s, seg1[20].timestamp_eof_s, seg1[30].timestamp_eof_s]
    aligned = map_window_to_frames("route_8d", monos, max_delta_s=0.03)
    assert [a.segment_id for a in aligned] == [10, 20, 30]
    assert all(a.segment_num == 1 for a in aligned)
    with pytest.raises(ValueError):
        map_window_to_frames("route_8d", [seg1[-1].timestamp_eof_s + 100.0], max_delta_s=0.03)

@pytest.mark.skipif(_route() is None, reason="no route_8d frames")
def test_boundary_crossing_window_is_accepted():
    # a window spanning a segment boundary maps to frames in BOTH per-segment HEVC files
    tl = build_frame_timeline("route_8d")
    by = {}
    for r in tl: by.setdefault(r.segment_num, []).append(r)
    s1 = sorted(by[1], key=lambda r: r.segment_id); s2 = sorted(by[2], key=lambda r: r.segment_id)
    monos = [s1[-2].timestamp_eof_s, s1[-1].timestamp_eof_s, s2[0].timestamp_eof_s, s2[1].timestamp_eof_s]
    aligned = map_window_to_frames("route_8d", monos, max_delta_s=0.06)
    assert [a.segment_num for a in aligned] == [1, 1, 2, 2]     # boundary crossing OK (FIX 1)
    assert [a.segment_id for a in aligned] == [s1[-2].segment_id, s1[-1].segment_id, 0, 1]
```

- [ ] **Step 2: Run → FAIL** (module missing).

- [ ] **Step 3: Implement `alignment.py`.**
```python
from __future__ import annotations
import sys, glob, functools
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # FIX 5: openpilot import robust to CWD
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader
from model_replay_sim import config as C

@dataclass(frozen=True)
class FrameRow:
    segment_num: int
    segment_id: int          # == HEVC frame index in that segment's fcamera.hevc
    frame_id: int            # global, monotonic
    timestamp_eof_s: float
    hevc_frame_count: int
    ecamera_segment_id: int | None   # wide-cam index if present at the same time

def _seg_dirs(route_id: str) -> list[Path]:
    return sorted(Path(C.LOG_ROOT/route_id).glob("000000*--*--*"))

@functools.lru_cache(maxsize=8)
def build_frame_timeline(route_id: str) -> tuple[FrameRow, ...]:     # cached (FIX 1 perf)
    rows: list[FrameRow] = []
    for seg in _seg_dirs(route_id):
        fcam = seg/"fcamera.hevc"
        if not fcam.exists():
            continue
        try:
            fcount = int(FrameReader(str(fcam), pix_fmt="nv12").frame_count)
        except Exception:
            continue
        wide_by_eof: dict[int, int] = {}
        road: list[tuple] = []
        for m in LogReader(str(seg/"rlog.zst")):
            w = m.which()
            if w == "roadEncodeIdx":
                e = m.roadEncodeIdx
                road.append((int(e.segmentNum), int(e.segmentId), int(e.frameId), e.timestampEof*1e-9))
            elif w == "wideRoadEncodeIdx":
                e = m.wideRoadEncodeIdx
                wide_by_eof[round(e.timestampEof*1e-9, 3)] = int(e.segmentId)
        for snum, sid, fid, eof in road:
            if sid >= fcount:        # truncation guard (FIX 1)
                continue
            rows.append(FrameRow(snum, sid, fid, eof, fcount, wide_by_eof.get(round(eof, 3))))
    rows.sort(key=lambda r: r.timestamp_eof_s)
    return tuple(rows)

@dataclass(frozen=True)
class FrameAlignment:
    segment_num: int
    segment_id: int
    ecamera_index: int | None

def map_window_to_frames(route_id: str, mono_times, max_delta_s: float = C.MAX_FRAME_DELTA_S):
    tl = build_frame_timeline(route_id)
    if not tl:
        raise ValueError(f"{route_id}: empty frame timeline")
    eofs = [r.timestamp_eof_s for r in tl]
    import bisect
    out: list[FrameAlignment] = []
    for t in mono_times:
        i = bisect.bisect_left(eofs, t)
        cands = [j for j in (i-1, i) if 0 <= j < len(tl)]
        best = min(cands, key=lambda j: abs(eofs[j]-t), default=None)
        if best is None or abs(eofs[best]-t) > max_delta_s:
            raise ValueError(f"{route_id}: mono {t:.3f}s has no frame within {max_delta_s}s")
        r = tl[best]
        out.append(FrameAlignment(r.segment_num, r.segment_id, r.ecamera_segment_id))
    return out
```
(Continuity is implicit: matching by `timestamp_eof_s` accepts boundary crossings because `segment_id` resets but time stays continuous. `read_frame(route, seg_num, seg_id)` reads `FrameReader(<…--seg_num>/fcamera.hevc, pix_fmt="nv12").get(seg_id)` — add the helper here.)

- [ ] **Step 4: Pass + commit.** `git commit -m "model-sim: FIX1 roadEncodeIdx frame alignment (off-by-one, cross-seg, truncation, cached)"`.

---

### Task 4: Route context + per-bundle constants from the pinned commit (FIX 2)

**Files:** Create `model_replay_sim/context.py`, `tests/test_context.py`.

- [ ] **Step 1: Read** how production builds the lateral delay/inputs: `sunnypilot/modeld_v2/modeld.py` (warp transform = `get_warp_matrix(liveCalibration.rpyCalib, intrinsics, bigmodel)` passed directly; `lat_delay = liveDelay.lateralDelay + LAT_SMOOTH_SECONDS`), `sunnypilot/livedelay/helpers.py` (`LagdToggle→LagdValueCache else liveDelay`), `camera_offset_helper.py`. Confirm param/message names (all verified real).

- [ ] **Step 2: Failing test** (FIX 2 = the load-bearing one):
```python
from model_replay_sim.context import bundle_lat_smooth_seconds
def test_opm7_lat_smooth_is_zero_from_pinned_commit():
    # sourced from 052692b2...:selfdrive/modeld/modeld.py, NOT a literal
    assert bundle_lat_smooth_seconds("OPM7") == 0.0
def test_cd210_nevada_lat_smooth_from_their_commits():
    for b in ("CD210", "Nevada"):
        v = bundle_lat_smooth_seconds(b)
        assert isinstance(v, float)     # whatever the commit says, not assumed
```

- [ ] **Step 3: Implement** `bundle_lat_smooth_seconds(bundle)` = parse `git show <BUNDLES[bundle]['full_sha']>:selfdrive/modeld/modeld.py` for `LAT_SMOOTH_SECONDS = <float>` (regex), cached. Implement `route_context(route_id)` reading from the route's boot rlog `initData.params` + messages: `rpyCalib`/`height` (liveCalibration), `liveDelay.lateralDelay`, `CameraOffset`/`PlanplusControl`/`LagdValueCache`/`LagdToggle`/`ModelManager_ActiveBundle` (params), `deviceState.deviceType`, `roadCameraState.sensor`, and `traffic_convention_input(ctx)` (RHD/LHD). The replay lateral-delay input = `(LagdValueCache if LagdToggle else liveDelay.lateralDelay) + bundle_lat_smooth_seconds(candidate_bundle)` — never a literal.

- [ ] **Step 4: Pass + commit.** `git commit -m "model-sim: route context + FIX2 per-bundle LAT_SMOOTH from pinned commit"`.

---

### Task 5: Bundle materialization + provenance (FIX 6 lfs syntax)

**Files:** Create `model_replay_sim/bundles.py`, `tests/test_bundles.py`. Reuse the hardening Task 5 design with two corrections: `git lfs ls-files --long --include=<path>` (not `-- <path>`); ONNX written under `results/model_replay/onnx/<bundle>/` (never source dirs). Materialize per bundle: `git show <full_sha>:<onnx_path>` is an LFS pointer → `git -C <clone> lfs pull --include=<path>` (or fetch the oid) → copy real ONNX out; generate `*_metadata.pkl` via `selfdrive/modeld/get_model_metadata.py` (standalone, no msgq); write `provenance.json` (`full_sha`, `repo`, per-model `onnx_sha256`, `lfs_oid`, `tinygrad_sha`). Test on a bundle whose ONNX is reachable from the local object store (CD210/Nevada/OPM7 all resolve — verified). Commit.

---

### Task 6: Anchor assets wired to the fixed alignment (FIX 4 + FIX 5)

**Files:** Create `model_replay_sim/assets.py`, `tests/test_assets.py`. Reuse hardening Task 6 design, with FIX 4: `_eligible_aligned_window_count(route)` selects replay-scene windows (straight/gentle `|curv|<GENTLE`, 10–85 mph, no blinker/lane-change — the replay gate; engagement is recorded but not gated) from the NPZ, restricted to the route's frame-present mono range, and aligned via the **fixed** `map_window_to_frames` (boundary-crossing allowed). Strict-anchor gate = `active_bundle_match AND local frames AND ≥1 aligned ≥20 s window`; sanity gate drops provenance.

- [ ] Key tests: `anchor_candidates("CD210")` → `route_b5` with `eligible_aligned_windows_20s >= 1` and `active_bundle_match is True`; `require_same_model_anchor_asset("CD210")` **passes** (FIX 4 — the 68.6 s window is now found); `require_same_model_anchor_asset("Nevada")` raises `missing local fcamera`; `require_same_model_anchor_asset("OPM7")` raises `active-bundle provenance`; `require_sanity_asset("CD210")` returns a candidate. Document the optional read-only **Nevada frame-pull** (`tar-over-ssh` one route's hevc) to later enable Nevada's strict anchor. Commit.

---

### Task 7: Warp pixel pipeline (replay core)

**Files:** Create `model_replay_sim/warp.py`, `tests/test_warp.py`. Implement per original plan **Task 9a** + the grounded note: the 3×3 `get_warp_matrix(rpyCalib, intrinsics, bigmodel)` (rpyCalib **passed directly**, `modeld.py:294`) + the `camera_offset` shear (`apply_camera_offset` + `0.9/0.1` EMA) is **applied to the NV12 buffer** and packed into the model's 12-channel current+prev tensor (mirror `sunnypilot/modeld_v2/warp.py` `make_frame_prepare`/`frames_to_tensor`; a numpy/`cv2.warpPerspective` equivalent is acceptable). Unit-test the pure shear/EMA + the output tensor shape against the bundle's `*_metadata.pkl` `input_shapes`. Frames are fed via `alignment.read_frame(route, segment_num, segment_id)`. Commit.

---

### Task 8: Inference loop — vision→policy + post-step (replay core)

**Files:** Create `model_replay_sim/infer.py`, `tests/test_infer.py`. Implement per original plan **Tasks 8/9b**: load the compiled pkl(s) (lift `TinygradRunner`/`TinygradSplitRunner` inference behind a `Params` stub — but note bundle/metadata come from `bundles.py`, not `get_active_bundle`); run the recurrent vision→policy loop (real `features_buffer`/`hidden_state`/`prev_desired_curv`/`desire` per `modeld.py:80-178`, `FULL_HISTORY_BUFFER_LEN`), 2-model and OPM7 3-model (`plan = plan + planplus`) paths, and the `get_action_from_model` post-step (`get_curvature_from_output` + the bundle's `LAT_SMOOTH_SECONDS` from Task 4 + gating) so the replayed `desiredCurvature` matches `controlsState.desiredCurvature`. Structural test on the temporal buffer; warmup from `FULL_HISTORY_BUFFER_LEN/FS`. Commit.

---

### Task 9: Parse + weave metric (replay core)

**Files:** Create `model_replay_sim/{parse,metrics}.py`, tests. `parse.py` wraps `selfdrive/modeld/parse_model_outputs.Parser` → `desired_curvature` (policy `plan`, post-step), `orientation_rate`, `model_y20`, and vision-only `lane_lines`/`road_edges` (for sanity). `metrics.py` = weave-band (0.10–0.35 Hz) duration-weighted RMS over the aligned window, reusing `retrospective_lateral.code.signal_utils`. Synthetic-signal tests. Commit.

---

### Task 10: **Run-ready gate — CD210 same-model fidelity anchor**

**Files:** Create `model_replay_sim/{anchor,run}.py`, `tests/test_fidelity_cd210.py`.

This is the milestone. Replay the **CD210** ONNX on `route_b5`'s aligned eligible window and confirm the replayed `desiredCurvature` reproduces `route_b5`'s **logged** `controlsState.desiredCurvature` (CD210 ran on route_b5) within the strict tolerance.

- [ ] **Step 1:** `anchor.replay_vs_logged("CD210", "route_b5", window)` = `assets`→pick the ≥20 s aligned window; `bundles`→materialize+compile CD210; `alignment`→frames; `warp`→tensors; `infer`→recurrent run + post-step; `parse`→`desired_curvature`; compare to the NPZ logged `desired_curvature` over the window (Pearson + weave-band RMS ratio).
- [ ] **Step 2: Test** (`require_real_asset` gates on `route_b5` frames; skips in unit mode):
```python
import pytest
from model_replay_sim.pytest_helpers import require_real_asset
from model_replay_sim.assets import anchor_candidates
from model_replay_sim.anchor import replay_vs_logged
from model_replay_sim import config as C
def test_cd210_same_model_fidelity():
    cands = [c for c in anchor_candidates("CD210") if c.route_id=="route_b5" and c.local_fcamera_segments>0]
    require_real_asset(bool(cands) and cands[0].eligible_aligned_windows_20s>=1, "route_b5 aligned window")
    rep = replay_vs_logged("CD210", "route_b5")
    assert rep["corr_desired_curv"] >= C.ANCHOR_CORR_MIN
    lo, hi = C.ANCHOR_BAND_RATIO; assert lo <= rep["weave_rms_ratio"] <= hi
```
- [ ] **Step 3: Run the gate.** PASS ⇒ **run-ready** (the pipeline is faithful end-to-end on the locally-available CD210 anchor). If `corr` falls short with the warp/temporal/post-step correct, debug those (not the data — the window exists). Document the result.
- [ ] **Step 4: `run.py`** orchestrator stub (decode-once/infer-N×, per-(bundle,segment) outputs) + the optional cross-model comparison (enabled once Nevada frames are pulled / OPM7 sanity). Commit `git commit -m "model-sim: CD210 same-model fidelity anchor (run-ready gate)"`.

---

## Self-Review

**Fix coverage:** FIX 1 → Task 3 (full TDD: off-by-one regression, cross-seg, truncation, cache); FIX 2 → Task 4 (`bundle_lat_smooth_seconds` from the pinned commit, OPM7=0.0 tested); FIX 3 → this plan *is* the reconciliation (one config Task 1, one module set in File Structure, supersedes both prior plans); FIX 4 → Task 6 (CD210 anchor on `route_b5`'s real window) + Task 10; FIX 5 → Task 1 (`require_real_asset` skips) + Task 3 (`sys.path` insert) + test ordering (consistency checks live in their own tasks, not before a rewrite); FIX 6 → Task 2 (compile-once) + Task 5 (`--include=`) + config wording. Run-ready definition (scope) → Task 10 gate.

**Placeholders (intentional):** Tasks 7–9 reference the original plan's already-QA'd replay-core approach + execution-time reads of the live runner/modeld (same pattern the original plan used, because the tinygrad pkl-load / temporal-stride details must be read against the initialized submodule). All fix-critical and foundation tasks (0–6, 10) have complete TDD code or precise grounded specs.

**Type consistency:** `FrameRow`/`FrameAlignment` (Task 3) consumed by `assets`/`warp`/`anchor`; `bundle_lat_smooth_seconds` (Task 4) consumed by `infer` post-step; `compile_onnx_to_pkl` rec (Task 2) consumed by `bundles`; `config.BUNDLES` full-SHA shape (Task 1) consumed by `bundles`/`context`; `require_real_asset` (Task 1) used in Tasks 6/10.

---

## Execution Handoff

Plan saved to `docs/superpowers/plans/2026-06-28-same-scene-model-replay-run-ready-implementation.md`. **Gates in order:** Task 0 (data + stock compile) → Task 3 (alignment — verify the off-by-one fix on real frames) → **Task 10 (CD210 fidelity anchor = run-ready)**. Tasks 0–6 + the unit parts of 7–10 need no device pull; only the optional Nevada strict anchor needs a read-only hevc pull.

Two execution options:
1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks.
2. **Inline Execution** — in-session with checkpoints.
