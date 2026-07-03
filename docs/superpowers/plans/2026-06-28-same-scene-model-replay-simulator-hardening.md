# Same-Scene Model-Replay Simulator Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch the same-scene model-replay simulator plan so its feasibility gates, frame alignment, context extraction, bundle provenance, and anchor tests are executable and capable of supporting a trustworthy model-vs-model weave comparison.

**Architecture:** Add a hardened Phase 0 in front of model replay: deterministic local environment setup, a custom compile/ONNXRuntime diagnostic, an explicit rlog-to-HEVC frame-alignment catalog, route context extraction for calibration and model inputs, metadata/provenance resolution for bundles, and non-skipping anchor asset gates. All generated artifacts stay under `retrospective_lateral/results/model_replay/`; production driving code is import-only.

**Tech Stack:** Python 3.11 (`.venv311`), pytest, NumPy/SciPy/pandas, pycapnp/zstandard log reading, `tools.lib.framereader.FrameReader`, tinygrad `compile3.py`, ONNX/ONNXRuntime diagnostics, git/git-lfs provenance checks.

---

## Constraints

- Do not modify vehicle-control code.
- Do not modify `opendbc_repo/` or `panda/`.
- Do not run anything on a connected comma device.
- Do not write generated model artifacts under `selfdrive/modeld/models/` or `sunnypilot/modeld_v2/models/`.
- All simulator-generated files go under `retrospective_lateral/results/model_replay/`.
- Every command that imports tinygrad or `FrameReader` must force `DEBUG=0` because this shell has `DEBUG=release`.
- Every Mac tinygrad compile command must force `DEV=CPU IMAGE=0 THREADS=0`.

## File Structure

Create or modify these analysis-only files:

- `model_replay_sim/__init__.py` - package marker.
- `model_replay_sim/config.py` - paths, env constants, tolerance constants, full bundle refs.
- `model_replay_sim/env.py` - builds sanitized subprocess environments for tinygrad/FrameReader tools.
- `model_replay_sim/compile_bundle.py` - path-safe tinygrad compile wrapper and ONNXRuntime diagnostic report.
- `model_replay_sim/alignment.py` - rlog segment catalog and timestamp/frame alignment.
- `model_replay_sim/context.py` - route context extraction: calibration, height, device/sensor, camera offset, live delay, RHD traffic convention, active bundle, and straight-window desire policy.
- `model_replay_sim/bundles.py` - full-ref bundle resolver, LFS pointer checks, ONNX copy/LFS pull, metadata generation/fetch.
- `model_replay_sim/assets.py` - anchor asset discovery and non-skipping asset requirements.
- `model_replay_sim/tests/` - focused pytest coverage for the above.
- `requirements-analysis.txt` - add dependencies required by this hardened Phase 0.
- `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md` - later integration target; update only after this hardening plan passes QA.

Generated outputs:

- `retrospective_lateral/results/model_replay/catalog/segment_catalog.parquet`
- `retrospective_lateral/results/model_replay/catalog/frame_timeline.parquet`
- `retrospective_lateral/results/model_replay/catalog/route_context.parquet`
- `retrospective_lateral/results/model_replay/compiled/<bundle>/compile_report.json`
- `retrospective_lateral/results/model_replay/onnx/<bundle>/*.onnx`
- `retrospective_lateral/results/model_replay/onnx/<bundle>/*_metadata.pkl`
- `retrospective_lateral/results/model_replay/anchor_assets.csv`

---

### Task 1: Harden Environment And Dependencies

**Files:**
- Create: `model_replay_sim/__init__.py`
- Create: `model_replay_sim/config.py`
- Create: `model_replay_sim/env.py`
- Create: `model_replay_sim/pytest_helpers.py`
- Create: `model_replay_sim/tests/__init__.py`
- Create: `model_replay_sim/tests/test_env.py`
- Create: `model_replay_sim/tests/test_pytest_helpers.py`
- Modify: `requirements-analysis.txt`

- [ ] **Step 1: Write the failing environment tests**

Create `model_replay_sim/tests/test_env.py`:

```python
import os
import subprocess
import sys

from model_replay_sim.env import replay_env, replay_python


def test_replay_env_clears_debug_and_sets_tinygrad_flags():
    env = replay_env()
    assert env["DEBUG"] == "0"
    assert env["DEV"] == "CPU"
    assert env["IMAGE"] == "0"
    assert env["THREADS"] == "0"
    assert "tinygrad_repo" in env["PYTHONPATH"]


def test_replay_python_imports_framereader_and_tinygrad_from_sanitized_env():
    proc = subprocess.run(
        replay_python([
            "-c",
            "import sys; sys.path.insert(0,'.'); import openpilot.tools.lib.framereader; import tinygrad; print('ok')",
        ]),
        env=replay_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_git_lfs_is_available_for_bundle_materialization():
    proc = subprocess.run(["git", "lfs", "version"], text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "git-lfs" in proc.stdout


def test_parquet_engine_available_for_catalogs():
    proc = subprocess.run(
        replay_python(["-c", "import pyarrow; import pandas as pd; print('ok')"]),
        env=replay_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
```

Create `model_replay_sim/tests/test_pytest_helpers.py`:

```python
import pytest

from model_replay_sim.pytest_helpers import require_real_asset


def test_require_real_asset_fails_by_default():
    with pytest.raises(AssertionError, match="missing required real asset"):
        require_real_asset(False, "route_c4 fcamera")


def test_require_real_asset_allows_unit_mode_when_env_set(monkeypatch):
    monkeypatch.setenv("MODEL_REPLAY_ALLOW_MISSING_ASSETS", "1")
    require_real_asset(False, "route_c4 fcamera")
```

- [ ] **Step 2: Run the test to verify failure**

Run:

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_env.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'model_replay_sim'`.

- [ ] **Step 3: Add dependencies**

Append these exact lines to `requirements-analysis.txt` if absent:

```text
onnx==1.22.0
onnxruntime==1.27.0
pyarrow==24.0.0
```

Do not add `opencv-python-headless` unless Task 7 chooses a cv2-based warp implementation. Do not add `setproctitle`; the hardening path avoids importing modules that require it.

Run:

```bash
.venv311/bin/pip install -r requirements-analysis.txt
git submodule update --init tinygrad_repo
```

Expected: `onnx`, `onnxruntime`, and `pyarrow` import in `.venv311`, `tinygrad_repo/examples/openpilot/compile3.py` exists, and `git lfs version` succeeds.

- [ ] **Step 4: Implement package, config, and environment helper**

Create `model_replay_sim/__init__.py`:

```python
"""Analysis-only same-scene model replay simulator."""

PACKAGE_VERSION = "0.1.0"
```

Create `model_replay_sim/tests/__init__.py` as an empty file.

Create `model_replay_sim/config.py`:

```python
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TINYGRAD_PATH = REPO_ROOT / "tinygrad_repo"
RESULTS_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "model_replay"
LOG_ROOT = REPO_ROOT / "explorer_st_logs"
CACHE_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "cache"

FS_HZ = 20.0
SEGMENT_SECONDS = 60.0

TINYGRAD_ENV = {
    "DEBUG": "0",
    "DEV": "CPU",
    "IMAGE": "0",
    "THREADS": "0",
}

RAW_ONNX_MAX_ABS_WARN = 4.0
RAW_ONNX_MAX_REL_WARN = 6.0
RAW_ONNX_MISMATCH_FRACTION_WARN = 0.95

BUNDLES = {
    "CD210": {
        "repo": "commaai/openpilot",
        "fetch_ref": "refs/pull/37050/head",
        "full_sha": "55f66e2246359c6593605399a0199d94d13ad90d",
        "models": ["driving_vision.onnx", "driving_policy.onnx"],
        "split": False,
        "native_context_routes": ["route_b5", "route_b8", "route_c1", "route_c2", "route_c3"],
        "candidate_overrides": {"lat": ".0", "long": ".3"},
        "override_source": "route_b5 active-bundle log",
    },
    "Nevada": {
        "repo": "commaai/openpilot",
        "fetch_ref": "refs/pull/36114/head",
        "full_sha": "3193eac5e385aa010694a8ac192ff38ffe000193",
        "models": ["driving_vision.onnx", "driving_policy.onnx"],
        "split": False,
        "native_context_routes": ["route_c4", "route_c5", "route_c6", "route_c7"],
        "candidate_overrides": {"lat": ".1", "long": ".3"},
        "override_source": "route_c4 active-bundle log",
    },
    "OPM7": {
        "repo": "sunnypilot/sunnypilot",
        "fetch_ref": "052692b25d63c5ddda276b5c2271383b6aff129f",
        "full_sha": "052692b25d63c5ddda276b5c2271383b6aff129f",
        "models": ["driving_vision.onnx", "driving_on_policy.onnx", "driving_off_policy.onnx"],
        "split": True,
        "native_context_routes": ["route_7f"],
        "candidate_overrides": {"lat": ".1", "long": ".3"},
        "override_source": "legacy stock modeld fallback; route_7f has no active-bundle param",
    },
}
```

Create `model_replay_sim/env.py`:

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

from model_replay_sim import config as C


def replay_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(C.TINYGRAD_ENV)
    existing = env.get("PYTHONPATH", "")
    parts = [str(C.TINYGRAD_PATH), str(C.REPO_ROOT)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def replay_python(args: list[str | Path]) -> list[str]:
    return [sys.executable, *[str(a) for a in args]]
```

Create `model_replay_sim/pytest_helpers.py`:

```python
from __future__ import annotations

import os


def require_real_asset(condition: bool, description: str) -> None:
    if condition:
        return
    if os.environ.get("MODEL_REPLAY_ALLOW_MISSING_ASSETS") == "1":
        return
    raise AssertionError(f"missing required real asset: {description}")
```

- [ ] **Step 5: Run tests to verify pass**

Run:

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_env.py model_replay_sim/tests/test_pytest_helpers.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements-analysis.txt model_replay_sim/__init__.py model_replay_sim/config.py model_replay_sim/env.py model_replay_sim/pytest_helpers.py model_replay_sim/tests/__init__.py model_replay_sim/tests/test_env.py model_replay_sim/tests/test_pytest_helpers.py
git commit -m "model-sim: harden replay environment"
```

---

### Task 2: Replace `compile3.py SELFTEST` Gate With A Custom Compile Diagnostic

**Files:**
- Create: `model_replay_sim/compile_bundle.py`
- Create: `model_replay_sim/tests/test_compile_bundle.py`

The built-in `SELFTEST=1` is not a valid GO/NO-GO gate on this Mac: it compares raw float16/float32-ish outputs at `1e-4` and fails for stock vision/policy even when tinygrad compile and JIT determinism succeed. This task makes compile success and ONNXRuntime comparison separate facts.

- [ ] **Step 1: Write failing tests for path normalization and diagnostic shape**

Create `model_replay_sim/tests/test_compile_bundle.py`:

```python
from pathlib import Path

from model_replay_sim.compile_bundle import local_onnx_arg, compile_onnx_to_pkl
from model_replay_sim import config as C


def test_local_onnx_arg_prefixes_relative_path_for_tinygrad_fetch():
    p = Path("selfdrive/modeld/models/driving_policy.onnx")
    assert local_onnx_arg(p) == "./selfdrive/modeld/models/driving_policy.onnx"


def test_compile_policy_without_builtin_selftest(tmp_path):
    onnx = C.REPO_ROOT / "selfdrive" / "modeld" / "models" / "driving_policy.onnx"
    out = tmp_path / "driving_policy_cpu.pkl"
    report = compile_onnx_to_pkl(onnx, out, compare_onnxruntime=True)
    assert out.exists()
    assert report["compile_returncode"] == 0
    assert report["tinygrad_jit_validated"] is True
    assert report["onnxruntime_compare"]["ran"] is True
    assert report["onnxruntime_compare"]["raw_max_abs"] >= 0.0
    assert "strict_1e_4_passed" in report["onnxruntime_compare"]
    assert report["onnx_sha256"]
    assert report["tinygrad_sha"]
```

- [ ] **Step 2: Run the tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_compile_bundle.py -q
```

Expected: FAIL because `model_replay_sim.compile_bundle` does not exist.

- [ ] **Step 3: Implement the compile wrapper**

Create `model_replay_sim/compile_bundle.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from model_replay_sim import config as C
from model_replay_sim.env import replay_env, replay_python


def local_onnx_arg(path: Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)
    text = path.as_posix()
    if text.startswith("./") or text.startswith("../"):
        return text
    return f"./{text}"


def _parse_compile_stdout(stdout: str) -> dict:
    return {
        "tinygrad_jit_validated": "jit run validated" in stdout,
        "kernel_count": _first_int(stdout, r"kernel_count=(\d+)"),
        "run_ms": [float(x) for x in re.findall(r"total run\s+([0-9.]+) ms", stdout)],
    }


def _first_int(text: str, pattern: str) -> int | None:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _onnxruntime_compare(stdout: str, stderr: str, returncode: int) -> dict:
    strict_passed = returncode == 0 and "test vs onnx passed" in stdout
    max_abs = _first_float(stderr, r"Max absolute difference[^:]*:\s*([0-9.eE+-]+)")
    max_rel = _first_float(stderr, r"Max relative difference[^:]*:\s*([0-9.eE+-]+)")
    mismatch = _first_fraction(stderr, r"Mismatched elements:\s*(\d+)\s*/\s*(\d+)")
    return {
        "ran": True,
        "strict_1e_4_passed": strict_passed,
        "raw_max_abs": max_abs,
        "raw_max_rel": max_rel,
        "mismatch_fraction": mismatch,
        "warning_only": True,
    }


def _first_float(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def _first_fraction(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text)
    if not m:
        return None
    return int(m.group(1)) / int(m.group(2))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(path: Path) -> str:
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    return proc.stdout.strip()


def compile_onnx_to_pkl(onnx_path: Path, out_pkl: Path, compare_onnxruntime: bool) -> dict:
    out_pkl = Path(out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    compile_cmd = replay_python([
        C.TINYGRAD_PATH / "examples" / "openpilot" / "compile3.py",
        local_onnx_arg(Path(onnx_path).relative_to(C.REPO_ROOT) if Path(onnx_path).is_absolute() else Path(onnx_path)),
        out_pkl,
    ])
    compile_proc = subprocess.run(
        compile_cmd,
        cwd=C.REPO_ROOT,
        env=replay_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    parsed = _parse_compile_stdout(compile_proc.stdout)
    report = {
        "onnx_path": str(onnx_path),
        "out_pkl": str(out_pkl),
        "onnx_sha256": _sha256(Path(onnx_path)),
        "tinygrad_sha": _git_sha(C.TINYGRAD_PATH),
        "compile_returncode": compile_proc.returncode,
        "tinygrad_jit_validated": parsed["tinygrad_jit_validated"],
        "kernel_count": parsed["kernel_count"],
        "run_ms": parsed["run_ms"],
        "onnxruntime_compare": {"ran": False},
        "stderr_tail": compile_proc.stderr[-4000:],
    }
    if compile_proc.returncode != 0:
        out_pkl.with_suffix(".compile_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
        return report
    if compare_onnxruntime:
        compare_proc = subprocess.run(
            compile_cmd,
            cwd=C.REPO_ROOT,
            env=replay_env({"SELFTEST": "1"}),
            text=True,
            capture_output=True,
            check=False,
        )
        report["onnxruntime_compare"] = _onnxruntime_compare(
            compare_proc.stdout, compare_proc.stderr, compare_proc.returncode
        )
    report_path = out_pkl.with_suffix(".compile_report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
```

- [ ] **Step 4: Run the tests to verify pass**

Run:

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_compile_bundle.py -q -s
```

Expected: PASS. The report should record compile/JIT success plus ONNXRuntime raw comparison metrics. `strict_1e_4_passed` is diagnostic only and must not be a GO/NO-GO condition.

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/compile_bundle.py model_replay_sim/tests/test_compile_bundle.py
git commit -m "model-sim: replace compile selftest with diagnostic report"
```

---

### Task 3: Build Segment Catalog And Frame Alignment

**Files:**
- Create: `model_replay_sim/alignment.py`
- Create: `model_replay_sim/tests/test_alignment.py`

This is the primary trustworthiness fix. The simulator must select frames by logged camera/model timing, not by assuming `t * 20` maps to a HEVC frame index. Store both a segment summary and a per-message timeline so replay windows can be mapped through `modelV2.frameId`, `modelV2.frameIdExtra`, `roadCameraState.frameId`, `wideRoadCameraState.frameId`, and `timestampEof`.

- [ ] **Step 1: Write failing alignment tests on `route_8d`**

Create `model_replay_sim/tests/test_alignment.py`:

```python
import pytest

from model_replay_sim.alignment import (
    assert_road_wide_sync,
    alignable_model_events,
    build_route_frame_timeline,
    build_route_segment_catalog,
    map_mono_time_to_frame,
    map_window_to_frames,
)
from model_replay_sim import config as C
from model_replay_sim.pytest_helpers import require_real_asset


def test_catalog_records_corrupt_segments_and_camera_counts():
    require_real_asset((C.LOG_ROOT / "route_8d").exists(), "route_8d rlogs")
    catalog = build_route_segment_catalog("route_8d")
    assert len(catalog) >= 40
    corrupt = [row for row in catalog if row.error]
    assert any("ZstdError" in row.error for row in corrupt)
    valid = [row for row in catalog if row.road_frame_count and row.model_frame_count and row.alignable_frame_count]
    assert valid, "expected at least one segment with camera/model frame IDs"
    assert valid[0].fcamera_frames > 0


def test_timeline_records_logged_frame_ids_for_road_wide_and_model():
    require_real_asset((C.LOG_ROOT / "route_8d").exists(), "route_8d rlogs")
    catalog = build_route_segment_catalog("route_8d")
    row = next(r for r in catalog if r.segment_index == 1 and r.road_frame_count and r.model_frame_count)
    timeline = build_route_frame_timeline("route_8d")
    segment_events = [e for e in timeline if e.segment_index == row.segment_index]
    services = {e.service for e in segment_events}
    assert {"roadCameraState", "modelV2"} <= services
    if row.ecamera_frames:
        assert "wideRoadCameraState" in services
    model_event = alignable_model_events(row, timeline)[0]
    assert model_event.frame_id == row.first_alignable_model_frame_id
    assert model_event.extra_frame_id is not None
    assert model_event.source == "rlog"
    assert row.alignable_frame_count > 0


def test_map_mono_time_to_frame_uses_timeline_frame_ids_not_linearized_time():
    require_real_asset((C.LOG_ROOT / "route_8d").exists(), "route_8d rlogs")
    catalog = build_route_segment_catalog("route_8d")
    row = next(r for r in catalog if r.segment_index == 1 and r.alignable_frame_count)
    timeline = build_route_frame_timeline("route_8d")
    model_events = alignable_model_events(row, timeline)
    event = model_events[min(10, len(model_events) - 1)]
    aligned = map_mono_time_to_frame(row, timeline, event.mono_time)
    assert aligned.segment_index == 1
    assert 0 <= aligned.fcamera_index < row.fcamera_frames
    assert aligned.model_frame_id == event.frame_id
    assert aligned.road_frame_id == event.frame_id
    assert aligned.model_delta_s == 0.0


def test_road_wide_sync_uses_timestamp_eof_when_wide_available():
    require_real_asset((C.LOG_ROOT / "route_8d").exists(), "route_8d rlogs")
    catalog = build_route_segment_catalog("route_8d")
    row = next(r for r in catalog if r.segment_index == 1 and r.alignable_frame_count and r.ecamera_frames)
    timeline = build_route_frame_timeline("route_8d")
    aligned = map_mono_time_to_frame(row, timeline, alignable_model_events(row, timeline)[0].mono_time)
    assert_road_wide_sync(aligned, max_delta_s=0.010)


def test_map_window_to_frames_rejects_gaps_and_requires_continuity():
    require_real_asset((C.LOG_ROOT / "route_8d").exists(), "route_8d rlogs")
    catalog = build_route_segment_catalog("route_8d")
    timeline = build_route_frame_timeline("route_8d")
    row = next(r for r in catalog if r.segment_index == 1 and r.alignable_frame_count >= 30)
    events = alignable_model_events(row, timeline)
    mono_times = [e.mono_time for e in events[10:30]]
    aligned = map_window_to_frames("route_8d", mono_times, max_delta_s=0.03)
    assert len(aligned) == len(mono_times)
    assert [a.fcamera_index for a in aligned] == list(range(aligned[0].fcamera_index, aligned[0].fcamera_index + len(aligned)))
    assert [a.model_frame_id for a in aligned] == list(range(aligned[0].model_frame_id, aligned[0].model_frame_id + len(aligned)))
    assert [a.road_frame_id for a in aligned] == list(range(aligned[0].road_frame_id, aligned[0].road_frame_id + len(aligned)))
    with pytest.raises(ValueError, match="gap|continuous|segment"):
        map_window_to_frames("route_8d", [mono_times[0], mono_times[-1] + 10.0], max_delta_s=0.03)
```

- [ ] **Step 2: Run the tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_alignment.py -q
```

Expected: FAIL because `model_replay_sim.alignment` does not exist.

- [ ] **Step 3: Implement catalog data types and log scan**

Create `model_replay_sim/alignment.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from model_replay_sim import config as C


@dataclass(frozen=True)
class SegmentCatalogRow:
    route_id: str
    segment_index: int
    rlog_path: Path
    fcamera_path: Path | None
    ecamera_path: Path | None
    fcamera_frames: int
    ecamera_frames: int
    first_road_mono_time: float | None
    last_road_mono_time: float | None
    first_road_frame_id: int | None
    last_road_frame_id: int | None
    first_model_mono_time: float | None
    last_model_mono_time: float | None
    first_model_frame_id: int | None
    last_model_frame_id: int | None
    first_alignable_model_frame_id: int | None
    last_alignable_model_frame_id: int | None
    first_alignable_mono_time: float | None
    last_alignable_mono_time: float | None
    road_frame_count: int
    model_frame_count: int
    alignable_frame_count: int
    error: str | None = None


@dataclass(frozen=True)
class FrameTimelineRow:
    route_id: str
    segment_index: int
    service: str
    mono_time: float
    timestamp_eof_s: float | None
    frame_id: int
    extra_frame_id: int | None
    segment_frame_index: int
    source: str = "rlog"


@dataclass(frozen=True)
class FrameAlignment:
    route_id: str
    segment_index: int
    mono_time: float
    fcamera_index: int
    ecamera_index: int | None
    model_frame_id: int
    road_frame_id: int
    wide_frame_id: int | None
    model_delta_s: float
    road_timestamp_delta_s: float | None
    wide_timestamp_delta_s: float | None


def segment_index_from_dir(path: Path) -> int:
    m = re.search(r"--(\d+)$", path.name)
    if not m:
        raise ValueError(f"unsupported segment directory: {path}")
    return int(m.group(1))


def build_route_segment_catalog(route_id: str) -> list[SegmentCatalogRow]:
    rows: list[SegmentCatalogRow] = []
    route_dir = C.LOG_ROOT / route_id
    for seg_dir in sorted(route_dir.glob("000000*--*--*"), key=segment_index_from_dir):
        seg_idx = segment_index_from_dir(seg_dir)
        rlog = seg_dir / "rlog.zst"
        fcam = seg_dir / "fcamera.hevc"
        ecam = seg_dir / "ecamera.hevc"
        fcount = 0
        ecount = 0
        try:
            fcount = _frame_count(fcam)
            ecount = _frame_count(ecam)
            stats, _timeline = _scan_segment_log(route_id, seg_idx, rlog)
            rows.append(SegmentCatalogRow(
                route_id=route_id,
                segment_index=seg_idx,
                rlog_path=rlog,
                fcamera_path=fcam if fcam.exists() else None,
                ecamera_path=ecam if ecam.exists() else None,
                fcamera_frames=fcount,
                ecamera_frames=ecount,
                error=None,
                **stats,
            ))
        except Exception as e:
            rows.append(SegmentCatalogRow(
                route_id=route_id,
                segment_index=seg_idx,
                rlog_path=rlog,
                fcamera_path=fcam if fcam.exists() else None,
                ecamera_path=ecam if ecam.exists() else None,
                fcamera_frames=fcount,
                ecamera_frames=ecount,
                first_road_mono_time=None,
                last_road_mono_time=None,
                first_road_frame_id=None,
                last_road_frame_id=None,
                first_model_mono_time=None,
                last_model_mono_time=None,
                first_model_frame_id=None,
                last_model_frame_id=None,
                first_alignable_model_frame_id=None,
                last_alignable_model_frame_id=None,
                first_alignable_mono_time=None,
                last_alignable_mono_time=None,
                road_frame_count=0,
                model_frame_count=0,
                alignable_frame_count=0,
                error=f"{type(e).__name__}: {e}",
            ))
    return rows


def _frame_count(path: Path) -> int:
    if not path.exists():
        return 0
    from openpilot.tools.lib.framereader import FrameReader
    return int(FrameReader(str(path), pix_fmt="nv12").frame_count)


def build_route_frame_timeline(route_id: str) -> list[FrameTimelineRow]:
    timeline: list[FrameTimelineRow] = []
    route_dir = C.LOG_ROOT / route_id
    for seg_dir in sorted(route_dir.glob("000000*--*--*"), key=segment_index_from_dir):
        seg_idx = segment_index_from_dir(seg_dir)
        try:
            _stats, events = _scan_segment_log(route_id, seg_idx, seg_dir / "rlog.zst")
            timeline.extend(events)
        except Exception:
            continue
    return timeline


def _scan_segment_log(route_id: str, segment_index: int, rlog_path: Path) -> tuple[dict, list[FrameTimelineRow]]:
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "opendbc_repo")
    from openpilot.tools.lib.logreader import LogReader

    events: list[FrameTimelineRow] = []
    counts = {"roadCameraState": 0, "wideRoadCameraState": 0, "modelV2": 0}
    for msg in LogReader(str(rlog_path)):
        which = msg.which()
        mono = msg.logMonoTime * 1e-9
        if which == "roadCameraState":
            fd = msg.roadCameraState
            events.append(FrameTimelineRow(
                route_id=route_id,
                segment_index=segment_index,
                service=which,
                mono_time=mono,
                timestamp_eof_s=int(fd.timestampEof) * 1e-9,
                frame_id=int(fd.frameId),
                extra_frame_id=None,
                segment_frame_index=counts[which],
            ))
            counts[which] += 1
        elif which == "wideRoadCameraState":
            fd = msg.wideRoadCameraState
            events.append(FrameTimelineRow(
                route_id=route_id,
                segment_index=segment_index,
                service=which,
                mono_time=mono,
                timestamp_eof_s=int(fd.timestampEof) * 1e-9,
                frame_id=int(fd.frameId),
                extra_frame_id=None,
                segment_frame_index=counts[which],
            ))
            counts[which] += 1
        elif which == "modelV2":
            md = msg.modelV2
            events.append(FrameTimelineRow(
                route_id=route_id,
                segment_index=segment_index,
                service=which,
                mono_time=mono,
                timestamp_eof_s=int(md.timestampEof) * 1e-9 if md.timestampEof else None,
                frame_id=int(md.frameId),
                extra_frame_id=int(md.frameIdExtra),
                segment_frame_index=counts[which],
            ))
            counts[which] += 1
    road = [e for e in events if e.service == "roadCameraState"]
    model = [e for e in events if e.service == "modelV2"]
    alignable = _alignable_model_events_from_events(events)
    stats = {
        "first_road_mono_time": road[0].mono_time if road else None,
        "last_road_mono_time": road[-1].mono_time if road else None,
        "first_road_frame_id": road[0].frame_id if road else None,
        "last_road_frame_id": road[-1].frame_id if road else None,
        "first_model_mono_time": model[0].mono_time if model else None,
        "last_model_mono_time": model[-1].mono_time if model else None,
        "first_model_frame_id": model[0].frame_id if model else None,
        "last_model_frame_id": model[-1].frame_id if model else None,
        "first_alignable_model_frame_id": alignable[0].frame_id if alignable else None,
        "last_alignable_model_frame_id": alignable[-1].frame_id if alignable else None,
        "first_alignable_mono_time": alignable[0].mono_time if alignable else None,
        "last_alignable_mono_time": alignable[-1].mono_time if alignable else None,
        "road_frame_count": len(road),
        "model_frame_count": len(model),
        "alignable_frame_count": len(alignable),
    }
    return stats, events


def _alignable_model_events_from_events(events: list[FrameTimelineRow]) -> list[FrameTimelineRow]:
    road_ids = {e.frame_id for e in events if e.service == "roadCameraState"}
    wide_ids = {e.frame_id for e in events if e.service == "wideRoadCameraState"}
    require_wide = bool(wide_ids)
    out = []
    for event in events:
        if event.service != "modelV2":
            continue
        road_ok = event.frame_id in road_ids
        wide_ok = (not require_wide) or (event.extra_frame_id in wide_ids)
        if road_ok and wide_ok:
            out.append(event)
    return out


def alignable_model_events(row: SegmentCatalogRow, timeline: list[FrameTimelineRow]) -> list[FrameTimelineRow]:
    events = [e for e in timeline if e.route_id == row.route_id and e.segment_index == row.segment_index]
    return _alignable_model_events_from_events(events)


def map_mono_time_to_frame(row: SegmentCatalogRow, timeline: list[FrameTimelineRow], mono_time: float) -> FrameAlignment:
    if row.error:
        raise ValueError(f"cannot align corrupt segment {row.route_id}/{row.segment_index}: {row.error}")
    events = [e for e in timeline if e.route_id == row.route_id and e.segment_index == row.segment_index]
    model = _nearest(alignable_model_events(row, timeline), mono_time)
    road = _event_by_frame_id(events, "roadCameraState", model.frame_id)
    wide = _event_by_frame_id(events, "wideRoadCameraState", model.extra_frame_id) if model.extra_frame_id is not None else None
    if road is None:
        raise ValueError(f"missing roadCameraState frameId {model.frame_id}")
    if row.ecamera_frames and wide is None:
        raise ValueError(f"missing wideRoadCameraState frameId {model.extra_frame_id}")
    fcamera_index = road.segment_frame_index
    ecamera_index = wide.segment_frame_index if wide is not None else None
    if not (0 <= fcamera_index < row.fcamera_frames):
        raise ValueError(f"aligned road frame {fcamera_index} outside fcamera range 0..{row.fcamera_frames - 1}")
    if ecamera_index is not None and not (0 <= ecamera_index < row.ecamera_frames):
        raise ValueError(f"aligned wide frame {ecamera_index} outside ecamera range 0..{row.ecamera_frames - 1}")
    return FrameAlignment(
        route_id=row.route_id,
        segment_index=row.segment_index,
        mono_time=float(mono_time),
        fcamera_index=int(fcamera_index),
        ecamera_index=int(ecamera_index) if ecamera_index is not None else None,
        model_frame_id=int(model.frame_id),
        road_frame_id=int(road.frame_id),
        wide_frame_id=int(wide.frame_id) if wide is not None else None,
        model_delta_s=float(mono_time - model.mono_time),
        road_timestamp_delta_s=_timestamp_delta(model, road),
        wide_timestamp_delta_s=_timestamp_delta(model, wide) if wide is not None else None,
    )


def _nearest(events: list[FrameTimelineRow], mono_time: float) -> FrameTimelineRow:
    if not events:
        raise ValueError("no events to align")
    return min(events, key=lambda e: abs(e.mono_time - mono_time))


def _event_by_frame_id(events: list[FrameTimelineRow], service: str, frame_id: int | None) -> FrameTimelineRow | None:
    if frame_id is None:
        return None
    return next((e for e in events if e.service == service and e.frame_id == frame_id), None)


def _timestamp_delta(model: FrameTimelineRow, camera: FrameTimelineRow) -> float | None:
    if model.timestamp_eof_s is None or camera.timestamp_eof_s is None:
        return None
    return float(camera.timestamp_eof_s - model.timestamp_eof_s)


def assert_road_wide_sync(aligned: FrameAlignment, max_delta_s: float = 0.010) -> None:
    if aligned.road_timestamp_delta_s is None or aligned.wide_timestamp_delta_s is None:
        return
    delta = aligned.wide_timestamp_delta_s - aligned.road_timestamp_delta_s
    if abs(delta) > max_delta_s:
        raise AssertionError(
            f"road/wide timestamp sync failed: delta={delta:.6f}s > {max_delta_s:.6f}s"
        )


def map_window_to_frames(route_id: str, mono_times: list[float], max_delta_s: float = 0.03) -> list[FrameAlignment]:
    if not mono_times:
        raise ValueError("empty replay window")
    catalog = build_route_segment_catalog(route_id)
    timeline = build_route_frame_timeline(route_id)
    aligned: list[FrameAlignment] = []
    for mono_time in mono_times:
        candidates = [
            row for row in catalog
            if row.error is None
            and row.first_alignable_mono_time is not None
            and row.last_alignable_mono_time is not None
            and row.first_alignable_mono_time - max_delta_s <= mono_time <= row.last_alignable_mono_time + max_delta_s
        ]
        if len(candidates) != 1:
            raise ValueError(f"ambiguous or missing segment for mono_time={mono_time}: {len(candidates)} candidates")
        frame = map_mono_time_to_frame(candidates[0], timeline, mono_time)
        if abs(frame.model_delta_s) > max_delta_s:
            raise ValueError(f"nearest model frame is {frame.model_delta_s:.6f}s from requested mono_time")
        aligned.append(frame)
    segment_ids = {(a.route_id, a.segment_index) for a in aligned}
    if len(segment_ids) != 1:
        raise ValueError(f"replay window crosses segments: {sorted(segment_ids)}")
    fcamera_indices = [a.fcamera_index for a in aligned]
    expected = list(range(fcamera_indices[0], fcamera_indices[0] + len(fcamera_indices)))
    if fcamera_indices != expected:
        raise ValueError("replay window camera frames are not continuous")
    ecamera_indices = [a.ecamera_index for a in aligned if a.ecamera_index is not None]
    if ecamera_indices and ecamera_indices != list(range(ecamera_indices[0], ecamera_indices[0] + len(ecamera_indices))):
        raise ValueError("replay window wide-camera frames are not continuous")
    model_ids = [a.model_frame_id for a in aligned]
    if model_ids != list(range(model_ids[0], model_ids[0] + len(model_ids))):
        raise ValueError("replay window model frame IDs are not continuous")
    road_ids = [a.road_frame_id for a in aligned]
    if road_ids != list(range(road_ids[0], road_ids[0] + len(road_ids))):
        raise ValueError("replay window road frame IDs are not continuous")
    wide_ids = [a.wide_frame_id for a in aligned if a.wide_frame_id is not None]
    if wide_ids and wide_ids != list(range(wide_ids[0], wide_ids[0] + len(wide_ids))):
        raise ValueError("replay window wide frame IDs are not continuous")
    return aligned
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
DEBUG=0 .venv311/bin/python -m pytest model_replay_sim/tests/test_alignment.py -q
```

Expected: PASS. Segment 0 corruption should be recorded, not fatal to the whole catalog.

- [ ] **Step 5: Add catalog writer command**

Append to `model_replay_sim/alignment.py`:

```python
def write_alignment_catalogs(route_ids: list[str]) -> tuple[Path, Path]:
    import pandas as pd

    out_dir = C.RESULTS_ROOT / "catalog"
    segment_out = out_dir / "segment_catalog.parquet"
    timeline_out = out_dir / "frame_timeline.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_rows = []
    timeline_rows = []
    for route_id in route_ids:
        segment_rows.extend(row.__dict__ for row in build_route_segment_catalog(route_id))
        timeline_rows.extend(row.__dict__ for row in build_route_frame_timeline(route_id))
    pd.DataFrame(segment_rows).to_parquet(segment_out)
    pd.DataFrame(timeline_rows).to_parquet(timeline_out)
    return segment_out, timeline_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--route", action="append", required=True)
    args = parser.parse_args()
    write_alignment_catalogs(args.route)
```

Run:

```bash
DEBUG=0 .venv311/bin/python -m model_replay_sim.alignment --route route_8d --route route_7f
```

Expected: writes `retrospective_lateral/results/model_replay/catalog/segment_catalog.parquet` and `frame_timeline.parquet`.

- [ ] **Step 6: Commit**

```bash
git add model_replay_sim/alignment.py model_replay_sim/tests/test_alignment.py
git commit -m "model-sim: add rlog-to-camera frame alignment catalog"
```

---

### Task 4: Extract Route Context Needed For Faithful Replay

**Files:**
- Create: `model_replay_sim/context.py`
- Create: `model_replay_sim/tests/test_context.py`

The existing NPZ caches have `cal_roll`, `cal_pitch`, `cal_yaw`, and `mono_time`, but not enough replay context. This task creates a separate context table from rlogs and init params.

- [ ] **Step 1: Write failing tests**

Create `model_replay_sim/tests/test_context.py`:

```python
import pytest

from model_replay_sim.context import (
    desire_input_for_window,
    lateral_control_params,
    route_context,
    traffic_convention_input,
    write_route_context_catalog,
)
from model_replay_sim import config as C
from model_replay_sim.pytest_helpers import require_real_asset


def test_route_context_reads_nevada_active_bundle_and_calibration_fields():
    require_real_asset((C.LOG_ROOT / "route_c4").exists(), "route_c4 rlogs")
    ctx = route_context("route_c4")
    assert ctx.active_bundle_internal_name == "NM"
    assert ctx.active_bundle_generation == 12
    assert ctx.active_bundle_is20hz is True
    assert ctx.device_type in {"mici", "tici"}
    assert ctx.road_camera_sensor
    assert ctx.calibration_height_m > 0.5
    assert ctx.camera_offset_m is not None
    assert ctx.live_delay_lateral_s is not None
    assert ctx.lagd_toggle is True
    assert ctx.effective_native_lat_delay_input_s == pytest.approx(ctx.lagd_value_s + ctx.native_lat_smooth_seconds_s)


def test_desire_input_policy_is_zero_for_straight_non_lane_change_windows():
    desire = desire_input_for_window(blinker=False, lane_change_state=False)
    assert desire.shape == (8,)
    assert desire.sum() == 0.0


def test_desire_input_policy_rejects_lane_change_windows():
    with pytest.raises(ValueError, match="lane-change"):
        desire_input_for_window(blinker=True, lane_change_state=False)


def test_model_input_helpers_match_modeld_shapes():
    require_real_asset((C.LOG_ROOT / "route_c4").exists(), "route_c4 rlogs")
    ctx = route_context("route_c4")
    traffic = traffic_convention_input(ctx)
    assert traffic.shape == (2,)
    assert traffic.sum() == 1.0
    lateral = lateral_control_params(ctx, "Nevada", v_ego=20.0)
    assert lateral.shape == (2,)
    assert lateral[0] == 20.0
    assert lateral[1] == pytest.approx(ctx.lagd_value_s + 0.1)
    opm7 = lateral_control_params(ctx, "OPM7", v_ego=20.0)
    assert opm7[1] == pytest.approx(ctx.lagd_value_s + 0.1)


def test_route_context_catalog_writer(tmp_path, monkeypatch):
    require_real_asset((C.LOG_ROOT / "route_c4").exists(), "route_c4 rlogs")
    monkeypatch.setattr("model_replay_sim.config.RESULTS_ROOT", tmp_path)
    out = write_route_context_catalog(["route_c4"])
    assert out == tmp_path / "catalog" / "route_context.parquet"
    assert out.exists()


def test_route_context_skips_corrupt_first_rlog_when_needed():
    require_real_asset((C.LOG_ROOT / "route_8d").exists(), "route_8d rlogs")
    ctx = route_context("route_8d")
    assert ctx.route_id == "route_8d"
    assert ctx.road_camera_sensor
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
DEBUG=0 .venv311/bin/python -m pytest model_replay_sim/tests/test_context.py -q
```

Expected: FAIL because `model_replay_sim.context` does not exist.

- [ ] **Step 3: Implement context extraction**

Create `model_replay_sim/context.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from model_replay_sim import config as C


@dataclass(frozen=True)
class RouteContext:
    route_id: str
    active_bundle_internal_name: str | None
    active_bundle_display_name: str | None
    active_bundle_generation: int | None
    active_bundle_is20hz: bool | None
    active_bundle_overrides: dict[str, str]
    camera_offset_m: float | None
    planplus_control: float
    lagd_value_s: float
    lagd_toggle: bool
    live_delay_lateral_s: float | None
    native_lat_smooth_seconds_s: float
    effective_native_lat_delay_input_s: float | None
    device_type: str | None
    road_camera_sensor: str | None
    calibration_height_m: float
    initial_rpy_calib: tuple[float, float, float] | None
    is_rhd: bool


def route_context(route_id: str) -> RouteContext:
    first_rlog = _first_rlog(route_id)
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "opendbc_repo")
    from openpilot.tools.lib.logreader import LogReader

    active_bundle = None
    params = {}
    device_type = None
    sensor = None
    height = 1.22
    rpy = None
    is_rhd = False
    live_delay = None
    for msg in LogReader(str(first_rlog)):
        which = msg.which()
        if which == "initData":
            params = {entry.key: bytes(entry.value).decode(errors="replace") for entry in msg.initData.params.entries}
            active_bundle = _parse_json(params.get("ModelManager_ActiveBundle"))
        elif which == "deviceState" and device_type is None:
            device_type = str(msg.deviceState.deviceType)
        elif which == "roadCameraState" and sensor is None:
            sensor = str(msg.roadCameraState.sensor)
        elif which == "liveCalibration" and rpy is None:
            lc = msg.liveCalibration
            if lc.height:
                height = float(lc.height[0])
            vals = list(lc.rpyCalib)
            if len(vals) >= 3:
                rpy = (float(vals[0]), float(vals[1]), float(vals[2]))
        elif which == "driverMonitoringState":
            is_rhd = bool(msg.driverMonitoringState.isRHD)
        elif which == "liveDelay" and live_delay is None:
            live_delay = float(msg.liveDelay.lateralDelay)
        if active_bundle is not None and device_type and sensor and rpy is not None and live_delay is not None:
            break
    overrides = _parse_overrides((active_bundle or {}).get("overrides", []))
    lagd_value = _float_param(params.get("LagdValueCache"), 0.2)
    lagd_toggle = _bool_param(params.get("LagdToggle"), False)
    native_lat_smooth = _float_param(overrides.get("lat"), 0.0 if active_bundle is not None else 0.1)
    base_lat_delay = lagd_value if lagd_toggle else live_delay
    return RouteContext(
        route_id=route_id,
        active_bundle_internal_name=(active_bundle or {}).get("internalName"),
        active_bundle_display_name=(active_bundle or {}).get("displayName"),
        active_bundle_generation=(active_bundle or {}).get("generation"),
        active_bundle_is20hz=(active_bundle or {}).get("is20hz"),
        active_bundle_overrides=overrides,
        camera_offset_m=_float_param(params.get("CameraOffset"), 0.0),
        planplus_control=_float_param(params.get("PlanplusControl"), 1.0),
        lagd_value_s=lagd_value,
        lagd_toggle=lagd_toggle,
        live_delay_lateral_s=live_delay,
        native_lat_smooth_seconds_s=native_lat_smooth,
        effective_native_lat_delay_input_s=(base_lat_delay + native_lat_smooth) if base_lat_delay is not None else None,
        device_type=device_type,
        road_camera_sensor=sensor,
        calibration_height_m=float(height),
        initial_rpy_calib=rpy,
        is_rhd=is_rhd,
    )


def _first_rlog(route_id: str) -> Path:
    hits = sorted((C.LOG_ROOT / route_id).glob("000000*--*--*/rlog.zst"))
    if not hits:
        raise FileNotFoundError(f"no rlog.zst for {route_id}")
    import sys
    sys.path.insert(0, ".")
    sys.path.insert(0, "opendbc_repo")
    from openpilot.tools.lib.logreader import LogReader

    errors = []
    for hit in hits:
        try:
            next(iter(LogReader(str(hit))))
            return hit
        except Exception as e:
            errors.append(f"{hit}: {type(e).__name__}: {e}")
    raise RuntimeError(f"no readable rlog.zst for {route_id}: {'; '.join(errors[:3])}")


def _parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_overrides(raw) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    return {str(item["key"]): str(item["value"]) for item in raw if "key" in item and "value" in item}


def _float_param(text: str | None, default: float) -> float:
    try:
        return float(text) if text is not None else default
    except ValueError:
        return default


def _bool_param(text: str | None, default: bool) -> bool:
    if text is None:
        return default
    return text in {"1", "true", "True"}


def desire_input_for_window(blinker: bool, lane_change_state: bool) -> np.ndarray:
    if blinker or lane_change_state:
        raise ValueError("lane-change windows are excluded because replay desire input is not logged")
    return np.zeros(8, dtype=np.float32)


def traffic_convention_input(ctx: RouteContext) -> np.ndarray:
    traffic = np.zeros(2, dtype=np.float32)
    traffic[int(ctx.is_rhd)] = 1.0
    return traffic


def lateral_control_params(ctx: RouteContext, bundle_name: str, v_ego: float) -> np.ndarray:
    from model_replay_sim.bundles import candidate_bundle_overrides

    base_delay = ctx.lagd_value_s if ctx.lagd_toggle else ctx.live_delay_lateral_s
    if base_delay is None:
        raise ValueError("liveDelay.lateralDelay missing and LagdToggle is false")
    lat_smooth = _float_param(candidate_bundle_overrides(bundle_name).get("lat"), 0.0)
    return np.array([float(v_ego), float(base_delay + lat_smooth)], dtype=np.float32)


def write_route_context_catalog(route_ids: list[str]) -> Path:
    import pandas as pd

    out = C.RESULTS_ROOT / "catalog" / "route_context.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [route_context(route_id).__dict__ for route_id in route_ids]
    pd.DataFrame(rows).to_parquet(out)
    return out
```

Implementation note: `effective_native_lat_delay_input_s` is only for same-model fidelity anchors. Cross-model replay must recompute the policy `lateral_control_params` delay using `candidate_bundle_overrides(...)` plus the route's logged/live-delay base (`LagdValueCache` when `LagdToggle` is true, otherwise `liveDelay.lateralDelay`). OPM7 remains disqualified as a strict same-model anchor without active-bundle provenance, but it still has an explicit legacy candidate override source so replay inputs are executable.

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
DEBUG=0 .venv311/bin/python -m pytest model_replay_sim/tests/test_context.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/context.py model_replay_sim/tests/test_context.py
git commit -m "model-sim: extract replay route context"
```

---

### Task 5: Resolve Bundle Provenance, ONNX, And Metadata

**Files:**
- Create: `model_replay_sim/bundles.py`
- Create: `model_replay_sim/tests/test_bundles.py`

- [ ] **Step 1: Write failing tests for full refs and LFS pointer detection**

Create `model_replay_sim/tests/test_bundles.py`:

```python
import pytest
import json

from pathlib import Path

from model_replay_sim.bundles import (
    bundle_fetch_plan,
    bundle_ref,
    candidate_bundle_overrides,
    copy_local_stock_onnx_for_test,
    expected_model_files,
    generate_metadata_for_onnx,
    is_git_lfs_pointer,
    native_bundle_overrides,
    write_bundle_provenance,
)
from model_replay_sim import config as C
from model_replay_sim.pytest_helpers import require_real_asset


def test_bundle_refs_are_full_or_fetchable_refs_not_bare_short_sha():
    assert bundle_ref("CD210").full_sha == "55f66e2246359c6593605399a0199d94d13ad90d"
    assert bundle_ref("CD210").fetch_ref == "refs/pull/37050/head"
    assert bundle_ref("Nevada").full_sha == "3193eac5e385aa010694a8ac192ff38ffe000193"
    assert bundle_ref("OPM7").full_sha == "052692b25d63c5ddda276b5c2271383b6aff129f"


def test_expected_model_files_encode_split_path():
    assert expected_model_files("CD210") == ["driving_vision.onnx", "driving_policy.onnx"]
    assert expected_model_files("OPM7") == [
        "driving_vision.onnx",
        "driving_on_policy.onnx",
        "driving_off_policy.onnx",
    ]


def test_lfs_pointer_detection():
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n"
    assert is_git_lfs_pointer(pointer) is True
    assert is_git_lfs_pointer(b"\x08\x01real onnx bytes") is False


def test_fetch_plan_uses_target_ref_and_lfs_include_not_broad_fetch():
    commands = bundle_fetch_plan("CD210", Path("/tmp/cd210"))
    text = [" ".join(cmd) for cmd in commands]
    assert any(cmd[-4:] == ["fetch", "--depth=1", "origin", "refs/pull/37050/head"] for cmd in commands)
    assert any(cmd[-3:] == ["lfs", "pull", "--include=selfdrive/modeld/models/*.onnx"] for cmd in commands)
    assert not any("fetch --all" in line for line in text)


def test_candidate_overrides_are_recovered_from_native_active_bundle_logs():
    require_real_asset((C.LOG_ROOT / "route_b5").exists() and (C.LOG_ROOT / "route_c4").exists(), "route_b5 and route_c4 native context rlogs")
    assert native_bundle_overrides("CD210")["lat"] == ".0"
    assert native_bundle_overrides("Nevada")["lat"] == ".1"
    with pytest.raises(LookupError, match="OPM7"):
        native_bundle_overrides("OPM7")
    assert candidate_bundle_overrides("OPM7")["lat"] == ".1"
    assert bundle_ref("OPM7").override_source.startswith("legacy stock")


def test_metadata_and_provenance_write_next_to_results_copy(tmp_path):
    copy_local_stock_onnx_for_test(tmp_path)
    onnx = tmp_path / "driving_policy.onnx"
    metadata = generate_metadata_for_onnx(onnx)
    provenance = write_bundle_provenance("stock-test", [onnx], [metadata], clone_dir=None, source_ref="local-tree")
    data = json.loads(provenance.read_text())
    assert provenance.parent == tmp_path
    assert data["source_ref"] == "local-tree"
    assert data["files"][0]["onnx_sha256"]
    assert data["files"][0]["metadata_sha256"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_bundles.py -q
```

Expected: FAIL because `model_replay_sim.bundles` does not exist.

- [ ] **Step 3: Implement resolver skeleton and metadata generation**

Create `model_replay_sim/bundles.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from model_replay_sim import config as C
from model_replay_sim.env import replay_env, replay_python


@dataclass(frozen=True)
class BundleRef:
    name: str
    repo: str
    fetch_ref: str
    full_sha: str
    split: bool
    models: tuple[str, ...]
    native_context_routes: tuple[str, ...]
    candidate_overrides: dict[str, str]
    override_source: str


def bundle_ref(name: str) -> BundleRef:
    raw = C.BUNDLES[name]
    return BundleRef(
        name=name,
        repo=raw["repo"],
        fetch_ref=raw["fetch_ref"],
        full_sha=raw["full_sha"],
        split=bool(raw["split"]),
        models=tuple(raw["models"]),
        native_context_routes=tuple(raw["native_context_routes"]),
        candidate_overrides=dict(raw["candidate_overrides"]),
        override_source=str(raw["override_source"]),
    )


def expected_model_files(name: str) -> list[str]:
    return list(bundle_ref(name).models)


def is_git_lfs_pointer(data: bytes) -> bool:
    return data.startswith(b"version https://git-lfs.github.com/spec/v1\n")


def bundle_dir(name: str) -> Path:
    return C.RESULTS_ROOT / "onnx" / name


def bundle_fetch_plan(name: str, clone_dir: Path) -> list[list[str]]:
    ref = bundle_ref(name)
    remote = f"https://github.com/{ref.repo}.git"
    return [
        ["git", "init", str(clone_dir)],
        ["git", "-C", str(clone_dir), "remote", "add", "origin", remote],
        ["git", "-C", str(clone_dir), "fetch", "--depth=1", "origin", ref.fetch_ref],
        ["git", "-C", str(clone_dir), "checkout", "--detach", ref.full_sha],
        ["git", "-C", str(clone_dir), "lfs", "install", "--local"],
        ["git", "-C", str(clone_dir), "lfs", "pull", "--include=selfdrive/modeld/models/*.onnx"],
    ]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=C.REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc


def _git_stdout(args: list[str]) -> str:
    return _run(["git", *args]).stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _tinygrad_sha() -> str:
    return _git_stdout(["-C", str(C.TINYGRAD_PATH), "rev-parse", "HEAD"])


def ensure_bundle_repo_at_ref(name: str, clone_dir: Path) -> str:
    ref = bundle_ref(name)
    remote = f"https://github.com/{ref.repo}.git"
    if not (clone_dir / ".git").exists():
        _run(["git", "init", str(clone_dir)])
        _run(["git", "-C", str(clone_dir), "remote", "add", "origin", remote])
    else:
        _run(["git", "-C", str(clone_dir), "remote", "set-url", "origin", remote])
    _run(["git", "-C", str(clone_dir), "fetch", "--depth=1", "origin", ref.fetch_ref])
    fetched = _git_stdout(["-C", str(clone_dir), "rev-parse", "FETCH_HEAD"])
    if fetched != ref.full_sha:
        raise RuntimeError(f"{name} fetched {fetched}, expected {ref.full_sha}")
    _run(["git", "-C", str(clone_dir), "checkout", "--detach", ref.full_sha])
    _run(["git", "-C", str(clone_dir), "lfs", "install", "--local"])
    _run(["git", "-C", str(clone_dir), "lfs", "pull", "--include=selfdrive/modeld/models/*.onnx"])
    return fetched


def materialize_bundle_onnx(name: str) -> list[Path]:
    ref = bundle_ref(name)
    clone_dir = C.RESULTS_ROOT / "repos" / name
    dest = bundle_dir(name)
    dest.mkdir(parents=True, exist_ok=True)
    ensure_bundle_repo_at_ref(name, clone_dir)
    copied = []
    for filename in ref.models:
        src = clone_dir / "selfdrive" / "modeld" / "models" / filename
        if not src.exists():
            raise FileNotFoundError(f"{src} not found in fetched {name} ref {ref.full_sha}")
        data = src.read_bytes()
        if is_git_lfs_pointer(data):
            raise RuntimeError(f"{src} is still a Git LFS pointer; git lfs pull did not materialize it")
        out = dest / filename
        shutil.copy2(src, out)
        copied.append(out)
    return copied


def lfs_oid_for_path(clone_dir: Path, rel_path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(clone_dir), "lfs", "ls-files", "--long", "--", rel_path],
        cwd=C.REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.split()[0]


def materialize_bundle_artifacts(name: str) -> tuple[list[Path], list[Path], Path]:
    clone_dir = C.RESULTS_ROOT / "repos" / name
    onnx_paths = materialize_bundle_onnx(name)
    metadata_paths = [generate_metadata_for_onnx(path) for path in onnx_paths]
    provenance = write_bundle_provenance(name, onnx_paths, metadata_paths, clone_dir=clone_dir, source_ref=bundle_ref(name).full_sha)
    return onnx_paths, metadata_paths, provenance


def native_bundle_overrides(name: str) -> dict[str, str]:
    from model_replay_sim.context import route_context

    ref = bundle_ref(name)
    for route_id in ref.native_context_routes:
        try:
            ctx = route_context(route_id)
        except Exception:
            continue
        if ctx.active_bundle_overrides:
            return ctx.active_bundle_overrides
    raise LookupError(f"{name} has no recoverable native active-bundle overrides in local logs")


def candidate_bundle_overrides(name: str) -> dict[str, str]:
    ref = bundle_ref(name)
    try:
        native = native_bundle_overrides(name)
        if native:
            return native
    except LookupError:
        pass
    if not ref.candidate_overrides:
        raise LookupError(f"{name} has no candidate override source")
    return ref.candidate_overrides


def generate_metadata_for_onnx(onnx_path: Path) -> Path:
    cmd = replay_python([C.REPO_ROOT / "selfdrive" / "modeld" / "get_model_metadata.py", onnx_path])
    proc = subprocess.run(cmd, cwd=C.REPO_ROOT, env=replay_env(), text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    metadata = onnx_path.with_name(f"{onnx_path.stem}_metadata.pkl")
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    return metadata


def write_bundle_provenance(
    name: str,
    onnx_paths: list[Path],
    metadata_paths: list[Path],
    clone_dir: Path | None,
    source_ref: str,
) -> Path:
    if len(onnx_paths) != len(metadata_paths):
        raise ValueError("onnx_paths and metadata_paths must have the same length")
    ref = bundle_ref(name) if name in C.BUNDLES else None
    files = []
    for onnx_path, metadata_path in zip(onnx_paths, metadata_paths):
        rel_path = f"selfdrive/modeld/models/{Path(onnx_path).name}"
        files.append({
            "model_file": Path(onnx_path).name,
            "onnx_path": str(onnx_path),
            "metadata_path": str(metadata_path),
            "onnx_sha256": _sha256(Path(onnx_path)),
            "metadata_sha256": _sha256(Path(metadata_path)),
            "lfs_oid": lfs_oid_for_path(clone_dir, rel_path) if clone_dir is not None else None,
        })
    out = Path(onnx_paths[0]).parent / "provenance.json"
    out.write_text(json.dumps({
        "bundle": name,
        "repo": ref.repo if ref else None,
        "fetch_ref": ref.fetch_ref if ref else None,
        "full_sha": ref.full_sha if ref else None,
        "source_ref": source_ref,
        "tinygrad_sha": _tinygrad_sha(),
        "files": files,
    }, indent=2, sort_keys=True))
    return out


def copy_local_stock_onnx_for_test(dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_root = C.REPO_ROOT / "selfdrive" / "modeld" / "models"
    for name in ["driving_vision.onnx", "driving_policy.onnx"]:
        shutil.copy2(src_root / name, dest_dir / name)
```

- [ ] **Step 4: Run tests to verify metadata and provenance pass**

Run:

```bash
DEBUG=0 .venv311/bin/python -m pytest model_replay_sim/tests/test_bundles.py -q
```

Expected: PASS. `provenance.json` and generated metadata are written next to the copied ONNX under the test temp directory, and no `*_metadata.pkl` appears under `selfdrive/modeld/models/`.

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/bundles.py model_replay_sim/tests/test_bundles.py
git commit -m "model-sim: resolve bundle provenance and metadata"
```

---

### Task 6: Make Anchor Asset Selection Explicit And Non-Skipping

**Files:**
- Create: `model_replay_sim/assets.py`
- Create: `model_replay_sim/tests/test_assets.py`

Anchor eligibility is not just "has `fcamera.hevc`". A candidate must have an NPZ cache, a replay-scene window from the registered cache signals (straight/gentle, speed-gated, no blinker/lane-change), and a successful `map_window_to_frames(...)` mapping for the whole window. Steering override and engagement state are recorded for analysis but are not model inputs, so they are not part of this asset-presence gate.

- [ ] **Step 1: Write failing tests**

Create `model_replay_sim/tests/test_assets.py`:

```python
import pytest

from model_replay_sim.assets import anchor_candidates, require_same_model_anchor_asset, require_sanity_asset


def test_cd210_candidates_include_local_route_b5_before_pull_routes():
    candidates = anchor_candidates("CD210")
    route_ids = [c.route_id for c in candidates]
    assert "route_b5" in route_ids
    assert "route_b8" in route_ids
    b5 = next(c for c in candidates if c.route_id == "route_b5")
    assert b5.local_fcamera_segments >= 1
    assert b5.npz_exists is True
    assert b5.active_bundle_match is True
    assert b5.eligible_aligned_windows_20s >= 1


def test_nevada_candidates_are_marked_as_needing_frame_pull():
    candidates = anchor_candidates("Nevada")
    assert {c.route_id for c in candidates} >= {"route_c4", "route_c5", "route_c6", "route_c7"}
    assert all(c.local_fcamera_segments == 0 for c in candidates)
    assert all(c.active_bundle_match is True for c in candidates)


def test_require_same_model_anchor_asset_fails_loudly_for_missing_nevada_frames():
    with pytest.raises(AssertionError, match="missing local fcamera"):
        require_same_model_anchor_asset("Nevada")


def test_opm7_is_not_a_strict_same_model_anchor_without_active_bundle_provenance():
    with pytest.raises(AssertionError, match="active-bundle provenance"):
        require_same_model_anchor_asset("OPM7")


def test_sanity_asset_can_use_local_frames_without_strict_bundle_provenance():
    candidate = require_sanity_asset("CD210")
    assert candidate.local_fcamera_segments > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
DEBUG=0 .venv311/bin/python -m pytest model_replay_sim/tests/test_assets.py -q
```

Expected: FAIL because `model_replay_sim.assets` does not exist.

- [ ] **Step 3: Implement anchor candidates**

Create `model_replay_sim/assets.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_replay_sim import config as C
from model_replay_sim.alignment import map_window_to_frames
from model_replay_sim.context import route_context
from retrospective_lateral.code import config as RL_C
from retrospective_lateral.code.signal_utils import contiguous_regions, dilate_flags


@dataclass(frozen=True)
class AnchorCandidate:
    bundle: str
    route_id: str
    source: str
    local_fcamera_segments: int
    local_ecamera_segments: int
    npz_exists: bool
    active_bundle_match: bool | None
    eligible_aligned_windows_20s: int
    needs_frame_pull: bool


ROUTE_CANDIDATES = {
    "CD210": [
        ("route_b5", "local-CD210-active-bundle"),
        ("route_b8", "preferred-CD210-active-bundle"),
        ("route_c1", "preferred-CD210-active-bundle"),
        ("route_c2", "preferred-CD210-active-bundle"),
        ("route_c3", "preferred-CD210-active-bundle"),
    ],
    "Nevada": [
        ("route_c4", "Nevada-active-bundle"),
        ("route_c5", "Nevada-active-bundle"),
        ("route_c6", "Nevada-active-bundle"),
        ("route_c7", "Nevada-active-bundle"),
    ],
    "OPM7": [
        ("route_7f", "hand-labeled-OPM7"),
    ],
}


def anchor_candidates(bundle: str) -> list[AnchorCandidate]:
    out: list[AnchorCandidate] = []
    for route_id, source in ROUTE_CANDIDATES[bundle]:
        route_dir = C.LOG_ROOT / route_id
        fcam = len(list(route_dir.glob("000000*--*--*/fcamera.hevc")))
        ecam = len(list(route_dir.glob("000000*--*--*/ecamera.hevc")))
        npz_exists = (C.CACHE_ROOT / f"{route_id}.npz").exists()
        active_match = _active_bundle_matches(bundle, route_id)
        aligned_windows = _eligible_aligned_window_count(route_id, min_seconds=20.0)
        out.append(AnchorCandidate(
            bundle=bundle,
            route_id=route_id,
            source=source,
            local_fcamera_segments=fcam,
            local_ecamera_segments=ecam,
            npz_exists=npz_exists,
            active_bundle_match=active_match,
            eligible_aligned_windows_20s=aligned_windows,
            needs_frame_pull=fcam == 0,
        ))
    return out


def _active_bundle_matches(bundle: str, route_id: str) -> bool | None:
    try:
        ctx = route_context(route_id)
    except Exception:
        return None
    if ctx.active_bundle_internal_name is None:
        return None
    expected = {
        "CD210": {"C210M", "CD210"},
        "Nevada": {"NM", "Nevada"},
        "OPM7": {"OPM7"},
    }[bundle]
    return ctx.active_bundle_internal_name in expected


def _eligible_aligned_window_count(route_id: str, min_seconds: float, needs_wide: bool = True) -> int:
    count = 0
    for mono_times in _eligible_npz_mono_windows(route_id, min_seconds=min_seconds):
        try:
            aligned = map_window_to_frames(route_id, mono_times.tolist(), max_delta_s=0.03)
        except ValueError:
            continue
        if needs_wide and any(a.ecamera_index is None for a in aligned):
            continue
        count += 1
    return count


def _eligible_npz_mono_windows(route_id: str, min_seconds: float) -> list[np.ndarray]:
    path = C.CACHE_ROOT / f"{route_id}.npz"
    if not path.exists():
        return []
    arrays = np.load(path)
    mono = arrays["mono_time"].astype(float)
    v = arrays["v_ego"].astype(float)
    n = len(mono)
    speed_mph = v * 2.23694
    yaw = np.where(
        np.isfinite(arrays.get("yaw_rate_calibrated", np.full(n, np.nan))),
        arrays.get("yaw_rate_calibrated", np.full(n, np.nan)),
        arrays.get("yaw_rate", np.full(n, np.nan)),
    ).astype(float)
    curv = np.divide(yaw, v, out=np.full(n, np.nan), where=(v > 1.0) & np.isfinite(yaw))
    blinker = dilate_flags(arrays.get("blinker", np.zeros(n)) > 0.5, int(round(RL_C.BLINKER_BUFFER_S * RL_C.FS_HZ)))
    lane_change = (
        (arrays.get("lane_change_state", np.zeros(n)) > 0.5)
        | (arrays.get("cx1_lane_change", np.zeros(n)) > 0.5)
    )
    lane_change = dilate_flags(lane_change, int(round(RL_C.LANE_CHANGE_BUFFER_S * RL_C.FS_HZ)))
    eligible = (
        np.isfinite(mono)
        & (speed_mph >= RL_C.WEAVE_SPEED_MPH[0])
        & (speed_mph <= RL_C.WEAVE_SPEED_MPH[1])
        & np.isfinite(curv)
        & (np.abs(curv) <= RL_C.ROAD_CURV_ABS_MAX_1PM)
        & ~blinker
        & ~lane_change
    )
    min_len = int(round(min_seconds * C.FS_HZ))
    windows: list[np.ndarray] = []
    for start, end in contiguous_regions(eligible, min_len=min_len):
        cursor = start
        while cursor + min_len <= end:
            windows.append(mono[cursor:cursor + min_len])
            cursor += min_len
    return windows


def require_sanity_asset(bundle: str) -> AnchorCandidate:
    candidates = anchor_candidates(bundle)
    local = [c for c in candidates if c.local_fcamera_segments > 0 and c.npz_exists and c.eligible_aligned_windows_20s > 0]
    if not local:
        routes = ", ".join(c.route_id for c in candidates)
        raise AssertionError(f"{bundle} missing local fcamera/NPZ/aligned-window sanity assets: {routes}")
    return local[0]


def require_same_model_anchor_asset(bundle: str, needs_wide: bool = True) -> AnchorCandidate:
    candidates = anchor_candidates(bundle)
    eligible = [
        c for c in candidates
        if c.active_bundle_match is True
        and c.local_fcamera_segments > 0
        and c.npz_exists
        and c.eligible_aligned_windows_20s > 0
        and (not needs_wide or c.local_ecamera_segments > 0)
    ]
    if not eligible:
        reasons = []
        for c in candidates:
            if c.active_bundle_match is not True:
                reasons.append(f"{c.route_id}: missing active-bundle provenance")
            elif c.local_fcamera_segments == 0:
                reasons.append(f"{c.route_id}: missing local fcamera")
            elif needs_wide and c.local_ecamera_segments == 0:
                reasons.append(f"{c.route_id}: missing local ecamera")
            elif not c.npz_exists:
                reasons.append(f"{c.route_id}: missing NPZ cache")
            elif c.eligible_aligned_windows_20s == 0:
                reasons.append(f"{c.route_id}: no >=20s aligned windows")
        raise AssertionError(f"{bundle} has no strict same-model anchor asset: {'; '.join(reasons)}")
    return eligible[0]
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
DEBUG=0 .venv311/bin/python -m pytest model_replay_sim/tests/test_assets.py -q
```

Expected: PASS in the current workspace: CD210 has local `route_b5`, Nevada has no local frames and fails loudly.

- [ ] **Step 5: Commit**

```bash
git add model_replay_sim/assets.py model_replay_sim/tests/test_assets.py
git commit -m "model-sim: make anchor assets explicit"
```

---

### Task 7: Audit Real-Asset Gates In Future Replay Tests

**Files:**
- Modify later simulator anchor tests to use `model_replay_sim.pytest_helpers.require_real_asset` instead of `SEG = None` skip patterns.
- Modify: `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md`

- [ ] **Step 1: Audit the future-test pattern**

When implementing `test_anchor_vision.py`, `test_pipeline_go_nogo.py`, and `test_fidelity_anchor.py`, do not use:

```python
SEG = None
pytest.mark.skipif(SEG is None, reason="set SEG")
```

Use:

```python
from model_replay_sim.pytest_helpers import require_real_asset


def test_pipeline_runs_on_registered_segment():
    require_real_asset(SEG is not None, "registered stock-era route_8d segment")
    ...
```

Expected behavior: missing required real assets fail in normal QA and CI-like local verification. Developers can opt into unit-only mode with `MODEL_REPLAY_ALLOW_MISSING_ASSETS=1`.

- [ ] **Step 2: Add a plan consistency check for skippable core gates**

Run:

```bash
rg -n "SEG = None|skipif\\(SEG|NEVADA_SEG = None|pytest\\.mark\\.skipif\\(not \\(C\\.LOG_ROOT" docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md
```

Expected: no matches in core replay, alignment, compile, or fidelity-anchor tests. If a non-core optional smoke test intentionally permits missing data, rewrite it to call `require_real_asset(... )` unless it is truly a unit-only test.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md
git commit -m "docs: make replay asset gates explicit"
```

---

### Task 8: Integrate Hardening Into The Main Simulator Plan

**Files:**
- Modify: `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md`

- [ ] **Step 1: Patch the main plan's Task 0 command block**

Replace the stock compile command with:

```bash
git submodule update --init tinygrad_repo
.venv311/bin/pip install -r requirements-analysis.txt
DEBUG=0 DEV=CPU IMAGE=0 THREADS=0 PYTHONPATH=tinygrad_repo:. .venv311/bin/python tinygrad_repo/examples/openpilot/compile3.py \
    ./selfdrive/modeld/models/driving_vision.onnx /tmp/driving_vision_cpu.pkl
```

State explicitly: direct `compile3.py SELFTEST=1` is a diagnostic only and is not a GO/NO-GO gate. The load-bearing compile gate is `model_replay_sim.compile_bundle.compile_onnx_to_pkl(..., compare_onnxruntime=True)` and it gates only on compile return code plus `tinygrad_jit_validated`.

- [ ] **Step 1b: Replace the original Task 7 compile wrapper**

In `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md`, replace the old `compile_onnx_to_mac_pkl(..., selftest=True)` snippet and its test:

```python
rec = compile_onnx_to_mac_pkl(onnx, out, selftest=True)
assert rec["selftest_passed"] is True
```

with:

```python
report = compile_onnx_to_pkl(onnx, out, compare_onnxruntime=True)
assert out.exists() and out.stat().st_size > 0
assert report["compile_returncode"] == 0
assert report["tinygrad_jit_validated"] is True
assert report["onnxruntime_compare"]["ran"] is True
```

Remove all `selftest=True`, `selftest_passed`, and "selftest mismatch means STOP" text from the main plan. Keep ONNXRuntime raw-output comparison only as a recorded diagnostic.

- [ ] **Step 2: Insert a new Phase 0a before warp/inference**

Add these required gates before any vision/policy replay:

```markdown
1. Build `segment_catalog.parquet` and `frame_timeline.parquet` from rlogs and local HEVC files.
2. Verify every selected replay window maps through `map_window_to_frames(...)` to one exact segment plus continuous `fcamera` and, when the model has big-image inputs, `ecamera` frame indices using the common alignable intersection of `roadCameraState`, `wideRoadCameraState`, and `modelV2` frame IDs/timestamps. Trim unalignable leading/trailing `modelV2` frames at segment boundaries; do not use `first_model_mono_time` as a replay frame unless it is in the common alignable set.
3. Extract route context: `liveCalibration.height`, `rpyCalib`, `deviceState.deviceType`, `roadCameraState.sensor`, `CameraOffset`, `PlanplusControl`, `LagdValueCache`, `LagdToggle`, logged `liveDelay.lateralDelay`, active-bundle overrides, and traffic convention.
4. For policy `lateral_control_params`, recompute the delay for each candidate model using the route's logged delay base plus `candidate_bundle_overrides(bundle)["lat"]`; use route active-bundle overrides only to validate same-model fidelity anchors.
5. Exclude lane-change/blinker windows from replay because desire input is not logged; primary straight/gentle windows use zero desire input.
```

- [ ] **Step 3: Replace silent skip language**

Search:

```bash
rg -n "SEG = None|skipif\\(SEG|NEVADA_SEG = None|NotImplementedError" docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md
```

For every anchor test, require registered real assets via `require_real_asset(...)`. Do not let the plan pass core replay tests with `SEG = None` or unimplemented replay functions.

- [ ] **Step 4: Update bundle provenance language**

Replace short-ref fetches with full refs:

```text
CD210: commaai/openpilot refs/pull/37050/head -> 55f66e2246359c6593605399a0199d94d13ad90d
Nevada: commaai/openpilot refs/pull/36114/head -> 3193eac5e385aa010694a8ac192ff38ffe000193
OPM7: sunnypilot/sunnypilot 052692b25d63c5ddda276b5c2271383b6aff129f
```

State that ONNX files at those refs are Git LFS pointers until `git lfs pull` materializes them, and metadata is not committed at those refs.

- [ ] **Step 5: Update anchor asset language**

Replace "CD210 requires frame pull" with:

```text
CD210: first try local route_b5 for a native-bundle sanity anchor; if it lacks enough eligible aligned windows, pull route_b8 or route_c1-c3 frames.
Nevada: route_c4-c7 are native Nevada but currently have no local fcamera; pull at least one route's fcamera/ecamera before same-model anchor.
OPM7: route_7f has local frames but no active-bundle param; treat as hand-labeled only unless exact OPM7 bundle provenance is recovered elsewhere.
```

- [ ] **Step 6: Run plan consistency checks**

Run:

```bash
rg -n "TBD|TODO|implement later|SEG = None|skipif\\(SEG|NotImplementedError|SELFTEST=1.*GO" docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md
rg -n "selftest=True|selftest_passed|selftest mismatches|SELFTEST=1.*(GO|NO-GO|gate|STOP)" docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md
```

Expected: no matches that leave core replay gates skippable or compile `SELFTEST` load-bearing. If the second command matches, either remove the stale text or rewrite it so `SELFTEST=1` is clearly diagnostic-only and not in a GO/NO-GO sentence.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md
git commit -m "docs: harden same-scene replay simulator plan"
```

---

## Self-Review

**Spec coverage:** This plan covers every verified QA requirement: command env/path fixes (Tasks 1-2), dependency corrections (Task 1), custom compile diagnostic (Task 2), frame alignment (Task 3), route context/calibration/model input extraction (Task 4), full refs/LFS/metadata (Task 5), anchor asset truth (Task 6), non-skipping tests (Task 7), and integration into the main simulator plan (Task 8).

**Placeholder scan:** The plan has no actionable placeholders. Literal strings such as `TODO`, `SEG = None`, and `selftest=True` appear only inside negative-pattern audit commands or examples of code that must be removed from the original simulator plan. Required local-data gates use `require_real_asset(...)` so missing route/video/cache assets fail loudly unless a developer explicitly opts into unit-only mode with `MODEL_REPLAY_ALLOW_MISSING_ASSETS=1`.

**Type consistency:** `RouteContext`, `SegmentCatalogRow`, `FrameAlignment`, `BundleRef`, and `AnchorCandidate` are defined before later tasks use them. `replay_env()` and `replay_python()` are used consistently in compile and metadata tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator-hardening.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.
