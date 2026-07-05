"""Parse the LOGGED model-action curvature from rlogs, aligned to frame ids.

The replay (Task 8) reproduces the model's pre-controller `action.desiredCurvature`,
so the fidelity anchor's ground truth is the LOGGED model action — keyed by frameId,
carried identically on `modelV2` and (fallback) `drivingModelData`. This is NOT the
NPZ `desired_curvature` (that is the lag-adjusted downstream `controlsState` value).
"""
from __future__ import annotations
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # openpilot import robust to CWD
from openpilot.tools.lib.logreader import LogReader
from model_replay_sim import config as C
from model_replay_sim.alignment import build_frame_timeline, _seg_dirs, _seg_num


@functools.lru_cache(maxsize=8)
def _route_action_map(route_id: str) -> dict:
    """{frame_id: logged modelV2.action.desiredCurvature} for the whole route.

    Scans each frame-present segment's rlog once. modelV2 is primary; drivingModelData
    fills any frameId the modelV2 stream is missing (they carry identical values).
    Corrupt/truncated rlogs are skipped per-segment (mirrors alignment.py).
    """
    tl = build_frame_timeline(route_id)
    seg_nums = sorted({r.segment_num for r in tl})
    seg_by_num = {_seg_num(s): s for s in _seg_dirs(route_id)}
    out: dict[int, float] = {}
    for snum in seg_nums:
        seg = seg_by_num.get(snum)
        if seg is None:
            continue
        dmd: dict[int, float] = {}
        try:                                          # corrupt/truncated rlog → skip segment
            for m in LogReader(str(seg / "rlog.zst")):
                w = m.which()
                if w == "modelV2":
                    out[int(m.modelV2.frameId)] = float(m.modelV2.action.desiredCurvature)
                elif w == "drivingModelData":
                    dmd[int(m.drivingModelData.frameId)] = float(m.drivingModelData.action.desiredCurvature)
        except Exception:
            continue
        for fid, c in dmd.items():                    # fallback only where modelV2 absent
            out.setdefault(fid, c)
    return out


def logged_action_curvature(route_id: str, frame_ids) -> np.ndarray:
    """LOGGED modelV2.action.desiredCurvature for `frame_ids`, in the SAME order.

    Missing frame_id → np.nan (kept in place, never dropped — Task 10 needs aligned arrays).
    """
    amap = _route_action_map(route_id)
    fids = np.asarray(frame_ids).reshape(-1)
    return np.array([amap.get(int(f), np.nan) for f in fids], dtype=float)
