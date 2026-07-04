"""Re-runnable integration gate for the S0 offset plumbing (and, after Task 4,
the M2a capture). Replays a short real window of ROUTE on SP002.

GATE A (offset plumbing): camera_offset=None and camera_offset=0.0 must produce
IDENTICAL desired_curvature (the route's logged param IS 0.0), while
camera_offset=0.05 must produce a DIFFERENT series (the shear reached the warp).

RUN: .venv311/bin/python stock_lateral_toolkit/centering/smoke_replay.py
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC

N_FRAMES = 60  # short: model load dominates; replay ~25 s


def _window_monos():
    from model_replay_sim.anchor import select_anchor_span
    span = select_anchor_span(CC.ROUTE, warmup_s=1.0, min_compare_s=2.0)
    return [float(t) for t in span.mono_times[:N_FRAMES]]


def main():
    from model_replay_sim.infer import replay_window
    mono = _window_monos()

    r_none = replay_window("SP002", CC.ROUTE, mono, camera_offset=None)
    r_zero = replay_window("SP002", CC.ROUTE, mono, camera_offset=0.0)
    r_off = replay_window("SP002", CC.ROUTE, mono, camera_offset=0.05)

    assert r_none["camera_offset_used"] == 0.0, r_none["camera_offset_used"]
    same = np.nanmax(np.abs(r_none["desired_curvature"] - r_zero["desired_curvature"]))
    diff = np.nanmax(np.abs(r_none["desired_curvature"] - r_off["desired_curvature"]))
    print(f"GATE A: max|none-zero| = {same:.3e} (expect 0), max|none-0.05| = {diff:.3e} (expect > 0)")
    assert same == 0.0, "offset=0.0 must equal the route default (param was 0.0)"
    assert diff > 0.0, "offset=0.05 must change the replayed curvature"
    print("GATE A PASS")


if __name__ == "__main__":
    main()
