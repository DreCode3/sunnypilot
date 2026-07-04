"""Shared scene-window selection: N disjoint (warmup+compare) frame windows over the
replay-eligible contiguous runs of a route. Pure (no model). The SAME windows.json is
read by the M1 sampler (frame preference), M2 consensus replays, and the S1 sweep, so
all phases see identical frames.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/windows.py [--route <name>]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def _eligible_runs(route_id: str):
    """All maximal runs that are BOTH eligible and frame-id contiguous, longest first.
    (Same run predicate as anchor._longest_contiguous_run, but keeps every run.)"""
    import model_replay_sim.anchor as anchor_mod
    good, tl = anchor_mod._eligible_aligned_timeline_mask(route_id)
    fids = [r.frame_id for r in tl]
    runs = []
    i, n = 0, len(good)
    while i < n:
        if good[i]:
            j = i + 1
            while j < n and good[j] and fids[j] == fids[j - 1] + 1:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    runs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    return runs, tl


def select_scene_windows(route_id: str, n_windows: int, warmup_frames: int,
                         compare_frames: int, min_compare_frames: int) -> list[dict]:
    runs, tl = _eligible_runs(route_id)
    out: list[dict] = []
    for a, b in runs:
        pos = a
        while len(out) < n_windows:
            remaining = b - pos
            if remaining >= warmup_frames + compare_frames:
                take = warmup_frames + compare_frames
            elif remaining >= warmup_frames + min_compare_frames:
                take = remaining
            else:
                break
            rows = tl[pos:pos + take]
            out.append({
                "window_id": len(out),
                "route_id": route_id,
                "mono_times": [float(r.timestamp_eof_s) for r in rows],
                "frame_ids": [int(r.frame_id) for r in rows],
                "split_index": int(warmup_frames),
                "n_compare": int(take - warmup_frames),
            })
            pos += take
        if len(out) >= n_windows:
            break
    if not out:
        raise RuntimeError(f"{route_id}: no eligible contiguous run of >= "
                           f"{warmup_frames + min_compare_frames} frames")
    return out


def windows_path(route: str = CC.ROUTE) -> Path:
    return CC.windows_json(route)


def write_windows(windows: list[dict], route: str = CC.ROUTE) -> Path:
    p = windows_path(route)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(windows))
    return p


def load_windows(route: str = CC.ROUTE) -> list[dict]:
    return json.loads(windows_path(route).read_text())


if __name__ == "__main__":
    route = sys.argv[sys.argv.index("--route") + 1] if "--route" in sys.argv else CC.ROUTE
    ws = select_scene_windows(route, CC.N_WINDOWS, CC.WARMUP_FRAMES,
                              CC.COMPARE_FRAMES, CC.MIN_COMPARE_FRAMES)
    p = write_windows(ws, route)
    for w in ws:
        m = w["mono_times"]
        print(f"window {w['window_id']}: {len(m)} frames ({w['n_compare']} compare) "
              f"mono [{m[0]:.1f}, {m[-1]:.1f}]")
    print(f"wrote {p}")
