"""R1: does SP002 misplace the LANE (perception translation) or correctly place the
lane and target off its center (trained preference)? Per-frame per-line comparison of
replayed SP002 lane lines vs M1 video annotations on identical frames, pooled across
the primary route and any ROUTES_EXTRA route with completed M1 + SP002 replays.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r1_discriminator.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering import ground_plane as G
from stock_lateral_toolkit.centering.m1_offsets import _accepted_y


def _video_lines_per_frame(route: str = CC.ROUTE) -> dict[int, dict]:
    """frame_idx -> {'mono_time', 'y_left_cal', 'y_right_cal'} from accepted proposals
    (median across eval distances; >= 2 accepted distances per side, as in m1_offsets)."""
    m1 = CC.m1_dir(route)
    manifest = {int(r["frame_idx"]): r for r in csv.DictReader(open(m1 / "frames_manifest.csv"))}
    from model_replay_sim.context import route_context
    height = float(route_context(route).height)
    review = {}
    rev = m1 / "review_subset.csv"
    if rev.exists():
        for r in csv.DictReader(open(rev)):
            review[(int(r["frame_idx"]), float(r["x_m"]), r["side"])] = r
    by_frame: dict[int, dict] = {}
    for r in csv.DictReader(open(m1 / "proposals.csv")):
        fi = int(r["frame_idx"])
        key = (fi, float(r["x_m"]), r["side"])
        if key in review:
            r["verdict"] = review[key]["verdict"]
            r["corrected_u_px"] = review[key]["corrected_u_px"]
        man = manifest[fi]
        r["cal_roll"], r["cal_pitch"], r["cal_yaw"] = man["cal_roll"], man["cal_pitch"], man["cal_yaw"]
        r["height"] = height
        y = _accepted_y(r)
        if y is None:
            continue
        e = by_frame.setdefault(fi, {"mono_time": float(man["mono_time"]), "left": [], "right": []})
        e[r["side"]].append(float(y))
    out = {}
    for fi, e in by_frame.items():
        if len(e["left"]) >= 2 and len(e["right"]) >= 2:
            man = manifest[fi]
            out[fi] = {"mono_time": e["mono_time"],
                       "seg_num": int(man["seg_num"]), "seg_id": int(man["seg_id"]),
                       "y_left_cal": G.y_cal_from_y_road(float(np.median(e["left"]))),
                       "y_right_cal": G.y_cal_from_y_road(float(np.median(e["right"])))}
    return out


def _sp002_lines(route: str = CC.ROUTE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated post-warmup (mono, y_left, y_right) at x=0 from the route's M2
    SP002 replays. Raises FileNotFoundError when windows or replay npzs are missing."""
    from stock_lateral_toolkit.centering.consensus import _npz_path
    from stock_lateral_toolkit.centering.windows import load_windows
    monos, yls, yrs = [], [], []
    for w in load_windows(route):
        p = _npz_path("SP002", w["window_id"], route)
        if not p.exists():
            raise FileNotFoundError(str(p))
        z = np.load(p)
        s = int(z["split_index"])
        monos.append(np.asarray(z["mono_time"])[s:])
        yls.append(z["lane_lines"][s:, 1, 0, 0])
        yrs.append(z["lane_lines"][s:, 2, 0, 0])
    return np.concatenate(monos), np.concatenate(yls), np.concatenate(yrs)


def main():
    # Match by FRAME IDENTITY via the frame timeline's eof timestamps — the manifest
    # mono_time is the modelV2 publish time (eof + latency) and can never mono-match.
    # d_left/d_right pool across the primary route and any completed extra route.
    from model_replay_sim.alignment import build_frame_timeline
    d_left, d_right = [], []
    n_per_route: dict[str, int] = {}
    for route in (CC.ROUTE,) + CC.ROUTES_EXTRA:
        try:
            video = _video_lines_per_frame(route)
            mono, yl, yr = _sp002_lines(route)
        except FileNotFoundError as e:
            print(f"NOTE: {route}: skipped (missing input: {e})")
            continue
        eof_by_seg = {(row.segment_num, row.segment_id): row.timestamp_eof_s
                      for row in build_frame_timeline(route)}
        n_route = 0
        for fi, v in sorted(video.items()):
            eof = eof_by_seg.get((v["seg_num"], v["seg_id"]))
            if eof is None:
                continue
            k = int(np.argmin(np.abs(mono - eof)))
            if abs(mono[k] - eof) > 1e-3:
                continue
            d_left.append(float(yl[k]) - v["y_left_cal"])
            d_right.append(float(yr[k]) - v["y_right_cal"])
            n_route += 1
        n_per_route[route] = n_route
    d_left = np.array(d_left); d_right = np.array(d_right)
    if len(d_left) < 7:  # amended 2026-07-04b from 10 (M0 addendum; log-based dmid is the primary cross-check)
        raise SystemExit(f"R1 blocked: only {len(d_left)} overlap frames (< 7) — "
                         "increase in-window M1 sampling or add the second route.")
    dmid = (d_left + d_right) / 2.0
    dwidth = d_right - d_left
    rng = np.random.default_rng(3)
    boots = [np.median(rng.choice(dmid, len(dmid))) for _ in range(2000)]
    res = {
        "n_overlap": int(len(dmid)),
        "n_overlap_per_route": n_per_route,
        "delta_left_median_m": float(np.median(d_left)),
        "delta_right_median_m": float(np.median(d_right)),
        "dmid_median_m": float(np.median(dmid)),
        "dmid_ci95_m": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "dwidth_median_m": float(np.median(dwidth)),
        "translation_component": bool(abs(np.median(dmid)) >= CC.DMID_MEANINGFUL_M
                                      and abs(np.median(dwidth)) <= CC.DWIDTH_COHERENT_M),
    }
    # cross-check vs the log-based estimate of the same quantity (m1_results.json)
    m1 = json.loads((CC.RESULTS_DIR / "m1" / "m1_results.json").read_text())
    res["dmid_logbased_m"] = m1["overlap_logged"]["median_delta_logged_minus_video_m"]
    res["dmid_replay_vs_logbased_delta_m"] = float(res["dmid_median_m"] - res["dmid_logbased_m"])
    out = CC.RESULTS_DIR / "r1_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
