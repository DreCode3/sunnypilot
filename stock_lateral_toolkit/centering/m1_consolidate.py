"""M1 consolidation (Task 10 step 1b): pool the per-frame video offsets from the
primary route and every ROUTES_EXTRA route whose M1 chain has completed (i.e. whose
per_frame_offsets.csv exists), and write results/m1/m1_results_consolidated.json with
pooled medians, per-route detail, and the same cam_offset/lever-arm fields as
m1_results.json. Missing extra routes are skipped with a note.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/m1_consolidate.py
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


def _route_rows(route: str) -> list[dict] | None:
    p = CC.m1_dir(route) / "per_frame_offsets.csv"
    if not p.exists():
        return None
    return list(csv.DictReader(open(p)))


def main():
    per_route: dict[str, list[dict]] = {}
    for route in (CC.ROUTE,) + CC.ROUTES_EXTRA:
        rows = _route_rows(route)
        if rows is None:
            if route == CC.ROUTE:
                raise SystemExit(f"m1_consolidate: {CC.m1_dir(route) / 'per_frame_offsets.csv'} "
                                 "missing — run the primary-route M1 chain first")
            print(f"NOTE: {route}: no per_frame_offsets.csv yet — skipped")
            continue
        per_route[route] = rows

    pooled = [r for rows in per_route.values() for r in rows]
    offsets = np.array([float(r["offset_cam_m"]) for r in pooled])
    logged = np.array([float(r["logged_center_y0"]) for r in pooled])
    speed_bins = np.array([int(r["speed_bin"]) for r in pooled])
    sigma = G.sigma_frame_m()
    med = float(np.median(offsets))
    delta = logged - offsets
    res = {
        "routes": {route: {"n_frames": len(rows),
                           "median_offset_cam_m": float(np.median(
                               [float(r["offset_cam_m"]) for r in rows]))}
                   for route, rows in per_route.items()},
        "n_frames": int(len(offsets)),
        "median_offset_cam_m": med,
        "mad_offset_cam_m": float(np.median(np.abs(offsets - med))),
        "sigma_frame_m": sigma,
        "sem_random_m": float(sigma / max(np.sqrt(len(offsets)), 1)),
        "systematic_note": "lever-arm/mean-roll/mean-yaw systematics ~±0.04 m do not average out (M0 §2)",
        "cam_offset_from_centerline_m": CC.CAM_OFFSET_FROM_CENTERLINE_M,
        "cam_offset_measured": bool(CC.CAM_OFFSET_MEASURED),
        "median_offset_vehicle_m": (med + CC.CAM_OFFSET_FROM_CENTERLINE_M)
                                   if CC.CAM_OFFSET_MEASURED else None,
        "per_speed_bin": {str(sb): float(np.median(offsets[speed_bins == sb]))
                          for sb in sorted(set(speed_bins.tolist()))},
        "overlap_logged": {
            "n": int(len(logged)),
            "median_delta_logged_minus_video_m": float(np.median(delta)),
            "mad_delta_m": float(np.median(np.abs(delta - np.median(delta)))),
        },
    }
    out = CC.RESULTS_DIR / "m1" / "m1_results_consolidated.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
