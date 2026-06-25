"""Corridor-level repeat-pass reproducibility (refined B test).

Analysis-only. Reads existing route NPZ caches and writes only to
retrospective_lateral/results/. Imports nothing from the vehicle-control stack.

Why this exists: the episode-level B test in `discrimination.py` binned the
weave-band lane curvature onto 80 m GPS cells. At highway speed a ~0.2 Hz weave
has a ~110 m wavelength, so 80 m cell-averaging washes the signal out and the test
reads "not reproducible" even on genuinely repeat-traversed corridors. This module
re-runs the road (B) test at a finer along-track resolution, is direction-aware so
it can use out-and-back (opposite-direction) passes, and parallelizes the per-route
profile build across CPU cores.

Discrimination logic: for a pair of routes that share a corridor, pair the
fine-cell weave-band lane-curvature profiles by GPS cell and correlate. A weave
that is locked to fixed roadway geometry (hypothesis B) reproduces at the same
cells across passes -> high correlation. A model-artifact (A) or fixed-period loop
limit-cycle (C) weave does not lock to location -> low correlation. For
opposite-direction passes the travel-frame curvature is sign-flipped, so one
profile is negated before correlating.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.discrimination import gps_course_deg, offset_to_curvature
from retrospective_lateral.code.signal_utils import filter_continuous, gps_cells

# Finer along-track resolution than the 80 m GPS_CELL_M used by the episode test.
CORRIDOR_CELL_M = 15.0
# Require a longer shared stretch than the coarse 4-cell episode floor.
MIN_CORRIDOR_CELLS = 8
SAME_DIR_MAX_DEG = 45.0      # heading diff below this => same-direction pass
OPP_DIR_MIN_DEG = 135.0      # heading diff above this => opposite-direction pass
WEAVE_SPEED_LO_MPH, WEAVE_SPEED_HI_MPH = 10.0, 70.0


def profile_from_arrays(arrays, cell_m: float = CORRIDOR_CELL_M,
                        lo_mph: float = WEAVE_SPEED_LO_MPH, hi_mph: float = WEAVE_SPEED_HI_MPH,
                        band=None) -> dict:
    """Mean weave-band lane curvature + median heading + count, per fine GPS cell.

    Returns {gps_cell: (lane_curv_mean, course_deg_median, n_samples)} over engaged,
    weave-speed, finite samples. Empty dict if the cache lacks the needed channels.
    """
    if band is None:
        band = C.DEFAULT_WEAVE_BAND_HZ
    keys = arrays.files if hasattr(arrays, "files") else arrays
    if "lane_center_y20" not in keys or "lat" not in keys:
        return {}
    lat = np.asarray(arrays["lat"], dtype=float)
    lon = np.asarray(arrays["lon"], dtype=float)
    v = np.asarray(arrays["v_ego"], dtype=float)
    lat_active = np.asarray(arrays["lat_active"], dtype=float) if "lat_active" in keys else np.ones_like(v)
    mph = v * C.MPS_TO_MPH
    lane_curv = offset_to_curvature(arrays["lane_center_y20"], C.DISCRIM_LOOKAHEAD_M)
    lane_band = filter_continuous(lane_curv, C.FS_HZ, band=band)
    cells = gps_cells(lat, lon, cell_m)
    course = gps_course_deg(lat, lon)
    mask = (np.isfinite(lat) & np.isfinite(lon) & (lat_active > 0.5)
            & (mph >= lo_mph) & (mph <= hi_mph) & np.isfinite(lane_band) & np.isfinite(cells))
    buckets: dict[float, list] = {}
    for c, val, h in zip(cells[mask], lane_band[mask], course[mask]):
        b = buckets.setdefault(float(c), [[], []])
        b[0].append(float(val))
        if np.isfinite(h):
            b[1].append(float(h))
    return {c: (float(np.mean(vals)), float(np.median(hs)) if hs else math.nan, len(vals))
            for c, (vals, hs) in buckets.items()}


def route_fine_profile(args):
    """ProcessPoolExecutor worker: load a cache and build its fine-cell profile.

    args is a picklable tuple (npz_path, cell_m, lo_mph, hi_mph). Returns
    (route_id, profile_dict). Top-level + picklable args for macOS spawn safety.
    """
    npz_path, cell_m, lo_mph, hi_mph = args
    route_id = Path(npz_path).stem
    try:
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}
    except Exception:
        return route_id, {}
    return route_id, profile_from_arrays(arrays, cell_m=cell_m, lo_mph=lo_mph, hi_mph=hi_mph)


def _circular_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def corridor_reproducibility(profile_a: dict, profile_b: dict,
                             min_cells: int = MIN_CORRIDOR_CELLS) -> dict | None:
    """Direction-aware reproducibility of the weave between two route profiles.

    Pairs fine cells by location, classifies the shared stretch as same- or
    opposite-direction by median heading, correlates the per-cell weave-band lane
    curvature (negating one profile for opposite-direction passes), and flags
    `reproducible` when the corrected correlation >= DISCRIM_REPRO_FRACTION_ROAD.
    Returns None if the shared stretch is too short or degenerate.
    """
    shared = sorted(set(profile_a) & set(profile_b))
    if len(shared) < min_cells:
        return None
    n_same = n_opp = 0
    for c in shared:
        ha, hb = profile_a[c][1], profile_b[c][1]
        if np.isfinite(ha) and np.isfinite(hb):
            dd = _circular_diff_deg(ha, hb)
            if dd < SAME_DIR_MAX_DEG:
                n_same += 1
            elif dd > OPP_DIR_MIN_DEG:
                n_opp += 1
    direction = "same" if n_same >= n_opp else "opposite"
    va = np.array([profile_a[c][0] for c in shared])
    vb = np.array([profile_b[c][0] for c in shared])
    if va.std() == 0 or vb.std() == 0:
        return None
    corr_raw = float(np.corrcoef(va, vb)[0, 1])
    corr_corrected = corr_raw if direction == "same" else -corr_raw
    return {
        "n_shared_cells": len(shared),
        "n_same": n_same,
        "n_opp": n_opp,
        "direction": direction,
        "corr_raw": corr_raw,
        "corr_corrected": corr_corrected,
        "reproducible": bool(corr_corrected >= C.DISCRIM_REPRO_FRACTION_ROAD),
    }


def build_corridor_repro(cache_root: Path = C.DEFAULT_CACHE_ROOT,
                         report_root: Path = C.DEFAULT_REPORT_ROOT,
                         cell_m: float = CORRIDOR_CELL_M,
                         min_cells: int = MIN_CORRIDOR_CELLS,
                         focus_prefixes: tuple | None = None,
                         workers: int | None = None) -> dict:
    """Build fine-cell profiles for all cached routes (in parallel) and score every
    route-pair's corridor reproducibility. Writes a per-pair CSV and a summary CSV.

    focus_prefixes: if given, only score pairs where at least one route_id starts
    with one of these prefixes (e.g. ("route_c1", "route_c2", "route_c3")).
    """
    import pandas as pd

    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(Path(cache_root) / "route_*.npz")))
    worker_args = [(p, cell_m, WEAVE_SPEED_LO_MPH, WEAVE_SPEED_HI_MPH) for p in paths]

    profiles: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for route_id, prof in ex.map(route_fine_profile, worker_args):
            if prof:
                profiles[route_id] = prof

    def in_focus(rid: str) -> bool:
        return focus_prefixes is None or any(rid.startswith(p) for p in focus_prefixes)

    routes = sorted(profiles)
    rows = []
    for i, a in enumerate(routes):
        for b in routes[i + 1:]:
            if focus_prefixes is not None and not (in_focus(a) or in_focus(b)):
                continue
            res = corridor_reproducibility(profiles[a], profiles[b], min_cells=min_cells)
            if res is not None:
                rows.append({"route_a": a, "route_b": b, **res})

    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(report_root / "corridor_repro_pairs.csv", index=False)

    summary_rows = []
    if len(pairs_df):
        for direction in ("same", "opposite"):
            sub = pairs_df[pairs_df["direction"] == direction]
            if len(sub):
                summary_rows.append({
                    "direction": direction,
                    "pairs": int(len(sub)),
                    "reproducible": int(sub["reproducible"].sum()),
                    "median_corr_corrected": float(sub["corr_corrected"].median()),
                    "p90_corr_corrected": float(sub["corr_corrected"].quantile(0.90)),
                    "max_corr_corrected": float(sub["corr_corrected"].max()),
                })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_root / "corridor_repro_summary.csv", index=False)

    return {
        "routes_profiled": len(profiles),
        "cell_m": cell_m,
        "pairs_scored": int(len(pairs_df)),
        "reproducible_pairs": int(pairs_df["reproducible"].sum()) if len(pairs_df) else 0,
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corridor-level repeat-pass reproducibility (refined B test)")
    parser.add_argument("--cache-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", type=Path, default=C.DEFAULT_REPORT_ROOT)
    parser.add_argument("--cell-m", type=float, default=CORRIDOR_CELL_M)
    parser.add_argument("--min-cells", type=int, default=MIN_CORRIDOR_CELLS)
    parser.add_argument("--focus", action="append", default=None,
                        help="route_id prefix to focus pairs on (repeatable)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    focus = tuple(args.focus) if args.focus else None
    result = build_corridor_repro(args.cache_root, args.report_root, cell_m=args.cell_m,
                                  min_cells=args.min_cells, focus_prefixes=focus, workers=args.workers)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
