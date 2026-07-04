"""M1 frame sampler: stratified (speed bin x heading quadrant), de-correlated (>= 5 s
apart), model-independent-straight sample from the route's retrospective cache,
preferring frames INSIDE the shared M2 scene windows. Writes results/m1/frames_manifest.csv.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/frame_sampler.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def _smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def road_curvature_gps(lat, lon, v_ego, mono_time, w=15):
    """Model-independent road curvature (1/m) + heading (deg) from GPS, the
    analyze_compare.py recipe (heading-rate / speed; sign irrelevant, |.| used)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    good = np.isfinite(lat) & np.isfinite(lon)
    if good.any() and not good.all():
        # GPS cold-start / dropout rows: interpolate ONLY for this curvature computation.
        # Eligibility separately requires raw isfinite(lat/lon), so these rows can never
        # be sampled — the fill just keeps np.unwrap's cumulative correction NaN-free.
        idx = np.arange(len(lat))
        lat = np.interp(idx, idx[good], lat[good])
        lon = np.interp(idx, idx[good], lon[good])
    x = (lon - np.nanmean(lon)) * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
    y = (lat - np.nanmean(lat)) * 110540.0
    dx = np.gradient(_smooth(x, w)); dy = np.gradient(_smooth(y, w))
    head = np.unwrap(np.arctan2(dy, dx))
    dt = np.maximum(np.gradient(np.asarray(mono_time, float)), 1e-3)
    headrate = np.gradient(_smooth(head, w)) / dt
    v = np.maximum(np.asarray(v_ego, float), 1.0)
    return np.clip(headrate / v, -0.02, 0.02), np.degrees(head) % 360.0


def eligibility(z):
    v = np.asarray(z["v_ego"], float)
    curv, head_deg = road_curvature_gps(z["lat"], z["lon"], v, z["mono_time"])
    ok = ((np.asarray(z["lat_active"]) > 0.5)
          & (np.asarray(z["steering_pressed"]) < 0.5)
          & (np.asarray(z["blinker"]) < 0.5)
          & (np.asarray(z["lane_change_state"]) <= 0.001)
          & (v >= CC.V_MIN_MPS)
          & (np.asarray(z["lane_prob_left"]) >= CC.LANE_PROB_MIN)
          & (np.asarray(z["lane_prob_right"]) >= CC.LANE_PROB_MIN)
          & (np.abs(curv) <= CC.STRAIGHT_CURV_MAX)
          & np.isfinite(np.asarray(z["lat"], float)) & np.isfinite(np.asarray(z["lon"], float)))
    return ok, head_deg


def stratum_of(v_mps: float, head_deg: float):
    sb = -1
    for i, (lo, hi) in enumerate(CC.SPEED_BINS_MPS):
        if lo <= v_mps < hi:
            sb = i
    hb = int(head_deg // CC.HEADING_BIN_DEG) % int(360 // CC.HEADING_BIN_DEG)
    return sb, hb


def sample(z, in_window_mask) -> list[int]:
    """Deterministic (seeded) round-robin stratified sample of cache indices."""
    ok, head = eligibility(z)
    mono = np.asarray(z["mono_time"], float)
    rng = np.random.default_rng(CC.SAMPLER_SEED)
    strata: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(ok):
        sb, hb = stratum_of(float(np.asarray(z["v_ego"])[i]), float(head[i]))
        if sb >= 0:
            strata.setdefault((sb, hb), []).append(int(i))

    chosen: list[int] = []
    chosen_t: list[float] = []

    def try_add(i: int) -> bool:
        t = mono[i]
        if all(abs(t - tt) >= CC.MIN_FRAME_SEPARATION_S for tt in chosen_t):
            chosen.append(i); chosen_t.append(t)
            return True
        return False

    keys = sorted(strata)
    pools = {k: rng.permutation(strata[k]).tolist() for k in keys}
    for prefer_window in (True, False):
        progress = True
        while progress and len(chosen) < CC.N_FRAMES_TARGET:
            progress = False
            for k in keys:
                for i in pools[k]:
                    if i in chosen or bool(in_window_mask[i]) != prefer_window:
                        continue
                    if try_add(i):
                        progress = True
                        break
                if len(chosen) >= CC.N_FRAMES_TARGET:
                    break
    return sorted(chosen)


def _in_window_mask(mono: np.ndarray) -> np.ndarray:
    from stock_lateral_toolkit.centering.windows import load_windows
    mask = np.zeros(len(mono), dtype=bool)
    try:
        wins = load_windows()
    except FileNotFoundError:
        print("WARNING: no windows.json (run windows.py first); sampling without preference")
        return mask
    for w in wins:
        wt = np.asarray(w["mono_times"], float)
        lo, hi = wt.min() - 0.03, wt.max() + 0.03
        mask |= (mono >= lo) & (mono <= hi)
    return mask


def main():
    z = dict(np.load(CC.CACHE_NPZ))
    mono = np.asarray(z["mono_time"], float)
    in_win = _in_window_mask(mono)
    idxs = sample(z, in_win)
    if len(idxs) < CC.N_FRAMES_MIN:
        print(f"WARNING: only {len(idxs)} frames (< pre-registered minimum {CC.N_FRAMES_MIN}); "
              f"the second-route video pull (Task 10 step 1b) is now REQUIRED, not optional")

    # map cache mono -> (segment_num, segment_id); frames without a decoded frame
    # within 0.03 s are dropped (logged but rare on this fully-local route)
    from model_replay_sim.alignment import map_window_to_frames
    _, head = road_curvature_gps(z["lat"], z["lon"], z["v_ego"], mono)   # once, O(n)
    rows, dropped = [], 0
    for i in idxs:
        try:
            al = map_window_to_frames(CC.ROUTE, [float(mono[i])])[0]
        except ValueError:
            dropped += 1
            continue
        sb, hb = stratum_of(float(z["v_ego"][i]), float(head[i]))
        rows.append(dict(frame_idx=len(rows), cache_index=i, mono_time=float(mono[i]),
                         seg_num=al.segment_num, seg_id=al.segment_id,
                         v_ego=float(z["v_ego"][i]), speed_bin=sb, heading_bin=hb,
                         in_m2_window=bool(in_win[i]),
                         lane_prob_left=float(z["lane_prob_left"][i]),
                         lane_prob_right=float(z["lane_prob_right"][i]),
                         cal_roll=float(z["cal_roll"][i]), cal_pitch=float(z["cal_pitch"][i]),
                         cal_yaw=float(z["cal_yaw"][i]),
                         logged_center_y0=float(z["lane_center_y0"][i])))
    if not rows:
        raise SystemExit("frame_sampler: 0 usable frames after mapping — check eligibility inputs")
    out = CC.RESULTS_DIR / "m1" / "frames_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    n_in = sum(r["in_m2_window"] for r in rows)
    print(f"{len(rows)} frames ({dropped} dropped, {n_in} inside M2 windows) -> {out}")
    per = {}
    for r in rows:
        per[(r["speed_bin"], r["heading_bin"])] = per.get((r["speed_bin"], r["heading_bin"]), 0) + 1
    print("strata:", dict(sorted(per.items())))


if __name__ == "__main__":
    main()
