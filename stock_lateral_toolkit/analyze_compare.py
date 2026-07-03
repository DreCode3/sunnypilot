#!/usr/bin/env python3
"""Matched stock-vs-custom lateral comparison on a shared corridor.

Discipline (memory feedback_lateral_ab_metrics — comparisons flip under poor control):
  * GPS-cell + speed-bin MATCHED: compare only where BOTH builds drove the same road at
    the same speed. Aggregate paired (per cell+speed-bin) medians, robustly.
  * Straight/curve classification is MODEL-INDEPENDENT (GPS heading-rate / vEgo), so the
    weave metric is measured only on genuinely straight road (where any 0.10-0.35 Hz lateral
    motion is weave, not curve-following — which shares that band).
  * Weave = band-pass (0.10-0.35 Hz, Butterworth filtfilt) of a lateral signal; report the
    model-INDEPENDENT one (yawRate) AND the position one (lane offset). Robust amplitude
    (median |bandpassed|). NEVER variance, NEVER aLat-as-primary.
  * Centering = |offset| median/p95 on good-perception active frames, matched.
  * Curve perf = achieved/commanded curvature + |offset| through GPS-classified curves.
  * EPS tracking = commanded->achieved curvature lag/gain (model-agnostic).

RUN: .venv311/bin/python stock_lateral_toolkit/analyze_compare.py <corridor>
  corridor in CORRIDORS below (e.g. hiram, marietta, hiram_null_04v05)
"""
import os, sys, glob
import numpy as np
from scipy import signal as sig

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
DT = 0.05
CELL = 0.0015          # ~150 m GPS cell
STRAIGHT_CURV = 0.0010 # 1/m; |road_curv| below => straight
V_MIN = 8.0            # m/s, ignore near-stops
SPEED_BINS = [(8, 15), (15, 22), (22, 30)]  # m/s

CORRIDORS = {
    "marietta": {"stock": ["stock_marietta"], "custom": ["custom_ce", "custom_cf"],
                 "custom_label": "custom migration (ce/cf)"},
    "hiram": {"stock": ["stock_hiram_04", "stock_hiram_05"], "custom": ["custom_c5", "custom_c7"],
              "custom_label": "custom Nevada (c5/c7)"},
    # NEGATIVE CONTROL (toolkit acceptance, validation/ACCEPTANCE.md): same build, same
    # corridor, opposite directions — expect null. Deeper null band: validation/self_split.py.
    "hiram_null_04v05": {"stock": ["stock_hiram_04"], "custom": ["stock_hiram_05"],
                         "custom_label": "stock hiram_05 (NEGATIVE CONTROL — expect null)"},
}

COLS = "t lat lon vEgo steerDeg yawRate pressed latActive enabled cmd_curv model_curv ach_curv offset innerProb".split()
IDX = {c: i for i, c in enumerate(COLS)}


def load(names):
    drives = []
    for n in names:
        z = np.load(f"{CACHE}/{n}.npz", allow_pickle=True)
        drives.append((n, z["data"], float(z["steerRatio"])))
    return drives


def bandpass(x, lo=0.10, hi=0.35, fs=20.0):
    b, a = sig.butter(2, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return sig.filtfilt(b, a, x)


def resample_drive(data):
    """Uniform 20 Hz grid; interpolate signals; compute GPS road-curv; return dict of arrays."""
    t = data[:, IDX["t"]]
    # split into contiguous chunks (gaps > 1 s = new segment/log break)
    grids = []
    starts = [0] + list(np.where(np.diff(t) > 1.0)[0] + 1) + [len(t)]
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 200:  # need >=10 s
            continue
        ta = t[a:b]
        g = np.arange(ta[0], ta[-1], DT)
        if len(g) < 200:
            continue
        d = {"t": g}
        for c in ["lat", "lon", "vEgo", "steerDeg", "yawRate", "cmd_curv", "ach_curv", "offset", "innerProb"]:
            d[c] = np.interp(g, ta, data[a:b, IDX[c]])
        for c in ["pressed", "latActive", "enabled"]:
            d[c] = np.interp(g, ta, data[a:b, IDX[c]]) > 0.5
        # GPS road curvature: heading from lat/lon, rate / speed. + = right (clockwise).
        la, lo = d["lat"], d["lon"]
        # local meters
        x = (lo - lo.mean()) * 111320.0 * np.cos(np.radians(la.mean()))
        y = (la - la.mean()) * 110540.0
        # heading via smoothed finite diff over ~0.75 s
        w = 15
        dx = np.gradient(_smooth(x, w)); dy = np.gradient(_smooth(y, w))
        head = np.unwrap(np.arctan2(dy, dx))
        headrate = np.gradient(_smooth(head, w)) / DT   # rad/s
        v = np.maximum(d["vEgo"], 1.0)
        d["road_curv"] = np.clip(headrate / v, -0.02, 0.02)  # 1/m, +=left? sign set below
        # note: atan2(dy,dx) heading increases CCW; curvature +=left in that frame. We only
        # use |road_curv| for straight/curve classification, so sign is irrelevant here.
        grids.append(d)
    return grids


def _smooth(x, w):
    if w < 2:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def collect(drives):
    """Per drive -> resampled grids with band-passed weave signals + flags."""
    allg = []
    for name, data, sr in drives:
        for d in resample_drive(data):
            act = d["latActive"] & (~d["pressed"]) & (d["vEgo"] > V_MIN)
            if act.sum() < 100:
                # still keep; band-pass whole chunk (filtfilt needs continuity)
                pass
            d["active"] = act
            d["straight"] = np.abs(d["road_curv"]) < STRAIGHT_CURV
            d["curve"] = np.abs(d["road_curv"]) >= STRAIGHT_CURV
            # band-pass the continuous chunk (model-independent yaw + position offset)
            try:
                d["bp_yaw"] = bandpass(d["yawRate"])
            except Exception:
                d["bp_yaw"] = np.full_like(d["yawRate"], np.nan)
            off = d["offset"].copy()
            off[~np.isfinite(off)] = np.interp(np.flatnonzero(~np.isfinite(off)),
                                               np.flatnonzero(np.isfinite(off)),
                                               off[np.isfinite(off)]) if np.isfinite(off).any() else 0.0
            try:
                d["bp_off"] = bandpass(off)
            except Exception:
                d["bp_off"] = np.full_like(off, np.nan)
            d["cell"] = (np.round(d["lat"] / CELL).astype(np.int64) * 100000
                         + np.round(d["lon"] / CELL).astype(np.int64))
            allg.append(d)
    return allg


def cellmetric(grids, mask_fn, value_fn):
    """dict[(cell,speedbin)] -> list of per-frame values (for later robust aggregation)."""
    out = {}
    for d in grids:
        m = mask_fn(d)
        if not m.any():
            continue
        vals = value_fn(d)
        v = d["vEgo"]
        for si, (lo, hi) in enumerate(SPEED_BINS):
            mm = m & (v >= lo) & (v < hi) & np.isfinite(vals)
            if mm.sum() == 0:
                continue
            cells = d["cell"][mm]; vv = vals[mm]
            for c in np.unique(cells):
                out.setdefault((c, si), []).append(np.median(vv[cells == c]))
    return out


def matched_compare(stock_grids, custom_grids, mask_fn, value_fn, label, unit, lower_better=True):
    sm = cellmetric(stock_grids, mask_fn, value_fn)
    cm = cellmetric(custom_grids, mask_fn, value_fn)
    print(f"\n  {label} ({unit}):")
    print(f"   speedbin |  n_cells | stock med | custom med |  delta  | stock better?")
    any_overall = []
    for si, (lo, hi) in enumerate(SPEED_BINS):
        pairs = []
        for key in set(k for k in sm if k[1] == si) & set(k for k in cm if k[1] == si):
            pairs.append((np.median(sm[key]), np.median(cm[key])))
        if len(pairs) < 4:
            print(f"   {lo:>2}-{hi:<2} m/s |  {len(pairs):>5}   | (too few matched cells)")
            continue
        pairs = np.array(pairs)
        s_med = np.median(pairs[:, 0]); c_med = np.median(pairs[:, 1])
        delta = s_med - c_med
        better = ("STOCK" if (delta < 0) == lower_better else "custom")
        # paired sign test on per-cell differences
        diffs = pairs[:, 0] - pairs[:, 1]
        frac_stock_better = np.mean((diffs < 0) if lower_better else (diffs > 0))
        print(f"   {lo:>2}-{hi:<2} m/s |  {len(pairs):>5}   | {s_med:9.4f} | {c_med:10.4f} | {delta:+.4f} | "
              f"{better} ({frac_stock_better*100:.0f}% of cells)")
        any_overall.append((si, s_med, c_med, len(pairs), frac_stock_better))
    return any_overall


def main():
    corr = sys.argv[1] if len(sys.argv) > 1 else "marietta"
    cfg = CORRIDORS[corr]
    print(f"===== CORRIDOR: {corr}  |  stock (dev) vs {cfg['custom_label']} =====")
    stock = collect(load(cfg["stock"]))
    custom = collect(load(cfg["custom"]))

    # coverage sanity
    sc = set(c for d in stock for c in np.unique(d["cell"]))
    cc = set(c for d in custom for c in np.unique(d["cell"]))
    print(f"GPS cells: stock={len(sc)} custom={len(cc)} shared={len(sc & cc)}")
    sframes = sum(int(d["active"].sum()) for d in stock)
    cframes = sum(int(d["active"].sum()) for d in custom)
    print(f"active frames: stock={sframes} custom={cframes}")

    A = lambda d: d["active"]
    Astr = lambda d: d["active"] & d["straight"]
    Acur = lambda d: d["active"] & d["curve"]
    Agood = lambda d: d["active"] & (d["innerProb"] > 0.6)

    print("\n--- WEAVE (straight road only; lower = calmer) ---")
    matched_compare(stock, custom, Astr, lambda d: np.abs(d["bp_yaw"]), "band|yawRate| 0.10-0.35Hz", "rad/s")
    matched_compare(stock, custom, Astr, lambda d: np.abs(d["bp_off"]), "band|laneOffset| 0.10-0.35Hz", "m")

    print("\n--- LANE CENTERING (good perception; lower = better centered) ---")
    matched_compare(stock, custom, Agood, lambda d: np.abs(d["offset"]), "|lane offset|", "m")

    print("\n--- CURVE PERFORMANCE (GPS-classified curves) ---")
    matched_compare(stock, custom, Acur, lambda d: np.abs(d["offset"]), "|offset| in curves", "m")
    # achieved/commanded ratio in curves (closer to 1 = tracks better); use |cmd|>0.0015
    def acc_ratio(d):
        r = np.full_like(d["cmd_curv"], np.nan)
        m = np.abs(d["cmd_curv"]) > 0.0015
        r[m] = d["ach_curv"][m] / d["cmd_curv"][m]
        return r
    matched_compare(stock, custom, Acur, acc_ratio, "achieved/commanded curv (curves)", "ratio~1", lower_better=False)

    print("\nNOTE: confounds — different model bundles (stock comma vs custom Nevada) affect "
          "offset/perception; steerRatio differs (see extract); ce/cf were the incident drive "
          "(fewer/atypical active frames). Weave-on-straights (yawRate) is the most model-agnostic metric.")


if __name__ == "__main__":
    main()
