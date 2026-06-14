#!/usr/bin/env python3
"""Cross-check GPS-kappa magnitude. Lens A said ~0.0059; my first pass said ~0.009-0.011.
Investigate: GPS sample rate, noise, smoothing sensitivity, and consistency of two methods.
Also: compute the straight-segment yaw-rate bias directly (cmd~0, GPS~0) and the EB(left) pass.
"""
import sys, os, json, math
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/opendbc_repo')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs')
import numpy as np
from analyze_curves_v2 import load_cx1, IDX
from phase1_validation import route_prefix_for
from find_labeled_event import pull_gps_and_signals, haversine_m

TARGET_LAT, TARGET_LON = 33.89518, -84.58368

def latlon_to_en(lat, lon, lat0, lon0):
    R = 6371000.0
    x = np.radians(np.asarray(lon) - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R
    return x, y

def smooth(a, w):
    if w <= 1: return a
    k = np.ones(w)/w
    return np.convolve(a, k, mode='same')

def kappa_heading(x, y, t, smooth_w=1):
    x = smooth(x, smooth_w); y = smooth(y, smooth_w)
    xp = np.gradient(x, t); yp = np.gradient(y, t)
    speed = np.hypot(xp, yp)
    heading = np.unwrap(np.arctan2(yp, xp))
    return np.gradient(heading, t) / np.maximum(speed, 0.1), speed

def report(route, want_left=False):
    prefix = route_prefix_for(route)
    gps = pull_gps_and_signals(prefix)
    arr, t = load_cx1(prefix)
    glat = np.asarray(gps['lat']); glon = np.asarray(gps['lon']); gt = np.asarray(gps['t'])
    d = np.array([haversine_m(la, lo, TARGET_LAT, TARGET_LON) for la, lo in zip(glat, glon)])
    i_near = int(np.argmin(d)); tc = float(gt[i_near])
    gm = np.abs(gt - tc) < 4.0
    lat0, lon0 = float(np.mean(glat[gm])), float(np.mean(glon[gm]))
    x, y = latlon_to_en(glat[gm], glon[gm], lat0, lon0)
    gtw = gt[gm]
    gps_dt = float(np.median(np.diff(gtw)))
    # smoothing sensitivity
    out = {}
    for w in (1, 3, 5, 9):
        k, spd = kappa_heading(x, y, gtw, w)
        k_r = -k  # +=RIGHT
        # apex = peak in turn direction; for right use max, for left use min then report mag
        if want_left:
            i_ap = int(np.argmin(k_r))
        else:
            i_ap = int(np.argmax(k_r))
        out[f'smooth{w}'] = {'kappa_apex_RIGHTpos': round(float(k_r[i_ap]),6),
                             'speed_apex': round(float(spd[i_ap]),2)}
    # also robust: 90th pct of |k| with smooth5
    k5,_ = kappa_heading(x, y, gtw, 5)
    out['median_abs_k_smooth5'] = round(float(np.median(np.abs(k5))),6)
    out['p90_abs_k_smooth5'] = round(float(np.percentile(np.abs(k5),90)),6)
    out['gps_dt_median'] = round(gps_dt,4)
    out['n_gps'] = int(gm.sum())
    return out, tc

print("=== GPS-kappa magnitude cross-check (smoothing sensitivity) ===")
for r in ['route_b0','route_9b','route_ad']:
    o, tc = report(r)
    print(r, json.dumps(o))

# ---- Straight-segment yaw bias: whole-route, cmd~0, v>15 ----
print("\n=== Straight-segment meas-cmd bias (cmd~0, v>15) per route ===")
for r in ['route_b0','route_9b','route_ad','route_aa','route_ab']:
    prefix = route_prefix_for(r)
    if prefix is None:
        print(r, 'no prefix'); continue
    arr, t = load_cx1(prefix)
    if arr is None:
        print(r, 'no cx1'); continue
    cmd = arr[:, IDX['cmd']]; meas = arr[:, IDX['meas']]; v = arr[:, IDX['v']]
    sel = (np.abs(cmd) < 0.0005) & (v > 15)
    if sel.sum() < 30:
        print(r, f'n={int(sel.sum())} too few'); continue
    mc = meas[sel] - cmd[sel]
    print(r, f"n={int(sel.sum())} meas-cmd median={np.median(mc):+.6f} mean={np.mean(mc):+.6f} "
             f"meas_median={np.median(meas[sel]):+.6f} cmd_median={np.median(cmd[sel]):+.6f}")
