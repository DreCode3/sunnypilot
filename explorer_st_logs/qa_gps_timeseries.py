#!/usr/bin/env python3
"""Align GPS-curvature, meas, cmd, des as time series over the curve.

Goals:
 - Correct any GPS/CX1 time misalignment by xcorr of GPS-kappa vs meas.
 - Compare GPS-kappa(t) vs meas(t) vs cmd(t) vs des(t) at matched times.
 - Subtract the straight-segment meas bias and re-check meas vs GPS.
"""
import sys, os, math
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/opendbc_repo')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs')
import numpy as np
from analyze_curves_v2 import load_cx1, IDX
from phase1_validation import route_prefix_for
from find_labeled_event import pull_gps_and_signals, haversine_m

TGT_LAT, TGT_LON = 33.89518, -84.58368
ROUTES = ['route_b0', 'route_9b', 'route_ad']

def latlon_to_local_m(lat, lon, lat0, lon0):
    R = 6371000.0
    x = np.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y

def path_curvature(x, y, t):
    xp = np.gradient(x, t); yp = np.gradient(y, t)
    xpp = np.gradient(xp, t); ypp = np.gradient(yp, t)
    denom = (xp**2 + yp**2)**1.5
    denom = np.where(denom < 1e-6, np.nan, denom)
    return (xp * ypp - yp * xpp) / denom

def smooth(a, w=5):
    if len(a) < w: return a
    return np.convolve(a, np.ones(w)/w, mode='same')

for route in ROUTES:
    prefix = route_prefix_for(route)
    arr, t_cx1 = load_cx1(prefix)
    gps = pull_gps_and_signals(prefix)
    d = np.array([haversine_m(TGT_LAT, TGT_LON, la, lo) for la, lo in zip(gps['lat'], gps['lon'])])
    ci = int(np.argmin(d)); tc = gps['t'][ci]

    # --- straight bias for meas (whole route) ---
    cmd_all = arr[:, IDX['cmd']]; meas_all = arr[:, IDX['meas']]; v_all = arr[:, IDX['v']]
    sm = (np.abs(cmd_all) < 0.0005) & (v_all > 15.0)
    meas_bias = float(np.median(meas_all[sm])) if sm.sum() > 20 else 0.0

    # --- GPS kappa over wide window, +=RIGHT ---
    gm = np.abs(gps['t'] - tc) < 10.0
    gt = gps['t'][gm]; gla = gps['lat'][gm]; glo = gps['lon'][gm]
    o = np.argsort(gt); gt, gla, glo = gt[o], gla[o], glo[o]
    x, y = latlon_to_local_m(gla, glo, gla.mean(), glo.mean())
    kapR = -smooth(path_curvature(x, y, gt), 5)  # +=RIGHT

    # --- CX1 over wide window ---
    cm = np.abs(t_cx1 - tc) < 6.0
    tt = t_cx1[cm]; cmd = arr[cm, IDX['cmd']]; des = arr[cm, IDX['des']]; meas = arr[cm, IDX['meas']]
    lOff = arr[cm, IDX['lOff']]; ovr = arr[cm, IDX['ovr']]

    # interp GPS kappa onto CX1 times
    kap_on_cx1 = np.interp(tt, gt, kapR)

    # xcorr lag: shift GPS kappa to best match meas (search +-2s)
    best_lag, best_c = 0.0, -1e9
    for lag in np.arange(-2.0, 2.01, 0.05):
        k2 = np.interp(tt, gt + lag, kapR)
        # corr around the curve region only (|meas|>0.001)
        reg = np.abs(meas) > 0.0015
        if reg.sum() < 5: continue
        c = np.corrcoef(k2[reg], meas[reg])[0, 1]
        if c > best_c: best_c, best_lag = c, lag
    kap_aligned = np.interp(tt, gt + best_lag, kapR)

    # apex by aligned GPS kappa within curve
    ai = int(np.nanargmax(np.abs(kap_aligned)))
    print('=' * 80)
    print(f'{route}: meas_straight_bias(median)={meas_bias:+.6f}  xcorr_lag={best_lag:+.2f}s r={best_c:.3f}')
    print(f'  at GPS-kappa apex (t_rel={tt[ai]-tc:+.2f}s):')
    gk = kap_aligned[ai]; mm = meas[ai]; cc = cmd[ai]; dd = des[ai]
    print(f'    GPS_kappa(+=R)={gk:+.5f}  meas={mm:+.5f}  meas-bias={mm-meas_bias:+.5f}  cmd={cc:+.5f}  des={dd:+.5f}')
    print(f'    GPS - meas          = {gk-mm:+.5f}')
    print(f'    GPS - (meas-bias)   = {gk-(mm-meas_bias):+.5f}')
    print(f'    GPS - cmd           = {gk-cc:+.5f}')
    print(f'    meas - cmd          = {mm-cc:+.5f}   (meas-bias)-cmd = {(mm-meas_bias)-cc:+.5f}')
    print(f'    lOff @ apex={lOff[ai]:+.3f}')
    # peak values each signal over window
    def speak(a): return a[np.nanargmax(np.abs(a))]
    print(f'  peaks: GPS={speak(kap_aligned):+.5f} meas={speak(meas):+.5f} cmd={speak(cmd):+.5f} des={speak(des):+.5f}')
    print(f'  lOff: start={lOff[0]:+.3f} end={lOff[-1]:+.3f} min={lOff.min():+.3f} max={lOff.max():+.3f}  ovr_samples={int((ovr>0.5).sum())}')
