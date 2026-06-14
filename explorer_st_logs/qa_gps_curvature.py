#!/usr/bin/env python3
"""Adversarial QA: independently re-derive GPS path curvature vs meas/cmd/des.

For each pass through the WB right-hander GPS (33.89518, -84.58368):
  1. Window CX1 by nearest GPS sample time tc, |t - tc| < 4 s.
  2. Compute GPS-path curvature from lat/lon/time -> local meters -> kappa.
  3. Compare GPS-kappa vs meas(-yr/v) vs cmd vs des at apex.
  4. Check meas on straights (cmd~0) for constant bias vs GPS~0.
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
    """Equirectangular local meters: x=East, y=North."""
    R = 6371000.0
    x = np.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * R
    return x, y


def path_curvature(x, y, t):
    """Signed path curvature kappa = (x' y'' - y' x'') / (x'^2+y'^2)^1.5.

    Sign convention: with x=East, y=North, positive kappa = LEFT turn (CCW),
    consistent with standard math/ISO-8855 z-up. We will report this raw, then
    also report -kappa so it aligns with carcontroller's +meas=RIGHT convention.
    """
    # uniform-ish resample on arc length for stable derivatives
    xp = np.gradient(x, t)
    yp = np.gradient(y, t)
    xpp = np.gradient(xp, t)
    ypp = np.gradient(yp, t)
    denom = (xp**2 + yp**2)**1.5
    denom = np.where(denom < 1e-6, np.nan, denom)
    kappa = (xp * ypp - yp * xpp) / denom
    return kappa


def heading_curvature(x, y, t):
    """Alternative: kappa = dpsi/ds, psi=atan2(dy,dx). Positive=CCW=LEFT."""
    dx = np.gradient(x, t)
    dy = np.gradient(y, t)
    psi = np.unwrap(np.arctan2(dy, dx))
    speed = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))])
    dpsi_ds = np.gradient(psi, s)
    return dpsi_ds


def smooth(a, w=5):
    if len(a) < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode='same')


for route in ROUTES:
    prefix = route_prefix_for(route)
    print('=' * 90)
    print(f'ROUTE {route}  prefix={prefix}')
    arr, t_cx1 = load_cx1(prefix)
    gps = pull_gps_and_signals(prefix)
    if arr is None or gps is None or len(gps['t']) == 0:
        print('  MISSING DATA'); continue

    # nearest GPS sample to target
    d = np.array([haversine_m(TGT_LAT, TGT_LON, la, lo)
                  for la, lo in zip(gps['lat'], gps['lon'])])
    ci = int(np.argmin(d))
    tc = gps['t'][ci]
    print(f'  closest GPS dist={d[ci]:.1f}m  tc={tc:.1f}s  v_samples_near20m={(d<20).sum()}')

    # ---- GPS curvature over the pass window (use a wider GPS window for derivs) ----
    gmask = np.abs(gps['t'] - tc) < 8.0
    gt = gps['t'][gmask]; gla = gps['lat'][gmask]; glo = gps['lon'][gmask]
    order = np.argsort(gt)
    gt, gla, glo = gt[order], gla[order], glo[order]
    if len(gt) < 8:
        print(f'  too few GPS pts ({len(gt)})'); continue
    x, y = latlon_to_local_m(gla, glo, gla.mean(), glo.mean())
    # GPS speed for sanity
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    dt = np.diff(gt)
    vgps = np.median(ds / np.maximum(dt, 1e-3))
    kap_math = smooth(path_curvature(x, y, gt), 5)
    kap_head = smooth(heading_curvature(x, y, gt), 5)
    # raw sign: positive=LEFT. carcontroller +meas=RIGHT => compare against -kappa
    kap_math_R = -kap_math  # now positive=RIGHT
    kap_head_R = -kap_head

    # apex of the curve = where |GPS curvature| max within |t-tc|<4
    win = np.abs(gt - tc) < 4.0
    if win.sum() < 5:
        win = np.abs(gt - tc) < 6.0
    idx_apex = np.nanargmax(np.abs(np.where(win, kap_math_R, np.nan)))
    t_apex_gps = gt[idx_apex]
    print(f'  GPS median speed={vgps:.1f} m/s ({vgps*2.237:.0f} mph)  GPS apex t={t_apex_gps:.1f}s')
    print(f'  GPS kappa(math, +=RIGHT) at apex   = {kap_math_R[idx_apex]:+.5f} 1/m')
    print(f'  GPS kappa(head, +=RIGHT) at apex   = {kap_head_R[idx_apex]:+.5f} 1/m')
    # peak over window
    kw = kap_math_R[win]
    pk = kw[np.nanargmax(np.abs(kw))]
    print(f'  GPS kappa peak over window (+=R)   = {pk:+.5f} 1/m')

    # ---- CX1 in the curve window around the GPS apex ----
    cmask = np.abs(t_cx1 - t_apex_gps) < 4.0
    if cmask.sum() < 5:
        print(f'  too few CX1 ({cmask.sum()})'); continue
    sub = arr[cmask]; tt = t_cx1[cmask]
    cmd = sub[:, IDX['cmd']]; des = sub[:, IDX['des']]; meas = sub[:, IDX['meas']]
    lOff = sub[:, IDX['lOff']]; v = sub[:, IDX['v']]; yr = sub[:, IDX['yr']]
    ovr = sub[:, IDX['ovr']]
    # recompute meas from raw yr/v to double-check the logged 'meas'
    meas_recompute = -yr / np.maximum(v, 0.1)

    # apex = peak |cmd|
    ai = int(np.argmax(np.abs(cmd)))
    print(f'  --- CX1 at peak|cmd| (t={tt[ai]-t_apex_gps:+.2f}s rel GPS apex) ---')
    print(f'    cmd  = {cmd[ai]:+.5f}   des = {des[ai]:+.5f}   meas = {meas[ai]:+.5f}')
    print(f'    meas_recompute(-yr/v) = {meas_recompute[ai]:+.5f}  (logged yr={yr[ai]:+.4f} v={v[ai]:.1f})')
    print(f'    lOff = {lOff[ai]:+.3f}   ovr={ovr[ai]:.0f}')
    # peaks over window (signed, take value at max |.|)
    def speak(a):
        return a[np.argmax(np.abs(a))]
    print(f'  --- peaks over CX1 window (signed @ max|.|) ---')
    print(f'    cmd_peak={speak(cmd):+.5f}  des_peak={speak(des):+.5f}  meas_peak={speak(meas):+.5f}')
    print(f'    meas-cmd @ apex = {meas[ai]-cmd[ai]:+.5f}   meas-des @ apex = {meas[ai]-des[ai]:+.5f}')
    print(f'    lOff start={lOff[0]:+.3f}  end={lOff[-1]:+.3f}  min={lOff.min():+.3f} max={lOff.max():+.3f}')
    print(f'    overrides in window: {int((ovr>0.5).sum())} samples')

    # ---- KEY TEST: GPS curvature vs meas vs cmd/des at apex ----
    print(f'  *** GPS-vs-sensor at apex ***')
    print(f'    GPS_kappa(+=R)={kap_math_R[idx_apex]:+.5f}   meas={meas[ai]:+.5f}   cmd={cmd[ai]:+.5f}  des={des[ai]:+.5f}')
    print(f'    GPS - meas = {kap_math_R[idx_apex]-meas[ai]:+.5f}   GPS - cmd = {kap_math_R[idx_apex]-cmd[ai]:+.5f}')

print('=' * 90)
print('STRAIGHTS BIAS CHECK (cmd~0 segments): meas vs GPS curvature')
for route in ROUTES:
    prefix = route_prefix_for(route)
    arr, t_cx1 = load_cx1(prefix)
    gps = pull_gps_and_signals(prefix)
    if arr is None or gps is None or len(gps['t']) == 0:
        continue
    cmd = arr[:, IDX['cmd']]; meas = arr[:, IDX['meas']]; v = arr[:, IDX['v']]
    yr = arr[:, IDX['yr']]
    # straight = |cmd|<0.0005 and v>15 m/s
    sm = (np.abs(cmd) < 0.0005) & (v > 15.0)
    n = int(sm.sum())
    if n < 20:
        print(f'  {route}: only {n} straight samples'); continue
    print(f'  {route}: n_straight={n}  meas_mean={meas[sm].mean():+.6f}  meas_median={np.median(meas[sm]):+.6f}  cmd_mean={cmd[sm].mean():+.6f}')
    # GPS curvature on straights: build full-route GPS curvature, then at straight times
    order = np.argsort(gps['t'])
    gt = gps['t'][order]; gla = gps['lat'][order]; glo = gps['lon'][order]
    x, y = latlon_to_local_m(gla, glo, gla.mean(), glo.mean())
    kap = -smooth(path_curvature(x, y, gt), 7)  # +=RIGHT
    # interp GPS kappa onto CX1 straight times
    st = t_cx1[sm]
    kap_at = np.interp(st, gt, kap)
    print(f'         GPS kappa(+=R) on straights: mean={np.nanmean(kap_at):+.6f}  median={np.nanmedian(kap_at):+.6f}')
    print(f'         => meas - GPS on straights  = {meas[sm].mean()-np.nanmean(kap_at):+.6f} 1/m')
