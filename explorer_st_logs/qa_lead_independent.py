#!/usr/bin/env python3
"""QA-LEAD independent re-derivation. Trust nothing; re-derive from raw CX1 + GPS.

Outputs JSON to stdout summarizing:
  - per-pass apex meas/cmd/des, meas-cmd, delay-aligned meas-cmd, integral ratio
  - straight-segment meas-cmd bias (cmd~0)
  - gate predicate fraction (|apply_curvature|>=0.005) in curve window
  - lInt state in curve
  - rise vs fall edge meas-cmd
  - delay scan residual D=0..400ms
"""
import sys, os, json, glob, math
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/opendbc_repo')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs')
import numpy as np
from analyze_curves_v2 import load_cx1, IDX
from phase1_validation import route_prefix_for
from find_labeled_event import pull_gps_and_signals, haversine_m

TARGET_LAT, TARGET_LON = 33.89518, -84.58368
ROUTES = ['route_b0', 'route_9b', 'route_ad']

def latlon_to_en(lat, lon, lat0, lon0):
    R = 6371000.0
    x = np.radians(np.asarray(lon) - lon0) * R * math.cos(math.radians(lat0))  # East
    y = np.radians(np.asarray(lat) - lat0) * R                                  # North
    return x, y

def gps_curvature(lat, lon, t):
    """Signed path curvature (1/m) two ways. Returns dict with kappa_math, kappa_head, heading_deg."""
    lat0, lon0 = float(np.mean(lat)), float(np.mean(lon))
    x, y = latlon_to_en(lat, lon, lat0, lon0)
    t = np.asarray(t, float)
    # uniform-ish; use gradient w.r.t. time
    xp = np.gradient(x, t); yp = np.gradient(y, t)
    xpp = np.gradient(xp, t); ypp = np.gradient(yp, t)
    speed = np.hypot(xp, yp)
    denom = np.power(np.maximum(speed**2, 1e-6), 1.5)
    kappa_math = (xp*ypp - yp*xpp) / denom   # ISO: +=CCW=LEFT in standard math frame
    # heading method
    heading = np.unwrap(np.arctan2(yp, xp))  # radians, math convention (CCW from East)
    ds = np.gradient(np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]))
    ds = np.maximum(np.abs(ds), 1e-3)
    kappa_head = np.gradient(heading, t) / np.maximum(speed, 0.1)
    heading_deg = (np.degrees(np.arctan2(np.diff(y, prepend=y[0]), np.diff(x, prepend=x[0]))))
    return kappa_math, kappa_head, speed

results = {}
for route in ROUTES:
    prefix = route_prefix_for(route)
    arr, t = load_cx1(prefix)
    gps = pull_gps_and_signals(prefix)
    if arr is None or gps is None:
        results[route] = {'error': 'no data'}
        continue
    # find GPS sample nearest target
    glat = np.asarray(gps['lat']); glon = np.asarray(gps['lon']); gt = np.asarray(gps['t'])
    d = np.array([haversine_m(la, lo, TARGET_LAT, TARGET_LON) for la, lo in zip(glat, glon)])
    i_near = int(np.argmin(d))
    tc = float(gt[i_near])
    min_d = float(d[i_near])

    # CX1 curve window
    m = np.abs(t - tc) < 4.0
    if m.sum() < 5:
        results[route] = {'error': f'too few CX1 in window (n={int(m.sum())})', 'min_gps_d_m': min_d}
        continue
    tw = t[m]
    cmd = arr[m, IDX['cmd']]
    meas = arr[m, IDX['meas']]
    des = arr[m, IDX['des']]
    lOff = arr[m, IDX['lOff']]
    lInt = arr[m, IDX['lInt']]
    ovr = arr[m, IDX['ovr']]
    tq = arr[m, IDX['tq']]
    v = arr[m, IDX['v']]
    yr = arr[m, IDX['yr']]

    # verify meas == -yr/v
    meas_recon = -yr / np.maximum(v, 0.1)
    meas_recon_err = float(np.nanmax(np.abs(meas_recon - meas)))

    # apex = index of max cmd (right turn => cmd>0)
    i_apex = int(np.argmax(cmd))
    apex = {
        'tc_rel': float(tw[i_apex] - tc),
        'cmd': float(cmd[i_apex]), 'meas': float(meas[i_apex]), 'des': float(des[i_apex]),
        'lOff': float(lOff[i_apex]), 'lInt': float(lInt[i_apex]),
        'meas_minus_cmd': float(meas[i_apex] - cmd[i_apex]),
        'meas_over_cmd_ratio': float(meas[i_apex] / cmd[i_apex]) if cmd[i_apex] != 0 else None,
    }

    # delay-aligned meas(t) - cmd(t-D): interpolate cmd onto shifted grid
    def resid_at_delay(D):
        # cmd at (t - D), evaluated at the same sample times
        cmd_shift = np.interp(tw - D, tw, cmd)
        # restrict to >50% peak cmd region (the actual curve)
        sel = cmd >= 0.5 * np.max(cmd)
        return float(np.mean((meas - cmd_shift)[sel])) if sel.sum() else float('nan')
    delay_scan = {f'{int(D*1000)}ms': resid_at_delay(D) for D in (0.0, 0.10, 0.25, 0.40)}

    # rise vs fall edge meas-cmd (in >40% peak region)
    sel = cmd >= 0.4 * np.max(cmd)
    if sel.sum() >= 4:
        idxs = np.where(sel)[0]
        peak_local = idxs[np.argmax(cmd[idxs])]
        rise = idxs[idxs <= peak_local]
        fall = idxs[idxs > peak_local]
        rise_excess = float(np.mean(meas[rise] - cmd[rise])) if len(rise) else float('nan')
        fall_excess = float(np.mean(meas[fall] - cmd[fall])) if len(fall) else float('nan')
    else:
        rise_excess = fall_excess = float('nan')

    # integral ratio (window integral of meas / window integral of cmd over >0 region)
    pos = cmd > 0
    int_ratio = float(np.trapezoid(meas[pos], tw[pos]) / np.trapezoid(cmd[pos], tw[pos])) if pos.sum() > 2 and np.trapezoid(cmd[pos], tw[pos]) != 0 else None

    # gate predicate fraction: |apply_curvature|>=0.005. Best proxy in CX1 is cmd
    # (apply_curvature pre-PI; cmd is post). Report fraction of curve-window frames.
    gate_frac_cmd = float(np.mean(np.abs(cmd) >= 0.005))
    max_abs_cmd = float(np.max(np.abs(cmd)))

    # override frames
    n_ovr = int(np.sum(ovr > 0.5))
    ovr_trel = [float(tw[i]-tc) for i in np.where(ovr>0.5)[0]]
    tq_max_abs = float(np.max(np.abs(tq)))

    # lOff drift extremes
    lOff_start = float(np.median(lOff[tw - tc < -2.0])) if (tw-tc<-2.0).any() else float(lOff[0])
    lOff_min = float(np.min(lOff))
    lOff_max = float(np.max(lOff))

    # GPS curvature in window
    gm = np.abs(gt - tc) < 4.0
    gps_block = {}
    if gm.sum() > 8:
        km, kh, spd = gps_curvature(glat[gm], glon[gm], gt[gm])
        gtw = gt[gm]
        # negate so +=RIGHT (math kappa +=LEFT)
        km_r = -km; kh_r = -kh
        i_gapex = int(np.argmax(km_r))
        gps_block = {
            'kappa_math_apex_RIGHTpos': float(km_r[i_gapex]),
            'kappa_head_apex_RIGHTpos': float(kh_r[i_gapex]),
            'gps_apex_trel': float(gtw[i_gapex] - tc),
            'n_gps': int(gm.sum()),
        }

    results[route] = {
        'min_gps_d_m': round(min_d, 2),
        'tc': tc,
        'n_cx1_window': int(m.sum()),
        'cx1_median_dt': float(np.median(np.diff(tw))),
        'meas_recon_err': meas_recon_err,
        'apex': apex,
        'delay_scan_resid': delay_scan,
        'rise_excess': rise_excess,
        'fall_excess': fall_excess,
        'int_ratio_meas_over_cmd': int_ratio,
        'gate_frac_cmd_ge_0.005': gate_frac_cmd,
        'max_abs_cmd': max_abs_cmd,
        'n_override': n_ovr,
        'override_trel': ovr_trel,
        'tq_max_abs': tq_max_abs,
        'lOff_start': lOff_start, 'lOff_min': lOff_min, 'lOff_max': lOff_max,
        'lInt_apex': apex['lInt'],
        'gps': gps_block,
    }

print(json.dumps(results, indent=2))
