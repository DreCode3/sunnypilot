#!/usr/bin/env python3
"""Discriminating test: is the over-turn in cmd/des (command-side) or in the plant?

Compare des (model desired) vs cmd (after PI/EMA/RL) vs GPS-kappa vs meas at apex.
If GPS ~ des/cmd and meas < GPS -> meas biased low; over-turn is real & commanded.
Also: where does the over-turn enter? des already > what a 0.7m-right drift needs?
Check integral state (lInt) + lOff sign during curve.
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
for route in ['route_b0', 'route_9b', 'route_ad']:
    prefix = route_prefix_for(route)
    arr, t_cx1 = load_cx1(prefix)
    gps = pull_gps_and_signals(prefix)
    d = np.array([haversine_m(TGT_LAT, TGT_LON, la, lo) for la, lo in zip(gps['lat'], gps['lon'])])
    ci = int(np.argmin(d)); tc = gps['t'][ci]
    cm = np.abs(t_cx1 - tc) < 4.0
    tt = t_cx1[cm]
    cmd = arr[cm, IDX['cmd']]; des = arr[cm, IDX['des']]; meas = arr[cm, IDX['meas']]
    lInt = arr[cm, IDX['lInt']]; lOff = arr[cm, IDX['lOff']]; lc = arr[cm, IDX['lc']]
    v = arr[cm, IDX['v']]
    ai = int(np.argmax(np.abs(des)))
    # straight bias
    cmd_a = arr[:, IDX['cmd']]; meas_a = arr[:, IDX['meas']]; v_a = arr[:, IDX['v']]
    sm = (np.abs(cmd_a) < 0.0005) & (v_a > 15.0)
    bias = float(np.median(meas_a[sm]))
    print(f'{route}: at peak|des| t_rel={tt[ai]-tc:+.2f}s v={v[ai]:.1f}')
    print(f'   des={des[ai]:+.5f} cmd={cmd[ai]:+.5f} meas={meas[ai]:+.5f} meas-bias={meas[ai]-bias:+.5f}')
    print(f'   des-cmd={des[ai]-cmd[ai]:+.5f}  cmd-meas={cmd[ai]-meas[ai]:+.5f}  cmd-(meas-bias)={cmd[ai]-(meas[ai]-bias):+.5f}')
    print(f'   lInt(integral)={lInt[ai]:+.4f} lOff={lOff[ai]:+.3f} lc(active)={lc[ai]:.0f}')
    # was integral gated off in the curve? gate = |cmd|>=0.005 -> integral stops growing/decays
    incurve = np.abs(cmd) >= 0.005
    print(f'   integral-gate: frames |cmd|>=0.005 = {int(incurve.sum())}/{len(cmd)}; lInt range in window [{lInt.min():+.4f},{lInt.max():+.4f}]')
    # PI I-term authority at this speed: ki=0.0002, cap=0.3 -> max 6e-5; vs des-need
    print(f'   PI max I-term = 0.0002*0.3 = {0.0002*0.3:.6f} 1/m (negligible vs des {des[ai]:.5f})')
    print()
