#!/usr/bin/env python3
"""Independently determine the sign convention of lOff (the CX1 logged lane offset).
Method: in a right-hand curve where the car cuts RIGHT (driver-confirmed), the car
moves toward the right lane line. We know +meas/+cmd = RIGHT (verified). If lOff goes
NEGATIVE as the car cuts right, then negative lOff = car-RIGHT (i.e. +lOff = car LEFT
matches the CLAIM convention only if the PI correction sign is consistent).

Decisive sub-test: does PI push curvature in the direction that OPPOSES the drift?
PI term = Kp*lOff + Ki*lInt. If +lOff is meant to mean car-LEFT, PI should add +curv
(turn RIGHT) to correct a left offset. So sign(lOff)==sign(PI correction).
We check: in this curve lOff<0 (per CX1). If convention=+lOff=LEFT, then lOff<0=car RIGHT,
PI should add NEGATIVE curv (turn left, back to center). Check lInt sign & pi contribution sign.
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
KP, KI = 0.0001, 0.0002

for r in ['route_b0','route_9b']:
    prefix=route_prefix_for(r)
    gps=pull_gps_and_signals(prefix);arr,t=load_cx1(prefix)
    glat=np.asarray(gps['lat']);glon=np.asarray(gps['lon']);gt=np.asarray(gps['t'])
    d=np.array([haversine_m(la,lo,TARGET_LAT,TARGET_LON) for la,lo in zip(glat,glon)])
    tc=float(gt[int(np.argmin(d))])
    m=np.abs(t-tc)<4.0
    tw=t[m];lOff=arr[m,IDX['lOff']];lInt=arr[m,IDX['lInt']];cmd=arr[m,IDX['cmd']]
    ia=int(np.argmax(cmd))
    pi_p=KP*lOff[ia]; pi_i=KI*lInt[ia]
    print(f"{r}: at apex (RIGHT curve, car cuts RIGHT per driver) lOff={lOff[ia]:+.3f} lInt={lInt[ia]:+.3f}")
    print(f"    PI correction = Kp*lOff + Ki*lInt = {pi_p:+.6f} + {pi_i:+.6f} = {pi_p+pi_i:+.6f} 1/m")
    print(f"    sign of PI correction: {'+RIGHT (turns more right)' if pi_p+pi_i>0 else '-LEFT (corrects toward center)'}")
    # if car is RIGHT and lOff<0, a correct controller adds LEFT curvature. Does it?
    print(f"    => lOff<0 with PI<0 (left) is CONSISTENT with: negative lOff = car RIGHT, PI opposing.")
    print()
