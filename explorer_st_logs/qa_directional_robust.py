#!/usr/bin/env python3
"""Robust directional test. For each pass through the bend:
  - GPS-kappa apex (savgol) and des at that time -> GPS over-turn vs model (kappa-des)
  - delay-aligned (0.25s) meas-cmd over the curve plateau (not raw apex index)
  - whole-route straight bias subtracted
Goal: is the over-yaw (kappa>des) directional (right only) or symmetric? And is meas-cmd,
after delay-alignment + bias subtraction, a clean directional signal or noise?
"""
import sys, os, json, math
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/opendbc_repo')
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs')
import numpy as np
from scipy.signal import savgol_filter
from analyze_curves_v2 import load_cx1, IDX
from phase1_validation import route_prefix_for
from find_labeled_event import pull_gps_and_signals, haversine_m

TARGET_LAT, TARGET_LON = 33.89518, -84.58368
PASSES = {  # route -> turn
  'route_7f':'RIGHT','route_9b':'RIGHT','route_a1':'RIGHT','route_b0':'RIGHT','route_ad':'RIGHT',
  'route_80':'LEFT',
}

def latlon_to_en(lat, lon, lat0, lon0):
    R=6371000.0
    return (np.radians(np.asarray(lon)-lon0)*R*math.cos(math.radians(lat0)),
            np.radians(np.asarray(lat)-lat0)*R)

def straight_bias(arr):
    cmd=arr[:,IDX['cmd']];meas=arr[:,IDX['meas']];v=arr[:,IDX['v']]
    sel=(np.abs(cmd)<0.0005)&(v>15)
    return float(np.median(meas[sel]-cmd[sel])) if sel.sum()>30 else float('nan')

for r,turn in PASSES.items():
    prefix=route_prefix_for(r)
    gps=pull_gps_and_signals(prefix);arr,t=load_cx1(prefix)
    bias=straight_bias(arr)
    glat=np.asarray(gps['lat']);glon=np.asarray(gps['lon']);gt=np.asarray(gps['t'])
    d=np.array([haversine_m(la,lo,TARGET_LAT,TARGET_LON) for la,lo in zip(glat,glon)])
    tc=float(gt[int(np.argmin(d))])
    gm=np.abs(gt-tc)<5.0
    lat0,lon0=float(np.mean(glat[gm])),float(np.mean(glon[gm]))
    x,y=latlon_to_en(glat[gm],glon[gm],lat0,lon0)
    tu=np.arange(gt[gm][0],gt[gm][-1],0.05)
    xu=np.interp(tu,gt[gm],x);yu=np.interp(tu,gt[gm],y)
    win=13
    xp=savgol_filter(xu,win,3,1,delta=0.05);yp=savgol_filter(yu,win,3,1,delta=0.05)
    xpp=savgol_filter(xu,win,3,2,delta=0.05);ypp=savgol_filter(yu,win,3,2,delta=0.05)
    spd=np.hypot(xp,yp)
    km=-(xp*ypp-yp*xpp)/np.power(np.maximum(spd**2,1e-6),1.5)  # +=RIGHT
    interior=(tu>tu[0]+0.7)&(tu<tu[-1]-0.7)
    idxs=np.where(interior)[0]
    if turn=='RIGHT':
        ia=idxs[np.argmax(km[interior])]
    else:
        ia=idxs[np.argmin(km[interior])]
    tap=tu[ia]; kap=float(km[ia])
    m=np.abs(t-tc)<5.0
    tw=t[m];cmd=arr[m,IDX['cmd']];des=arr[m,IDX['des']];meas=arr[m,IDX['meas']]
    des_ap=float(np.interp(tap,tw,des));cmd_ap=float(np.interp(tap,tw,cmd))
    # delay-aligned plateau meas-cmd: region |cmd|>0.6 peak, meas(t)-cmd(t-0.25)
    pk=np.max(np.abs(cmd))
    sel=np.abs(cmd)>=0.6*pk
    cmd_sh=np.interp(tw-0.25,tw,cmd)
    mc_da=float(np.mean((meas-cmd_sh)[sel])) if sel.sum() else float('nan')
    print(f"{r:10s} {turn:5s}: GPS-kappa_apex={kap:+.5f} des={des_ap:+.5f} "
          f"kappa-des={kap-des_ap:+.5f} | meas-cmd(0.25s-aligned plateau)={mc_da:+.5f} "
          f"bias-corr={mc_da-bias:+.5f} (straight_bias={bias:+.5f})")
