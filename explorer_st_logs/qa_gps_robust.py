#!/usr/bin/env python3
"""Robust GPS-kappa via Savitzky-Golay derivatives (no edge-blowup) + local polyfit.
Compare to cmd/meas/des at apex. Settle the GPS-kappa magnitude question.
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

def latlon_to_en(lat, lon, lat0, lon0):
    R = 6371000.0
    x = np.radians(np.asarray(lon) - lon0) * R * math.cos(math.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * R
    return x, y

def kappa_savgol(x, y, t, win_s=0.6):
    # uniform resample for stable savgol
    tu = np.arange(t[0], t[-1], 0.05)
    xu = np.interp(tu, t, x); yu = np.interp(tu, t, y)
    dt = 0.05
    win = int(win_s/dt); win = win+1 if win%2==0 else win; win=max(win,5)
    xp = savgol_filter(xu, win, 3, deriv=1, delta=dt)
    yp = savgol_filter(yu, win, 3, deriv=1, delta=dt)
    xpp = savgol_filter(xu, win, 3, deriv=2, delta=dt)
    ypp = savgol_filter(yu, win, 3, deriv=2, delta=dt)
    speed = np.hypot(xp, yp)
    kappa = (xp*ypp - yp*xpp)/np.power(np.maximum(speed**2,1e-6),1.5)  # +=LEFT
    # heading method too
    heading = np.unwrap(np.arctan2(yp, xp))
    dpsi = savgol_filter(heading, win, 3, deriv=1, delta=dt)
    kappa_h = dpsi/np.maximum(speed,0.1)
    return tu, -kappa, -kappa_h, speed  # negate => +=RIGHT

for r in ['route_b0','route_9b','route_ad']:
    prefix = route_prefix_for(r)
    gps = pull_gps_and_signals(prefix)
    arr, t = load_cx1(prefix)
    glat=np.asarray(gps['lat']);glon=np.asarray(gps['lon']);gt=np.asarray(gps['t'])
    d=np.array([haversine_m(la,lo,TARGET_LAT,TARGET_LON) for la,lo in zip(glat,glon)])
    tc=float(gt[int(np.argmin(d))])
    gm=np.abs(gt-tc)<5.0
    lat0,lon0=float(np.mean(glat[gm])),float(np.mean(glon[gm]))
    x,y=latlon_to_en(glat[gm],glon[gm],lat0,lon0)
    tu,km,kh,spd=kappa_savgol(x,y,gt[gm])
    # restrict to interior away from edges (>0.7s margin)
    interior=(tu>tu[0]+0.7)&(tu<tu[-1]-0.7)
    i=np.argmax(km[interior]); idxs=np.where(interior)[0]; ia=idxs[i]
    # cmd/des/meas at gps-apex time
    m=np.abs(t-tc)<5.0
    tw=t[m]; cmd=arr[m,IDX['cmd']];des=arr[m,IDX['des']];meas=arr[m,IDX['meas']]
    tap=tu[ia]
    cmd_ap=float(np.interp(tap,tw,cmd));des_ap=float(np.interp(tap,tw,des));meas_ap=float(np.interp(tap,tw,meas))
    print(f"{r}: GPS-kappa apex (savgol) math={km[ia]:+.5f} head={kh[ia]:+.5f} @trel={tap-tc:+.2f} v={spd[ia]:.1f} "
          f"| cmd={cmd_ap:+.5f} des={des_ap:+.5f} meas={meas_ap:+.5f} "
          f"| kappa-des={km[ia]-des_ap:+.5f} kappa-cmd={km[ia]-cmd_ap:+.5f}")
    print(f"    p90|km|interior={np.percentile(np.abs(km[interior]),90):.5f} max km interior={np.max(km[interior]):.5f}")
