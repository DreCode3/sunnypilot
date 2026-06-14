#!/usr/bin/env python3
"""Find ALL passes near the target bend across all routes; classify EB(left)/WB(right)
by GPS heading; measure apex meas-cmd sign for each. The decisive EB/WB asymmetry test.

Claim (EPAS directional over-delivery): meas-cmd flips sign with turn dir (right +, left -).
Alt (constant additive yaw bias): meas-cmd stays + on BOTH dirs (left under-turns in dir).
"""
import sys, os, json, math, glob
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

# enumerate route dirs
route_dirs = sorted([os.path.basename(p) for p in glob.glob('explorer_st_logs/route_*') if os.path.isdir(p)])
print(f"scanning {len(route_dirs)} route dirs for passes within 30m of target bend")

found = []
for r in route_dirs:
  try:
    prefix = route_prefix_for(r)
    if prefix is None: continue
    try:
        gps = pull_gps_and_signals(prefix)
    except Exception:
        continue
    if gps is None: continue
    glat=np.asarray(gps['lat']);glon=np.asarray(gps['lon']);gt=np.asarray(gps['t'])
    if len(glat)==0: continue
    d=np.array([haversine_m(la,lo,TARGET_LAT,TARGET_LON) for la,lo in zip(glat,glon)])
    # find all local minima < 30 m (separate passes >30s apart)
    near = np.where(d<30)[0]
    if len(near)==0: continue
    # cluster by time gap
    clusters=[]
    cur=[near[0]]
    for k in near[1:]:
        if gt[k]-gt[cur[-1]]>30:
            clusters.append(cur); cur=[k]
        else:
            cur.append(k)
    clusters.append(cur)
    for cl in clusters:
        i_near = cl[int(np.argmin(d[cl]))]
        tc=float(gt[i_near]); mind=float(d[i_near])
        # heading at tc to determine direction (E vs W)
        w=(np.abs(gt-tc)<2.0)
        lat0,lon0=float(np.mean(glat[w])),float(np.mean(glon[w]))
        x,y=latlon_to_en(glat[w],glon[w],lat0,lon0)
        xp=np.gradient(x);yp=np.gradient(y)
        brg=math.degrees(math.atan2(np.mean(yp),np.mean(xp)))  # math: 0=E,90=N
        # compass: convert
        compass=(90-brg)%360
        ew = 'W' if (compass>180 and compass<360) else 'E'
        # turn direction via savgol kappa
        ww=(np.abs(gt-tc)<5.0)
        if ww.sum()<20: continue
        x2,y2=latlon_to_en(glat[ww],glon[ww],lat0,lon0)
        tu=np.arange(gt[ww][0],gt[ww][-1],0.05)
        xu=np.interp(tu,gt[ww],x2);yu=np.interp(tu,gt[ww],y2)
        win=13
        xpp=savgol_filter(xu,win,3,deriv=1,delta=0.05);ypp_=savgol_filter(yu,win,3,deriv=1,delta=0.05)
        xpp2=savgol_filter(xu,win,3,deriv=2,delta=0.05);ypp2=savgol_filter(yu,win,3,deriv=2,delta=0.05)
        spd=np.hypot(xpp,ypp_)
        km=-(xpp*ypp2-ypp_*xpp2)/np.power(np.maximum(spd**2,1e-6),1.5)  # +=RIGHT
        interior=(tu>tu[0]+0.7)&(tu<tu[-1]-0.7)
        if interior.sum()<5: continue
        kmax=float(np.max(km[interior]));kmin=float(np.min(km[interior]))
        turn = 'RIGHT' if abs(kmax)>abs(kmin) else 'LEFT'
        kapex = kmax if turn=='RIGHT' else kmin
        if abs(kapex)<0.0030:  # not a real curve here
            continue
        # cmd/meas at apex
        arr,t=load_cx1(prefix)
        if arr is None: continue
        m=np.abs(t-tc)<5.0
        if m.sum()<8: continue
        tw=t[m];cmd=arr[m,IDX['cmd']];meas=arr[m,IDX['meas']];des=arr[m,IDX['des']];ovr=arr[m,IDX['ovr']]
        # apex of |cmd| in turn dir
        if turn=='RIGHT':
            ia=int(np.argmax(cmd))
        else:
            ia=int(np.argmin(cmd))
        mc=float(meas[ia]-cmd[ia])
        found.append({'route':r,'dir':ew,'turn':turn,'mind':round(mind,1),
                      'compass':round(compass,0),'kapex':round(kapex,5),
                      'cmd_apex':round(float(cmd[ia]),5),'meas_apex':round(float(meas[ia]),5),
                      'des_apex':round(float(des[ia]),5),'meas_minus_cmd':round(mc,5),
                      'n_ovr':int(np.sum(ovr>0.5))})
  except Exception as e:
    sys.stderr.write(f"  skip {r}: {e}\n")
    continue

print("\n=== passes through the bend ===")
for f in sorted(found,key=lambda z:(z['turn'],z['route'])):
    print(json.dumps(f))

# summary by turn dir
print("\n=== meas-cmd by turn direction (no-override only) ===")
for turn in ('RIGHT','LEFT'):
    vals=[f['meas_minus_cmd'] for f in found if f['turn']==turn and f['n_ovr']==0]
    if vals:
        print(f"{turn}: n={len(vals)} median meas-cmd={np.median(vals):+.5f} vals={vals}")
