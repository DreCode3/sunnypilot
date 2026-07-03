#!/usr/bin/env python3
"""How much qualifying data actually exists per route? Speed distribution of active-curve frames."""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT="/Users/dregilley/Documents/GitHub/sunnypilot"
def segs(route):
    return sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*"),
        key=lambda p:int(p.rsplit('--',1)[-1]) if p.rsplit('--',1)[-1].isdigit() else 0)
def extract(seg):
    import sys as _s
    if ROOT not in _s.path: _s.path.insert(0,ROOT)
    from openpilot.tools.lib.logreader import LogReader
    rl=os.path.join(seg,"rlog.zst")
    if not os.path.exists(rl): rl=os.path.join(seg,"rlog")
    if not os.path.exists(rl): return None
    co=[];cst=[];cc=[]
    try: lr=LogReader(rl)
    except Exception: return None
    for msg in lr:
        try: w=msg.which()
        except Exception: continue
        t=msg.logMonoTime
        try:
            if w=="carOutput": co.append((t,float(msg.carOutput.actuatorsOutput.curvature)))
            elif w=="carState":
                c=msg.carState; cst.append((t,float(c.vEgo),bool(c.steeringPressed),float(c.steeringTorque)))
            elif w=="carControl": cc.append((t,bool(msg.carControl.latActive)))
        except Exception: continue
    return (co,cst,cc)
def npv(times,vals,q):
    import bisect;i=bisect.bisect_right(times,q)-1;return vals[i] if i>=0 else None
def run(route):
    CO=[];CST=[];CC=[]
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(extract,segs(route)):
            if r: CO+=r[0];CST+=r[1];CC+=r[2]
    for L in (CO,CST,CC):L.sort(key=lambda x:x[0])
    if not CO or not CST: print(f"{route}: no data");return
    cst_t=[r[0] for r in CST];cc_t=[r[0] for r in CC]
    speeds_curve=[]
    n_active=0
    for (t,cmd) in CO:
        st=npv(cst_t,CST,t);la=npv(cc_t,CC,t)
        if st is None:continue
        _,v,sp,tq=st
        if la and la[1]:
            n_active+=1
            if abs(cmd)>0.0015 and not sp and abs(tq)<0.5:
                speeds_curve.append(v)
    speeds_curve=np.array(speeds_curve)
    tot=len(speeds_curve)
    b_low=((speeds_curve>=5)&(speeds_curve<18)).sum()
    b_mid=((speeds_curve>=18)&(speeds_curve<=30)).sum()
    b_hi=(speeds_curve>30).sum()
    print(f"{route}: activeCarOut={n_active}  curveFrames(clean)={tot}  | v5-18={b_low} v18-30={b_mid} v>30={b_hi}")
if __name__=="__main__":
    for r in (sys.argv[1:] or ["route_c5","route_c7","route_b5","route_7f","route_ce","route_cf"]):
        run(r)
