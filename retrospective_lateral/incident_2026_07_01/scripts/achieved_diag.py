#!/usr/bin/env python3
"""Deeper diagnostic on ce/cf contradiction:
 - Is the steerAngle-vs-yaw disagreement a yaw-signal or geometry issue?
 - Per-segment ratio spread (is ce n=90 dominated by 1 seg?)
 - Distribution of achieved/commanded, and whether commanded itself differs old vs new.
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT="/Users/dregilley/Documents/GitHub/sunnypilot"
WB=3.025; SR=17.2
def segs(route):
    return sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*"),
        key=lambda p:int(p.rsplit('--',1)[-1]) if p.rsplit('--',1)[-1].isdigit() else 0)
def extract(seg):
    import sys as _s
    if ROOT not in _s.path: _s.path.insert(0,ROOT)
    from openpilot.tools.lib.logreader import LogReader
    rl=os.path.join(seg,"rlog.zst")
    if not os.path.exists(rl): rl=os.path.join(seg,"rlog")
    if not os.path.exists(rl): return (seg,[])
    co=[];cst=[];cc=[]
    try: lr=LogReader(rl)
    except Exception: return (seg,[])
    for msg in lr:
        try: w=msg.which()
        except Exception: continue
        t=msg.logMonoTime
        try:
            if w=="carOutput": co.append((t,float(msg.carOutput.actuatorsOutput.curvature)))
            elif w=="carState":
                c=msg.carState
                cst.append((t,float(c.vEgo),float(c.yawRate),float(c.steeringAngleDeg),
                            bool(c.steeringPressed),float(c.steeringTorque)))
            elif w=="carControl": cc.append((t,bool(msg.carControl.latActive)))
        except Exception: continue
    return (seg,[("co",co),("cst",cst),("cc",cc)])
def np_(times,vals,q):
    import bisect;i=bisect.bisect_right(times,q)-1;return vals[i] if i>=0 else None
def per_seg(route):
    print(f"\n===== {route} per-seg =====")
    with ProcessPoolExecutor(max_workers=10) as ex:
        res=list(ex.map(extract,segs(route)))
    for seg,data in res:
        if not data: continue
        d={t:rows for t,rows in data}
        CO=d.get("co",[]);CST=d.get("cst",[]);CC=d.get("cc",[])
        if not CO or not CST: continue
        cst_t=[r[0] for r in CST];cc_t=[r[0] for r in CC]
        sel=[]
        for (t,cmd) in CO:
            st=np_(cst_t,CST,t); la=np_(cc_t,CC,t)
            if st is None: continue
            _,v,yaw,sang,sp,tq=st
            if (la and la[1]) and 18<=v<=30 and abs(cmd)>0.0015 and not sp and abs(tq)<0.5:
                sel.append((cmd,v,yaw,sang))
        if len(sel)<10: continue
        cmd=np.array([s[0] for s in sel]);v=np.array([s[1] for s in sel])
        yaw=np.array([s[2] for s in sel]);sang=np.array([s[3] for s in sel])
        ach_yaw=-yaw/np.maximum(v,0.1)
        ach_ang=np.tan(np.radians(sang)/SR)/WB
        ry=np.median(np.abs(ach_yaw)/np.abs(cmd)); ra=np.median(np.abs(ach_ang)/np.abs(cmd))
        sn=int(seg.rsplit('--',1)[-1])
        print(f"  seg {sn:3d} n={len(sel):4d} v~{np.median(v):.1f} yawRatio={ry:.2f} angRatio={ra:.2f} "
              f"medCmd={np.median(np.abs(cmd)):.4f} medYaw={np.median(np.abs(yaw)):.4f} medAng={np.median(np.abs(sang)):.1f}deg")
if __name__=="__main__":
    for r in (sys.argv[1:] or ["route_ce","route_cf"]):
        per_seg(r)
