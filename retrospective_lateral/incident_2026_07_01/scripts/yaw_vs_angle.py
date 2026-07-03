#!/usr/bin/env python3
"""Cross-check the two achieved estimators against each other (yaw-curv vs angle-curv).
If on OLD builds yaw-curv ≈ angle-curv (slope~1), the estimators agree and we can trust the
disagreement on cf as a real yaw-vs-steering discrepancy. Uses ALL active moving frames
(not just commanded curves) to get a clean regression, driver-override excluded."""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT="/Users/dregilley/Documents/GitHub/sunnypilot"
WB=3.025; SR=17.2
def segs(route,cap=None):
    s=sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*"),
        key=lambda p:int(p.rsplit('--',1)[-1]) if p.rsplit('--',1)[-1].isdigit() else 0)
    return s[:cap] if cap else s
def extract(seg):
    import sys as _s
    if ROOT not in _s.path: _s.path.insert(0,ROOT)
    from openpilot.tools.lib.logreader import LogReader
    rl=os.path.join(seg,"rlog.zst")
    if not os.path.exists(rl): rl=os.path.join(seg,"rlog")
    if not os.path.exists(rl): return None
    cst=[]
    try: lr=LogReader(rl)
    except Exception: return None
    for msg in lr:
        try: w=msg.which()
        except Exception: continue
        if w=="carState":
            c=msg.carState
            cst.append((float(c.vEgo),float(c.yawRate),float(c.steeringAngleDeg),
                        bool(c.steeringPressed),float(c.steeringTorque)))
    return cst
def run(route,cap=None):
    CST=[]
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(extract,segs(route,cap)):
            if r: CST+=r
    A=np.array(CST)
    if len(A)==0: print(f"{route}: none");return
    v,yaw,sang,sp,tq=A[:,0],A[:,1],A[:,2],A[:,3],A[:,4]
    m=(v>=18)&(v<=30)&(sp<0.5)&(np.abs(tq)<0.5)&(np.abs(sang)>1.0)
    yc=-yaw[m]/np.maximum(v[m],0.1)
    ac=np.tan(np.radians(sang[m])/SR)/WB
    if m.sum()<20: print(f"{route}: n={m.sum()} too few");return
    # slope of yc vs ac through origin
    slope=np.sum(yc*ac)/np.sum(ac*ac)
    corr=np.corrcoef(yc,ac)[0,1]
    print(f"{route}: n={m.sum()}  yawCurv/angleCurv slope={slope:.3f}  corr={corr:.3f}  medYawCurv={np.median(np.abs(yc)):.5f} medAngCurv={np.median(np.abs(ac)):.5f}")
if __name__=="__main__":
    for r,c in [("route_c5",None),("route_b5",12),("route_ce",None),("route_cf",None)]:
        run(r,c)
