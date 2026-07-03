#!/usr/bin/env python3
"""On the ce seg2 / cf seg10 'clean curve' frames, dump the full chain:
model desiredCurvature -> carControl.actuators.curvature -> carOutput -> steering angle -> yaw.
Answers: is carOutput (EPS command) actually equal to what was commanded, or already clipped?
Was there an override right before? What's the time structure (is it curve-ENTRY)?
"""
import glob, os, sys, bisect
import numpy as np
ROOT="/Users/dregilley/Documents/GitHub/sunnypilot"
WB=3.025; SR=17.2
def seg_path(route,segnum):
    for p in glob.glob(f"{ROOT}/explorer_st_logs/{route}/*"):
        if p.rsplit('--',1)[-1]==str(segnum): return p
    return None
def run(route,segnum):
    sys.path.insert(0,ROOT)
    from openpilot.tools.lib.logreader import LogReader
    seg=seg_path(route,segnum)
    rl=os.path.join(seg,"rlog.zst")
    if not os.path.exists(rl): rl=os.path.join(seg,"rlog")
    mdl=[];cc=[];co=[];cst=[]
    for msg in LogReader(rl):
        try:w=msg.which()
        except Exception:continue
        t=msg.logMonoTime
        try:
            if w=="modelV2": mdl.append((t,float(msg.modelV2.action.desiredCurvature)))
            elif w=="carControl": cc.append((t,float(msg.carControl.actuators.curvature),bool(msg.carControl.latActive)))
            elif w=="carOutput": co.append((t,float(msg.carOutput.actuatorsOutput.curvature)))
            elif w=="carState":
                c=msg.carState;cst.append((t,float(c.vEgo),float(c.yawRate),float(c.steeringAngleDeg),bool(c.steeringPressed),float(c.steeringTorque)))
        except Exception:continue
    for L in (mdl,cc,co,cst):L.sort(key=lambda x:x[0])
    def npv(L,q):
        ts=[r[0] for r in L];i=bisect.bisect_right(ts,q)-1;return L[i] if i>=0 else None
    print(f"\n== {route} seg{segnum}: clean-curve frames (v18-30,|carOut|>0.0015,!press,|tq|<0.5) ==")
    print(f"{'idx':>3} {'v':>5} {'desC':>8} {'ccCurv':>8} {'outCurv':>8} {'angC':>8} {'yawC':>8} {'ang°':>6} {'tq':>5} {'prs':>3}")
    cnt=0
    for (t,cmd) in co:
        st=npv(cst,t)
        if not st: continue
        _,v,yaw,sang,sp,tq=st
        la=npv(cc,t)
        if not (la and la[2] and 18<=v<=30 and abs(cmd)>0.0015 and not sp and abs(tq)<0.5): continue
        m=npv(mdl,t); ccv=la[1]
        angc=np.tan(np.radians(sang)/SR)/WB
        yawc=-yaw/max(v,0.1)
        cnt+=1
        if cnt%8==1:  # sample every 8th to keep output short
            print(f"{cnt:>3} {v:5.1f} {m[1] if m else float('nan'):8.5f} {ccv:8.5f} {cmd:8.5f} {angc:8.5f} {yawc:8.5f} {sang:6.1f} {tq:5.2f} {int(sp)}")
    print(f"total clean curve frames: {cnt}")
if __name__=="__main__":
    run("route_ce",2)
    run("route_cf",10)
