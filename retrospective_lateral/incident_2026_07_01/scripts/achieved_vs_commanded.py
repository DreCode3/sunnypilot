#!/usr/bin/env python3
"""Independent re-check of the plan's linchpin metric:
   |achieved_curvature / commanded_curvature| — old build vs new build.

Addresses confounds:
  - achieved from yawRate/vEgo AND independent estimate from steeringAngleDeg (steerRatio/wheelbase)
  - exclude driver-override frames (steeringPressed OR |steeringTorque|>0.5)
  - speed band 18-30 m/s; real curves |cmd|>0.0015
  - steady-state vs curve-entry split via |d(cmd)/dt|
  - sign: use signed ratio median AND |median| to detect sign issues
Commanded = carOutput.actuatorsOutput.curvature (what EPAS was actually told).
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
WHEELBASE = 3.025
STEER_RATIO = 17.2

def seg_files(route, cap=None):
    segs = sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*"),
                  key=lambda p: int(p.rsplit("--", 1)[-1]) if p.rsplit("--",1)[-1].isdigit() else 0)
    return segs[:cap] if cap else segs

def extract(seg):
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from openpilot.tools.lib.logreader import LogReader
    rl = os.path.join(seg, "rlog.zst")
    if not os.path.exists(rl):
        rl = os.path.join(seg, "rlog")
    if not os.path.exists(rl):
        return []
    co=[]; cst=[]; cc=[]
    try:
        lr = LogReader(rl)
    except Exception:
        return []
    for msg in lr:
        try:
            w = msg.which()
        except Exception:
            continue
        t = msg.logMonoTime
        try:
            if w == "carOutput":
                co.append((t, float(msg.carOutput.actuatorsOutput.curvature)))
            elif w == "carState":
                c = msg.carState
                cst.append((t, float(c.vEgo), float(c.yawRate), float(c.steeringAngleDeg),
                            bool(c.steeringPressed), float(c.steeringTorque)))
            elif w == "carControl":
                c = msg.carControl
                cc.append((t, bool(c.latActive)))
        except Exception:
            continue
    return [("co",co),("cst",cst),("cc",cc)]

def nearest_prev(times, vals, q):
    import bisect
    i = bisect.bisect_right(times, q) - 1
    return vals[i] if i >= 0 else None

def analyze(route, cap=None):
    segs = seg_files(route, cap)
    CO=[]; CST=[]; CC=[]
    with ProcessPoolExecutor(max_workers=10) as ex:
        for res in ex.map(extract, segs):
            for tag, rows in res:
                if tag=="co": CO+=rows
                elif tag=="cst": CST+=rows
                elif tag=="cc": CC+=rows
    for L in (CO,CST,CC): L.sort(key=lambda x:x[0])
    if not CO or not CST:
        print(f"{route}: no data"); return
    cst_t=[r[0] for r in CST]; cc_t=[r[0] for r in CC]
    rows=[]
    for (t, cmd) in CO:
        st = nearest_prev(cst_t, CST, t)
        la = nearest_prev(cc_t, CC, t)
        if st is None: continue
        _,v,yaw,sang,spress,storq = st
        latActive = la[1] if la else False
        rows.append(dict(t=t, cmd=cmd, v=v, yaw=yaw, sang=sang, spress=spress, storq=storq, latActive=latActive))
    # curvature rate (per-frame) for entry vs steady split — use commanded time series
    for i in range(1,len(rows)):
        dt=(rows[i]["t"]-rows[i-1]["t"])/1e9
        rows[i]["dcmd"] = (rows[i]["cmd"]-rows[i-1]["cmd"])/dt if dt>1e-3 else 0.0
    if rows: rows[0]["dcmd"]=0.0

    def filt(r):
        return (r["latActive"] and 18<=r["v"]<=30 and abs(r["cmd"])>0.0015
                and not r["spress"] and abs(r["storq"])<0.5)
    sel=[r for r in rows if filt(r)]
    if not sel:
        print(f"{route}: 0 selected frames"); return
    # achieved from yaw
    ach_yaw = np.array([ -r["yaw"]/max(r["v"],0.1) for r in sel])   # sign to match commanded convention (see note)
    ach_yaw_raw = np.array([ r["yaw"]/max(r["v"],0.1) for r in sel])
    # achieved from steering angle: curvature = tan(angle/steerRatio)/wheelbase ≈ (angle_rad/steerRatio)/wheelbase
    ach_ang = np.array([ np.tan(np.radians(r["sang"])/STEER_RATIO)/WHEELBASE for r in sel])
    cmd = np.array([r["cmd"] for r in sel])
    dcmd = np.array([abs(r["dcmd"]) for r in sel])
    v = np.array([r["v"] for r in sel])

    # ratio (guard tiny cmd already >0.0015)
    def med_ratio(ach):
        rr = np.abs(ach)/np.abs(cmd)
        return np.median(rr), len(rr)
    # signed ratio to check sign convention: (ach*cmd)/(cmd^2) median sign
    def signed_ratio(ach):
        return np.median((ach*cmd)/(cmd*cmd))

    # steady vs entry split on |dcmd|
    thr = np.median(dcmd)
    steady = dcmd <= thr
    entry = dcmd > thr

    print(f"\n===== {route} (segs={len(segs)}) selected={len(sel)}  medspeed={np.median(v):.1f} m/s =====")
    for name, ach in [("yaw(-yaw/v)", ach_yaw), ("yaw(+yaw/v)", ach_yaw_raw), ("steerAngle", ach_ang)]:
        m,n = med_ratio(ach)
        sr = signed_ratio(ach)
        ms,_ = (np.median(np.abs(ach[steady])/np.abs(cmd[steady])), None) if steady.any() else (np.nan,None)
        me,_ = (np.median(np.abs(ach[entry])/np.abs(cmd[entry])), None) if entry.any() else (np.nan,None)
        print(f"  {name:14s} |med ratio|={m:.3f}  signedMed={sr:+.3f}  steady={ms:.3f}(n={steady.sum()})  entry={me:.3f}(n={entry.sum()})")

if __name__=="__main__":
    # cap segments per route to bound runtime; new routes small anyway
    plan = [("route_c5",None),("route_b5",12),("route_ce",None),("route_cf",None)]
    for route,cap in plan:
        analyze(route,cap)
