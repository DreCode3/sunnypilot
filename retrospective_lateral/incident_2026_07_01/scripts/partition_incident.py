#!/usr/bin/env python3
"""Partition the ce/cf incident failures: model-commanded-straight vs control-clipped.
Extracts per-seg signals (parallel), resamples onto modelV2 (20Hz) times, classifies
failure events, and correlates laneLineProb crashes + alert presence.
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"

def seg_files(route):
    segs = sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*"),
                  key=lambda p: int(p.rsplit("--", 1)[-1]) if p.rsplit("--",1)[-1].isdigit() else 0)
    return segs

def extract(seg):
    import sys as _sys
    if ROOT not in _sys.path:
        _sys.path.insert(0, ROOT)
    from openpilot.tools.lib.logreader import LogReader
    rl = os.path.join(seg, "rlog.zst")
    if not os.path.exists(rl):
        rl = os.path.join(seg, "rlog")
    segnum = int(seg.rsplit("--", 1)[-1])
    # per-type (t, tuple) rows
    mdl=[]; cc=[]; co=[]; cs_ctrl=[]; cst=[]; sds=[]; road_idx=0
    try:
        lr = LogReader(rl)
    except Exception as e:
        return (segnum, {}, f"open_fail:{e}")
    for msg in lr:
        try:
            w = msg.which()
        except Exception:
            continue
        t = msg.logMonoTime
        try:
            if w == "modelV2":
                m = msg.modelV2
                dc = float(m.action.desiredCurvature)
                probs = list(m.laneLineProbs) if len(m.laneLineProbs) else []
                lp_min = min(probs) if probs else float("nan")
                lp_inner = min(probs[1:3]) if len(probs) >= 3 else lp_min
                # far-horizon lateral position (path curvature proxy)
                posy = list(m.position.y)
                py_far = float(posy[-1]) if posy else float("nan")
                mdl.append((t, dc, lp_min, lp_inner, py_far))
            elif w == "carControl":
                c = msg.carControl
                cc.append((t, float(c.actuators.curvature), bool(c.latActive), bool(c.enabled)))
            elif w == "carOutput":
                co.append((t, float(msg.carOutput.actuatorsOutput.curvature)))
            elif w == "controlsState":
                c = msg.controlsState
                cs_ctrl.append((t, float(c.desiredCurvature), float(c.curvature)))
            elif w == "carState":
                c = msg.carState
                cst.append((t, float(c.vEgo), float(c.yawRate), float(c.steeringAngleDeg),
                            bool(c.steeringPressed), float(c.steeringTorque)))
            elif w == "selfdriveState":
                s = msg.selfdriveState
                at = (str(s.alertText1) + "|" + str(s.alertText2)).strip("|")
                sds.append((t, at, str(s.alertStatus)))
            elif w == "roadEncodeIdx":
                road_idx += 1
        except Exception:
            continue
    return (segnum, dict(mdl=mdl, cc=cc, co=co, cs_ctrl=cs_ctrl, cst=cst, sds=sds, road_idx=road_idx), None)

def nearest_prev(times, vals, q):
    """for query time q, return val at greatest t<=q (times sorted)."""
    import bisect
    i = bisect.bisect_right(times, q) - 1
    return vals[i] if i >= 0 else None

def analyze(route):
    segs = seg_files(route)
    with ProcessPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(extract, segs))
    results.sort(key=lambda r: r[0])
    # concat with global mono time (already absolute ns)
    MDL=[]; CC=[]; CO=[]; CTRL=[]; CST=[]; SDS=[]; road_total=0; seg_road={}
    for segnum, d, err in results:
        if err:
            print(f"  seg {segnum}: {err}"); continue
        MDL += d["mdl"]; CC += d["cc"]; CO += d["co"]; CTRL += d["cs_ctrl"]; CST += d["cst"]; SDS += d["sds"]
        road_total += d["road_idx"]; seg_road[segnum]=d["road_idx"]
    for L in (MDL,CC,CO,CTRL,CST,SDS): L.sort(key=lambda x:x[0])
    print(f"\n===== ROUTE {route} =====")
    print(f"segs={len(segs)}  modelV2={len(MDL)}  carControl={len(CC)}  carOutput={len(CO)}  carState={len(CST)}")
    print(f"roadEncodeIdx TOTAL={road_total}  per-seg nonzero: {sorted([s for s,c in seg_road.items() if c])}")
    print(f"  per-seg zero roadIdx: {sorted([s for s,c in seg_road.items() if not c])}")
    if not MDL:
        return
    # build aligned arrays on modelV2 times
    cc_t=[r[0] for r in CC]; co_t=[r[0] for r in CO]; ctrl_t=[r[0] for r in CTRL]; cst_t=[r[0] for r in CST]; sds_t=[r[0] for r in SDS]
    rows=[]
    for (t, dc, lp_min, lp_inner, py_far) in MDL:
        cc_v = nearest_prev(cc_t, CC, t); co_v = nearest_prev(co_t, CO, t)
        ct_v = nearest_prev(ctrl_t, CTRL, t); st_v = nearest_prev(cst_t, CST, t)
        sd_v = nearest_prev(sds_t, SDS, t)
        rows.append(dict(t=t, dc=dc, lp_min=lp_min, lp_inner=lp_inner, py_far=py_far,
            act_curv=cc_v[1] if cc_v else np.nan, latActive=cc_v[2] if cc_v else False, enabled=cc_v[3] if cc_v else False,
            out_curv=co_v[1] if co_v else np.nan,
            ctrl_dc=ct_v[1] if ct_v else np.nan, ctrl_curv=ct_v[2] if ct_v else np.nan,
            vEgo=st_v[1] if st_v else np.nan, yaw=st_v[2] if st_v else np.nan, sang=st_v[3] if st_v else np.nan,
            spress=st_v[4] if st_v else False, storque=st_v[5] if st_v else np.nan,
            alert=sd_v[1] if sd_v else "", astat=sd_v[2] if sd_v else ""))
    # ---- aggregate stats over ACTIVE frames ----
    act = [r for r in rows if r["latActive"] and r["vEgo"]>5]
    print(f"active(latActive & v>5) modelV2 frames: {len(act)}  (dt≈{(act[-1]['t']-act[0]['t'])/1e9:.0f}s)" if act else "no active frames")
    if not act: return
    CURVE_DC=0.0030   # |desiredCurvature| above this = model intends a real curve
    STRAIGHT_DC=0.0012 # |desiredCurvature| below this = model commanding ~straight
    LP_CRASH=0.30
    # laneProb crashes
    lp_crash=[r for r in act if r["lp_min"]<LP_CRASH]
    print(f"laneProb(min)<{LP_CRASH}: {len(lp_crash)} frames = {len(lp_crash)/len(act)*100:.2f}% of active")
    lp_inner_crash=[r for r in act if r["lp_inner"]<LP_CRASH]
    print(f"laneProb(inner two)<{LP_CRASH}: {len(lp_inner_crash)} frames = {len(lp_inner_crash)/len(act)*100:.2f}%")
    # model-straight while lane-prob-crashed (dead-straight-blackout signature)
    ms_black=[r for r in act if abs(r["dc"])<STRAIGHT_DC and r["lp_min"]<LP_CRASH]
    print(f"MODEL-STRAIGHT (|dc|<{STRAIGHT_DC}) DURING laneProb crash: {len(ms_black)} frames")
    # control-clip: model wants curve but output much less
    clip=[r for r in act if abs(r["dc"])>CURVE_DC and abs(r["out_curv"])<0.6*abs(r["dc"])]
    print(f"CONTROL-CLIP (|dc|>{CURVE_DC} & |out|<0.6|dc|): {len(clip)} frames = {len(clip)/len(act)*100:.2f}%")
    # driver overrides
    ov=[r for r in act if r["spress"] and abs(r["storque"])>2.0 and r["vEgo"]>15]
    print(f"driver overrides (press & |torque|>2 & v>15): {len(ov)} frames")
    # alerts present during active?
    alerts=set(r["alert"] for r in act if r["alert"])
    print(f"distinct alertText during active: {alerts if alerts else '(NONE)'}")
    # ---- classify each override cluster ----
    print("--- override clusters (gap>1s splits) ---")
    ov.sort(key=lambda r:r["t"]); clusters=[]; cur=[]
    for r in ov:
        if cur and (r["t"]-cur[-1]["t"])>1e9: clusters.append(cur); cur=[]
        cur.append(r)
    if cur: clusters.append(cur)
    for ci,cl in enumerate(clusters):
        # look at model/output in the 1.5s BEFORE the cluster start
        t0=cl[0]["t"]; pre=[r for r in act if t0-1.5e9<=r["t"]<t0]
        seg=cl[0].get("seg")
        dc_pre=np.nanmax([abs(r["dc"]) for r in pre]) if pre else np.nan
        out_pre=np.nanmax([abs(r["out_curv"]) for r in pre]) if pre else np.nan
        lp_pre=np.nanmin([r["lp_min"] for r in pre]) if pre else np.nan
        v=cl[0]["vEgo"]
        kind = "MODEL-STRAIGHT" if dc_pre<CURVE_DC else ("CONTROL-CLIP" if out_pre<0.6*dc_pre else "both-ok?")
        print(f"  #{ci}: v={v*2.237:.0f}mph  max|dc_model|(pre)={dc_pre:.5f}  max|out|(pre)={out_pre:.5f}  min laneProb(pre)={lp_pre:.2f}  torque={cl[0]['storque']:.1f}  => {kind}")

if __name__=="__main__":
    for route in (sys.argv[1:] or ["route_ce","route_cf"]):
        analyze(route)
