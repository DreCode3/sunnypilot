#!/usr/bin/env python3
"""Consolidated reproducible analyses for the 2026-07-01 AOL lateral incident (routes ce+cf).
Each function is the exact logic that produced a finding in the handoff. Run individual
functions from __main__.

RUN:  cd <repo> && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
        retrospective_lateral/incident_2026_07_01/scripts/incident_analyses.py <analysis>

Data: explorer_st_logs/route_{ce,cf}=NEW build (2026-07-01 incident), route_{c5,c7}=OLD
Nevada, route_b5=OLD CD210, route_7f=OLD OPM7. Video: ce seg1,3,4 + cf seg9,10,11.
LogReader: wrap msg.which() in try/except (some events corrupt on these new-format rlogs).
"""
import sys, glob, bisect, math
import numpy as np
ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader

NEW = [("route_ce", "NEW Nev"), ("route_cf", "NEW Nev")]
OLD = [("route_c7", "OLD Nev"), ("route_b5", "OLD CD210"), ("route_7f", "OLD OPM7")]


def _pv(arr, ts, t):
    i = bisect.bisect_right(ts, t) - 1
    return arr[i] if i >= 0 else None


def _load(route, want):
    """Return dict of sorted (t, ...) lists for the requested message types."""
    out = {k: [] for k in want}
    for rl in glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst"):
        try:
            lr = LogReader(rl)
        except Exception:
            continue
        for m in lr:
            try:
                w = m.which()
            except Exception:
                continue
            if w not in want:
                continue
            t = m.logMonoTime
            if w == "modelV2":
                M = m.modelV2; p = list(M.laneLineProbs); ll = list(M.laneLines)
                lY = ll[1].y[0] if len(ll) >= 2 and len(ll[1].y) else np.nan
                rY = ll[2].y[0] if len(ll) >= 3 and len(ll[2].y) else np.nan
                out[w].append((t, int(M.frameId), min(p[1:3]) if len(p) >= 3 else 1.0,
                               float(M.action.desiredCurvature), (lY + rY) / 2))
            elif w == "carState":
                c = m.carState
                out[w].append((t, float(c.vEgo), float(c.steeringAngleDeg), bool(c.steeringPressed),
                               float(c.steeringTorque), bool(c.leftBlinker), bool(c.rightBlinker), float(c.yawRate)))
            elif w == "carControl":
                out[w].append((t, bool(m.carControl.latActive), bool(m.carControl.enabled)))
            elif w == "carOutput":
                out[w].append((t, float(m.carOutput.actuatorsOutput.curvature)))
            elif w == "roadCameraState":
                r = m.roadCameraState
                try:
                    out[w].append((t, int(r.frameId), float(r.measuredGreyFraction),
                                   float(r.targetGreyFraction), int(r.integLines)))
                except Exception:
                    pass
            elif w == "liveLocationKalman":
                g = m.liveLocationKalman
                try:
                    pg = g.positionGeodetic.value
                    if len(pg) >= 2 and g.positionGeodetic.valid:
                        out[w].append((t, float(pg[0]), float(pg[1])))
                except Exception:
                    pass
    for k in out:
        out[k].sort()
    return out


def _vm():
    from opendbc.car.vehicle_model import VehicleModel
    for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/route_cf/*/rlog.zst"))[:2]:
        for m in LogReader(rl):
            try:
                if m.which() == "carParams":
                    return VehicleModel(m.carParams)
            except Exception:
                continue
    raise RuntimeError("no carParams")


# ---------------------------------------------------------------------------
# 1. ACHIEVED vs COMMANDED curvature — angle(VehicleModel) AND yaw. THE analysis
#    that OVERTURNED the feed-forward hypothesis (wheel DID follow command on cf).
# ---------------------------------------------------------------------------
def achieved_vs_commanded():
    VM = _vm()
    print("route     build     n   yaw_ratio  angle_ratio(VM)  med|steerDeg|  (system-only, 18-30 m/s, |cmd|>0.0015)")
    for route, b in OLD + NEW:
        d = _load(route, {"carOutput", "carState", "carControl"})
        cstt = [x[0] for x in d["carState"]]; cct = [x[0] for x in d["carControl"]]
        yr, ar, ang = [], [], []
        for (t, c) in d["carOutput"]:
            st = _pv(d["carState"], cstt, t); cc = _pv(d["carControl"], cct, t)
            if not st or not cc or not cc[1]:
                continue
            v, sa, sp = st[1], st[2], st[3]
            if sp or v < 18 or v > 30 or abs(c) < 0.0015:
                continue
            yr.append(abs((-st[7] / v) / c))
            ar.append(abs(VM.calc_curvature(math.radians(sa), v, 0.0) / c))
            ang.append(abs(sa))
        n = len(yr)
        print(f"  {route:9s} {b:9s} {n:4d}  {np.median(yr) if n else float('nan'):.2f}      "
              f"{np.median(ar) if n else float('nan'):.2f}          {np.median(ang) if n else float('nan'):.1f}")
    print("EXPECT: OLD ~1.0-1.2 both; NEW cf angle~0.90 (wheel follows), ce~0.09 (n~100, entry-confounded).")


# ---------------------------------------------------------------------------
# 2. RECOVERY LAG — laneProb collapse(<0.15) -> recover(>0.7). THE leading signal
#    (same-model old-engine vs new-engine).
# ---------------------------------------------------------------------------
def recovery_lag():
    print("laneProb collapse -> recovery time, per route:")
    for route, b in OLD + NEW:
        d = _load(route, {"modelV2", "roadCameraState"})
        MV = d["modelV2"]; rt = [x[0] for x in d["roadCameraState"]]
        lags = []; i = 0; n = len(MV)
        while i < n:
            if MV[i][2] < 0.15:
                j = i
                while j < n and MV[j][2] < 0.7:
                    j += 1
                if j < n:
                    dur = (MV[j][0] - MV[i][0]) / 1e9
                    mid = (MV[i][0] + MV[j][0]) / 2
                    k = bisect.bisect_right(rt, mid) - 1
                    grey_ok = (k >= 0 and d["roadCameraState"][k][2] < 0.34)
                    lags.append((dur, grey_ok)); i = j; continue
            i += 1
        if not lags:
            print(f"  {route} ({b}): none"); continue
        durs = np.array([l[0] for l in lags]); clr = [l[0] for l in lags if l[1]]
        print(f"  {route:9s} {b:9s}: {len(lags)} collapses median={np.median(durs):.2f}s "
              f"p90={np.percentile(durs,90):.2f}s ; clear-image recoveries median={np.median(clr) if clr else 0:.2f}s")
    print("EXPECT: OLD-engine Nevada c7 ~2.2s vs NEW-engine cf 3.0s / ce 4.4s (clear-img ce 6.4s).")


# ---------------------------------------------------------------------------
# 3. OVERRIDE CHARACTERIZATION — the 9(ce)+1(cf) events: lane-change, laneProb, drift.
# ---------------------------------------------------------------------------
def overrides():
    for route in ("route_ce", "route_cf"):
        d = _load(route, {"modelV2", "carState", "carControl"})
        MV = d["modelV2"]; cct = [x[0] for x in d["carControl"]]
        bt = sorted(t for (t, v, sa, sp, st, lb, rb, yr) in d["carState"] if lb or rb)
        ov = [(t, st, v) for (t, v, sa, sp, st, lb, rb, yr) in d["carState"]
              if sp and abs(st) > 1.8 and v > 13 and (_pv(d["carControl"], cct, t) or (0, False))[1]]
        clusters = []; cur = []
        for o in ov:
            if cur and o[0] - cur[-1][0] > 2.5e9:
                clusters.append(cur); cur = []
            cur.append(o)
        if cur:
            clusters.append(cur)
        print(f"=== {route}: {len(clusters)} override events (v | laneChg | minLaneProb3s | offset drift | ovrTrq) ===")
        for cl in clusters:
            t0 = cl[0][0]; v = cl[0][2]; drv = cl[0][1]
            lo = bisect.bisect_left(bt, t0 - 6e9); hi = bisect.bisect_right(bt, t0 + 1e9)
            blk = hi > lo
            pre = [r for r in MV if t0 - 3e9 <= r[0] < t0]
            if not pre:
                continue
            lp = min(r[2] for r in pre)
            offs = [r[4] for r in pre if np.isfinite(r[4])]
            od = (offs[-1] - offs[0]) if offs else np.nan
            print(f"  {v*2.237:4.0f} | {'LANE-CHG' if blk else '   -    '} | {lp:.2f} | "
                  f"{offs[0] if offs else float('nan'):+.2f}->{offs[-1] if offs else float('nan'):+.2f} ({od:+.2f}m) | {drv:+.1f}")
    print("EXPECT: 4/9 ce lane-change; several drift 0.6-0.7m; the 62mph/laneProb0.96 one is the decisive weak-correction event.")


# ---------------------------------------------------------------------------
# 4. OFFSET EXCURSIONS when perception GOOD (laneProb>0.7) — centering strength.
# ---------------------------------------------------------------------------
def offset_excursions():
    print("|lane-center offset| when laneProb>0.7, active moving (bigger=weaker centering):")
    for route, b in [("route_b5", "OLD CD210"), ("route_c7", "OLD Nev")] + NEW:
        d = _load(route, {"modelV2", "carState", "carControl"})
        cstt = [x[0] for x in d["carState"]]; cct = [x[0] for x in d["carControl"]]
        offs = []
        for (t, fid, lp, dc, off) in d["modelV2"]:
            st = _pv(d["carState"], cstt, t); cc = _pv(d["carControl"], cct, t)
            if not np.isfinite(off) or lp < 0.7:
                continue
            if st and cc and cc[1] and st[1] > 13 and not st[3]:
                offs.append(abs(off))
        offs = np.array(offs)
        print(f"  {route:9s} {b:9s} n={len(offs):5d} median={np.median(offs):.3f} "
              f"p95={np.percentile(offs,95):.3f} max={offs.max():.3f}")
    print("EXPECT: NEW p95 0.5-0.7 vs OLD 0.34-0.36 (suggestive, small NEW samples).")


# ---------------------------------------------------------------------------
# 5. UNDERPASS EXPOSURE TIMELINE (cf seg10, the one-off blowout).
# ---------------------------------------------------------------------------
def underpass():
    import os
    seg = f"{ROOT}/explorer_st_logs/route_cf/000000cf--4d24fc7149--10"
    rows = []
    for m in LogReader(os.path.join(seg, "rlog.zst")):
        try:
            w = m.which()
        except Exception:
            continue
        if w == "roadCameraState":
            r = m.roadCameraState
            if 12630 <= int(r.frameId) <= 12760 and int(r.frameId) % 5 == 0:
                rows.append((int(r.frameId), float(r.measuredGreyFraction), int(r.integLines)))
    rows.sort()
    print("cf seg10 underpass exposure (frameId | measuredGreyFraction | integLines):")
    for fid, mg, il in rows:
        flag = "  <-- BLOWOUT" if mg > 0.42 else ""
        print(f"  {fid} | {mg:.3f} | {il}{flag}")
    print("EXPECT: integLines 8->237 (shadow) then grey spikes to 0.50 emerging into sun (blowout); "
          "laneProb collapses 12691->12720; image clear by 12720 but laneProb stuck <0.1 until ~12766.")


if __name__ == "__main__":
    fns = {"achieved": achieved_vs_commanded, "recovery": recovery_lag, "overrides": overrides,
           "offsets": offset_excursions, "underpass": underpass}
    if len(sys.argv) < 2 or sys.argv[1] not in fns:
        print("usage: incident_analyses.py [" + "|".join(fns) + "]")
    else:
        fns[sys.argv[1]]()
