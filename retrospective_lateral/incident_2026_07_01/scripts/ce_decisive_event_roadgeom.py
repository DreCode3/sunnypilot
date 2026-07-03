#!/usr/bin/env python3
"""Dissect the decisive weak-correction event on route_ce (t0=212.535s mono, 62mph,
minLaneProb3s=0.96): was the road CURVING (model under-commands a curve) or STRAIGHT
(model fails to correct a growing offset)?

Ground truth road curvature from three independent sources:
  (a) liveLocationKalman.angularVelocityCalibrated.value[2] / vEgo  (achieved yaw curvature)
  (b) model's own lane-line geometry: quadratic fit of (laneLines[1]+laneLines[2])/2 y(x)
  (c) steeringAngleDeg -> VehicleModel.calc_curvature (achieved wheel curvature)
plus desiredCurvature (model command) and carOutput.curvature (actuator echo).

Sign conventions are verified empirically (correlations over the active driving in
segments 1-3) before the timeline is interpreted.
"""
import sys, glob, bisect, math
import numpy as np
ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader

T0 = 212.535e9  # mono ns, cluster 3 from ce_decisive_locate.py
SEGS = ["000000ce--f8a8d4f570--1", "000000ce--f8a8d4f570--2", "000000ce--f8a8d4f570--3"]


def _pv(ts, t):
    i = bisect.bisect_right(ts, t) - 1
    return i if i >= 0 else None


def load():
    d = {"modelV2": [], "carState": [], "carControl": [], "carOutput": [], "llk": []}
    for seg in SEGS:
        for m in LogReader(f"{ROOT}/explorer_st_logs/route_ce/{seg}/rlog.zst"):
            try:
                w = m.which()
            except Exception:
                continue
            t = m.logMonoTime
            if w == "modelV2":
                M = m.modelV2
                p = list(M.laneLineProbs); ll = list(M.laneLines)
                if len(ll) < 3 or not len(ll[1].y):
                    continue
                lx = np.array(ll[1].x); ly = np.array(ll[1].y); ry = np.array(ll[2].y)
                ds = list(M.meta.desireState)
                d["modelV2"].append(dict(t=t, fid=int(M.frameId),
                                         lp=min(p[1:3]) if len(p) >= 3 else 1.0,
                                         lpl=p[1], lpr=p[2],
                                         dc=float(M.action.desiredCurvature),
                                         lx=lx, ly=ly, ry=ry, ds=ds))
            elif w == "carState":
                c = m.carState
                d["carState"].append((t, float(c.vEgo), float(c.steeringAngleDeg),
                                      bool(c.steeringPressed), float(c.steeringTorque),
                                      bool(c.leftBlinker), bool(c.rightBlinker), float(c.yawRate)))
            elif w == "carControl":
                cc = m.carControl
                d["carControl"].append((t, bool(cc.latActive), bool(cc.enabled),
                                        float(cc.actuators.curvature)))
            elif w == "carOutput":
                d["carOutput"].append((t, float(m.carOutput.actuatorsOutput.curvature)))
            elif w == "liveLocationKalman":
                g = m.liveLocationKalman
                try:
                    av = g.angularVelocityCalibrated
                    on = g.calibratedOrientationNED
                    if len(av.value) >= 3:
                        d["llk"].append((t, float(av.value[2]), bool(av.valid),
                                         float(on.value[2]) if len(on.value) >= 3 else np.nan))
                except Exception:
                    pass
    for k in d:
        d[k].sort(key=lambda r: r["t"] if isinstance(r, dict) else r[0])
    return d


def vm():
    from opendbc.car.vehicle_model import VehicleModel
    for m in LogReader(f"{ROOT}/explorer_st_logs/route_ce/{SEGS[0]}/rlog.zst"):
        try:
            if m.which() == "carParams":
                return VehicleModel(m.carParams)
        except Exception:
            continue
    raise RuntimeError("no carParams")


def lane_fit_curv(mv, xmax=60.0):
    """Quadratic fit of lane-center y(x) over 0..xmax. Returns (2*c2, c1, c0) in MODEL y sign."""
    x = mv["lx"]; yc = (mv["ly"] + mv["ry"]) / 2
    sel = x <= xmax
    if sel.sum() < 5:
        return np.nan, np.nan, np.nan
    c2, c1, c0 = np.polyfit(x[sel], yc[sel], 2)
    return 2 * c2, c1, c0


def main():
    VM = vm()
    d = load()
    cst = [r[0] for r in d["carState"]]
    cct = [r[0] for r in d["carControl"]]
    cot = [r[0] for r in d["carOutput"]]
    llt = [r[0] for r in d["llk"]]
    mvt = [r["t"] for r in d["modelV2"]]

    # ---------------- sign verification over active driving ------------------
    A = []  # angle_curv, dc, yawRate, llk_avz, off, v, t
    for mv in d["modelV2"]:
        t = mv["t"]
        i = _pv(cst, t); j = _pv(cct, t); k = _pv(llt, t)
        if i is None or j is None or k is None:
            continue
        (_, v, sa, sp, stq, lb, rb, yr) = d["carState"][i]
        (_, la, en, acur) = d["carControl"][j]
        (_, avz, avok, hdg) = d["llk"][k]
        if v < 13:
            continue
        ac = VM.calc_curvature(math.radians(sa), v, 0.0)
        off = (mv["ly"][0] + mv["ry"][0]) / 2
        A.append((ac, mv["dc"], yr, avz, off, v, t, sp, la))
    A = np.array([(a, b, c, e, f, g, h) for (a, b, c, e, f, g, h, sp, la) in A if la and not sp])
    ac, dc, yr, avz, off, v, tt = A.T
    def corr(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return np.corrcoef(x[m], y[m])[0, 1]
    print("=== SIGN VERIFICATION (active, unpressed, v>13, segs 1-3, n=%d) ===" % len(ac))
    print(f"  corr(angle_curv[VM], desiredCurvature)          = {corr(ac, dc):+.3f}")
    print(f"  corr(angle_curv[VM], carState.yawRate/v)        = {corr(ac, yr / v):+.3f}")
    print(f"  corr(angle_curv[VM], llk.angVelCalib[2]/v)      = {corr(ac, avz / v):+.3f}")
    # lane line y convention: mean of inner-left vs inner-right y[0]
    l0 = np.array([mv["ly"][0] for mv in d["modelV2"]])
    r0 = np.array([mv["ry"][0] for mv in d["modelV2"]])
    print(f"  laneLines[1](inner-LEFT ).y[0] mean = {np.nanmean(l0):+.2f} m")
    print(f"  laneLines[2](inner-RIGHT).y[0] mean = {np.nanmean(r0):+.2f} m")
    # offset dynamics vs curvature: d(off)/dt vs (achieved - road) curvature * v.
    doff = np.gradient(off, tt / 1e9)
    print(f"  corr(angle_curv, d(offset)/dt)                  = {corr(ac, doff):+.3f}")
    print(f"  corr(desiredCurvature, lane-fit 2*c2 @same t)   = ", end="")
    lf = []
    for mv in d["modelV2"]:
        t = mv["t"]
        i = _pv(cst, t)
        if i is None or d["carState"][i][1] < 13:
            continue
        c2x2, _, _ = lane_fit_curv(mv)
        lf.append((mv["dc"], c2x2))
    lf = np.array(lf)
    print(f"{corr(lf[:, 0], lf[:, 1]):+.3f}")

    # ------------------------- event timeline --------------------------------
    print("\n=== EVENT TIMELINE t0-10s .. t0+5s (t0=%.3f s mono), 0.25 s steps ===" % (T0 / 1e9))
    hdr = ("  t-t0 |  mph | laneP |  off_m | desCrv   | outCrv   | angCrv   | yawCrv   | laneFit  "
           "| desire      | blk | prs | lat | trq")
    print(hdr)
    rows = []
    DESIRES = ["none", "turnL", "turnR", "lcL", "lcR", "keepL", "keepR", "?7"]
    for step in np.arange(-10.0, 5.01, 0.25):
        t = T0 + step * 1e9
        im = _pv(mvt, t); ic = _pv(cst, t); jj = _pv(cct, t); ko = _pv(cot, t); kl = _pv(llt, t)
        if im is None or ic is None:
            continue
        mv = d["modelV2"][im]
        (_, v, sa, sp, stq, lb, rb, yr) = d["carState"][ic]
        (_, la, en, acur) = d["carControl"][jj] if jj is not None else (0, False, False, np.nan)
        oc = d["carOutput"][ko][1] if ko is not None else np.nan
        avz = d["llk"][kl][1] if kl is not None else np.nan
        angc = VM.calc_curvature(math.radians(sa), v, 0.0)
        yawc = (avz / v) if v > 1 else np.nan
        offm = (mv["ly"][0] + mv["ry"][0]) / 2
        c2x2, c1, c0 = lane_fit_curv(mv)
        dsi = int(np.argmax(mv["ds"])); dsp = mv["ds"][dsi]
        blk = ("L" if lb else "") + ("R" if rb else "") or "-"
        rows.append((step, v, mv["lp"], offm, mv["dc"], oc, angc, yawc, c2x2,
                     f"{DESIRES[dsi]}:{dsp:.2f}", blk, sp, la, stq))
        print(f"  {step:+5.2f} | {v*2.237:4.0f} | {mv['lp']:.3f} | {offm:+.3f} | {mv['dc']:+.5f} | "
              f"{oc:+.5f} | {angc:+.5f} | {yawc:+.5f} | {c2x2:+.5f} | {rows[-1][9]:<11s} | "
              f"{blk:>2s}  |  {'Y' if sp else '.'}  |  {'Y' if la else '.'}  | {stq:+5.0f}")

    # ---------------- pre-drift lane geometry + needed curvature -------------
    print("\n=== PRE-DRIFT ROAD GEOMETRY (window start t0-10..t0-4) ===")
    pre = [mv for mv in d["modelV2"] if T0 - 10e9 <= mv["t"] <= T0 - 4e9]
    for xmax in (40, 60, 90):
        cs = [lane_fit_curv(mv, xmax)[0] for mv in pre]
        print(f"  lane-center fit 2*c2, x<= {xmax:2d} m: mean={np.nanmean(cs):+.6f} "
              f"median={np.nanmedian(cs):+.6f} (model-y sign)")
    pre_llk = [(r[1], r[0]) for r in d["llk"] if T0 - 10e9 <= r[0] <= T0 - 4e9]
    pre_v = np.mean([r[1] for r in d["carState"] if T0 - 10e9 <= r[0] <= T0 - 4e9])
    print(f"  llk yaw curvature avz/v over same span: mean={np.mean([a for a, _ in pre_llk]) / pre_v:+.6f}")
    pre_ang = [VM.calc_curvature(math.radians(r[2]), r[1], 0.0)
               for r in d["carState"] if T0 - 10e9 <= r[0] <= T0 - 4e9]
    print(f"  angle curvature (VM) same span:         mean={np.mean(pre_ang):+.6f}")
    pre_dc = [mv["dc"] for mv in pre]
    print(f"  desiredCurvature same span:             mean={np.mean(pre_dc):+.6f}")

    print("\n=== NEEDED-CURVATURE SCALE ===")
    # offset excursion during drift: from timeline offsets
    drift = [(mv["t"], (mv["ly"][0] + mv["ry"][0]) / 2) for mv in d["modelV2"]
             if T0 - 3e9 <= mv["t"] <= T0 + 0.5e9]
    o0, o1 = drift[0][1], min(x[1] for x in drift) if drift[0][1] > drift[-1][1] else max(x[1] for x in drift)
    dT = (drift[-1][0] - drift[0][0]) / 1e9
    vref = 27.7
    for off_amt in (0.35, 0.72):
        for T in (1.5, 2.0, 3.0):
            k = 2 * off_amt / (vref * T) ** 2
            print(f"  |offset|={off_amt:.2f} m, horizon T={T:.1f}s  ->  kappa_need = {k:.6f} (1/m)")
    # implied sustained curvature DEFICIT from the drift kinematics (0.5*a*t^2)
    a = 2 * abs(drift[0][1] - min(x[1] for x in drift)) / dT ** 2
    print(f"  drift {drift[0][1]:+.2f}->{min(x[1] for x in drift):+.2f} m in {dT:.1f}s: "
          f"implied lateral accel {a:.3f} m/s^2 -> curvature deficit ~ {a / vref**2:.6f} (1/m)")


if __name__ == "__main__":
    main()
