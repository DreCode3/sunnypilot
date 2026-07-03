#!/usr/bin/env python3
"""Follow-ups for the route_ce decisive weak-correction event (t0=212.535s):
1. Lane-line ordering sanity: laneLines[0]/[3] y[0] means (expect ~-5 / ~+5 if +y=right).
2. Clean pre-drift road fit over t0-6..t0-3.5 (post-lane-change, pre-drift).
3. Lateral velocity of the drift + refined needed curvature (position+velocity terms).
4. Small-command EPAS following: achieved yaw-curvature vs carOutput.curvature ratio,
   binned by |cmd|, OLD build (c7) vs NEW build (ce, cf) — does the wheel deliver
   small (|k|<0.0015) curvature commands at highway speed?
5. desireState full vector + enabled/latActive over the drift window.
"""
import sys, glob, bisect, math
import numpy as np
ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader

T0 = 212.535e9


def _pv(ts, t):
    i = bisect.bisect_right(ts, t) - 1
    return i if i >= 0 else None


def load(route, segs=None):
    d = {"modelV2": [], "carState": [], "carControl": [], "carOutput": [], "llk": []}
    files = sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst"))
    if segs is not None:
        files = [f for f in files if f.split("/")[-2].split("--")[-1] in segs]
    for rl in files:
        try:
            lr = LogReader(rl)
        except Exception:
            continue
        for m in lr:
            try:
                w = m.which()
            except Exception:
                continue
            t = m.logMonoTime
            if w == "modelV2":
                M = m.modelV2
                ll = list(M.laneLines)
                if len(ll) < 4 or not len(ll[1].y):
                    continue
                p = list(M.laneLineProbs)
                d["modelV2"].append(dict(t=t, lp=min(p[1:3]),
                                         dc=float(M.action.desiredCurvature),
                                         y0=[ll[i].y[0] for i in range(4)],
                                         lx=np.array(ll[1].x),
                                         ly=np.array(ll[1].y), ry=np.array(ll[2].y),
                                         ds=list(M.meta.desireState)))
            elif w == "carState":
                c = m.carState
                d["carState"].append((t, float(c.vEgo), float(c.steeringAngleDeg),
                                      bool(c.steeringPressed), float(c.steeringTorque)))
            elif w == "carControl":
                d["carControl"].append((t, bool(m.carControl.latActive), bool(m.carControl.enabled)))
            elif w == "carOutput":
                d["carOutput"].append((t, float(m.carOutput.actuatorsOutput.curvature)))
            elif w == "liveLocationKalman":
                g = m.liveLocationKalman
                try:
                    if len(g.angularVelocityCalibrated.value) >= 3:
                        d["llk"].append((t, float(g.angularVelocityCalibrated.value[2])))
                except Exception:
                    pass
    for k in d:
        d[k].sort(key=lambda r: r["t"] if isinstance(r, dict) else r[0])
    return d


def main():
    ce = load("route_ce", segs={"1", "2", "3"})

    # 1. lane line ordering
    y0 = np.array([mv["y0"] for mv in ce["modelV2"]])
    print("=== 1. laneLines y[0] means (segs1-3, all frames) ===")
    for i in range(4):
        print(f"  laneLines[{i}].y[0]: mean={np.nanmean(y0[:, i]):+.2f}  median={np.nanmedian(y0[:, i]):+.2f}")

    # 2. clean pre-drift road fit
    print("\n=== 2. PRE-DRIFT (clean, t0-6..t0-3.5) road curvature, model-y(right+) sign ===")
    pre = [mv for mv in ce["modelV2"] if T0 - 6e9 <= mv["t"] <= T0 - 3.5e9]
    for xmax in (40, 60, 90):
        cs = []
        for mv in pre:
            x = mv["lx"]; yc = (mv["ly"] + mv["ry"]) / 2
            sel = x <= xmax
            c2, c1, c0 = np.polyfit(x[sel], yc[sel], 2)
            cs.append(2 * c2)
        print(f"  x<={xmax:2d} m: mean={np.nanmean(cs):+.6f}  median={np.nanmedian(cs):+.6f}  "
              f"|max|={np.nanmax(np.abs(cs)):.6f}")

    # 3. lateral velocity + refined needed curvature
    print("\n=== 3. DRIFT KINEMATICS + NEEDED CURVATURE ===")
    offs = [(mv["t"], (mv["ly"][0] + mv["ry"][0]) / 2) for mv in ce["modelV2"]
            if T0 - 6e9 <= mv["t"] <= T0 + 0.5e9]
    ot = np.array([o[0] for o in offs]) / 1e9
    oy = np.array([o[1] for o in offs])
    t0s = T0 / 1e9
    for (a, b, tag) in [(-5.5, -3.0, "recentering approach (left of center)"),
                        (-3.0, 0.0, "the drift (center -> -0.72)"),
                        (-1.0, 0.0, "final second")]:
        m = (ot >= t0s + a) & (ot <= t0s + b)
        if m.sum() > 3:
            vlat = np.polyfit(ot[m], oy[m], 1)[0]
            print(f"  [{a:+.1f}..{b:+.1f}s] offset {oy[m][0]:+.2f}->{oy[m][-1]:+.2f} m, "
                  f"v_lat={vlat:+.3f} m/s  ({tag})")
    v = 27.7
    print("  needed curvature magnitude (position 2|y|/(vT)^2 + velocity 2|ydot|/(v^2 T)):")
    for (y_off, ydot, when) in [(0.02, 0.21, "at center-crossing t0-3s"),
                                (0.40, 0.24, "mid-drift t0-1s"),
                                (0.72, 0.25, "at override t0")]:
        for T in (1.5, 2.0):
            k = 2 * y_off / (v * T) ** 2 + 2 * ydot / (v ** 2 * T)
            print(f"    {when:26s} T={T:.1f}s: kappa_need~{k:.6f}")

    # 4. small-command following ratio, OLD vs NEW
    print("\n=== 4. EPAS FOLLOWING of SMALL commands: achieved yaw-curv (llk avz/v, right+) vs carOutput.curvature ===")
    print("      (latActive, unpressed, v>18 m/s; signed ratio median per |cmd| bin)")
    for route in ("route_c7", "route_ce", "route_cf"):
        d = load(route) if route != "route_ce" else ce
        cst = [r[0] for r in d["carState"]]; cct = [r[0] for r in d["carControl"]]
        llt = [r[0] for r in d["llk"]]
        samp = []
        for (t, cmd) in d["carOutput"]:
            i = _pv(cst, t); j = _pv(cct, t); k = _pv(llt, t)
            if i is None or j is None or k is None:
                continue
            (_, v_, sa, sp, stq) = d["carState"][i]
            (_, la, en) = d["carControl"][j]
            if not la or sp or v_ < 18:
                continue
            avz = d["llk"][k][1]
            samp.append((cmd, avz / v_))
        samp = np.array(samp)
        if not len(samp):
            print(f"  {route}: no samples"); continue
        print(f"  {route} (n={len(samp)}):")
        for lo, hi in [(0.0002, 0.0005), (0.0005, 0.001), (0.001, 0.0015), (0.0015, 0.003), (0.003, 0.01)]:
            m = (np.abs(samp[:, 0]) >= lo) & (np.abs(samp[:, 0]) < hi)
            if m.sum() < 10:
                print(f"    |cmd| {lo:.4f}-{hi:.4f}: n={m.sum()} (too few)"); continue
            r = samp[m, 1] / samp[m, 0]
            print(f"    |cmd| {lo:.4f}-{hi:.4f}: n={m.sum():5d}  median ratio={np.median(r):+.2f}  "
                  f"IQR=[{np.percentile(r,25):+.2f},{np.percentile(r,75):+.2f}]")

    # 5. desire vector + enabled over the drift window
    print("\n=== 5. desireState + enabled/latActive, t0-4..t0+1 (1 s steps) ===")
    cct = [r[0] for r in ce["carControl"]]
    mvt = [mv["t"] for mv in ce["modelV2"]]
    NAMES = ["none", "turnL", "turnR", "lcL", "lcR", "keepL", "keepR", "x7"]
    for step in np.arange(-4, 1.01, 1.0):
        t = T0 + step * 1e9
        mv = ce["modelV2"][_pv(mvt, t)]
        (_, la, en) = ce["carControl"][_pv(cct, t)]
        ds = ", ".join(f"{NAMES[i]}={p:.3f}" for i, p in enumerate(mv["ds"]) if p > 0.01)
        print(f"  t0{step:+.0f}s: latActive={la} enabled={en} desire[{ds}]")


if __name__ == "__main__":
    main()
