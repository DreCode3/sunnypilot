#!/usr/bin/env python3
"""Locate the decisive weak-correction override cluster on route_ce
(the one with 3s-min laneProb ~0.96 at ~62 mph), print its t0 + segment,
and probe which pose/orientation messages exist on this NEW-build log.
"""
import sys, glob, bisect
import numpy as np
ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader


def _pv(arr, ts, t):
    i = bisect.bisect_right(ts, t) - 1
    return arr[i] if i >= 0 else None


def main():
    route = "route_ce"
    want = {"modelV2", "carState", "carControl"}
    out = {k: [] for k in want}
    seg_of_t = []  # (t_start, segname) to map t0 -> segment
    which_counts = {}
    for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst")):
        seg = rl.split("/")[-2]
        first_t = None
        for m in LogReader(rl):
            try:
                w = m.which()
            except Exception:
                continue
            t = m.logMonoTime
            if first_t is None:
                first_t = t
                seg_of_t.append((t, seg))
            which_counts[w] = which_counts.get(w, 0) + 1
            if w not in want:
                continue
            if w == "modelV2":
                M = m.modelV2; p = list(M.laneLineProbs); ll = list(M.laneLines)
                lY = ll[1].y[0] if len(ll) >= 2 and len(ll[1].y) else np.nan
                rY = ll[2].y[0] if len(ll) >= 3 and len(ll[2].y) else np.nan
                out[w].append((t, int(M.frameId), min(p[1:3]) if len(p) >= 3 else 1.0,
                               float(M.action.desiredCurvature), (lY + rY) / 2))
            elif w == "carState":
                c = m.carState
                out[w].append((t, float(c.vEgo), float(c.steeringAngleDeg), bool(c.steeringPressed),
                               float(c.steeringTorque), bool(c.leftBlinker), bool(c.rightBlinker),
                               float(c.yawRate)))
            elif w == "carControl":
                out[w].append((t, bool(m.carControl.latActive), bool(m.carControl.enabled)))
    for k in out:
        out[k].sort()
    seg_of_t.sort()

    print("pose/orientation-ish message availability on route_ce:")
    for k in sorted(which_counts):
        kl = k.lower()
        if any(s in kl for s in ("pose", "location", "kalman", "orientation", "gnss", "gps", "imu",
                                 "accel", "gyro", "calibration")):
            print(f"  {k}: {which_counts[k]}")

    MV = out["modelV2"]; cct = [x[0] for x in out["carControl"]]
    bt = sorted(t for (t, v, sa, sp, st, lb, rb, yr) in out["carState"] if lb or rb)
    ov = [(t, st, v) for (t, v, sa, sp, st, lb, rb, yr) in out["carState"]
          if sp and abs(st) > 1.8 and v > 13 and (_pv(out["carControl"], cct, t) or (0, False))[1]]
    clusters = []; cur = []
    for o in ov:
        if cur and o[0] - cur[-1][0] > 2.5e9:
            clusters.append(cur); cur = []
        cur.append(o)
    if cur:
        clusters.append(cur)
    t_all0 = MV[0][0]
    print(f"\n=== {route}: {len(clusters)} override clusters ===")
    print("  idx |   t0(mono s) | t0-rel(s) | segment | mph | laneChg | minLaneProb3s | offset drift | ovrTrq")
    for i, cl in enumerate(clusters):
        t0 = cl[0][0]; v = cl[0][2]; drv = cl[0][1]
        lo = bisect.bisect_left(bt, t0 - 6e9); hi = bisect.bisect_right(bt, t0 + 1e9)
        blk = hi > lo
        pre = [r for r in MV if t0 - 3e9 <= r[0] < t0]
        if not pre:
            continue
        lp = min(r[2] for r in pre)
        offs = [r[4] for r in pre if np.isfinite(r[4])]
        od = (offs[-1] - offs[0]) if offs else np.nan
        j = bisect.bisect_right([s[0] for s in seg_of_t], t0) - 1
        seg = seg_of_t[j][1] if j >= 0 else "?"
        print(f"  {i:3d} | {t0/1e9:12.3f} | {(t0-t_all0)/1e9:9.1f} | {seg} | {v*2.237:4.0f} | "
              f"{'LANE-CHG' if blk else '   -    '} | {lp:.2f} | "
              f"{offs[0] if offs else float('nan'):+.2f}->{offs[-1] if offs else float('nan'):+.2f} "
              f"({od:+.2f}m) | {drv:+.1f}")


if __name__ == "__main__":
    main()
