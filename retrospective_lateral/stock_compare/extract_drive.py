#!/usr/bin/env python3
"""Extract per-modelV2-frame lateral records for a drive (stock or custom) -> npz cache.

Signals per frame (model cadence ~20Hz): t, lat, lon, vEgo, steerDeg, yawRate,
steeringPressed, latActive, enabled, cmd_curv (carOutput, +=RIGHT), model_curv
(modelV2 action.desiredCurvature), ach_curv (achieved via THIS build's VehicleModel,
+=RIGHT), offset (laneLine midpoint, +=car LEFT of center), innerProb, gps_headrate (deg/s,
+=clockwise/RIGHT, from GPS bearing, for model-independent curve classification).

Each build uses its OWN carParams (stock steerRatio 16.8 vs custom 17.2) — critical for a
fair achieved-curvature comparison. Position from liveLocationKalman.positionGeodetic
(custom) OR gpsLocationExternal (stock), whichever the log carries.

RUN: PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
       retrospective_lateral/stock_compare/extract_drive.py <route_glob> <out.npz>
e.g. ... extract_drive.py "explorer_st_logs/stock/00000002--5da5840d8d" cache/stock_marietta.npz
"""
import sys, glob, bisect, math, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader
from opendbc.car.vehicle_model import VehicleModel


def _seg_num(p):
    m = re.search(r'--(\d+)/rlog', p)
    return int(m.group(1)) if m else 0


def _carparams(seg_paths):
    for rl in seg_paths[:3]:
        try:
            for m in LogReader(rl):
                try:
                    if m.which() == "carParams":
                        return m.carParams
                except Exception:
                    continue
        except Exception:
            continue
    return None


def _extract_seg(args):
    rl, sr, wb = args
    VM = VehicleModel_from(sr, wb)
    mv = []; cs = []; co = []; cc = []; gps = []
    try:
        lr = LogReader(rl)
    except Exception:
        return []
    for m in lr:
        try:
            w = m.which()
        except Exception:
            continue
        t = m.logMonoTime * 1e-9
        if w == "modelV2":
            M = m.modelV2; ll = list(M.laneLines); p = list(M.laneLineProbs)
            if len(ll) >= 3 and len(ll[1].y) and len(ll[2].y):
                off = (ll[1].y[0] + ll[2].y[0]) / 2.0
            else:
                off = np.nan
            inner = min(p[1], p[2]) if len(p) >= 3 else 1.0
            mv.append((t, inner, off, float(M.action.desiredCurvature)))
        elif w == "carState":
            c = m.carState
            cs.append((t, float(c.vEgo), float(c.steeringAngleDeg), bool(c.steeringPressed), float(c.yawRate)))
        elif w == "carControl":
            cc.append((t, bool(m.carControl.latActive), bool(m.carControl.enabled)))
        elif w == "carOutput":
            co.append((t, float(m.carOutput.actuatorsOutput.curvature)))
        elif w == "liveLocationKalman":
            try:
                pg = m.liveLocationKalman.positionGeodetic
                if pg.valid and len(pg.value) >= 2 and abs(pg.value[0]) > 1:
                    gps.append((t, float(pg.value[0]), float(pg.value[1])))
            except Exception:
                pass
        elif w == "gpsLocationExternal":
            g = m.gpsLocationExternal
            try:
                if g.hasFix and g.horizontalAccuracy < 25 and abs(g.latitude) > 1:
                    gps.append((t, float(g.latitude), float(g.longitude)))
            except Exception:
                pass
    cs.sort(); co.sort(); cc.sort(); gps.sort()
    cst = [x[0] for x in cs]; cot = [x[0] for x in co]; cct = [x[0] for x in cc]; gpst = [x[0] for x in gps]

    def pv(arr, ts, t):
        i = bisect.bisect_right(ts, t) - 1
        return arr[i] if i >= 0 else None

    def gps_at(t):
        if not gps:
            return (np.nan, np.nan)
        i = bisect.bisect_right(gpst, t) - 1
        if i < 0:
            return (gps[0][1], gps[0][2])
        if i + 1 < len(gps) and gpst[i + 1] - gpst[i] < 3.0:
            t0, la0, lo0 = gps[i]; t1, la1, lo1 = gps[i + 1]
            f = (t - t0) / max(t1 - t0, 1e-6); f = min(max(f, 0), 1)
            return (la0 + f * (la1 - la0), lo0 + f * (lo1 - lo0))
        return (gps[i][1], gps[i][2])

    out = []
    for (t, inner, off, mcurv) in mv:
        st = pv(cs, cst, t); cc_ = pv(cc, cct, t); c_o = pv(co, cot, t)
        if st is None or cc_ is None:
            continue
        v, sa, sp, yr = st[1], st[2], st[3], st[4]
        lat, en = cc_[1], cc_[2]
        cmd = c_o[1] if c_o else np.nan
        ach = -VM.calc_curvature(math.radians(sa), max(v, 1.0), 0.0)  # +=RIGHT
        la, lo = gps_at(t)
        out.append((t, la, lo, v, sa, yr, 1.0 if sp else 0.0, 1.0 if lat else 0.0,
                    1.0 if en else 0.0, cmd, mcurv, ach, off, inner))
    return out


# Each drive uses its OWN logged carParams (set into _BASE_CP via the pool initializer),
# so steerRatio/wheelbase already reflect that build (stock 16.8 vs custom 17.2). No override.
def VehicleModel_from(steer_ratio, wheelbase):
    global _BASE_CP
    return VehicleModel(_BASE_CP)


_BASE_CP = None


def main():
    route_glob = sys.argv[1]
    out_npz = sys.argv[2]
    segs = sorted(glob.glob(f"{route_glob}--*/rlog.zst"), key=_seg_num)
    if not segs:
        segs = sorted(glob.glob(f"{route_glob}/*/rlog.zst"), key=_seg_num)
    assert segs, f"no rlogs for {route_glob}"
    cp = _carparams(segs)
    assert cp is not None, "no carParams"
    sr, wb = float(cp.steerRatio), float(cp.wheelbase)
    global _BASE_CP
    _BASE_CP = cp
    print(f"{route_glob}: {len(segs)} segs, steerRatio={sr:.2f} wheelbase={wb:.3f} fp={cp.carFingerprint}")

    rows = []
    with ProcessPoolExecutor(max_workers=12, initializer=_init_worker, initargs=(cp,)) as ex:
        for seg_rows in ex.map(_extract_seg, [(s, sr, wb) for s in segs]):
            rows.extend(seg_rows)
    rows.sort()
    arr = np.array(rows, dtype=np.float64)
    cols = "t lat lon vEgo steerDeg yawRate pressed latActive enabled cmd_curv model_curv ach_curv offset innerProb".split()
    np.savez(out_npz, data=arr, cols=np.array(cols), steerRatio=sr, wheelbase=wb, fp=str(cp.carFingerprint))
    print(f"  -> {out_npz}: {len(arr)} frames, {np.sum(arr[:,7]>0.5)} latActive, "
          f"{np.sum(np.isfinite(arr[:,1]))} with GPS")


def _init_worker(cp):
    global _BASE_CP
    _BASE_CP = cp


if __name__ == "__main__":
    main()
