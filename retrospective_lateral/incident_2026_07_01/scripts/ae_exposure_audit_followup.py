#!/usr/bin/env python3
"""Follow-up to ae_exposure_audit.py:
  (a) where/when whole-drive maxGrey happens per route + blowout frame counts
      (moving frames with measuredGrey > 0.55 / > 0.48), and cluster span
  (b) AE tracking-error metric per route: % moving frames |meas - targ| > 0.10
      and > 0.05 (lighting-robust AE quality)
  (c) matched-severity AE step recovery: only steps with trailing-1s ratio in
      [3.0, 4.0] AND isolated (no ratio>3 trigger in prior 15 s) to strip the
      overpass-cluster severity confound
"""
import os
import sys
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/opendbc_repo")

ROUTES = {
    "route_c5": "OLD/Nevada", "route_c7": "OLD/Nevada", "route_b5": "OLD/CD210",
    "route_7f": "OLD/OPM7", "route_ce": "NEW/Nevada", "route_cf": "NEW/Nevada",
}


def seg_sort_key(p):
    return int(p.rstrip("/").rsplit("--", 1)[-1])


def load_segment(path):
    from openpilot.tools.lib.logreader import LogReader
    cam, vego = [], []
    for m in LogReader(path):
        try:
            w = m.which()
        except Exception:
            continue
        try:
            if w == "roadCameraState":
                c = m.roadCameraState
                cam.append((m.logMonoTime, float(c.measuredGreyFraction),
                            float(c.targetGreyFraction), int(c.integLines)))
            elif w == "carState":
                vego.append((m.logMonoTime, float(m.carState.vEgo)))
        except Exception:
            continue
    return path, cam, vego


def main():
    tasks = []
    for route in ROUTES:
        for s in sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*--*"), key=seg_sort_key):
            p = os.path.join(s, "rlog.zst")
            if os.path.exists(p):
                tasks.append(p)
    data = {r: {"cam": [], "vego": []} for r in ROUTES}
    with ProcessPoolExecutor(max_workers=14) as ex:
        for path, cam, vego in ex.map(load_segment, tasks):
            route = path.split("explorer_st_logs/")[1].split("/")[0]
            data[route]["cam"].extend(cam)
            data[route]["vego"].extend(vego)

    print("route      build       |m-t|>0.10%  |m-t|>0.05%  nGrey>0.55 nGrey>0.48  maxGrey  t(max)s  blowoutSpan(s)")
    for route in ROUTES:
        cam = sorted(data[route]["cam"])
        veg = sorted(data[route]["vego"])
        t = np.array([r[0] for r in cam], dtype=np.float64)
        meas = np.array([r[1] for r in cam])
        targ = np.array([r[2] for r in cam])
        vt = np.array([r[0] for r in veg], dtype=np.float64)
        vv = np.array([r[1] for r in veg])
        mv = np.interp(t, vt, vv) > 3.0
        mm, tg, tm = meas[mv], targ[mv], t[mv]
        err = np.abs(mm - tg)
        imax = int(np.argmax(mm))
        hi55 = mm > 0.55
        hi48 = mm > 0.48
        span = (tm[hi55].max() - tm[hi55].min()) / 1e9 if hi55.any() else 0.0
        print(f"{route:<10} {ROUTES[route]:<11} {100*(err>0.10).mean():11.3f} {100*(err>0.05).mean():12.3f} "
              f"{int(hi55.sum()):10d} {int(hi48.sum()):10d} {mm.max():8.3f} {tm[imax]/1e9:8.1f} {span:10.1f}")

    print("\nmatched-severity isolated AE steps (ratio 3.0-4.0, no prior trigger within 15 s):")
    print("route      t(s)   ratio  recovFrames(|m-t|<=0.05)")
    for route in ROUTES:
        cam = sorted(data[route]["cam"])
        veg = sorted(data[route]["vego"])
        t = np.array([r[0] for r in cam], dtype=np.float64)
        meas = np.array([r[1] for r in cam])
        targ = np.array([r[2] for r in cam])
        integ = np.array([r[3] for r in cam], dtype=np.float64)
        vt = np.array([r[0] for r in veg], dtype=np.float64)
        vv = np.array([r[1] for r in veg])
        mv = np.interp(t, vt, vv) > 3.0
        n = len(t)
        i0 = 0
        trig_times = []
        rows = []
        for i in range(n):
            while t[i] - t[i0] > 1.0e9:
                i0 += 1
            if i0 == i:
                continue
            w = integ[i0:i + 1]
            if w.min() <= 0:
                continue
            ratio = w.max() / w.min()
            if ratio > 3.0 and (not trig_times or t[i] - trig_times[-1] > 2.0e9) and mv[i]:
                isolated = not any(t[i] - tt < 15.0e9 for tt in trig_times[:-1] + trig_times[-1:])
                trig_times.append(t[i])
                if ratio <= 4.0 and isolated:
                    k, nf = i, None
                    while k < n and (t[k] - t[i]) < 30e9:
                        if abs(meas[k] - targ[k]) <= 0.05:
                            nf = k - i
                            break
                        k += 1
                    rows.append((t[i] / 1e9, ratio, nf))
        for r in rows:
            print(f"{route:<10} {r[0]:6.0f} {r[1]:6.2f}  {r[2]}")
        if not rows:
            print(f"{route:<10} (none)")


if __name__ == "__main__":
    main()
