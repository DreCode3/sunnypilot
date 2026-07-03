#!/usr/bin/env python3
"""AE / exposure audit across OLD vs NEW builds.

Per route (c5, c7, b5, 7f = OLD; ce, cf = NEW), while moving (vEgo > 3 m/s):
  - roadCameraState.measuredGreyFraction: median, p90, p99, %>0.42, max
  - targetGreyFraction: median
  - integLines: median, p95;  gain: median, p95
  - AE response speed: for each large integLines step (max/min ratio > 3x within a
    trailing 1 s window), frames until |measuredGrey - targetGrey| <= 0.05. Median.
Also: initData (osVersion, version, gitCommit, gitBranch, kernelVersion) for one
OLD and one NEW route, and a +/-10 s exposure timeline around the decisive
weak-correction event on route_ce (T0 = 212.535 s mono, segments 1-3).
"""
import os
import sys
import glob
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/opendbc_repo")

ROUTES = {
    "route_c5": "OLD/Nevada",
    "route_c7": "OLD/Nevada",
    "route_b5": "OLD/CD210",
    "route_7f": "OLD/OPM7",
    "route_ce": "NEW/Nevada",
    "route_cf": "NEW/Nevada",
}

CE_T0 = 212.535e9  # mono ns, decisive weak-correction event (ce_decisive_locate.py cluster 3)


def seg_sort_key(p):
    return int(p.rstrip("/").rsplit("--", 1)[-1])


def load_segment(args):
    """Return (cam, vego, init) arrays for one segment rlog."""
    from openpilot.tools.lib.logreader import LogReader
    path, want_init = args
    cam = []    # (t, frameId, measGrey, targGrey, integLines, gain, sensor)
    vego = []   # (t, vEgo)
    init = None
    try:
        lr = LogReader(path)
    except Exception as e:
        return path, [], [], None, f"open fail: {e}"
    err = None
    for m in lr:
        try:
            w = m.which()
        except Exception:
            continue
        try:
            if w == "roadCameraState":
                c = m.roadCameraState
                cam.append((m.logMonoTime, int(c.frameId),
                            float(c.measuredGreyFraction), float(c.targetGreyFraction),
                            int(c.integLines), float(c.gain), str(c.sensor)))
            elif w == "carState":
                vego.append((m.logMonoTime, float(m.carState.vEgo)))
            elif w == "initData" and want_init:
                d = m.initData
                init = {
                    "osVersion": str(d.osVersion),
                    "version": str(d.version),
                    "gitCommit": str(d.gitCommit),
                    "gitBranch": str(d.gitBranch),
                    "kernelVersion": str(d.kernelVersion),
                    "deviceType": str(d.deviceType),
                }
        except Exception as e:
            err = str(e)
            continue
    return path, cam, vego, init, err


def analyze_route(route, cam, vego):
    cam.sort(key=lambda r: r[0])
    vego.sort(key=lambda r: r[0])
    t = np.array([r[0] for r in cam], dtype=np.float64)
    meas = np.array([r[2] for r in cam])
    targ = np.array([r[3] for r in cam])
    integ = np.array([r[4] for r in cam], dtype=np.float64)
    gain = np.array([r[5] for r in cam])
    sensors = set(r[6] for r in cam[:: max(1, len(cam) // 50)])

    vt = np.array([r[0] for r in vego], dtype=np.float64)
    vv = np.array([r[1] for r in vego])
    v_at_cam = np.interp(t, vt, vv)
    moving = v_at_cam > 3.0

    mm, tg, ig, gn, tmov = meas[moving], targ[moving], integ[moving], gain[moving], t[moving]

    stats = {
        "route": route,
        "build": ROUTES[route],
        "sensor": ",".join(sorted(sensors)),
        "n_frames_total": len(meas),
        "n_frames_moving": int(moving.sum()),
        "grey_median": float(np.median(mm)),
        "grey_p90": float(np.percentile(mm, 90)),
        "grey_p99": float(np.percentile(mm, 99)),
        "grey_max": float(mm.max()),
        "grey_pct_gt_0.42": float((mm > 0.42).mean() * 100),
        "target_median": float(np.median(tg)),
        "target_p99": float(np.percentile(tg, 99)),
        "integ_median": float(np.median(ig)),
        "integ_p95": float(np.percentile(ig, 95)),
        "gain_median": float(np.median(gn)),
        "gain_p95": float(np.percentile(gn, 95)),
        "grey_minus_target_median": float(np.median(mm - tg)),
        "grey_minus_target_p99": float(np.percentile(mm - tg, 99)),
    }

    # AE response speed: large integLines steps (>3x within trailing 1 s), on ALL
    # frames (steps can start while stopped too, but require moving at trigger).
    # Trailing-window max/min ratio; group triggers < 2 s apart into one event.
    events = []
    n = len(t)
    i0 = 0
    last_trig = -1e18
    recov = []
    for i in range(n):
        while t[i] - t[i0] > 1.0e9:
            i0 += 1
        if i0 == i:
            continue
        w_int = integ[i0:i + 1]
        wmin = w_int.min()
        if wmin <= 0:
            continue
        ratio = w_int.max() / wmin
        if ratio > 3.0 and (t[i] - last_trig) > 2.0e9 and moving[i]:
            last_trig = t[i]
            # recovery: frames from trigger until |meas - targ| <= 0.05
            k = i
            nf = None
            while k < n and (t[k] - t[i]) < 30e9:
                if abs(meas[k] - targ[k]) <= 0.05:
                    nf = k - i
                    break
                k += 1
            events.append({"t": float(t[i] / 1e9), "ratio": float(ratio),
                           "recov_frames": nf})
            if nf is not None:
                recov.append(nf)
    stats["ae_steps_n"] = len(events)
    stats["ae_recov_frames_median"] = float(np.median(recov)) if recov else None
    stats["ae_recov_frames_p90"] = float(np.percentile(recov, 90)) if recov else None
    stats["ae_steps_unrecovered_30s"] = sum(1 for e in events if e["recov_frames"] is None)
    return stats, events, (t, meas, targ, integ, gain, v_at_cam)


def main():
    tasks = []
    for route in ROUTES:
        segs = sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*--*"), key=seg_sort_key)
        for j, s in enumerate(segs):
            p = os.path.join(s, "rlog.zst")
            if os.path.exists(p):
                tasks.append((p, j == 0))

    results = {}
    inits = {}
    errs = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for path, cam, vego, init, err in ex.map(load_segment, tasks):
            route = path.split("explorer_st_logs/")[1].split("/")[0]
            results.setdefault(route, {"cam": [], "vego": []})
            results[route]["cam"].extend(cam)
            results[route]["vego"].extend(vego)
            if init:
                inits[route] = init
            if err:
                errs.append((path, err))

    print("=== initData per route ===")
    for route in ROUTES:
        d = inits.get(route, {})
        print(f"{route} [{ROUTES[route]}]: os={d.get('osVersion')} version={d.get('version')} "
              f"git={d.get('gitCommit', '')[:12]} branch={d.get('gitBranch')} "
              f"kernel={d.get('kernelVersion')} device={d.get('deviceType')}")

    print("\n=== whole-route moving-frame exposure stats ===")
    hdr = ("route      build       sensor    nMov   greyMed greyP90 greyP99 greyMax %>0.42 "
           "targMed  d(g-t)Med integMed integP95 gainMed gainP95 aeSteps recovMedF recovP90F unrec")
    print(hdr)
    all_stats = {}
    route_arrays = {}
    for route in ROUTES:
        st, events, arrs = analyze_route(route, results[route]["cam"], results[route]["vego"])
        all_stats[route] = {"stats": st, "events": events}
        route_arrays[route] = arrs
        print(f"{route:<10} {st['build']:<11} {st['sensor'].split('.')[-1]:<9} {st['n_frames_moving']:>6} "
              f"{st['grey_median']:7.3f} {st['grey_p90']:7.3f} {st['grey_p99']:7.3f} {st['grey_max']:7.3f} "
              f"{st['grey_pct_gt_0.42']:6.2f} {st['target_median']:7.3f} {st['grey_minus_target_median']:9.3f} "
              f"{st['integ_median']:8.0f} {st['integ_p95']:8.0f} {st['gain_median']:7.2f} {st['gain_p95']:7.2f} "
              f"{st['ae_steps_n']:7d} "
              f"{st['ae_recov_frames_median'] if st['ae_recov_frames_median'] is not None else 'n/a':>9} "
              f"{st['ae_recov_frames_p90'] if st['ae_recov_frames_p90'] is not None else 'n/a':>9} "
              f"{st['ae_steps_unrecovered_30s']:5d}")

    print("\n=== AE step events detail (per route) ===")
    for route in ROUTES:
        evs = all_stats[route]["events"]
        det = ", ".join(f"t={e['t']:.0f}s r={e['ratio']:.1f} rec={e['recov_frames']}" for e in evs)
        print(f"{route}: {det if det else '(none)'}")

    # decisive event on route_ce
    print("\n=== route_ce decisive event (T0=212.535 s mono, +/-10 s; override ~T0+3.2s) ===")
    t, meas, targ, integ, gain, v = route_arrays["route_ce"]
    m = (t >= CE_T0 - 10e9) & (t <= CE_T0 + 13e9)
    idx = np.where(m)[0]
    print("  t-T0(s)  vEgo(mph)  measGrey  targGrey  |m-t|  integLines  gain")
    for k in idx[::5]:  # every 5th frame (~0.25 s at 20 fps)
        print(f"  {(t[k]-CE_T0)/1e9:7.2f} {v[k]*2.23694:9.1f} {meas[k]:9.3f} {targ[k]:9.3f} "
              f"{abs(meas[k]-targ[k]):6.3f} {integ[k]:10.0f} {gain[k]:6.2f}")
    w = (t >= CE_T0 - 10e9) & (t <= CE_T0 + 13e9)
    print(f"  window summary: measGrey min={meas[w].min():.3f} max={meas[w].max():.3f} "
          f"median={np.median(meas[w]):.3f}; |m-t| max={np.abs(meas[w]-targ[w]).max():.3f}; "
          f"integ min={integ[w].min():.0f} max={integ[w].max():.0f}; "
          f"gain min={gain[w].min():.2f} max={gain[w].max():.2f}")

    if errs:
        print(f"\n(non-fatal read errors in {len(errs)} segments)")

    out = {r: all_stats[r]["stats"] for r in ROUTES}
    scratch = os.environ.get("SCRATCH", "/private/tmp/claude-501/-Users-dregilley-Documents-GitHub-sunnypilot/29f90cf8-9f2a-4abc-94c6-33e5771ceae1/scratchpad")
    with open(os.path.join(scratch, "ae_audit_stats.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
