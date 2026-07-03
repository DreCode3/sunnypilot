#!/usr/bin/env python3
"""Stress-test of the headline recovery-lag metric (incident_analyses.py recovery_lag()).

Reproduces the exact event extraction (collapse = min(laneLineProbs[1:3]) < 0.15,
recovery = > 0.7) for routes c7, c5 (OLD Nevada), ce, cf (NEW Nevada), but records
per-event covariates so the metric can be scene/exposure-matched:
  duration, vEgo@collapse, measuredGreyFraction@collapse and @recovery (nearest
  roadCameraState), integLines@collapse, blinker within +-3 s (lane-change proxy),
  min laneProb reached (depth), and whether the event spans a modelV2 gap >1 s.

Analyses:
  1) per-event table for each route
  2) per-route medians before/after excluding lane-change-associated events
  3) covariate-matched pairwise comparison (grey +-0.08, vEgo +-5 m/s, depth +-0.05)
  4) Mann-Whitney U + KS on excluded-set durations, old-pooled vs new-pooled

RUN: cd <repo> && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
       retrospective_lateral/incident_2026_07_01/scripts/recovery_lag_stress_test.py
"""
import sys, glob, bisect, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")

ROUTES = [("route_c7", "OLD"), ("route_c5", "OLD"), ("route_ce", "NEW"), ("route_cf", "NEW")]
CACHE = os.environ.get("RLST_CACHE",
    "/private/tmp/claude-501/-Users-dregilley-Documents-GitHub-sunnypilot/29f90cf8-9f2a-4abc-94c6-33e5771ceae1/scratchpad/recovery_lag_events.json")


def load_route(route):
    """Load modelV2 laneprob series, roadCameraState, and blinker/vEgo series."""
    from openpilot.tools.lib.logreader import LogReader
    mv, cam, cs = [], [], []
    for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst")):
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
                p = list(m.modelV2.laneLineProbs)
                mv.append((t, min(p[1:3]) if len(p) >= 3 else 1.0))
            elif w == "roadCameraState":
                r = m.roadCameraState
                try:
                    cam.append((t, float(r.measuredGreyFraction), int(r.integLines)))
                except Exception:
                    pass
            elif w == "carState":
                c = m.carState
                cs.append((t, float(c.vEgo), bool(c.leftBlinker) or bool(c.rightBlinker)))
    mv.sort(); cam.sort(); cs.sort()
    return mv, cam, cs


def extract_events(route):
    mv, cam, cs = load_route(route)
    camt = [x[0] for x in cam]
    cst = [x[0] for x in cs]
    blink_t = [t for (t, v, b) in cs if b]  # times blinker on

    def nearest_cam(t):
        i = bisect.bisect_left(camt, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(cam):
                if best is None or abs(cam[j][0] - t) < abs(cam[best][0] - t):
                    best = j
        return cam[best] if best is not None else (None, np.nan, -1)

    def nearest_cs(t):
        i = bisect.bisect_left(cst, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(cs):
                if best is None or abs(cs[j][0] - t) < abs(cs[best][0] - t):
                    best = j
        return cs[best] if best is not None else (None, np.nan, False)

    events = []
    i, n = 0, len(mv)
    while i < n:
        if mv[i][1] < 0.15:
            j = i
            gap = False
            while j < n and mv[j][1] < 0.7:
                if j > i and (mv[j][0] - mv[j - 1][0]) > 1e9:
                    gap = True
                j += 1
            if j < n:
                t0, t1 = mv[i][0], mv[j][0]
                dur = (t1 - t0) / 1e9
                depth = min(x[1] for x in mv[i:j])
                _, g0, il0 = nearest_cam(t0)
                _, g1, _ = nearest_cam(t1)
                _, v0, _ = nearest_cs(t0)
                # blinker within +-3 s of collapse onset (primary, per task wording)
                lo = bisect.bisect_left(blink_t, t0 - 3e9)
                hi = bisect.bisect_right(blink_t, t0 + 3e9)
                blk_onset = hi > lo
                # sensitivity: blinker anywhere in [collapse-3s, recovery+3s]
                lo2 = bisect.bisect_left(blink_t, t0 - 3e9)
                hi2 = bisect.bisect_right(blink_t, t1 + 3e9)
                blk_span = hi2 > lo2
                events.append(dict(route=route, t0=t0 / 1e9, dur=round(dur, 2),
                                   vego=round(v0, 1), grey0=round(g0, 3), grey1=round(g1, 3),
                                   integ=il0, depth=round(depth, 3),
                                   blk=bool(blk_onset), blk_span=bool(blk_span), gap=gap))
                i = j
                continue
        i += 1
    return events


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def main():
    if os.path.exists(CACHE):
        ev = json.load(open(CACHE))
    else:
        with ProcessPoolExecutor(max_workers=4) as ex:
            res = list(ex.map(extract_events, [r for r, _ in ROUTES]))
        ev = {r: e for (r, _), e in zip(ROUTES, res)}
        json.dump(ev, open(CACHE, "w"))

    build = dict(ROUTES)

    # ---- 1) per-event table ----
    print("=" * 100)
    print("1) PER-EVENT TABLE (collapse<0.15 -> recovery>0.7)")
    print("   blk = blinker within +-3s of collapse onset; blkSpan = blinker in [collapse-3s, recovery+3s]")
    hdr = f"{'route':9s} {'build':4s} {'t0(s)':>9s} {'dur(s)':>7s} {'vEgo':>5s} {'grey@c':>7s} {'grey@r':>7s} {'integ':>6s} {'depth':>6s} {'blk':>4s} {'blkSpan':>7s} {'gap':>4s}"
    for r, b in ROUTES:
        print(f"-- {r} ({b} Nevada): {len(ev[r])} events")
        print(hdr)
        for e in ev[r]:
            print(f"{e['route']:9s} {b:4s} {e['t0']:9.1f} {e['dur']:7.2f} {e['vego']:5.1f} "
                  f"{e['grey0']:7.3f} {e['grey1']:7.3f} {e['integ']:6d} {e['depth']:6.3f} "
                  f"{'Y' if e['blk'] else '.':>4s} {'Y' if e['blk_span'] else '.':>7s} {'Y' if e['gap'] else '.':>4s}")

    # ---- 2) exclude lane-change events, recompute medians ----
    print("\n" + "=" * 100)
    print("2) PER-ROUTE MEDIANS: all vs lane-change-excluded (two exclusion definitions)")
    print(f"{'route':9s} {'build':4s} {'n_all':>5s} {'med_all':>8s} | {'n_noBlk':>7s} {'med_noBlk':>9s} | {'n_noBlkSpan':>11s} {'med_noBlkSpan':>13s}")
    pooled = {"OLD": {"all": [], "noblk": [], "nospan": []}, "NEW": {"all": [], "noblk": [], "nospan": []}}
    for r, b in ROUTES:
        durs = [e["dur"] for e in ev[r]]
        noblk = [e["dur"] for e in ev[r] if not e["blk"]]
        nospan = [e["dur"] for e in ev[r] if not e["blk_span"]]
        pooled[b]["all"] += durs; pooled[b]["noblk"] += noblk; pooled[b]["nospan"] += nospan
        print(f"{r:9s} {b:4s} {len(durs):5d} {med(durs):8.2f} | {len(noblk):7d} {med(noblk):9.2f} | "
              f"{len(nospan):11d} {med(nospan):13.2f}")
    for b in ("OLD", "NEW"):
        p = pooled[b]
        print(f"{'POOLED':9s} {b:4s} {len(p['all']):5d} {med(p['all']):8.2f} | {len(p['noblk']):7d} "
              f"{med(p['noblk']):9.2f} | {len(p['nospan']):11d} {med(p['nospan']):13.2f}")

    # ---- 3) covariate matching ----
    print("\n" + "=" * 100)
    print("3) COVARIATE-MATCHED PAIRS (grey@c +-0.08, vEgo +-5 m/s, depth +-0.05), lane-change-excluded (blk-onset def)")
    old_ev = [e for r, b in ROUTES if b == "OLD" for e in ev[r] if not e["blk"]]
    new_ev = [e for r, b in ROUTES if b == "NEW" for e in ev[r] if not e["blk"]]
    cands = []
    for ni, ne in enumerate(new_ev):
        for oi, oe in enumerate(old_ev):
            if (np.isfinite(ne["grey0"]) and np.isfinite(oe["grey0"])
                    and abs(ne["grey0"] - oe["grey0"]) <= 0.08
                    and abs(ne["vego"] - oe["vego"]) <= 5.0
                    and abs(ne["depth"] - oe["depth"]) <= 0.05):
                d = (abs(ne["grey0"] - oe["grey0"]) / 0.08 + abs(ne["vego"] - oe["vego"]) / 5.0
                     + abs(ne["depth"] - oe["depth"]) / 0.05)
                cands.append((d, ni, oi))
    cands.sort()
    used_n, used_o, pairs = set(), set(), []
    for d, ni, oi in cands:
        if ni in used_n or oi in used_o:
            continue
        used_n.add(ni); used_o.add(oi); pairs.append((new_ev[ni], old_ev[oi]))
    print(f"matched pairs: {len(pairs)} (of {len(new_ev)} NEW, {len(old_ev)} OLD non-lane-change events)")
    print(f"{'NEWroute':9s} {'dur':>6s} {'vEgo':>5s} {'grey':>6s} {'depth':>6s}  <->  "
          f"{'OLDroute':9s} {'dur':>6s} {'vEgo':>5s} {'grey':>6s} {'depth':>6s}  {'d(N-O)':>8s}")
    diffs = []
    for ne, oe in pairs:
        diffs.append(ne["dur"] - oe["dur"])
        print(f"{ne['route']:9s} {ne['dur']:6.2f} {ne['vego']:5.1f} {ne['grey0']:6.3f} {ne['depth']:6.3f}  <->  "
              f"{oe['route']:9s} {oe['dur']:6.2f} {oe['vego']:5.1f} {oe['grey0']:6.3f} {oe['depth']:6.3f}  {ne['dur']-oe['dur']:+8.2f}")
    if diffs:
        diffs = np.array(diffs)
        print(f"paired: n={len(diffs)} median(NEW-OLD)={np.median(diffs):+.2f}s  "
              f"NEW>OLD in {int((diffs>0).sum())}/{len(diffs)}")
        from scipy.stats import wilcoxon
        if len(diffs) >= 5:
            try:
                w = wilcoxon(diffs)
                print(f"Wilcoxon signed-rank: stat={w.statistic:.1f} p={w.pvalue:.4f}")
            except Exception as e2:
                print(f"Wilcoxon failed: {e2}")

    # ---- 4) pooled stats on excluded set ----
    print("\n" + "=" * 100)
    print("4) OLD-pooled (c5+c7) vs NEW-pooled (ce+cf), lane-change-EXCLUDED durations")
    from scipy.stats import mannwhitneyu, ks_2samp
    for name, key in (("blk-onset exclusion", "noblk"), ("blk-span exclusion", "nospan"), ("no exclusion", "all")):
        o, nw = np.array(pooled["OLD"][key]), np.array(pooled["NEW"][key])
        if len(o) and len(nw):
            mw = mannwhitneyu(o, nw, alternative="two-sided")
            ks = ks_2samp(o, nw)
            print(f"[{name:20s}] OLD n={len(o)} med={np.median(o):.2f}  NEW n={len(nw)} med={np.median(nw):.2f}  "
                  f"MWU p={mw.pvalue:.4f}  KS D={ks.statistic:.3f} p={ks.pvalue:.4f}")

    # ---- 5) sensitivities: moving-only (vEgo>=5) and covariate balance ----
    print("\n" + "=" * 100)
    print("5) SENSITIVITY: moving-only (vEgo>=5 m/s) + covariate balance of the compared sets")
    for label, keep in (("moving, all", lambda e: e["vego"] >= 5),
                        ("moving, no-blinker", lambda e: e["vego"] >= 5 and not e["blk"])):
        sets = {}
        for b in ("OLD", "NEW"):
            sets[b] = [e["dur"] for r, bb in ROUTES if bb == b for e in ev[r] if keep(e)]
        o, nw = np.array(sets["OLD"]), np.array(sets["NEW"])
        mw = mannwhitneyu(o, nw, alternative="two-sided"); ks = ks_2samp(o, nw)
        print(f"[{label:20s}] OLD n={len(o)} med={np.median(o):.2f} p90={np.percentile(o,90):.2f}  "
              f"NEW n={len(nw)} med={np.median(nw):.2f} p90={np.percentile(nw,90):.2f}  "
              f"MWU p={mw.pvalue:.4f}  KS p={ks.pvalue:.4f}")
        for r, b in ROUTES:
            d = [e["dur"] for e in ev[r] if keep(e)]
            print(f"    {r} ({b}): n={len(d)} med={med(d):.2f}")
    print("\ncovariate balance (no-blinker sets used in step 4):")
    for b in ("OLD", "NEW"):
        es = [e for r, bb in ROUTES if bb == b for e in ev[r] if not e["blk"]]
        print(f"  {b}: n={len(es)} med vEgo={med([e['vego'] for e in es]):.1f}  "
              f"med grey@c={med([e['grey0'] for e in es]):.3f}  med depth={med([e['depth'] for e in es]):.3f}")
    print("\nevent-rate exposure (all events / route duration from first-to-last event t0):")
    for r, b in ROUTES:
        n = len(ev[r]); span = (ev[r][-1]["t0"] - ev[r][0]["t0"]) / 60 if n > 1 else float("nan")
        blk = sum(1 for e in ev[r] if e["blk"])
        print(f"  {r} ({b}): {n} events over {span:.1f} min = {n/span:.2f}/min ; blinker-assoc {blk}/{n}")


if __name__ == "__main__":
    main()
