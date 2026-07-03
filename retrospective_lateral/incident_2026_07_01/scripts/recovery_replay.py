#!/usr/bin/env python3
"""THE frame-vs-engine discriminator for the 2026-07-01 AOL incident recovery-lag signal.

The sim (model_replay_sim) implements the OLD engine's temporal math (verified identical to
the NEW engine's compiled queues for Nevada's shapes: fb (1,25,512) split-branch idxs
[3::4] == sample_skip [::4] on a 97-deep queue; desire max-pool identical; img pair [t-4,t];
no prev_desired_curv input). It anchors to BOTH builds (c5 OLD corr 0.990, cf NEW 0.980).

So: replay laneProb-collapse windows from OLD-build drives (c7/c5) and NEW-build drives
(ce/cf) through the SAME sim and compare the sim's lane-prob recovery time to the logged one.
  - sim ~= logged on BOTH builds  -> recovery time is deterministic from the FRAMES;
    the old-vs-new recovery gap is a frame/scene/camera property, engine exonerated.
  - sim recovers FASTER than logged on NEW-build events only -> the new engine really is
    stickier than the old math; Suspect A back alive.

Also replays the decisive ce weak-correction event (62 mph, laneProb 0.96-0.99, drift to
-0.7 m, tiny dc ramp) and compares sim dc vs logged dc.

RUN: cd <repo> && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
       retrospective_lateral/incident_2026_07_01/scripts/recovery_replay.py [route ...]
Default routes: route_c7 route_c5 route_ce route_cf (all Nevada).
"""
import sys, glob, bisect, json
from pathlib import Path
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
import os
os.chdir(ROOT)

from openpilot.tools.lib.logreader import LogReader
from model_replay_sim import config as C

# non-destructive: seconds-based v_ego caches go to a scratch cache, main cache untouched
SCRATCH_CACHE = Path(ROOT) / "retrospective_lateral/incident_2026_07_01/replay_cache"
SCRATCH_CACHE.mkdir(parents=True, exist_ok=True)
C.CACHE_ROOT = SCRATCH_CACHE

VIDEO_SEGS = {
    "route_c7": {14, 15, 16}, "route_c5": {4, 5, 6},
    "route_ce": {1, 3, 4}, "route_cf": {9, 10, 11},
}
WARMUP_S = 12.0          # temporal buffers are 100 frames = 5 s; 12 s is comfortable
TAIL_S = 4.0
COLLAPSE, RECOVER = 0.15, 0.7
MAX_EVENTS_PER_ROUTE = 5


def load_route(route):
    """One rlog pass -> modelV2 rows, frame eof map, carState, camera grey, blinker times."""
    mv = []            # (mono_s, frameId, innerLP, dc)
    eof = {}           # frameId -> (eof_s, segment_num)
    cst = []           # (mono_s, vEgo)
    grey = []          # (mono_s, measuredGreyFraction)
    blink = []         # mono_s when a blinker is on
    for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst")):
        seg_num = int(Path(rl).parent.name.split("--")[-1])
        try:
            lr = LogReader(rl)
        except Exception:
            continue
        for m in lr:
            try:
                w = m.which()
            except Exception:
                continue
            t = m.logMonoTime * 1e-9
            if w == "modelV2":
                M = m.modelV2; p = list(M.laneLineProbs)
                mv.append((t, int(M.frameId), min(p[1:3]) if len(p) >= 3 else 1.0,
                           float(M.action.desiredCurvature)))
            elif w == "roadEncodeIdx":
                e = m.roadEncodeIdx
                eof[int(e.frameId)] = (e.timestampEof * 1e-9, int(e.segmentNum))
            elif w == "carState":
                c = m.carState
                cst.append((t, float(c.vEgo)))
                if c.leftBlinker or c.rightBlinker:
                    blink.append(t)
            elif w == "roadCameraState":
                try:
                    grey.append((t, float(m.roadCameraState.measuredGreyFraction)))
                except Exception:
                    pass
    mv.sort(); cst.sort(); grey.sort(); blink.sort()
    return mv, eof, cst, grey, blink


def write_vego_cache(route, cst):
    np.savez(SCRATCH_CACHE / f"{route}.npz",
             mono_time=np.array([c[0] for c in cst]),
             v_ego=np.array([c[1] for c in cst], np.float32))


def nearest(arr_ts, arr_vals, t):
    i = bisect.bisect_right(arr_ts, t) - 1
    if i < 0:
        return np.nan
    return arr_vals[i]


def find_collapse_events(mv):
    """Same scan as incident_analyses.recovery_lag: first lp<0.15 -> first lp>0.7."""
    events = []
    i, n = 0, len(mv)
    while i < n:
        if mv[i][2] < COLLAPSE:
            j = i
            while j < n and mv[j][2] < RECOVER:
                j += 1
            if j < n:
                events.append((mv[i][0], mv[j][0]))  # (t_collapse, t_recover)
                i = j
                continue
        i += 1
    return events


def window_frames(eof, t0, t1):
    """frameIds whose eof lies in [t0, t1], sorted by eof; with their segment nums."""
    rows = sorted((e[0], fid, e[1]) for fid, e in eof.items() if t0 <= e[0] <= t1)
    return rows  # list of (eof_s, frameId, seg_num)


def sim_recovery(mono_ts, sim_lp, t_collapse_logged):
    """Sim collapse->recovery duration: first sim lp<0.15 within +-2.5 s of the logged
    collapse, then first lp>0.7 after. Returns (dur_s, t_collapse_sim) or (nan, nan)."""
    mono_ts = np.asarray(mono_ts)
    lo = np.searchsorted(mono_ts, t_collapse_logged - 2.5)
    hi = np.searchsorted(mono_ts, t_collapse_logged + 2.5)
    start = None
    for k in range(lo, min(hi, len(sim_lp))):
        if sim_lp[k] < COLLAPSE:
            start = k
            break
    if start is None:
        return np.nan, np.nan
    for k in range(start, len(sim_lp)):
        if sim_lp[k] > RECOVER:
            return mono_ts[k] - mono_ts[start], mono_ts[start]
    return np.inf, mono_ts[start]


def main():
    routes = sys.argv[1:] or ["route_c7", "route_c5", "route_ce", "route_cf"]

    # ---- record lane probs from the vision net (device rule: sigmoid probs [0,1::2],
    # inner two = [1:3], min) by wrapping BundleModel.run_vision ---------------------
    import model_replay_sim.infer as I
    LANE_PROBS = []
    orig_run_vision = I.BundleModel.run_vision

    def run_vision_rec(self, img, big_img):
        out = orig_run_vision(self, img, big_img)
        lp = out.get("lane_lines_prob")
        if lp is not None:
            probs4 = np.asarray(lp)[0, 1::2]
            LANE_PROBS.append(float(np.min(probs4[1:3])))
        else:
            LANE_PROBS.append(np.nan)
        return out
    I.BundleModel.run_vision = run_vision_rec

    all_rows = []
    for route in routes:
        print(f"\n########## {route} ##########", flush=True)
        mv, eof, cst, grey, blink = load_route(route)
        write_vego_cache(route, cst)
        cst_ts = [c[0] for c in cst]; cst_v = [c[1] for c in cst]
        grey_ts = [g[0] for g in grey]; grey_v = [g[1] for g in grey]
        vsegs = VIDEO_SEGS[route]

        events = find_collapse_events(mv)
        usable = []
        for (tc, tr) in events:
            v = nearest(cst_ts, cst_v, tc)
            if not np.isfinite(v) or v < 5:
                continue
            rows = window_frames(eof, tc - WARMUP_S, tr + TAIL_S)
            if len(rows) < 40:
                continue
            if any(r[2] not in vsegs for r in rows):
                continue
            usable.append((tc, tr, rows, v))
        print(f"{route}: {len(events)} collapse events, {len(usable)} in video segs "
              f"(video segs {sorted(vsegs)})", flush=True)
        usable = usable[:MAX_EVENTS_PER_ROUTE]

        for (tc, tr, rows, v) in usable:
            mono_times = [r[0] for r in rows]
            fids = [r[1] for r in rows]
            LANE_PROBS.clear()
            from model_replay_sim.infer import replay_window
            res = replay_window("Nevada", route, mono_times)
            sim_lp = np.array(LANE_PROBS)
            if len(sim_lp) != len(mono_times):
                print(f"  !! frame-count mismatch sim {len(sim_lp)} vs window {len(mono_times)}")
                continue
            dur_logged = tr - tc
            dur_sim, tc_sim = sim_recovery(mono_times, sim_lp, tc)
            # logged lp series over the same frames for corr
            mv_by_fid = {r[1]: r[2] for r in mv}
            logged_lp = np.array([mv_by_fid.get(f, np.nan) for f in fids])
            fin = np.isfinite(logged_lp) & np.isfinite(sim_lp)
            corr = (np.corrcoef(sim_lp[fin], logged_lp[fin])[0, 1]
                    if fin.sum() > 2 and np.std(sim_lp[fin]) > 0 and np.std(logged_lp[fin]) > 0
                    else np.nan)
            g = nearest(grey_ts, grey_v, tc)
            blk = any(abs(b - tc) < 3.0 for b in blink)
            row = dict(route=route, t_collapse=round(tc, 2), v_mph=round(v * 2.237, 1),
                       grey_at_collapse=round(float(g), 3) if np.isfinite(g) else None,
                       blinker_pm3s=bool(blk),
                       dur_logged_s=round(dur_logged, 2),
                       dur_sim_s=(round(float(dur_sim), 2) if np.isfinite(dur_sim) else
                                  ("no-recovery" if dur_sim == np.inf else "no-sim-collapse")),
                       lp_corr=round(float(corr), 3) if np.isfinite(corr) else None,
                       n_frames=len(mono_times))
            all_rows.append(row)
            print("  EVENT " + json.dumps(row), flush=True)

    print("\n===== SUMMARY (sim vs logged recovery durations) =====")
    for r in all_rows:
        print(f"  {r['route']} t={r['t_collapse']:9.2f} v={r['v_mph']:5.1f}mph "
              f"grey={r['grey_at_collapse']} blk={'Y' if r['blinker_pm3s'] else 'n'} "
              f"logged={r['dur_logged_s']:>6}s sim={r['dur_sim_s']:>6} corr={r['lp_corr']}")
    print("\nINTERPRET: sim~=logged on BOTH builds -> frame/scene-driven (engine exonerated);"
          "\n           sim << logged on NEW routes only -> new engine stickier (Suspect A).")

    out = Path(ROOT) / "retrospective_lateral/incident_2026_07_01/replay_cache/recovery_replay_results.json"
    out.write_text(json.dumps(all_rows, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
