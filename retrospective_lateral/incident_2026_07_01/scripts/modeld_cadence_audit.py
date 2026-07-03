#!/usr/bin/env python3
"""modeld cadence/exec-time audit: NEW build (route_ce, route_cf) vs OLD build
(route_c5, route_c7 Nevada; route_b5 CD210; route_7f OPM7).

Question: does the NEW build's model daemon run slower or drop frames vs OLD?
Per route, while moving (vEgo > 5 m/s):
  1) modelV2.modelExecutionTime: median / p95 / max (s)
  2) modelV2 publish cadence via logMonoTime: median & p99 gap (ms), gaps > 150 ms
  3) modelV2.frameDropPerc: median / p95 (guarded try/except)
  4) modelV2.frameId step between consecutive msgs: fraction with delta > 1
     (skipped camera frames), fraction delta == 0 (repeat), median delta
  5) roadCameraState cadence (~20 Hz expected) for reference

RUN: cd /Users/dregilley/Documents/GitHub/sunnypilot && \
     PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
     retrospective_lateral/incident_2026_07_01/scripts/modeld_cadence_audit.py
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/opendbc_repo")

ROUTES = [
    ("route_c5", "OLD Nevada"),
    ("route_c7", "OLD Nevada"),
    ("route_b5", "OLD CD210"),
    ("route_7f", "OLD OPM7"),
    ("route_ce", "NEW Nevada"),
    ("route_cf", "NEW Nevada"),
]

V_MOVING = 5.0          # m/s
DISCONT_S = 5.0         # gaps larger than this = missing log data, excluded from cadence stats
STALE_V_S = 1.0         # carState sample must be within this of msg time to trust vEgo


def load_segment(rlog_path):
    """One rlog -> dict of raw arrays. Corrupt events skipped, never abort."""
    from openpilot.tools.lib.logreader import LogReader
    cs_t, cs_v = [], []
    mv_t, mv_fid, mv_exec, mv_drop = [], [], [], []
    cam_t = []
    try:
        lr = LogReader(rlog_path)
    except Exception:
        return None
    for m in lr:
        try:
            w = m.which()
        except Exception:
            continue  # "Corrupted events detected" mid-iteration: skip message
        try:
            if w == "carState":
                cs_t.append(m.logMonoTime)
                cs_v.append(float(m.carState.vEgo))
            elif w == "modelV2":
                M = m.modelV2
                mv_t.append(m.logMonoTime)
                try:
                    mv_fid.append(int(M.frameId))
                except Exception:
                    mv_fid.append(-1)
                try:
                    mv_exec.append(float(M.modelExecutionTime))
                except Exception:
                    mv_exec.append(np.nan)
                try:
                    mv_drop.append(float(M.frameDropPerc))
                except Exception:
                    mv_drop.append(np.nan)
            elif w == "roadCameraState":
                cam_t.append(m.logMonoTime)
        except Exception:
            continue
    return {
        "cs_t": np.array(cs_t, dtype=np.int64), "cs_v": np.array(cs_v),
        "mv_t": np.array(mv_t, dtype=np.int64), "mv_fid": np.array(mv_fid, dtype=np.int64),
        "mv_exec": np.array(mv_exec), "mv_drop": np.array(mv_drop),
        "cam_t": np.array(cam_t, dtype=np.int64),
    }


def vego_at(query_t_ns, cs_t, cs_v):
    """Previous-sample vEgo for each query time; NaN if no sample within STALE_V_S."""
    if len(cs_t) == 0:
        return np.full(len(query_t_ns), np.nan)
    idx = np.searchsorted(cs_t, query_t_ns, side="right") - 1
    out = np.full(len(query_t_ns), np.nan)
    ok = idx >= 0
    age_s = np.full(len(query_t_ns), np.inf)
    age_s[ok] = (query_t_ns[ok] - cs_t[idx[ok]]) / 1e9
    fresh = ok & (age_s <= STALE_V_S)
    out[fresh] = cs_v[idx[fresh]]
    return out


def pctl(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def analyze_route(route):
    seg_dirs = sorted(
        glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst"),
        key=lambda p: int(os.path.basename(os.path.dirname(p)).rsplit("--", 1)[-1]),
    )
    with ProcessPoolExecutor(max_workers=min(14, len(seg_dirs))) as ex:
        segs = [s for s in ex.map(load_segment, seg_dirs) if s is not None]
    if not segs:
        return None

    def cat(key):
        return np.concatenate([s[key] for s in segs]) if segs else np.array([])

    cs_t, cs_v = cat("cs_t"), cat("cs_v")
    mv_t, mv_fid = cat("mv_t"), cat("mv_fid")
    mv_exec, mv_drop = cat("mv_exec"), cat("mv_drop")
    cam_t = cat("cam_t")

    # sort everything by logMonoTime (segments share one boot clock)
    o = np.argsort(cs_t); cs_t, cs_v = cs_t[o], cs_v[o]
    o = np.argsort(mv_t, kind="stable")
    mv_t, mv_fid, mv_exec, mv_drop = mv_t[o], mv_fid[o], mv_exec[o], mv_drop[o]
    cam_t = np.sort(cam_t)

    mv_moving = vego_at(mv_t, cs_t, cs_v) > V_MOVING
    cam_moving = vego_at(cam_t, cs_t, cs_v) > V_MOVING

    r = {"route": route, "n_segs": len(segs), "n_mv": int(mv_moving.sum()),
         "n_cam": int(cam_moving.sum())}

    # 1) modelExecutionTime while moving
    e = mv_exec[mv_moving]
    e = e[np.isfinite(e)]
    r["exec_med"], r["exec_p95"], r["exec_max"] = pctl(e, 50), pctl(e, 95), (float(e.max()) if len(e) else float("nan"))
    r["exec_n"] = len(e)

    # 2) modelV2 cadence: consecutive msgs, both moving, gap < DISCONT_S
    gap_ms = np.diff(mv_t) / 1e6
    both = mv_moving[:-1] & mv_moving[1:]
    disc = gap_ms >= DISCONT_S * 1e3
    use = both & ~disc
    g = gap_ms[use]
    r["mv_gap_med"], r["mv_gap_p99"] = pctl(g, 50), pctl(g, 99)
    r["mv_gap_max"] = float(g.max()) if len(g) else float("nan")
    r["mv_gt150"] = int((g > 150).sum())
    r["mv_gt150_per_min"] = r["mv_gt150"] / (g.sum() / 60e3) if len(g) else float("nan")
    r["mv_disc"] = int((both & disc).sum())
    r["mv_hz"] = 1000.0 / r["mv_gap_med"] if r["mv_gap_med"] else float("nan")
    r["mv_min"] = g.sum() / 60e3 if len(g) else 0.0  # minutes of moving cadence data

    # 3) frameDropPerc while moving
    d = mv_drop[mv_moving]
    d = d[np.isfinite(d)]
    r["drop_med"], r["drop_p95"], r["drop_max"] = pctl(d, 50), pctl(d, 95), (float(d.max()) if len(d) else float("nan"))

    # 4) frameId deltas across consecutive modelV2 msgs (same window as cadence)
    fid_ok = (mv_fid[:-1] >= 0) & (mv_fid[1:] >= 0)
    fu = use & fid_ok
    df = (mv_fid[1:] - mv_fid[:-1])[fu]
    r["fid_n"] = len(df)
    r["fid_gt1"] = float((df > 1).mean()) if len(df) else float("nan")
    r["fid_eq0"] = float((df == 0).mean()) if len(df) else float("nan")
    r["fid_lt0"] = float((df < 0).mean()) if len(df) else float("nan")
    r["fid_med"] = float(np.median(df)) if len(df) else float("nan")
    r["fid_max"] = int(df.max()) if len(df) else -1

    # 5) roadCameraState cadence while moving
    cg = np.diff(cam_t) / 1e6
    cboth = cam_moving[:-1] & cam_moving[1:]
    cuse = cboth & (cg < DISCONT_S * 1e3)
    c = cg[cuse]
    r["cam_gap_med"], r["cam_gap_p99"] = pctl(c, 50), pctl(c, 99)
    r["cam_gt150"] = int((c > 150).sum())
    r["cam_hz"] = 1000.0 / r["cam_gap_med"] if r["cam_gap_med"] else float("nan")
    return r


def main():
    rows = []
    for route, label in ROUTES:
        res = analyze_route(route)
        if res is None:
            print(f"{route}: NO DATA")
            continue
        res["label"] = label
        rows.append(res)
        print(f"done {route}")

    hdr = (f"{'route':9s} {'build':11s} {'segs':>4s} {'min':>6s} "
           f"{'exec_med':>8s} {'exec_p95':>8s} {'exec_max':>8s} "
           f"{'gap_med':>7s} {'gap_p99':>7s} {'gapMax':>7s} {'>150ms':>6s} {'/min':>5s} "
           f"{'Hz':>5s} {'drop_med':>8s} {'drop_p95':>8s} "
           f"{'fid>1%':>7s} {'fid=0%':>7s} {'fidMax':>6s} "
           f"{'camHz':>6s} {'cam_p99':>7s} {'cam>150':>7s}")
    print("\n=== modeld cadence audit (vEgo > 5 m/s) — exec in ms, gaps in ms ===")
    print(hdr)
    for r in rows:
        print(f"{r['route']:9s} {r['label']:11s} {r['n_segs']:4d} {r['mv_min']:6.1f} "
              f"{r['exec_med']*1e3:8.1f} {r['exec_p95']*1e3:8.1f} {r['exec_max']*1e3:8.1f} "
              f"{r['mv_gap_med']:7.1f} {r['mv_gap_p99']:7.1f} {r['mv_gap_max']:7.0f} {r['mv_gt150']:6d} {r['mv_gt150_per_min']:5.2f} "
              f"{r['mv_hz']:5.2f} {r['drop_med']:8.3f} {r['drop_p95']:8.3f} "
              f"{r['fid_gt1']*100:7.3f} {r['fid_eq0']*100:7.3f} {r['fid_max']:6d} "
              f"{r['cam_hz']:6.2f} {r['cam_gap_p99']:7.1f} {r['cam_gt150']:7d}")
    print("\nnotes: min = minutes of moving modelV2 cadence data; drop = frameDropPerc units as logged;")
    print("fid>1% = % of consecutive modelV2 steps where frameId advanced by >1 (camera frame skipped);")
    disc_str = ", ".join("{}={}".format(r["route"], r["mv_disc"]) for r in rows)
    print(f"gaps >= {DISCONT_S:.0f}s treated as log discontinuities and excluded (counts: {disc_str})")


if __name__ == "__main__":
    main()
