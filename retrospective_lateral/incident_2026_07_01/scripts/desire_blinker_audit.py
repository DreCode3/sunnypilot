#!/usr/bin/env python3
"""Audit lane-change DESIRE pulses vs blinker intervals on new-build (route_ce/route_cf)
vs old-build (route_c7) drives.

For each blinker-on interval (leftBlinker != rightBlinker) with mean vEgo > 13 m/s:
  - max modelV2.meta.desireState[laneChangeLeft=3] and [laneChangeRight=4]
  - laneChangeState sequence reached (off/pre/starting/finishing)
  - latActive / enabled coverage during the interval
  - steeringPressed (nudge) presence
Desire enum indices verified from cereal: laneChangeLeft=3, laneChangeRight=4.
"""
import os
import sys
import glob
import numpy as np
from concurrent.futures import ProcessPoolExecutor

LOG_ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs"

LCS_NAMES = {0: "off", 1: "pre", 2: "starting", 3: "finishing"}


def load_segment(rlog_path):
    from openpilot.tools.lib.logreader import LogReader
    cs_rows = []   # t, vEgo, leftBlinker, rightBlinker, steeringPressed, steeringTorque, brakePressed
    cc_rows = []   # t, latActive, enabled
    mdl_rows = []  # t, desireState(8), laneChangeState, laneChangeDirection
    sds_rows = []  # t, enabled, active
    try:
        lr = LogReader(rlog_path)
    except Exception as e:
        return (rlog_path, str(e), cs_rows, cc_rows, mdl_rows, sds_rows)
    it = iter(lr)
    err = None
    while True:
        try:
            msg = next(it)
        except StopIteration:
            break
        except Exception as e:
            err = str(e)
            break
        try:
            which = msg.which()
            t = msg.logMonoTime * 1e-9
            if which == "carState":
                cs = msg.carState
                cs_rows.append((t, cs.vEgo, cs.leftBlinker, cs.rightBlinker,
                                cs.steeringPressed, cs.steeringTorque, cs.brakePressed))
            elif which == "carControl":
                cc = msg.carControl
                cc_rows.append((t, cc.latActive, cc.enabled))
            elif which == "modelV2":
                m = msg.modelV2
                ds = list(m.meta.desireState)
                if len(ds) < 8:
                    ds = ds + [0.0] * (8 - len(ds))
                mdl_rows.append((t, tuple(ds[:8]), int(m.meta.laneChangeState.raw),
                                 int(m.meta.laneChangeDirection.raw)))
            elif which == "selfdriveState":
                s = msg.selfdriveState
                sds_rows.append((t, s.enabled, s.active))
        except Exception:
            continue
    return (rlog_path, err, cs_rows, cc_rows, mdl_rows, sds_rows)


def seg_index(path):
    return int(path.split("--")[-1].split("/")[0])


def analyze_route(route_key, v_min=13.0):
    rlogs = sorted(glob.glob(os.path.join(LOG_ROOT, route_key, "*", "rlog.zst")),
                   key=lambda p: seg_index(os.path.dirname(p)))
    print(f"\n=== {route_key}: {len(rlogs)} segments ===")
    with ProcessPoolExecutor(max_workers=12) as ex:
        results = list(ex.map(load_segment, rlogs))

    cs_all, cc_all, mdl_all, sds_all = [], [], [], []
    for path, err, cs, cc, mdl, sds in results:
        if err:
            print(f"  [warn] {os.path.basename(os.path.dirname(path))}: iteration stopped early: {err[:80]}")
        cs_all.extend(cs)
        cc_all.extend(cc)
        mdl_all.extend(mdl)
        sds_all.extend(sds)

    cs_all.sort(key=lambda r: r[0])
    cc_all.sort(key=lambda r: r[0])
    mdl_all.sort(key=lambda r: r[0])
    sds_all.sort(key=lambda r: r[0])

    t_cs = np.array([r[0] for r in cs_all])
    v = np.array([r[1] for r in cs_all])
    lb = np.array([r[2] for r in cs_all], dtype=bool)
    rb = np.array([r[3] for r in cs_all], dtype=bool)
    sp = np.array([r[4] for r in cs_all], dtype=bool)
    stq = np.array([r[5] for r in cs_all])

    t_cc = np.array([r[0] for r in cc_all])
    lat_act = np.array([r[1] for r in cc_all], dtype=bool)
    cc_en = np.array([r[2] for r in cc_all], dtype=bool)

    t_mdl = np.array([r[0] for r in mdl_all])
    ds = np.array([r[1] for r in mdl_all])  # (N, 8)
    lcs = np.array([r[2] for r in mdl_all])
    lcd = np.array([r[3] for r in mdl_all])

    t_sds = np.array([r[0] for r in sds_all])
    sds_en = np.array([r[1] for r in sds_all], dtype=bool)

    one_blinker = lb != rb
    # find intervals of one_blinker
    idx = np.where(one_blinker)[0]
    intervals = []
    if len(idx):
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if t_cs[i] - t_cs[prev] > 1.0:  # gap > 1 s => new interval
                intervals.append((start, prev))
                start = i
            prev = i
        intervals.append((start, prev))

    t0 = t_cs[0] if len(t_cs) else 0.0
    rows = []
    for (a, b) in intervals:
        ta, tb = t_cs[a], t_cs[b]
        dur = tb - ta
        m_cs = slice(a, b + 1)
        v_mean = float(np.mean(v[m_cs]))
        if v_mean <= v_min:
            continue
        side = "L" if lb[m_cs].mean() >= rb[m_cs].mean() else "R"
        # extend model window slightly past blinker-off to catch finishing ramp
        m_mdl = (t_mdl >= ta - 0.2) & (t_mdl <= tb + 2.0)
        m_cc = (t_cc >= ta) & (t_cc <= tb)
        m_cc_pre = (t_cc >= ta - 3.0) & (t_cc < ta)
        m_sds = (t_sds >= ta) & (t_sds <= tb)
        max_l = float(ds[m_mdl, 3].max()) if m_mdl.any() else float("nan")
        max_r = float(ds[m_mdl, 4].max()) if m_mdl.any() else float("nan")
        states = sorted(set(lcs[m_mdl].tolist())) if m_mdl.any() else []
        lat_frac = float(lat_act[m_cc].mean()) if m_cc.any() else float("nan")
        lat_pre_frac = float(lat_act[m_cc_pre].mean()) if m_cc_pre.any() else float("nan")
        en_frac = float(cc_en[m_cc].mean()) if m_cc.any() else float("nan")
        sds_frac = float(sds_en[m_sds].mean()) if m_sds.any() else float("nan")
        nudged = bool(sp[m_cs].any())
        max_tq = float(np.abs(stq[m_cs]).max())
        rows.append(dict(route=route_key, t_start=ta - t0, dur=dur, side=side,
                         v_mean=v_mean, max_dsL=max_l, max_dsR=max_r,
                         lcs_states="/".join(LCS_NAMES.get(s, str(s)) for s in states),
                         lat_frac=lat_frac, lat_pre_frac=lat_pre_frac,
                         en_frac=en_frac, sds_en_frac=sds_frac,
                         nudged=nudged, max_tq=max_tq))

    hdr = (f"{'t_start_s':>9} {'dur_s':>6} {'side':>4} {'v_mps':>6} "
           f"{'maxDesL':>8} {'maxDesR':>8} {'lcStates':>22} "
           f"{'latAct%':>8} {'pre3sLat%':>9} {'ccEn%':>6} {'sdsEn%':>7} {'nudge':>6} {'maxTq':>7}")
    print(hdr)
    for r in rows:
        print(f"{r['t_start']:9.1f} {r['dur']:6.2f} {r['side']:>4} {r['v_mean']:6.1f} "
              f"{r['max_dsL']:8.3f} {r['max_dsR']:8.3f} {r['lcs_states']:>22} "
              f"{100*r['lat_frac']:7.1f}% {100*r['lat_pre_frac']:8.1f}% "
              f"{100*r['en_frac']:5.1f}% {100*r['sds_en_frac']:6.1f}% "
              f"{str(r['nudged']):>6} {r['max_tq']:7.1f}")
    if not rows:
        print("  (no blinker intervals with vEgo > %.1f m/s)" % v_min)

    # global desire activity summary (sanity: does desireState EVER ramp on this route)
    if len(t_mdl):
        print(f"  route-wide: max desireState[3] (lcLeft) = {ds[:, 3].max():.3f}, "
              f"max desireState[4] (lcRight) = {ds[:, 4].max():.3f}, "
              f"frames laneChangeState!=off: {(lcs != 0).sum()} / {len(lcs)}")
        print(f"  route-wide: latActive frames {lat_act.sum()}/{len(lat_act)} "
              f"({100*lat_act.mean():.1f}%), carControl.enabled {cc_en.sum()}/{len(cc_en)} "
              f"({100*cc_en.mean():.1f}%)")
    return rows


if __name__ == "__main__":
    routes = sys.argv[1:] if len(sys.argv) > 1 else ["route_ce", "route_cf", "route_c7"]
    all_rows = []
    for rk in routes:
        all_rows.extend(analyze_route(rk))
    print(f"\nTotal qualifying blinker intervals: {len(all_rows)}")
