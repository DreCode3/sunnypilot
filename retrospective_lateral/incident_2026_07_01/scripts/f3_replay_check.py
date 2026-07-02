#!/usr/bin/env python3
"""F3 acceptance gate: replay recorded rlogs through AolSafeguardMonitor.

MUST-PASS gates (from handoff v3 §5/§9b):
  G1 ce decisive event (override at mono 212.54): departure alert >= 1.0 s before override.
  G2 cf blowout (deep-blind at 682.21): low-conf alert within 2.5 s of collapse onset.
  G3 ce blind-in-lane-change (blind 257.31, override 260.18): low-conf alert BEFORE the
     override despite the maneuver (suppression must lift after 2 s).
  G4 departure-alert count on OLD routes (c5/c7/b5/7f) == 0 (no real departures occurred).
REPORT-ONLY (user tunes thresholds if excessive):
  R1 low-conf alert episode count + total alert seconds per route (fatigue check).

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo:/Users/dregilley/Documents/GitHub/sp-aol-f3 \
      .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py
"""
import glob, re, sys
import importlib.util
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
WORKTREE = "/Users/dregilley/Documents/GitHub/sp-aol-f3"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo"); sys.path.append(WORKTREE)

from openpilot.tools.lib.logreader import LogReader
# HARNESS FIX (import): `openpilot.sunnypilot.selfdrive.selfdrived` is a regular package rooted
# in the MAIN checkout (its __init__.py does not export aol_monitor). Once `openpilot` is
# imported above for LogReader, a plain `from openpilot.sunnypilot...selfdrived import aol_monitor`
# resolves to MAIN and raises ImportError. Load the worktree file DIRECTLY by path so we exercise
# the exact committed monitor selfdrived would run. Monitor is pure (deque + numpy only).
sys.path.insert(0, WORKTREE)
_mon_path = WORKTREE + "/openpilot/sunnypilot/selfdrive/selfdrived/aol_monitor.py"
_spec = importlib.util.spec_from_file_location("aol_monitor_worktree_f3", _mon_path)
am_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am_mod)
AolSafeguardMonitor = am_mod.AolSafeguardMonitor
print("monitor module:", am_mod.__file__)
assert WORKTREE in am_mod.__file__, "monitor must come from the aol-fixes-f3 worktree"


def _seg_num(path):
    # HARNESS FIX (ordering): sort segments by numeric index so the monitor sees a continuous
    # chronological time stream, exactly like selfdrived. Lexical sort interleaves --10 before
    # --2, injecting large mono-time discontinuities across segment boundaries.
    m = re.search(r"--(\d+)/rlog\.zst$", path)
    return int(m.group(1)) if m else 0


def replay(route):
    """Feed the monitor exactly what selfdrived would see; return alert timelines."""
    rows = []      # (t, kind) kind in {"lc","dep"}
    cs_state = dict(v=0.0, lb=False, rb=False)
    lat = dict(active=False)
    mon = AolSafeguardMonitor()
    for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst"), key=_seg_num):
        try:
            lr = LogReader(rl)
        except Exception:
            continue
        for m in lr:
            try:
                w = m.which()
            except Exception:
                continue
            if w == "carState":
                c = m.carState
                cs_state = dict(v=float(c.vEgo), lb=bool(c.leftBlinker), rb=bool(c.rightBlinker))
            elif w == "carControl":
                lat = dict(active=bool(m.carControl.latActive))
            elif w == "modelV2":
                md = m.modelV2
                t = m.logMonoTime * 1e-9
                p = list(md.laneLineProbs)
                inner = min(p[1], p[2]) if len(p) >= 3 else 1.0
                ll = list(md.laneLines)
                if len(ll) >= 3 and len(ll[1].y) and len(ll[2].y):
                    ly, ry = ll[1].y[0], ll[2].y[0]
                    off, ld, rd = (ly + ry) / 2.0, abs(ly), abs(ry)
                else:
                    off, ld, rd = 0.0, 10.0, 10.0
                maneuver = (str(md.meta.laneChangeState) != "off") or cs_state["lb"] or cs_state["rb"]
                lc, dep = mon.update(t=t, lat_active=lat["active"], v_ego=cs_state["v"],
                                     inner_prob=inner, lane_offset=off,
                                     left_line_dist=ld, right_line_dist=rd, maneuver=maneuver)
                if dep:
                    rows.append((t, "dep"))
                elif lc:
                    rows.append((t, "lc"))
    return rows


def episodes(rows, kind):
    ts = [t for (t, k) in rows if k == kind]
    eps = []
    for t in ts:
        if eps and t - eps[-1][1] < 1.0:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


results = {r: replay(r) for r in ("route_ce", "route_cf", "route_c5", "route_c7", "route_b5", "route_7f")}

print("\n===== REPORT (R1: alert load per route) =====")
for r, rows in results.items():
    lc_eps, dep_eps = episodes(rows, "lc"), episodes(rows, "dep")
    lc_secs = sum(e[1] - e[0] for e in lc_eps)
    print(f"  {r}: low-conf episodes={len(lc_eps)} ({lc_secs:.1f}s total)  departure episodes={len(dep_eps)}")

print("\n===== GATES =====")
ce_dep = [t for (t, k) in results["route_ce"] if k == "dep"]
g1 = [t for t in ce_dep if 205.0 < t < 212.54]
print(f"G1 decisive-event departure alert: {'PASS' if g1 and (212.54 - g1[0]) >= 1.0 else 'FAIL'}"
      f"  (first alert {'%.2f' % g1[0] if g1 else 'none'}; need <= 211.54)")

cf_lc = [t for (t, k) in results["route_cf"] if k == "lc"]
g2 = [t for t in cf_lc if 682.21 <= t <= 684.71]
print(f"G2 blowout low-conf alert: {'PASS' if g2 else 'FAIL'}  (alerts in [682.2, 684.7]: {len(g2)})")

ce_lc = [t for (t, k) in results["route_ce"] if k == "lc"]
g3 = [t for t in ce_lc if 257.31 <= t <= 260.18]
print(f"G3 blind-in-lane-change alert before override: {'PASS' if g3 else 'FAIL'}"
      f"  (first {'%.2f' % g3[0] if g3 else 'none'}; override 260.18)")

old_dep = sum(len(episodes(results[r], "dep")) for r in ("route_c5", "route_c7", "route_b5", "route_7f"))
print(f"G4 departure alerts on OLD routes: {'PASS' if old_dep == 0 else 'FAIL'}  (count {old_dep})")
