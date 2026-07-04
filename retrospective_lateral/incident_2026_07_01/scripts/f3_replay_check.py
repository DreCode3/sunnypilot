#!/usr/bin/env python3
"""F3 acceptance gate: replay recorded rlogs through AolSafeguardMonitor.

STOCK-PORT VERSION (2026-07-03): the monitor imports from THIS checkout
(sunnypilot/selfdrive/selfdrived/aol_monitor.py via the openpilot/ symlink shim —
no worktree), and the rlog corpus reads from the NAS mount.

MUST-PASS gates (from handoff v3 §5/§9b):
  G1 ce decisive event (override at mono 212.54): departure alert >= 1.0 s before override.
  G2 cf blowout (deep-blind at 682.21): low-conf alert within 2.5 s of collapse onset.
  G3 ce blind-in-lane-change (blind 257.31, override 260.18): low-conf alert BEFORE the
     override despite the maneuver (suppression must lift after 2 s).
  G4 departure-alert count on OLD routes (c5/c7/b5/7f) == 0 (no real departures occurred).
REPORT-ONLY (user tunes thresholds if excessive):
  R1 low-conf alert episode count + total alert seconds per route (fatigue check).

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo \
      .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py
"""
import glob, os, re, sys

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
CORPUS = "/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")

from openpilot.tools.lib.logreader import LogReader
import openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor as am_mod
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
print("monitor module:", am_mod.__file__)
assert os.path.realpath(am_mod.__file__).startswith(os.path.realpath(ROOT)), \
    "monitor must come from this checkout"
assert os.path.isdir(CORPUS), \
    f"NAS corpus not mounted: {CORPUS} (mount smb://datacore.local/RAID_6_HDD first)"


def _seg_num(path):
    # Sort segments by numeric index so the monitor sees a continuous chronological
    # time stream, exactly like selfdrived. Lexical sort interleaves --10 before --2,
    # injecting large mono-time discontinuities across segment boundaries.
    m = re.search(r"--(\d+)/rlog\.zst$", path)
    return int(m.group(1)) if m else 0


def replay(route):
    """Feed the monitor exactly what selfdrived would see; return alert timelines."""
    rows = []      # (t, kind) kind in {"lc","dep"}
    cs_state = dict(v=0.0, lb=False, rb=False)
    lat = dict(active=False)
    mon = AolSafeguardMonitor()
    files = sorted(glob.glob(f"{CORPUS}/{route}/*/rlog.zst"), key=_seg_num)
    print(f"  {route}: {len(files)} segments")
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
sys.exit(0 if (g1 and (212.54 - g1[0]) >= 1.0 and g2 and g3 and old_dep == 0) else 1)
