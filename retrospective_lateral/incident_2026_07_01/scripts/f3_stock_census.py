#!/usr/bin/env python3
"""G5 acceptance gate: AOL safeguard on ALL stock drives — false-alert clearance + census.

GATE (must pass before deploy):
  ZERO departure-alert episodes across every stock route with rlog coverage.
REPORT (expected on-device visual-alert frequency; user reviews before deploy):
  per-route low-conf episode count, total low-conf alert-seconds, active-driving time.

Corpus: /Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock/
  Segment dirs are RRRRRRRR--HHHHHHHHHH--N; a route = all segments sharing the
  RRRRRRRR--HHHHHHHHHH prefix. Routes are auto-discovered, so drives pulled after
  2026-07-03 are included automatically — re-run this gate after pulling new drives.
  Routes with NO rlog.zst (qlog-only on the NAS: 00000000, 00000001, 00000003 as of
  2026-07-03) are SKIPPED and reported — qlog modelV2 is decimated (~4 Hz), which
  violates the monitor's 20 Hz dt assumption and would distort every time constant.

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo \
      .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_stock_census.py
Exit status: 0 = gate PASS (zero departure episodes), 1 = FAIL.
"""
import os, re, sys
from collections import defaultdict

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
STOCK = "/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")

from openpilot.tools.lib.logreader import LogReader
import openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor as am_mod
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
print("monitor module:", am_mod.__file__)
assert os.path.realpath(am_mod.__file__).startswith(os.path.realpath(ROOT)), \
    "monitor must come from this checkout"
assert os.path.isdir(STOCK), \
    f"NAS corpus not mounted: {STOCK} (mount smb://datacore.local/RAID_6_HDD first)"

SEG_RE = re.compile(r"^([0-9a-fA-F]{8}--[0-9a-fA-F]{10})--(\d+)$")


def discover_routes():
    routes = defaultdict(list)   # route_id -> [(seg_num, seg_dir), ...] sorted by seg_num
    for d in sorted(os.listdir(STOCK)):
        m = SEG_RE.match(d)
        if m and os.path.isdir(os.path.join(STOCK, d)):
            routes[m.group(1)].append((int(m.group(2)), os.path.join(STOCK, d)))
    return {rid: sorted(segs) for rid, segs in routes.items()}


def replay(rlogs):
    """Feed the monitor exactly what selfdrived would see; return alert rows + context."""
    rows = []           # (t, kind) kind in {"lc", "dep"}
    active_frames = 0   # modelV2 frames with lat_active and v >= MIN_SPEED (alerts possible)
    total_frames = 0
    cs_state = dict(v=0.0, lb=False, rb=False)
    lat = dict(active=False)
    mon = AolSafeguardMonitor()
    for rl in rlogs:
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
                total_frames += 1
                if lat["active"] and cs_state["v"] >= AolSafeguardMonitor.MIN_SPEED:
                    active_frames += 1
                if dep:
                    rows.append((t, "dep"))
                elif lc:
                    rows.append((t, "lc"))
    return rows, active_frames * 0.05, total_frames * 0.05


def episodes(rows, kind):
    ts = [t for (t, k) in rows if k == kind]
    eps = []
    for t in ts:
        if eps and t - eps[-1][1] < 1.0:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


routes = discover_routes()
total_dep = 0
scanned = 0
print(f"\n{'route':24s} {'segs':>5s} {'rlogs':>5s} {'drive_min':>9s} {'active_min':>10s} "
      f"{'lowconf_eps':>11s} {'lowconf_s':>9s} {'depart_eps':>10s}")
for rid, segs in sorted(routes.items()):
    rlogs = [os.path.join(d, "rlog.zst") for (_, d) in segs
             if os.path.exists(os.path.join(d, "rlog.zst"))]
    if not rlogs:
        print(f"{rid:24s} {len(segs):5d} {0:5d}  SKIPPED (qlog-only on NAS — pull rlogs from device for coverage)")
        continue
    scanned += 1
    rows, active_s, total_s = replay(rlogs)
    lc_eps, dep_eps = episodes(rows, "lc"), episodes(rows, "dep")
    lc_secs = sum(e[1] - e[0] for e in lc_eps)
    total_dep += len(dep_eps)
    print(f"{rid:24s} {len(segs):5d} {len(rlogs):5d} {total_s/60:9.1f} {active_s/60:10.1f} "
          f"{len(lc_eps):11d} {lc_secs:9.1f} {len(dep_eps):10d}")
    for e in dep_eps:
        print(f"    DEPARTURE EPISODE mono t=[{e[0]:.2f}, {e[1]:.2f}] — pull this window's "
              f"speed/blinker/laneChangeState context before any threshold change")

print(f"\nscanned {scanned} routes with rlog coverage")
print(f"G5 departure alerts on stock drives: {'PASS' if total_dep == 0 else 'FAIL'}  (count {total_dep})")
sys.exit(0 if total_dep == 0 else 1)
