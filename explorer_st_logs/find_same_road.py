#!/usr/bin/env python3
"""Find which strong-PI (golden) and weak-PI drives covered the SAME roads as today's b1/b2.
GPS glitch-filtered; cell overlap = fraction of a drive's ~55m cells that fall on today's road."""
import sys, glob, os
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

CELL = 0.0005  # ~55 m


def load_gps(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'),
                  key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    pts = []
    for sd in segs:
        f = sd + 'qlog.zst'
        if not os.path.exists(f):
            f = sd + 'rlog.zst'
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                w = msg.which()
                la = lo = None
                if w == 'liveLocationKalman':
                    g = msg.liveLocationKalman
                    if g.positionGeodetic.valid:
                        v = g.positionGeodetic.value; la, lo = float(v[0]), float(v[1])
                elif w == 'gpsLocationExternal':
                    g = msg.gpsLocationExternal
                    if abs(g.latitude) > 0.01:
                        la, lo = g.latitude, g.longitude
                if la is not None and 33.0 < la < 35.5 and -85.5 < lo < -83.5:
                    pts.append((la, lo))
        except Exception:
            continue
    return rid, pts


def cells(pts):
    return {(round(a / CELL), round(o / CELL)) for a, o in pts}


today = ['route_b1', 'route_b2']
golden = ['route_7a', 'route_7b', 'route_7c', 'route_7d', 'route_7f', 'route_80', 'route_86', 'route_87',
          'route_88', 'route_89', 'route_8a', 'route_8c', 'route_8e', 'route_8f', 'route_90', 'route_91',
          'route_92', 'route_93', 'route_94', 'route_95', 'route_96', 'route_97', 'route_98', 'route_99', 'route_9a']
weak = ['route_9b', 'route_9c', 'route_9d', 'route_9e', 'route_9f', 'route_a0', 'route_a4', 'route_a5',
        'route_a7', 'route_a8', 'route_a9', 'route_aa', 'route_ac', 'route_ad']
allr = today + golden + weak

with cf.ThreadPoolExecutor(max_workers=12) as ex:
    g = dict(ex.map(load_gps, allr))

today_cells = set()
for r in today:
    today_cells |= cells(g.get(r, []))
print(f'today (b1+b2) road cells: {len(today_cells)}  (b1={len(g.get("route_b1",[]))}pts b2={len(g.get("route_b2",[]))}pts)')


def report(name, lst):
    print(f'\n--- {name}: overlap with today road ---')
    out = []
    for r in lst:
        c = cells(g.get(r, []))
        inter = len(c & today_cells)
        frac = inter / len(c) if c else 0
        out.append((r, len(g.get(r, [])), len(c), inter, frac))
    for r, npts, nc, inter, frac in sorted(out, key=lambda x: -x[3]):
        flag = '  <== SAME ROAD' if inter >= 8 else ''
        print(f'  {r:<12} {npts:>5}pts {nc:>4}cells  {inter:>4} on-today ({frac:>4.0%}){flag}')


report('GOLDEN strong-PI', golden)
report('recent WEAK-PI', weak)
