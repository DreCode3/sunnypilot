#!/usr/bin/env python3
"""Same-road lateral A/B by GPS cell, three configs:
  GOLDEN = strong PI (Kp 0.0005, int_cap 1.0), OPM7   [routes 7f/95/99/98]
  WEAK   = weak PI  (Kp 0.0001, int_cap 0.3), OPM7    [routes a0/9b/9d]  (clean PI-only control vs golden)
  TODAY  = weak PI + CD210                              [b1/b2]
Metric per ~55m cell: mean|A| (centering error), mean A (bias), std A (wander).  A=-(L+R)/2 lane offset.
Paired per-cell across configs on cells they SHARE (definitely same road). Robust median + sign test."""
import sys, math
import numpy as np
from collections import defaultdict
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.straight_hunting_mv2 import load_mv2
from explorer_st_logs.gps_matched_ab import load_gps

CELL = 0.0005
GROUPS = {
    'GOLDEN': ['route_7f', 'route_95', 'route_99', 'route_98'],
    'WEAK':   ['route_a0', 'route_9b', 'route_9d'],
    'TODAY':  ['route_b1', 'route_b2'],
}


def ckey(la, lo):
    return (round(la / CELL), round(lo / CELL))


def per_cell(rid):
    d = load_mv2(rid); g = load_gps(rid)
    if d is None or g is None:
        return rid, {}
    t, A, spd = d['t'], d['A'], d['spd']
    gm = (g['lat'] > 33.0) & (g['lat'] < 35.5) & (g['lon'] > -85.5) & (g['lon'] < -83.5)
    gt, gla, glo = g['t'][gm], g['lat'][gm], g['lon'][gm]
    if len(gt) < 10:
        return rid, {}
    la = np.interp(t, gt, gla); lo = np.interp(t, gt, glo)
    valid = (t >= gt[0]) & (t <= gt[-1]) & (la > 33) & (la < 35.5)
    cd = defaultdict(list)
    idx = np.where(valid)[0]
    for i in idx:
        cd[ckey(la[i], lo[i])].append(A[i])
    out = {}
    for c, vals in cd.items():
        if len(vals) < 15:
            continue
        a = np.array(vals)
        out[c] = dict(absA=float(np.mean(np.abs(a))), meanA=float(np.mean(a)), stdA=float(np.std(a)), n=len(a))
    return rid, out


allr = [r for lst in GROUPS.values() for r in lst]
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    res = dict(ex.map(per_cell, allr))
for r in allr:
    print(f'  {r}: {len(res.get(r,{}))} cells')


def group_cells(names):
    agg = defaultdict(lambda: dict(absA=0., meanA=0., stdA=0., n=0))
    for nm in names:
        for c, s in res.get(nm, {}).items():
            a = agg[c]
            a['absA'] += s['absA'] * s['n']; a['meanA'] += s['meanA'] * s['n']
            a['stdA'] += s['stdA'] * s['n']; a['n'] += s['n']
    return {c: dict(absA=a['absA'] / a['n'], meanA=a['meanA'] / a['n'], stdA=a['stdA'] / a['n'], n=a['n'])
            for c, a in agg.items() if a['n'] >= 30}


G = {k: group_cells(v) for k, v in GROUPS.items()}
print(f'\ncells per group (>=30 samp): GOLDEN {len(G["GOLDEN"])}  WEAK {len(G["WEAK"])}  TODAY {len(G["TODAY"])}')


def compare(a, b, an, bn):
    common = [c for c in G[a] if c in G[b]]
    if not common:
        print(f'\n{an} vs {bn}: no shared cells'); return
    dabs = [(G[b][c]['absA'] - G[a][c]['absA']) for c in common]
    rabs = [G[b][c]['absA'] / G[a][c]['absA'] for c in common if G[a][c]['absA'] > 1e-4]
    b_worse = sum(1 for d in dabs if d > 0)
    gA = math.sqrt(np.mean([G[a][c]['absA'] ** 2 for c in common]))
    bA = math.sqrt(np.mean([G[b][c]['absA'] ** 2 for c in common]))
    gAm = np.median([G[a][c]['absA'] for c in common]); bAm = np.median([G[b][c]['absA'] for c in common])
    print(f'\n{an} vs {bn}  ({len(common)} shared cells):')
    print(f'  mean|A| centering error  pooled-RMS: {an} {gA*1000:.0f}mm -> {bn} {bA*1000:.0f}mm ({100*(bA-gA)/gA:+.0f}%)')
    print(f'  mean|A| centering error  MEDIAN cell: {an} {gAm*1000:.0f}mm -> {bn} {bAm*1000:.0f}mm ({100*(bAm-gAm)/gAm:+.0f}%)')
    print(f'  per-cell: {bn} worse (higher |A|) in {b_worse}/{len(common)} cells; median ratio {bn}/{an} = {np.median(rabs):.2f}')


compare('GOLDEN', 'TODAY', 'GOLDEN', 'TODAY')
compare('GOLDEN', 'WEAK', 'GOLDEN', 'WEAK')
compare('WEAK', 'TODAY', 'WEAK', 'TODAY')
