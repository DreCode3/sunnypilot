#!/usr/bin/env python3
"""INDEPENDENTLY VERIFY the adversary's conclusion-flipping claim: the +0.056m engaged centering gap is ~era-drift,
not the golden PI. The DECISIVE clean test = UNENGAGED (latActive==0, PI OFF, HUMAN driving in BOTH eras):
if GOLD(b8) still measures more-centered than WEAK(b1/b2) at matched locations with the controller OFF, the gap
is era/physical/calibration, NOT the controller.

Also: per-era engaged vs unengaged (edge-cleaned around transitions) + DiD = (GOLD_eng-GOLD_uneng)-(WEAK_eng-WEAK_uneng).
CAVEAT noted in output: engaged=MODEL+PI steering, unengaged=HUMAN steering, so the within-era eng-uneng shift mixes
'OpenPilot vs human' with 'PI on/off' — NOT a clean PI isolation. The UNENGAGED-ONLY location-paired gap (human both
eras) is the clean era-confound probe and does not have that issue."""
import os, math
import numpy as np
from collections import defaultdict
CACHE = 'explorer_st_logs/_cache_reassess'
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
FS = 50.0; rng = np.random.default_rng(99)
CELL_M = 55.0; M_PER_DEG_LAT = 111320.0
EDGE = int(3.0*FS)  # drop +-3s around latActive transitions


def load(rid):
    d = {k: np.load(f'{CACHE}/{rid}.npz')[k] for k in np.load(f'{CACHE}/{rid}.npz').files}
    return d


def masks(d):
    spd = d['spd'].astype(float)*2.237; al = d['yaw'].astype(float)*d['spd'].astype(float)
    eng = d['latact'].astype(int); pos = d['pos'].astype(float)
    base = (np.abs(al) < 0.6) & (spd >= 40) & (spd <= 80) & np.isfinite(pos)
    # edge-clean: exclude samples within EDGE of a latActive transition
    trans = np.zeros(len(eng), bool)
    ch = np.where(np.diff(eng) != 0)[0]
    for c in ch:
        trans[max(0, c-EDGE):min(len(eng), c+EDGE+1)] = True
    clean = base & ~trans
    return spd, pos, eng, clean


def pooled(group, want_eng):
    vals = []
    for rid in GROUPS[group]:
        d = load(rid); spd, pos, eng, clean = masks(d)
        m = clean & (eng == (1 if want_eng else 0))
        vals.append(pos[m])
    return np.concatenate(vals)


print('=== PER-ERA pooled mean offset (edge-cleaned ±3s around transitions), straight 40-80mph ===')
we, wu = pooled('WEAK', True), pooled('WEAK', False)
ge, gu = pooled('GOLD', True), pooled('GOLD', False)
print(f'  WEAK: engaged {we.mean():+.4f} (n={len(we)})   unengaged/human {wu.mean():+.4f} (n={len(wu)})   controller shift {we.mean()-wu.mean():+.4f}')
print(f'  GOLD: engaged {ge.mean():+.4f} (n={len(ge)})   unengaged/human {gu.mean():+.4f} (n={len(gu)})   controller shift {ge.mean()-gu.mean():+.4f}')
print(f'  ENGAGED gap GOLD-WEAK   = {ge.mean()-we.mean():+.4f}m   (the apparent centering "win")')
print(f'  UNENGAGED gap GOLD-WEAK = {gu.mean()-wu.mean():+.4f}m   <-- HUMAN both eras = pure era/physical/calibration')
print(f'  within-era DiD = (GOLD_eng-GOLD_uneng)-(WEAK_eng-WEAK_uneng) = {(ge.mean()-gu.mean())-(we.mean()-wu.mean()):+.4f}m')
print(f'    (CAVEAT: eng=MODEL+PI, uneng=HUMAN, so DiD mixes OP-vs-human with PI-on/off; the UNENGAGED gap above is the clean probe)')


def cells(group, want_eng):
    bd = defaultdict(list)
    for rid in GROUPS[group]:
        d = load(rid); spd, pos, eng, clean = masks(d)
        lat = d['lat']; lon = d['lon']
        idx = np.where(clean & (eng == (1 if want_eng else 0)))[0]
        for i in idx:
            la, lo = lat[i], lon[i]
            x = lo*M_PER_DEG_LAT*math.cos(math.radians(la)); y = la*M_PER_DEG_LAT
            j = min(i+10, len(lat)-1)
            dy = (lat[j]-la)*M_PER_DEG_LAT; dx = (lon[j]-lo)*M_PER_DEG_LAT*math.cos(math.radians(la))
            oct_ = int(((math.degrees(math.atan2(dx, dy)) % 360 + 22.5) % 360)//45)
            bd[((round(x/CELL_M), round(y/CELL_M)), oct_)].append(pos[i])
    return {k: (np.mean(v), len(v)) for k, v in bd.items()}


def paired(label, want_eng):
    W = cells('WEAK', want_eng); G = cells('GOLD', want_eng)
    shared = [k for k in W if k in G]
    if len(shared) < 4:
        print(f'  {label}: only {len(shared)} shared cells'); return
    diffs = np.array([G[k][0]-W[k][0] for k in shared])
    bs = [np.mean(diffs[rng.integers(0, len(diffs), len(diffs))]) for _ in range(8000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f'  {label}: n={len(shared)} cells | GOLD-WEAK {np.mean(diffs):+.4f}m  95%CI[{lo:+.4f},{hi:+.4f}] '
          f'{"SIG" if (lo>0 or hi<0) else "ns"} | GOLD-better {np.sum(diffs>0)}/{len(shared)}')

print('\n=== LOCATION-PAIRED (55m cell + heading octant) ===')
paired('ENGAGED  (model+PI)  ', True)
paired('UNENGAGED(human, PI OFF) <-- clean era probe', False)
print('\nINTERPRETATION: if UNENGAGED GOLD-WEAK is large & positive (~+0.1m), the engaged "+0.056m win" is era-drift,')
print('not the golden PI. b3-b7 (more weak-era unengaged data, extracting now) will add power to this probe.')
