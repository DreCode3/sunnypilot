#!/usr/bin/env python3
"""STRENGTHEN the era-baseline probe with the b3-b7 weak-era HUMAN data (b3/b6/b7 ~100% unengaged, b4/b5 partial).
The cleanest controller-free test: UNENGAGED (PI off, human driving) GOLD-era (b8) vs WEAK-era (b1..b7), location-
paired. If b8-unengaged still sits more-centered than the big weak-era human pool at matched cells, the centering
difference is a real between-era PHYSICAL change (calibration already shown stable), NOT the golden PI.

Also: corridor overlap (how many ~55m cells b3-b7 share with b8) so we know how much power they add."""
import os, math
import numpy as np
from collections import defaultdict
CACHE = 'explorer_st_logs/_cache_reassess'
WEAK_ERA = ['route_b1', 'route_b2', 'route_b4', 'route_b5', 'route_b6', 'route_b7', 'route_b3']
GOLD_ERA = ['route_b8']
FS = 50.0; rng = np.random.default_rng(7)
CELL_M = 55.0; M_PER_DEG_LAT = 111320.0
EDGE = int(3.0*FS)


def samples(rid, want_eng):
    d = {k: np.load(f'{CACHE}/{rid}.npz')[k] for k in np.load(f'{CACHE}/{rid}.npz').files}
    spd = d['spd'].astype(float)*2.237; al = d['yaw'].astype(float)*d['spd'].astype(float)
    eng = d['latact'].astype(int); pos = d['pos'].astype(float); lat = d['lat']; lon = d['lon']
    base = (np.abs(al) < 0.6) & (spd >= 40) & (spd <= 80) & np.isfinite(pos)
    trans = np.zeros(len(eng), bool)
    for c in np.where(np.diff(eng) != 0)[0]:
        trans[max(0, c-EDGE):min(len(eng), c+EDGE+1)] = True
    m = base & ~trans & (eng == (1 if want_eng else 0))
    return lat[m], lon[m], pos[m]


def cellmap(rids, want_eng):
    bd = defaultdict(list)
    for rid in rids:
        try: la, lo, pos = samples(rid, want_eng)
        except Exception: continue
        for i in range(len(la)):
            x = lo[i]*M_PER_DEG_LAT*math.cos(math.radians(la[i])); y = la[i]*M_PER_DEG_LAT
            j = min(i+10, len(la)-1)
            dy = (la[j]-la[i])*M_PER_DEG_LAT; dx = (lo[j]-lo[i])*M_PER_DEG_LAT*math.cos(math.radians(la[i]))
            oct_ = int(((math.degrees(math.atan2(dx, dy)) % 360 + 22.5) % 360)//45)
            bd[((round(x/CELL_M), round(y/CELL_M)), oct_)].append(pos[i])
    return {k: np.mean(v) for k, v in bd.items() if len(v) >= 3}


print('=== sample counts (engaged-straight / unengaged-straight, 40-80mph, edge-cleaned) ===')
for rid in WEAK_ERA+GOLD_ERA:
    try:
        e = len(samples(rid, True)[2]); u = len(samples(rid, False)[2])
        print(f'  {rid}: engaged {e}  unengaged {u}')
    except Exception as ex:
        print(f'  {rid}: ERR {ex}')

print('\n=== UNENGAGED (PI OFF, human) location-paired: GOLD-era b8 vs WEAK-era fleet ===')
Wu = cellmap(WEAK_ERA, False); Gu = cellmap(GOLD_ERA, False)
sh = [k for k in Wu if k in Gu]
print(f'  shared unengaged cells: {len(sh)}  (was 11 with b1/b2 only)')
if len(sh) >= 6:
    d = np.array([Gu[k]-Wu[k] for k in sh])
    bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(8000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f'  GOLD-WEAK unengaged offset = {np.mean(d):+.4f}m  95%CI[{lo:+.4f},{hi:+.4f}] {"SIG" if (lo>0 or hi<0) else "ns"} | GOLD-better {np.sum(d>0)}/{len(sh)}')

print('\n=== ENGAGED location-paired (for reference): GOLD b8 vs WEAK-era engaged fleet (b1,b2,b4,b5) ===')
We = cellmap(['route_b1','route_b2','route_b4','route_b5'], True); Ge = cellmap(GOLD_ERA, True)
she = [k for k in We if k in Ge]
print(f'  shared engaged cells: {len(she)}')
if len(she) >= 6:
    d = np.array([Ge[k]-We[k] for k in she])
    bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(8000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f'  GOLD-WEAK engaged offset = {np.mean(d):+.4f}m  95%CI[{lo:+.4f},{hi:+.4f}] {"SIG" if (lo>0 or hi<0) else "ns"} | GOLD-better {np.sum(d>0)}/{len(she)}')

print('\nNOTE: unengaged gap ~= engaged gap => the centering difference is between-ERA (present PI-OFF), real physical')
print('(calibration stable per check_calibration_era.py), NOT the golden PI. b3-b7 add weak-era human power.')
