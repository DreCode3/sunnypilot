#!/usr/bin/env python3
"""CONJECTURE-FREE magnitude: direct location-paired centering measurement (NO model, NO inversion).
At the SAME physical GPS cell + heading, compare GOLD(b8) vs WEAK(b1/b2) mean lane offset. Crown/geometry are
fixed per cell so they CANCEL in the per-cell difference. The only intrinsic caveat is the between-era nature
(b8 golden-era vs b1/b2 weak-era) — that is a data limitation, not a modeling assumption.

Centering offset is ~speed-independent (measured slope ~0.001 m/mph), so unlike oscillation the speed gap is a
weak confound here; we report WITH and WITHOUT per-cell speed matching to show it.

Autocorr-honest: each ~55m cell = one spatial unit; bootstrap OVER CELLS (and a coarser 220m-block variant,
since adjacent cells share crown). Per-cell value = mean offset over all engaged-straight samples of that group
in that cell. Difference d_cell = GOLD - WEAK (signed; +d = GOLD less-left = better centered, given left bias)."""
import os, math
import numpy as np
from collections import defaultdict
CACHE = 'explorer_st_logs/_cache_reassess'
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
FS = 50.0
rng = np.random.default_rng(2026)
CELL_M = 55.0
M_PER_DEG_LAT = 111320.0


def load(group):
    rows = []  # (cellkey, headingoct, spd_mph, pos)
    for rid in GROUPS[group]:
        f = f'{CACHE}/{rid}.npz'
        if not os.path.exists(f): continue
        d = {k: np.load(f)[k] for k in np.load(f).files}
        spd = d['spd'].astype(float)*2.237; al = d['yaw'].astype(float)*d['spd'].astype(float)
        eng = d['latact'].astype(bool); pos = d['pos'].astype(float)
        lat = d['lat']; lon = d['lon']
        m = eng & (np.abs(al) < 0.6) & (spd >= 40) & (spd <= 80) & np.isfinite(pos)
        idx = np.where(m)[0]
        # heading from local GPS gradient (smoothed)
        for i in idx:
            la = lat[i]; lo = lon[i]
            x = lo*M_PER_DEG_LAT*math.cos(math.radians(la)); y = la*M_PER_DEG_LAT
            cell = (round(x/CELL_M), round(y/CELL_M))
            # heading from a short lookahead
            j = min(i+10, len(lat)-1)
            dy = (lat[j]-la)*M_PER_DEG_LAT; dx = (lon[j]-lo)*M_PER_DEG_LAT*math.cos(math.radians(la))
            hdg = math.degrees(math.atan2(dx, dy)) % 360
            oct_ = int(((hdg+22.5) % 360)//45)
            rows.append((cell, oct_, spd[i], pos[i]))
    return rows


def cellstats(rows):
    bd = defaultdict(list)
    for cell, oct_, spd, pos in rows:
        bd[(cell, oct_)].append((spd, pos))
    out = {}
    for k, vs in bd.items():
        a = np.array(vs)
        out[k] = (np.mean(a[:, 1]), np.median(a[:, 0]), len(a))  # mean pos, median spd, n
    return out


W = cellstats(load('WEAK')); G = cellstats(load('GOLD'))
shared = [k for k in W if k in G]
print(f'shared cells (same ~55m cell + heading octant): {len(shared)}')


def report(cells, label):
    diffs = np.array([G[k][0]-W[k][0] for k in cells])   # GOLD-WEAK signed offset
    wv = np.array([W[k][0] for k in cells]); gv = np.array([G[k][0] for k in cells])
    if len(diffs) < 6:
        print(f'  {label}: only {len(diffs)} cells'); return
    # bootstrap over cells
    bs = [np.mean(diffs[rng.integers(0, len(diffs), len(diffs))]) for _ in range(8000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    better = np.sum(diffs > 0)  # gold less-left
    # sign test p (two-sided, binomial)
    from math import comb
    n = len(diffs); k = better
    p_sign = 2*min(sum(comb(n, j) for j in range(k, n+1)), sum(comb(n, j) for j in range(0, k+1)))/2**n
    p_sign = min(1.0, p_sign)
    print(f'  {label}: n={n} cells | WEAK mean {np.mean(wv):+.4f} GOLD {np.mean(gv):+.4f} | '
          f'mean diff {np.mean(diffs):+.4f}m  95%CI[{lo:+.4f},{hi:+.4f}] {"SIG" if (lo>0 or hi<0) else "ns"} | '
          f'GOLD better {better}/{n} sign-p {p_sign:.3f}')

print('\n=== per-cell GOLD-WEAK offset difference (+ = GOLD better-centered) ===')
report(shared, 'ALL shared cells (no speed match)')
sm = [k for k in shared if abs(W[k][1]-G[k][1]) <= 4]
report(sm, 'speed-matched |dspd|<=4mph')
sm2 = [k for k in shared if abs(W[k][1]-G[k][1]) <= 2.5]
report(sm2, 'speed-matched |dspd|<=2.5mph')

# coarser 220m spatial blocks (adjacent 55m cells share crown -> more honest independence)
def coarsen(rows, factor=4):
    rows2 = [((c[0]//factor, c[1]//factor), o, s, p) for (c, o, s, p) in rows]
    return rows2
Wc = cellstats(coarsen(load('WEAK'))); Gc = cellstats(coarsen(load('GOLD')))
sharedc = [k for k in Wc if k in Gc]
diffsc = np.array([Gc[k][0]-Wc[k][0] for k in sharedc])
if len(diffsc) >= 6:
    bs = [np.mean(diffsc[rng.integers(0, len(diffsc), len(diffsc))]) for _ in range(8000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    better = np.sum(diffsc > 0)
    print(f'\n  220m-block (coarser, more-independent): n={len(diffsc)} | mean diff {np.mean(diffsc):+.4f}m '
          f'95%CI[{lo:+.4f},{hi:+.4f}] {"SIG" if (lo>0 or hi<0) else "ns"} | GOLD better {better}/{len(diffsc)}')
print('\nNOTE: this is the conjecture-free magnitude. Intrinsic caveat = between-era (b8 vs b1/b2), unfixable')
print('without same-session A/B or more weak corridor passes (b3-b7, pending device-up).')
