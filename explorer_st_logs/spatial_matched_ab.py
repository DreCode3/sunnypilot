#!/usr/bin/env python3
"""Powered same-road smooth_tau A/B: bin the overlapping road into ~55m cells, compute hunt-band
(0.5-1.5Hz) lateral-offset RMS in short straight windows, pair b1(0.12/0.04) vs b2(0.25/0.12) cells
at matched location AND speed. Uses ALL overlapping data, not just 4 run-centroids."""
import sys, math
import numpy as np
from collections import defaultdict
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.straight_hunting_mv2 import load_mv2
from explorer_st_logs.gps_matched_ab import load_gps, band_var

CELL = 0.0005   # ~55 m
W = 4.0         # window seconds


def series(rid):
    d = load_mv2(rid); g = load_gps(rid)
    if d is None or g is None:
        return None
    t = d['t']; fs = 20.0
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    A = np.interp(tu, t, d['A']); dc = np.interp(tu, t, d['dc']); spd = np.interp(tu, t, d['spd'])
    dcroll = np.convolve(np.abs(dc), np.ones(40) / 40, mode='same')
    sm = (dcroll < 5e-4) & (spd > 13)
    lat = np.interp(tu, g['t'], g['lat']); lon = np.interp(tu, g['t'], g['lon'])
    return dict(t=tu, A=A, dc=dc, spd=spd * 2.237, sm=sm, lat=lat, lon=lon, fs=fs)


def windows(s):
    fs = s['fs']; n = int(W * fs); out = []; i = 0; N = len(s['t'])
    while i + n <= N:
        sl = slice(i, i + n)
        if s['sm'][sl].all():
            out.append(dict(lat=float(np.mean(s['lat'][sl])), lon=float(np.mean(s['lon'][sl])),
                            spd=float(np.median(s['spd'][sl])),
                            hunt=band_var(s['A'][sl], fs)['hunt'], slow=band_var(s['A'][sl], fs)['slow'],
                            dchunt=band_var(s['dc'][sl], fs)['hunt']))
        i += n
    return out


def ckey(la, lo):
    return (round(la / CELL), round(lo / CELL))


s1 = series('route_b1'); s2 = series('route_b2')
w1 = windows(s1); w2 = windows(s2)
print(f'straight 4s-windows: b1={len(w1)}  b2={len(w2)}')
c1 = defaultdict(list); c2 = defaultdict(list)
for w in w1:
    c1[ckey(w['lat'], w['lon'])].append(w)
for w in w2:
    c2[ckey(w['lat'], w['lon'])].append(w)
common = [k for k in c1 if k in c2]
print(f'overlapping ~55m cells with BOTH configs: {len(common)}')

rows = []
for k in common:
    a, b = c1[k], c2[k]
    sa = np.mean([w['spd'] for w in a]); sb = np.mean([w['spd'] for w in b])
    if abs(sa - sb) > 8:
        continue
    rows.append(dict(spd=(sa + sb) / 2,
                     ha=np.mean([w['hunt'] for w in a]), hb=np.mean([w['hunt'] for w in b]),
                     sla=np.mean([w['slow'] for w in a]), slb=np.mean([w['slow'] for w in b]),
                     dca=np.mean([w['dchunt'] for w in a]), dcb=np.mean([w['dchunt'] for w in b])))
print(f'speed-matched (<8mph) common cells: {len(rows)}')

if rows:
    def pooled(key):
        return math.sqrt(np.mean([r[key] for r in rows]))
    print(f'\n  POOLED over {len(rows)} same-road speed-matched cells:')
    h1, h2 = pooled('ha') * 1000, pooled('hb') * 1000
    s_1, s_2 = pooled('sla') * 1000, pooled('slb') * 1000
    d1, d2 = pooled('dca') * 1e4, pooled('dcb') * 1e4
    print(f'    A hunt 0.5-1.5Hz RMS: b1 {h1:.1f}mm -> b2 {h2:.1f}mm  ({100*(h2-h1)/h1:+.0f}%)   <- the felt center-hunt')
    print(f'    A slow 0.1-0.5Hz RMS: b1 {s_1:.1f}mm -> b2 {s_2:.1f}mm  ({100*(s_2-s_1)/s_1:+.0f}%)')
    print(f'    desC hunt RMS (ctrl): b1 {d1:.3f} -> b2 {d2:.3f}  ({100*(d2-d1)/d1:+.0f}%)')
    # per-cell paired sign test on hunt
    nb = sum(1 for r in rows if r['hb'] < r['ha'])
    ratios = sorted(math.sqrt(r['hb'] / r['ha']) for r in rows if r['ha'] > 0)
    med_ratio = ratios[len(ratios) // 2]
    print(f'\n    per-cell: b2<b1 hunt in {nb}/{len(rows)} cells; median b2/b1 hunt-RMS ratio = {med_ratio:.2f}')
    # crude binomial p (two-sided) vs 0.5
    from math import comb
    n = len(rows); k = nb
    p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) * 2 / (2 ** n)
    print(f'    sign-test p (b2<b1 vs 50/50) ~ {min(p,1.0):.3f}')
    # ROBUST central tendency (outlier-resistant) vs the outlier-driven pooled above
    mh1 = math.sqrt(np.median([r['ha'] for r in rows])) * 1000
    mh2 = math.sqrt(np.median([r['hb'] for r in rows])) * 1000
    print(f'\n    MEDIAN per-cell hunt RMS (robust): b1 {mh1:.1f}mm -> b2 {mh2:.1f}mm  ({100*(mh2-mh1)/mh1:+.0f}%)')
    # trimmed pooled: drop the 3 highest-hunt b1 cells (the suspected outliers)
    keep = sorted(rows, key=lambda r: r['ha'])[:-3]
    tp1 = math.sqrt(np.mean([r['ha'] for r in keep])) * 1000
    tp2 = math.sqrt(np.mean([r['hb'] for r in keep])) * 1000
    print(f'    TRIMMED pooled (drop top-3 b1 cells): b1 {tp1:.1f}mm -> b2 {tp2:.1f}mm  ({100*(tp2-tp1)/tp1:+.0f}%)')
    print(f'    => if trimmed flips toward 0/positive, the -55% pooled was 3 outlier cells, not the config.')
