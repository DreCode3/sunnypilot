#!/usr/bin/env python3
"""Independent verification of the smooth_tau QA's decisive claims (do NOT trust the agents).
For b1 vs b2 straight runs, band-decompose BOTH:
  A    = lateral offset (felt wander)         -> hunt band 0.5-1.5Hz should DROP if smooth_tau helped
  desC = model action.desiredCurvature        -> hunt band should be FLAT (it's upstream of smooth_tau);
         if flat => the A drop is attributable to the downstream filter, not a between-drive model confound.
Also: duration-weighted A_std (artifact check) + saturation of the A_cps/desC_cps crossings metrics."""
import sys, math
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.straight_hunting_mv2 import load_mv2, runs

BANDS = {'slow 0.1-0.5': (0.1, 0.5), 'hunt 0.5-1.5': (0.5, 1.5), 'hf 1.5-5': (1.5, 5.0)}


def band_var(x, fs):
    x = x - np.mean(x)
    n = len(x)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1.0 / fs)
    p = np.abs(X) ** 2
    return {name: 2.0 * np.sum(p[(f >= lo) & (f < hi)]) / (n * n) for name, (lo, hi) in BANDS.items()}


def analyze(rid):
    d = load_mv2(rid)
    t, dc, A, spd = d['t'], d['dc'], d['A'], d['spd']
    dcroll = np.convolve(np.abs(dc), np.ones(40) / 40, mode='same')
    sm = (dcroll < 5e-4) & (spd > 13)
    out = []
    for a, b in runs(sm, 120):
        ts = t[a:b]
        if ts[-1] - ts[0] < 6:
            continue
        fs = 20.0
        n = int((ts[-1] - ts[0]) * fs) + 1
        tu = ts[0] + np.arange(n) / fs
        Au = np.interp(tu, ts, A[a:b]); dcu = np.interp(tu, ts, dc[a:b])
        dts = np.diff(ts)
        out.append(dict(spd=float(np.median(spd[a:b]) * 2.237), n=n,
                        A_std=float(np.std(A[a:b])),
                        Avar=band_var(Au, fs), DCvar=band_var(dcu, fs),
                        sat_A=float(np.mean(np.abs(np.diff(A[a:b]) / dts) > 0.05)),
                        sat_dc=float(np.mean(np.abs(np.diff(dc[a:b]) / dts) > 4e-5))))
    return out


def pooled_rms(g, key, band):
    den = sum(r['n'] for r in g)
    return math.sqrt(sum(r[key][band] * r['n'] for r in g) / den) if den else float('nan')


def dwm(g, key):
    den = sum(r['n'] for r in g)
    return sum(r[key] * r['n'] for r in g) / den if den else float('nan')


b1 = analyze('route_b1'); b2 = analyze('route_b2')
for label, filt in [('ALL straights', lambda r: True), ('SPEED-MATCHED 45-60mph', lambda r: 45 <= r['spd'] < 60)]:
    g1 = [r for r in b1 if filt(r)]; g2 = [r for r in b2 if filt(r)]
    print(f'\n==================== {label}:  b1 n={len(g1)}  vs  b2 n={len(g2)} ====================')
    print('  A lateral-offset band RMS (mm), duration-weighted:')
    for band in BANDS:
        r1 = pooled_rms(g1, 'Avar', band) * 1000; r2 = pooled_rms(g2, 'Avar', band) * 1000
        print(f'    {band:<14} b1 {r1:7.1f}   b2 {r2:7.1f}   delta {100*(r2-r1)/r1:+.0f}%')
    print('  desC model-output band RMS (1/m x1e4), dur-weighted  [ATTRIBUTION CONTROL: want hunt FLAT]:')
    for band in BANDS:
        r1 = pooled_rms(g1, 'DCvar', band) * 1e4; r2 = pooled_rms(g2, 'DCvar', band) * 1e4
        print(f'    {band:<14} b1 {r1:7.3f}   b2 {r2:7.3f}   delta {100*(r2-r1)/r1:+.0f}%')
    print(f'  A_std dur-weighted:  b1 {dwm(g1, "A_std"):.4f}   b2 {dwm(g2, "A_std"):.4f}')
    print(f'  saturation |dA/dt|>0.05:    b1 {dwm(g1, "sat_A"):.1%}   b2 {dwm(g2, "sat_A"):.1%}   (A_cps validity)')
    print(f'  saturation |ddesC/dt|>4e-5: b1 {dwm(g1, "sat_dc"):.1%}   b2 {dwm(g2, "sat_dc"):.1%}   (desC_cps validity)')
