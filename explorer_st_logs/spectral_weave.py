#!/usr/bin/env python3
"""Spectral signature of the WEAVE. Hypothesis: weak has a low-freq weave (large lane excursion, visible) that golden
moves to higher freq / suppresses. Per config, average Welch PSD of steeringAngleDeg, lane position, and steering-rate
over engaged-straight bouts (>=30s). The KEY driver-felt channel is LANE POSITION (visible side-to-side); a low-freq
(0.1-0.3Hz) lane-position peak that golden reduces = the felt improvement. Also report band-integrated lane-position
weave by fine band, per config and split by speed (highway 55-65 = where the driver feels it; weak data may be thin)."""
import os, math
import numpy as np
from scipy import signal
CACHE = 'explorer_st_logs/_cache_reassess'
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']; GOLD = ['route_b8']
FS = 50.0; NPERSEG = 768


def bouts(rids, spd_lo=30, spd_hi=75):
    segs = []
    for rid in rids:
        d = {k: np.load(f'{CACHE}/{rid}.npz')[k] for k in np.load(f'{CACHE}/{rid}.npz').files}
        spd = d['spd'].astype(float)*2.237; al = d['yaw'].astype(float)*d['spd'].astype(float)
        eng = d['latact'].astype(int); pos = d['pos'].astype(float); steer = d['steer'].astype(float)
        m = (eng == 1) & (np.abs(al) < 1.0) & (spd >= spd_lo) & (spd <= spd_hi) & np.isfinite(pos)
        idx = np.where(m)[0]
        if len(idx) == 0: continue
        for run in np.split(idx, np.where(np.diff(idx) > 1)[0]+1):
            if len(run) >= int(20*FS):
                segs.append((pos[run], steer[run], float(np.median(spd[run]))))
    return segs


def avg_psd(segs, which):
    Ps = []; f = None
    for pos, steer, sp in segs:
        x = pos if which == 'pos' else (steer if which == 'steer' else np.gradient(steer, 1/FS))
        x = x - np.mean(x)
        f, P = signal.welch(x, FS, nperseg=min(NPERSEG, len(x)))
        Ps.append(np.interp(np.linspace(0, 2, 200), f, P) if f[-1] >= 2 else None)
    Ps = [p for p in Ps if p is not None]
    fg = np.linspace(0, 2, 200)
    return fg, np.mean(Ps, axis=0), len(Ps)


SW = bouts(WEAK); SG = bouts(GOLD)
print(f'engaged-straight bouts >=30s: WEAK {len(SW)}  GOLD {len(SG)}')
for ch in ['pos', 'steer', 'rate']:
    fg, pw, nw = avg_psd(SW, ch); _, pg, ng = avg_psd(SG, ch)
    print(f'\n=== avg PSD {ch} (WEAK n={nw} / GOLD n={ng} bouts) — power in fine bands ===')
    for lo, hi in [(0.05,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,0.8),(0.8,1.2),(1.2,2.0)]:
        mk = (fg >= lo) & (fg < hi)
        ipw = np.trapezoid(pw[mk], fg[mk]); ipg = np.trapezoid(pg[mk], fg[mk])
        chg = 100*(ipg-ipw)/ipw if ipw > 0 else float('nan')
        flag = '  <== GOLD LOWER' if ipg < ipw*0.9 else ('  (gold higher)' if ipg > ipw*1.1 else '')
        print(f'  {lo:.2f}-{hi:.2f}Hz: WEAK {ipw:.3e}  GOLD {ipg:.3e}  ({chg:+.0f}%){flag}')
    # peak in 0.1-0.8
    band = (fg >= 0.1) & (fg < 0.8)
    pf_w = fg[band][np.argmax(pw[band])]; pf_g = fg[band][np.argmax(pg[band])]
    print(f'  PEAK freq in 0.1-0.8Hz: WEAK {pf_w:.2f}Hz (P={pw[band].max():.2e})  GOLD {pf_g:.2f}Hz (P={pg[band].max():.2e})')

print('\n=== LANE-POSITION weave (0.1-0.3Hz RMS) by SPEED bin, per config (is weak weave worse at highway speed?) ===')
print(f'{"speed":<10}{"WEAK posW":>12}{"(n bouts)":>11}{"GOLD posW":>12}{"(n bouts)":>11}')
for lo, hi in [(30,45),(45,55),(55,65),(65,75)]:
    sw = bouts(WEAK, lo, hi); sg = bouts(GOLD, lo, hi)
    def w(segs):
        vals = []
        for pos, steer, sp in segs:
            x = pos-np.mean(pos); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1/FS); P = np.abs(X)**2
            vals.append(math.sqrt(2*np.sum(P[(f >= 0.1) & (f < 0.3)])/(len(x)**2)))
        return (np.median(vals), len(vals)) if vals else (float('nan'), 0)
    ww, nw = w(sw); wg, ng = w(sg)
    print(f'  {lo}-{hi:<5}{ww:>12.4f}{nw:>11}{wg:>12.4f}{ng:>11}')
print('\nKEY: if a low-freq (0.1-0.3Hz) lane-position PEAK in WEAK is reduced in GOLD, and/or weak posW rises with')
print('speed while gold stays flat at 55-65mph, that is the visible weave the driver felt golden fix.')
