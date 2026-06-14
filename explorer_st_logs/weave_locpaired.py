#!/usr/bin/env python3
"""THE ACTUAL PROBLEM: small left-right WEAVE of car/wheel on straights & gentle curves (uncomfortable, visible to
others). Driver: CD210-weak was UNUSABLE, golden made it USABLE = large real improvement. Driver holds consistent
speed per location => LOCATION-PAIRING is the correct control (auto speed-matches; avoids the speed-match-across-
different-roads confound that may have produced the prior 'flat-to-worse').

Per 20s engaged window (slow-band resolvable), exclude only SHARP curves (|yaw*v|>2.0; keep straights+gentle curves
since location-pairing matches geometry). Metrics = the WEAVE the driver feels:
  posW   = 0.1-0.5Hz RMS of lane position (the visible side-to-side weave)
  posVslow=0.05-0.2Hz RMS of lane position (very slow weave)
  posP2P = peak-to-peak of detrended lane position (the visible lurch amplitude)
  steerW = 0.1-0.5Hz RMS of steeringAngleDeg (the felt wheel weave)
  steerP2P = peak-to-peak detrended steer
Pair windows by ~400m block + heading octant (same road => same speed for this driver). Robust over blocks.
WEAK=b1,b2,b4,b5 (all weak-era CD210) ; GOLD=b8."""
import os, math
import numpy as np
from collections import defaultdict
CACHE = 'explorer_st_logs/_cache_reassess'
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']; GOLD = ['route_b8']
FS = 50.0; WIN = int(20*FS); HOP = int(10*FS); rng = np.random.default_rng(11)
BLOCK_M = 400.0; M_PER_DEG_LAT = 111320.0


def band_rms(x, lo, hi):
    n = len(x); t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1/FS); p = np.abs(X)**2
    return math.sqrt(2*np.sum(p[(f >= lo) & (f < hi)])/(n*n))


def detrend_p2p(x):
    n = len(x); t = np.arange(n); x = x - np.polyval(np.polyfit(t, x, 1), t)
    return float(np.percentile(x, 97.5) - np.percentile(x, 2.5))  # robust peak-to-peak


def windows(rids):
    bd = defaultdict(list)
    for rid in rids:
        d = {k: np.load(f'{CACHE}/{rid}.npz')[k] for k in np.load(f'{CACHE}/{rid}.npz').files}
        spd = d['spd'].astype(float)*2.237; yaw = d['yaw'].astype(float); veg = d['spd'].astype(float)
        eng = d['latact'].astype(int); pos = d['pos'].astype(float); steer = d['steer'].astype(float)
        lat = d['lat']; lon = d['lon']; aLat = yaw*veg
        for i in range(0, len(pos)-WIN, HOP):
            sl = slice(i, i+WIN)
            if np.mean(eng[sl]) < 0.95: continue
            if np.median(np.abs(aLat[sl])) > 2.0: continue   # exclude only SHARP curves
            if np.any(np.isnan(pos[sl])): continue
            sp = float(np.median(spd[sl]))
            if not (25 <= sp <= 80): continue
            la = float(np.median(lat[sl])); lo = float(np.median(lon[sl]))
            x = lo*M_PER_DEG_LAT*math.cos(math.radians(la)); y = la*M_PER_DEG_LAT
            c0, c1 = i, min(i+200, len(lat)-1)
            hdg = math.degrees(math.atan2((lon[c1]-lon[c0])*M_PER_DEG_LAT*math.cos(math.radians(la)),
                                          (lat[c1]-lat[c0])*M_PER_DEG_LAT)) % 360
            key = ((round(x/BLOCK_M), round(y/BLOCK_M)), int(((hdg+22.5) % 360)//45))
            bd[key].append(dict(spd=sp,
                posW=band_rms(pos[sl], 0.1, 0.5), posVslow=band_rms(pos[sl], 0.05, 0.2),
                posP2P=detrend_p2p(pos[sl]), steerW=band_rms(steer[sl], 0.1, 0.5),
                steerP2P=detrend_p2p(steer[sl])))
    return bd


def blockmed(bd):
    out = {}
    for k, ws in bd.items():
        out[k] = {m: float(np.median([w[m] for w in ws])) for m in ws[0]}
        out[k]['n'] = len(ws)
    return out


BW = blockmed(windows(WEAK)); BG = blockmed(windows(GOLD))
shared = [k for k in BW if k in BG]
print(f'WEAK windows-blocks {len(BW)}  GOLD {len(BG)}  SHARED (same ~400m block+heading) {len(shared)}')
if shared:
    dsp = [BG[k]['spd']-BW[k]['spd'] for k in shared]
    print(f'per-block speed diff GOLD-WEAK: median {np.median(dsp):+.1f}mph  IQR[{np.percentile(dsp,25):+.1f},{np.percentile(dsp,75):+.1f}]'
          f'  (if small => driver consistent-speed claim holds, location-pairing auto speed-matches)')
print(f'\n{"weave metric":<26}{"WEAK":>9}{"GOLD":>9}{"chg%":>7}{"95%CI on GOLD-WEAK":>24}{"gold-lower":>12}')
for m, lbl in [('posW', 'POS weave 0.1-0.5Hz (m)'), ('posVslow', 'POS very-slow .05-.2 (m)'),
               ('posP2P', 'POS peak-to-peak (m)'), ('steerW', 'STEER weave 0.1-0.5 (deg)'),
               ('steerP2P', 'STEER peak-to-peak (deg)')]:
    pairs = [(BW[k][m], BG[k][m]) for k in shared]
    w = np.array([p[0] for p in pairs]); g = np.array([p[1] for p in pairs])
    d = g - w
    bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(8000)]
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    lower = int(np.sum(g < w))
    print(f'  {lbl:<24}{np.median(w):>9.4f}{np.median(g):>9.4f}{100*(np.median(g)-np.median(w))/np.median(w):>+6.0f}%'
          f'  [{lo:>+8.4f},{hi:>+8.4f}] {"SIG" if (lo>0 or hi<0) else "ns "}{f"{lower}/{len(pairs)}":>11}')
print('\nNEG chg% / gold-lower>half / CI excludes 0 below 0 => golden REDUCES the felt weave at matched locations.')
print('Compare to the prior SPEED-MATCHED result (golden flat-to-worse): if this LOCATION-paired result shows golden')
print('better, the speed-match-across-different-roads was the artifact that hid the real improvement.')
