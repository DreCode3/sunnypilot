#!/usr/bin/env python3
"""VERIFY the driver-matching candidate (Angle 3): does GOLD have fewer BAD WHEEL-WEAVE EPISODES than WEAK, and
does it survive MATCHED SPEED (steering angle ~1/v^2, so gold being faster mechanically lowers steer amplitude —
MUST speed-bin to rule that out)? Metric = fraction of 8s gentle-engaged windows whose steeringAngleDeg 0.1-0.5Hz
RMS exceeds a threshold (the 'busy wheel' episodes the driver notices). Reported PER SPEED BIN (kills 1/v^2) and
location-paired. Also report the MEDIAN steer-weave per bin (to see if it's a tail effect vs a median effect)."""
import os, math
import numpy as np
from collections import defaultdict
CACHE = 'explorer_st_logs/_cache_reassess'
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']; GOLD = ['route_b8']
FS = 50.0; WIN = int(8*FS); HOP = int(4*FS)
M_PER_DEG_LAT = 111320.0


def band_rms(x, lo, hi):
    n = len(x); t = np.arange(n); x = x - np.polyval(np.polyfit(t, x, 1), t)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1/FS); p = np.abs(X)**2
    return math.sqrt(2*np.sum(p[(f >= lo) & (f < hi)])/(n*n))


def windows(rids):
    out = []
    for rid in rids:
        d = {k: np.load(f'{CACHE}/{rid}.npz')[k] for k in np.load(f'{CACHE}/{rid}.npz').files}
        spd = d['spd'].astype(float)*2.237; al = d['yaw'].astype(float)*d['spd'].astype(float)
        eng = d['latact'].astype(int); steer = d['steer'].astype(float); lat = d['lat']; lon = d['lon']
        for i in range(0, len(steer)-WIN, HOP):
            sl = slice(i, i+WIN)
            if np.mean(eng[sl]) < 0.95 or np.median(np.abs(al[sl])) > 1.0: continue
            sp = float(np.median(spd[sl]))
            if not (30 <= sp <= 70): continue
            la = float(np.median(lat[sl])); lo = float(np.median(lon[sl]))
            x = lo*M_PER_DEG_LAT*math.cos(math.radians(la)); y = la*M_PER_DEG_LAT
            out.append((sp, band_rms(steer[sl], 0.1, 0.5), (round(x/55), round(y/55))))
    return out


W = windows(WEAK); G = windows(GOLD)
print(f'gentle-engaged 8s windows: WEAK {len(W)}  GOLD {len(G)}')
Wsp = np.array([w[0] for w in W]); Wrms = np.array([w[1] for w in W])
Gsp = np.array([w[0] for w in G]); Grms = np.array([w[1] for w in G])
print(f'speed median: WEAK {np.median(Wsp):.1f}  GOLD {np.median(Gsp):.1f} mph')

print('\n=== per SPEED BIN: median steer-weave 0.1-0.5Hz (deg) AND frac windows >1.0deg (bad-weave episodes) ===')
print(f'{"bin":<9}{"WEAK med":>10}{"GOLD med":>10}{"medR":>7}   {"WEAK>1.0":>9}{"GOLD>1.0":>9}{"rateR":>7}{"  nW/nG":>10}')
for lo, hi in [(30,40),(40,48),(48,55),(55,62),(62,70)]:
    wm = (Wsp >= lo) & (Wsp < hi); gm = (Gsp >= lo) & (Gsp < hi)
    if wm.sum() < 8 or gm.sum() < 8:
        print(f'  {lo}-{hi:<4}  thin ({wm.sum()}/{gm.sum()})'); continue
    wmed, gmed = np.median(Wrms[wm]), np.median(Grms[gm])
    wbad, gbad = np.mean(Wrms[wm] > 1.0), np.mean(Grms[gm] > 1.0)
    print(f'  {lo}-{hi:<4}{wmed:>10.3f}{gmed:>10.3f}{gmed/wmed:>7.2f}   {wbad*100:>8.0f}%{gbad*100:>8.0f}%{(gbad/wbad if wbad>0 else float("nan")):>7.2f}{f"{wm.sum()}/{gm.sum()}":>10}')

print('\n=== LOCATION-PAIRED (same 55m cell) at matched speed (|dspd|<=6mph): bad-weave rate & median ===')
def cellmap(W):
    bd = defaultdict(list)
    for sp, rms, cell in W: bd[cell].append((sp, rms))
    return bd
BW, BG = cellmap(W), cellmap(G)
shared = [c for c in BW if c in BG]
pairs = []
for c in shared:
    wsp = np.median([v[0] for v in BW[c]]); gsp = np.median([v[0] for v in BG[c]])
    if abs(wsp-gsp) <= 6:
        pairs.append((np.mean([v[1] > 1.0 for v in BW[c]]), np.mean([v[1] > 1.0 for v in BG[c]]),
                      np.median([v[1] for v in BW[c]]), np.median([v[1] for v in BG[c]])))
if pairs:
    pairs = np.array(pairs)
    print(f'  {len(pairs)} speed-matched shared cells | bad-weave rate WEAK {pairs[:,0].mean()*100:.0f}% vs GOLD {pairs[:,1].mean()*100:.0f}%'
          f' | median steer-weave WEAK {np.median(pairs[:,2]):.3f} GOLD {np.median(pairs[:,3]):.3f}')
    print(f'  cells where WEAK bad-rate>GOLD: {np.sum(pairs[:,0]>pairs[:,1])}/{len(pairs)}; GOLD>WEAK: {np.sum(pairs[:,1]>pairs[:,0])}')
print('\nKEY: if at MATCHED speed (same bin / |dspd|<=6 paired) GOLD bad-weave rate is ~half WEAK, the wheel-weave')
print('improvement is REAL (not the 1/v^2 speed artifact). If it only shows pooled and vanishes per-bin, it was speed.')
