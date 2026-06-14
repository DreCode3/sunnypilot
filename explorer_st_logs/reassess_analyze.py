#!/usr/bin/env python3
"""FIRST-PASS multi-lens reassessment of the b8 (CD210+GOLDEN PI) vs b1/b2 (CD210+WEAK PI) disconnect.

The existing analysis measured AMPLITUDE only (steer std + detrended 0.1-0.5Hz RMS) and found flat/worse
at matched speed. But a driver reading "less ping-pong, confident, on-center" responds to things that
amplitude-with-mean-and-trend-removed CANNOT see:
  - HUNTING = reversal RATE & sharpness of the wheel (not displacement amplitude)
  - CONFIDENT/ON-CENTER = steady-state centering: mean offset + slow (<0.1Hz) drift -- which std/detrend DISCARD
  - CHARACTER = where the spectral energy sits (deliberate-low vs nervous-high)

This script computes, per engaged-straight 16s window, the FULL set, plus a SPEED-CURVE comparison
(regress each metric on speed within each group: does GOLD lie BELOW WEAK at equal speed = real win,
or ON the same curve = speed mediates?). Robust stats + bootstrap CI on the speed-matched difference.

Reads the 50Hz cache from reassess_extract.py. No log re-reads."""
import sys, glob, os, math
import numpy as np
from scipy import signal
from collections import defaultdict

FS = 50.0; WIN = 16.0; HOP = 4.0; ENG_MIN = 0.9
CACHE = 'explorer_st_logs/_cache_reassess'
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
rng = np.random.default_rng(12345)


def band_rms(x, lo, hi, fs=FS):
    """detrended power in [lo,hi) Hz expressed as RMS."""
    n = len(x); t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1/fs); p = np.abs(X)**2
    return math.sqrt(2*np.sum(p[(f >= lo) & (f < hi)]) / (n*n))


def low_drift_rms(x, hi=0.1, fs=FS):
    """power BELOW hi Hz with ONLY the mean removed (keeps slow drift the detrend-std analysis discards)."""
    n = len(x); x = x - np.mean(x)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1/fs); p = np.abs(X)**2
    return math.sqrt(2*np.sum(p[(f > 0) & (f < hi)]) / (n*n))


def spectral_centroid(x, lo=0.1, hi=5.0, fs=FS):
    n = len(x); t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1/fs); p = np.abs(X)**2
    m = (f >= lo) & (f < hi)
    if p[m].sum() <= 0: return np.nan
    return float(np.sum(f[m]*p[m]) / np.sum(p[m]))


# Butterworth low-pass for clean rate/jerk (avoid 50Hz diff noise blow-up)
_bw = signal.butter(2, 5.0/(FS/2), 'low', output='sos')

def rate_jerk(x):
    xf = signal.sosfiltfilt(_bw, x)
    rate = np.gradient(xf, 1/FS)
    jerk = np.gradient(rate, 1/FS)
    return float(np.std(rate)), float(np.std(jerk)), xf


def reversals_per_s(xf, prom):
    """count wheel direction reversals/s (peaks+troughs with prominence>prom). TRUE 'hunting' rate."""
    pk, _ = signal.find_peaks(xf, prominence=prom)
    tr, _ = signal.find_peaks(-xf, prominence=prom)
    return (len(pk)+len(tr)) / (len(xf)/FS)


def bearing(la0, lo0, la1, lo1):
    dlat = la1-la0; dlon = (lo1-lo0)*math.cos(math.radians((la0+la1)/2))
    return math.degrees(math.atan2(dlon, dlat)) % 360

def adiff(a, b):
    d = abs(a-b) % 360
    return d if d <= 180 else 360-d


def load(rid):
    f = f'{CACHE}/{rid}.npz'
    if not os.path.exists(f): return None
    return {k: np.load(f)[k] for k in np.load(f).files}


def windows(rid):
    d = load(rid)
    if d is None: return []
    n = int(WIN*FS); hop = int(HOP*FS)
    steer = d['steer'].astype(float); spd = d['spd'].astype(float); yaw = d['yaw'].astype(float)
    pos = d['pos'].astype(float); cmd = d['cmd'].astype(float); mp = d['modpath'].astype(float)
    lat = d['lat']; lon = d['lon']; eng = d['latact']; press = d['press']
    aLat = yaw*spd
    have_pos = not np.all(np.isnan(pos))
    W = []
    for i in range(0, len(steer)-n, hop):
        sl = slice(i, i+n)
        if np.mean(eng[sl]) < ENG_MIN: continue
        la = lat[sl]; lo = lon[sl]
        if not (np.all(la > 33) & np.all(la < 35.5) & np.all(lo > -85.5) & np.all(lo < -83.5)): continue
        if np.median(np.abs(aLat[sl])) >= 0.6: continue  # straight
        spdm = float(np.median(spd[sl]))*2.237
        if not (40 <= spdm <= 80): continue
        st = steer[sl]
        r_std, j_std, stf = rate_jerk(st)
        w = dict(
            rid=rid, blk=(round(np.median(la)/0.0005), round(np.median(lo)/0.0005)),
            brg=bearing(la[0], lo[0], la[-1], lo[-1]), spd=spdm,
            press=float(np.mean(press[sl])),
            # AMPLITUDE (reproduce existing)
            steer_std=float(np.std(st)),
            steer_slow=band_rms(st, 0.1, 0.5),
            # NEW: multi-band steer
            b_slow=band_rms(st, 0.1, 0.3), b_weave=band_rms(st, 0.3, 0.5),
            b_hunt=band_rms(st, 0.5, 1.5), b_saw=band_rms(st, 1.5, 5.0),
            # NEW: rate / jerk / reversals (hunting CHARACTER)
            rate_std=r_std, jerk_std=j_std,
            rev_s=reversals_per_s(stf, prom=0.10),       # reversals/s, 0.10deg prominence
            rev_s_strict=reversals_per_s(stf, prom=0.30),
            # NEW: spectral centroid (deliberate-low vs nervous-high)
            centroid=spectral_centroid(st),
            # cmd channel (controller output)
            cmd_std=float(np.std(cmd[sl])), cmd_slow=band_rms(cmd[sl], 0.1, 0.5),
        )
        if have_pos:
            ps = pos[sl]
            if not np.any(np.isnan(ps)):
                w['pos_std'] = float(np.std(ps))
                w['pos_slow'] = band_rms(ps, 0.1, 0.5)
                # NEW centering: mean offset + slow drift (the detrend-std DISCARDS these)
                w['pos_mean'] = float(np.mean(ps))
                w['pos_absmean'] = float(np.abs(np.mean(ps)))
                w['pos_drift'] = low_drift_rms(ps, 0.1)      # <0.1Hz drift power
                w['steer_drift'] = low_drift_rms(st, 0.1)
                w['pos_out30'] = float(np.mean(np.abs(ps) > 0.30))
        W.append(w)
    return W


def collect(group):
    return [w for rid in GROUPS[group] for w in windows(rid)]


print('loading windows ...')
WK = collect('WEAK'); GD = collect('GOLD')
print(f'  WEAK {len(WK)} eng-straight 16s win   GOLD {len(GD)} eng-straight 16s win')
print(f'  speed: WEAK median {np.median([w["spd"] for w in WK]):.1f}  IQR '
      f'[{np.percentile([w["spd"] for w in WK],25):.1f},{np.percentile([w["spd"] for w in WK],75):.1f}]'
      f'  | GOLD median {np.median([w["spd"] for w in GD]):.1f}  IQR '
      f'[{np.percentile([w["spd"] for w in GD],25):.1f},{np.percentile([w["spd"] for w in GD],75):.1f}]')

METRICS = [
    ('steer_std', 'AMPL steer std (deg)', 'lower=better'),
    ('steer_slow', 'AMPL steer 0.1-0.5 (deg)', 'lower'),
    ('b_slow', 'BAND 0.1-0.3 slow-weave', 'lower'),
    ('b_weave', 'BAND 0.3-0.5', 'lower'),
    ('b_hunt', 'BAND 0.5-1.5 hunt', 'lower'),
    ('b_saw', 'BAND 1.5-5 saw', 'lower'),
    ('rate_std', 'RATE std (deg/s) <-hunting', 'lower'),
    ('jerk_std', 'JERK std (deg/s2) <-sharpness', 'lower'),
    ('rev_s', 'REVERSALS/s prom0.1 <-HUNTING', 'lower'),
    ('rev_s_strict', 'REVERSALS/s prom0.3', 'lower'),
    ('centroid', 'SPECTRAL centroid Hz', 'lower=deliberate'),
    ('cmd_std', 'cmd curv std', 'lower'),
    ('pos_std', 'POS std (m)', 'lower'),
    ('pos_slow', 'POS 0.1-0.5 (m)', 'lower'),
    ('pos_absmean', 'CENTERING |mean offset| m', 'lower=better-centered'),
    ('pos_drift', 'POS <0.1Hz DRIFT m <-wander', 'lower'),
    ('steer_drift', 'STEER <0.1Hz drift deg', 'lower'),
    ('pos_out30', 'frac |pos|>0.30m', 'lower'),
    ('press', 'override frac (steeringPressed)', 'lower'),
]


def med(group, m):
    return [w[m] for w in group if m in w and np.isfinite(w[m])]


def boot_ci(a, b, nb=4000):
    """bootstrap 95% CI on median(b)-median(a)."""
    a = np.array(a); b = np.array(b)
    d = np.array([np.median(rng.choice(b, len(b))) - np.median(rng.choice(a, len(a))) for _ in range(nb)])
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


print('\n================ POOLED medians (unmatched — speed-confounded) ================')
print(f'{"metric":<34}{"WEAK":>10}{"GOLD":>10}{"chg%":>8}   note')
for m, lbl, note in METRICS:
    a, b = med(WK, m), med(GD, m)
    if len(a) >= 5 and len(b) >= 5:
        ma, mb = np.median(a), np.median(b)
        print(f'  {lbl:<32}{ma:>10.4f}{mb:>10.4f}{100*(mb-ma)/ma:>+7.0f}%   {note}')


print('\n================ SPEED-MATCHED (each GOLD win <-> WEAK win within +-2.5mph), bootstrap CI ================')
print('  ⚠️ THE "sig YES/ns" FLAGS BELOW ARE UNRELIABLE: boot_ci resamples pa,pb INDEPENDENTLY (this is a PAIRED')
print('     design) and overlapping GOLD windows reuse the same WEAK windows -> pa entries correlated, effective')
print('     N << len(pa) -> CI too narrow -> significance OVERSTATED. TRUST THE POINT ESTIMATES (medians/chg%) ONLY.')
print('     For honest significance use: centering=robustness block-bootstrap; oscillation=C4 MWU; DiD=verify_did_blockbootstrap.py.')
print(f'{"metric":<34}{"WEAKm":>9}{"GOLDm":>9}{"chg%":>7}  {"95% CI on GOLD-WEAK":>22}  sig')
for m, lbl, note in METRICS:
    pa, pb = [], []
    WKm = np.array([(w['spd'], w[m]) for w in WK if m in w and np.isfinite(w[m])])
    for w in GD:
        if m not in w or not np.isfinite(w[m]): continue
        cand = WKm[np.abs(WKm[:, 0]-w['spd']) <= 2.5]
        if len(cand) >= 2:
            pa.append(np.median(cand[:, 1])); pb.append(w[m])
    if len(pb) >= 8:
        ma, mb = np.median(pa), np.median(pb)
        lo, hi = boot_ci(pa, pb)
        sig = 'YES' if (lo > 0 or hi < 0) else 'ns'
        arrow = 'GOLD better' if mb < ma else 'GOLD worse'
        print(f'  {lbl:<32}{ma:>9.4f}{mb:>9.4f}{100*(mb-ma)/ma:>+6.0f}%  [{lo:>+8.4f},{hi:>+8.4f}]  {sig} {arrow if sig=="YES" else ""}')
    else:
        print(f'  {lbl:<32}  too few matched ({len(pb)})')


print('\n================ SPEED-CURVE: is GOLD shifted BELOW WEAK at equal speed, or on the SAME curve? ================')
print('  (linear fit metric~speed in each group; compare GOLD intercept-at-shared-speed vs WEAK)')
shared_lo, shared_hi = 50, 62  # overlap region
sp_eval = 56.0
for m, lbl, note in [x for x in METRICS if x[0] in
                     ('steer_std','rate_std','rev_s','b_hunt','b_weave','centroid','pos_absmean','pos_drift','press')]:
    aw = np.array([(w['spd'], w[m]) for w in WK if m in w and np.isfinite(w[m]) and shared_lo <= w['spd'] <= shared_hi])
    bg = np.array([(w['spd'], w[m]) for w in GD if m in w and np.isfinite(w[m]) and shared_lo <= w['spd'] <= shared_hi])
    if len(aw) >= 6 and len(bg) >= 6:
        pa = np.polyfit(aw[:, 0], aw[:, 1], 1); pb = np.polyfit(bg[:, 0], bg[:, 1], 1)
        va = np.polyval(pa, sp_eval); vb = np.polyval(pb, sp_eval)
        print(f'  {lbl:<32} @ {sp_eval:.0f}mph: WEAK {va:.4f}  GOLD {vb:.4f}  ({100*(vb-va)/va:+.0f}%)  '
              f'slopeW {pa[0]:+.4f} slopeG {pb[0]:+.4f}  nW/nG {len(aw)}/{len(bg)}')
    else:
        print(f'  {lbl:<32}  too few in {shared_lo}-{shared_hi}mph ({len(aw)}/{len(bg)})')


print('\n================ LOCATION-PAIRED (same GPS ~55m cell, same dir<=60deg, |dspd|<=8) robust ================')
def gblk(group):
    bd = defaultdict(list)
    for w in group: bd[w['blk']].append(w)
    out = {}
    for k, ws in bd.items():
        bx = np.mean([math.cos(math.radians(w['brg'])) for w in ws]); by = np.mean([math.sin(math.radians(w['brg'])) for w in ws])
        o = {'brg': math.degrees(math.atan2(by, bx)) % 360, 'spd': np.median([w['spd'] for w in ws]), 'n': len(ws)}
        for m, _, _ in METRICS:
            v = [w[m] for w in ws if m in w and np.isfinite(w[m])]
            if v: o[m] = float(np.median(v))
        out[k] = o
    return out
BW, BG = gblk(WK), gblk(GD)
shared = [k for k in BW if k in BG and adiff(BW[k]['brg'], BG[k]['brg']) <= 60 and abs(BW[k]['spd']-BG[k]['spd']) <= 8]
print(f'  shared same-dir cells: {len(shared)}  (median |dspd| {np.median([abs(BW[k]["spd"]-BG[k]["spd"]) for k in shared]):.1f} mph)')
print(f'{"metric":<34}{"WEAK":>9}{"GOLD":>9}{"chg%":>7}{"gold-better":>13}')
for m, lbl, note in METRICS:
    pairs = [(BW[k][m], BG[k][m]) for k in shared if m in BW[k] and m in BG[k]]
    if len(pairs) >= 6:
        wv = [p[0] for p in pairs]; gv = [p[1] for p in pairs]
        better = sum(1 for p in pairs if p[1] < p[0])  # gold lower
        print(f'  {lbl:<32}{np.median(wv):>9.4f}{np.median(gv):>9.4f}'
              f'{100*(np.median(gv)-np.median(wv))/np.median(wv):>+6.0f}%{f"{better}/{len(pairs)}":>13}')
