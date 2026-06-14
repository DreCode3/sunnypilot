#!/usr/bin/env python3
"""ROBUST per-window confirmation of the steer-per-position weave-band gain finding.
The pooled PSD said WEAK weave-band steer/pos amplitude gain ~12 deg/m vs GOLD ~6 (G/W~0.48).
Pooled metrics burned us 3x before -> confirm with PER-WINDOW median + sign test, speed matched.
For each engaged gentle window: bandpass steer & pos to 0.12-0.45Hz, gain = std(steer_bp)/std(pos_bp).
Report median gain per config + Mann-Whitney + speed bins. Also raw band steer-RMS & pos-RMS per window.
"""
import numpy as np
from scipy import signal, stats

FS = 50.0
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']
GOLD = ['route_b8']
ALAT_THRESH = 1.0
MIN_WIN_S = 14.0
SOS = signal.butter(4, [0.12, 0.45], btype='band', fs=FS, output='sos')


def load(r):
    return np.load(f'explorer_st_logs/_cache_reassess/{r}.npz')


def gentle_mask(d, slo, shi):
    alat = np.abs(d['yaw'] * d['spd'])
    m = (d['latact'] == 1) & (d['press'] <= 0.5) & (alat < ALAT_THRESH)
    m &= (d['spd'] > slo) & (d['spd'] < shi) & (np.abs(d['steer']) < 90.0)
    return m


def contiguous(mask, min_len):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], splits + 1))
    ends = np.concatenate((splits, [len(idx) - 1]))
    for s, e in zip(starts, ends):
        a, b = idx[s], idx[e]
        if (b - a + 1) >= min_len:
            yield a, b + 1


def per_window(routes, slo, shi):
    min_len = int(MIN_WIN_S * FS)
    gains, steerrms, posrms, spd = [], [], [], []
    for r in routes:
        d = load(r)
        m = gentle_mask(d, slo, shi)
        for a, b in contiguous(m, min_len):
            st = d['steer'][a:b].astype(float)
            po = d['pos'][a:b].astype(float)
            if np.any(~np.isfinite(st)) or np.any(~np.isfinite(po)):
                continue
            stb = signal.sosfiltfilt(SOS, st - st.mean())
            pob = signal.sosfiltfilt(SOS, po - po.mean())
            sr, pr = np.std(stb), np.std(pob)
            if pr < 1e-6:
                continue
            gains.append(sr / pr)
            steerrms.append(sr)
            posrms.append(pr)
            spd.append(np.mean(d['spd'][a:b]) * 2.237)
    return (np.array(gains), np.array(steerrms), np.array(posrms), np.array(spd))


def q(a):
    return np.median(a), np.percentile(a, 25), np.percentile(a, 75)


def run(tag, slo, shi):
    print(f'\n===== {tag} ({slo*2.237:.0f}-{shi*2.237:.0f}mph), bp 0.12-0.45Hz =====')
    gW, srW, prW, spW = per_window(WEAK, slo, shi)
    gG, srG, prG, spG = per_window(GOLD, slo, shi)
    print(f'  nW={len(gW)} (spd med {np.median(spW):.1f})   nG={len(gG)} (spd med {np.median(spG):.1f})')
    for name, aW, aG, unit in [('STEER/POS GAIN', gW, gG, 'deg/m'),
                               ('STEER weave-RMS', srW, srG, 'deg'),
                               ('POS weave-RMS', prW, prG, 'm')]:
        mW, lW, uW = q(aW)
        mG, lG, uG = q(aG)
        try:
            U, pv = stats.mannwhitneyu(aW, aG, alternative='two-sided')
        except Exception:
            pv = np.nan
        print(f'  {name:16s} WEAK med={mW:.4g} [{lW:.4g},{uW:.4g}] {unit}  '
              f'GOLD med={mG:.4g} [{lG:.4g},{uG:.4g}]  G/W={mG/mW:.2f}  MWU p={pv:.3f}')


run('FULL', 30/2.237, 75/2.237)
run('OVERLAP', 48/2.237, 62/2.237)
run('TIGHT', 55/2.237, 62/2.237)
