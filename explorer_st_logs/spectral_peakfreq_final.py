#!/usr/bin/env python3
"""FINAL: is the steering-weave reduction concentrated at the LOCKED 0.167Hz limit-cycle
frequency? Per-window narrowband steer-RMS in tight 0.12-0.22Hz bin (around the 0.167Hz
peak / 4.5s period) vs the rest. Speed-matched. Sign-test friendly (per-window medians)."""
import numpy as np
from scipy import signal, stats

FS = 50.0
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']
GOLD = ['route_b8']
ALAT_THRESH = 1.0
MIN_WIN_S = 14.0
SOS_LC = signal.butter(4, [0.12, 0.22], btype='band', fs=FS, output='sos')   # the 0.167Hz peak
SOS_HI = signal.butter(4, [0.22, 0.45], btype='band', fs=FS, output='sos')   # upper weave


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
    lc, hi = [], []
    for r in routes:
        d = load(r)
        m = gentle_mask(d, slo, shi)
        for a, b in contiguous(m, min_len):
            st = d['steer'][a:b].astype(float)
            if np.any(~np.isfinite(st)):
                continue
            st = st - st.mean()
            lc.append(np.std(signal.sosfiltfilt(SOS_LC, st)))
            hi.append(np.std(signal.sosfiltfilt(SOS_HI, st)))
    return np.array(lc), np.array(hi)


def q(a):
    return np.median(a), np.percentile(a, 25), np.percentile(a, 75)


def run(tag, slo, shi):
    print(f'\n===== {tag} ({slo*2.237:.0f}-{shi*2.237:.0f}mph) steering-angle narrowband RMS =====')
    lcW, hiW = per_window(WEAK, slo, shi)
    lcG, hiG = per_window(GOLD, slo, shi)
    print(f'  nW={len(lcW)} nG={len(lcG)}')
    for name, aW, aG in [('LIMIT-CYCLE 0.12-0.22Hz (~0.167Hz/4.5s)', lcW, lcG),
                         ('UPPER WEAVE 0.22-0.45Hz', hiW, hiG)]:
        mW = q(aW); mG = q(aG)
        U, pv = stats.mannwhitneyu(aW, aG, alternative='two-sided')
        print(f'  {name}')
        print(f'     WEAK med={mW[0]:.4f} [{mW[1]:.4f},{mW[2]:.4f}] deg   '
              f'GOLD med={mG[0]:.4f} [{mG[1]:.4f},{mG[2]:.4f}] deg   '
              f'G/W={mG[0]/mW[0]:.2f}  p={pv:.3f}')


run('FULL', 30/2.237, 75/2.237)
run('OVERLAP', 48/2.237, 62/2.237)
