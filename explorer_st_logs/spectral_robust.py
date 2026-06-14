#!/usr/bin/env python3
"""Robustness check for the 0.2-0.5Hz limit-cycle finding.
Concerns: (1) GOLD thin at 48-60mph (1 window). (2) pooled PSD could be outlier-driven.
Fixes:
  - Per-WINDOW band-power distribution (median + IQR), not just pooled mean PSD.
  - Show GOLD data availability across speed bands.
  - Compute a dimensionless 'weave concentration': fraction of 0.1-1.5Hz steering-angle
    power that sits in the 0.2-0.5Hz sub-band -> SPEED-ROBUST (ratio cancels overall scale).
    A self-sustaining limit cycle = a high concentration in a narrow band.
"""
import numpy as np
from scipy import signal

FS = 50.0
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']
GOLD = ['route_b8']
ALAT_THRESH = 1.0
MIN_WIN_S = 14.0


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


def win_psd(x, nper):
    seglen = min(nper, len(x))
    return signal.welch(x, fs=FS, nperseg=seglen, noverlap=seglen // 2,
                        detrend='constant', scaling='density')


def bp(f, p, lo, hi):
    sel = (f >= lo) & (f <= hi)
    return np.trapezoid(p[sel], f[sel])


def per_window_stats(routes, chan, slo, shi, rate=False):
    min_len = int(MIN_WIN_S * FS)
    nper = int(12 * FS)
    rows = []  # (dur, concentration, midband_pow, mean_spd)
    for r in routes:
        d = load(r)
        m = gentle_mask(d, slo, shi)
        for a, b in contiguous(m, min_len):
            x = d['steer'][a:b].astype(float)
            if np.any(~np.isfinite(x)):
                continue
            if rate:
                x = np.diff(x) * FS
            f, p = win_psd(x, nper)
            total = bp(f, p, 0.1, 1.5)
            mid = bp(f, p, 0.2, 0.5)
            conc = mid / total if total > 0 else np.nan
            rows.append((len(x) / FS, conc, mid, np.mean(d['spd'][a:b]) * 2.237))
    return rows


def summarize(name, rows):
    if not rows:
        print(f'{name}: NO windows')
        return
    rows = np.array(rows)
    dur, conc, mid, spd = rows.T
    print(f'{name}: n={len(rows)} totdur={dur.sum():.0f}s spd[mph] med={np.median(spd):.1f} ({spd.min():.0f}-{spd.max():.0f})')
    print(f'   0.2-0.5Hz CONCENTRATION (frac of 0.1-1.5 pow): median={np.median(conc):.3f} IQR=[{np.percentile(conc,25):.3f},{np.percentile(conc,75):.3f}]')
    print(f'   0.2-0.5Hz POWER: median={np.median(mid):.3e} IQR=[{np.percentile(mid,25):.3e},{np.percentile(mid,75):.3e}]')


print('=== GOLD data availability by speed band (gentle straights) ===')
for lo, hi in [(30, 48), (48, 55), (55, 62), (62, 75)]:
    rows = per_window_stats(GOLD, 'steer', lo / 2.237, hi / 2.237)
    tot = sum(r[0] for r in rows)
    print(f'  {lo}-{hi}mph: {len(rows)}win {tot:.0f}s')

print('\n=== STEERING ANGLE 0.2-0.5Hz concentration (speed-robust) ===')
print('-- full band 30-75mph --')
summarize('WEAK', per_window_stats(WEAK, 'steer', 30/2.237, 75/2.237))
summarize('GOLD', per_window_stats(GOLD, 'steer', 30/2.237, 75/2.237))
print('-- overlap 48-62mph --')
summarize('WEAK', per_window_stats(WEAK, 'steer', 48/2.237, 62/2.237))
summarize('GOLD', per_window_stats(GOLD, 'steer', 48/2.237, 62/2.237))

print('\n=== STEERING RATE 0.2-0.5Hz concentration ===')
print('-- overlap 48-62mph --')
summarize('WEAK', per_window_stats(WEAK, 'steer', 48/2.237, 62/2.237, rate=True))
summarize('GOLD', per_window_stats(GOLD, 'steer', 48/2.237, 62/2.237, rate=True))
