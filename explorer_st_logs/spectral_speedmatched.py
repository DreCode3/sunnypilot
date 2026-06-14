#!/usr/bin/env python3
"""ANGLE 2 refinement: speed-matched + finer-resolution Welch PSD.
The first pass showed steering-angle/rate PSD G/W ~0.5 (gold weaves HALF as much
at the wheel) but lane-pos G/W ~1.4. But GOLD is ~54mph and WEAK ~42-50mph -> speed
confound. Restrict BOTH groups to the OVERLAP band 48-60mph and recompute.
Also use longer Welch segments (16s -> 0.0625Hz res) to actually RESOLVE the peak,
and normalize steer angle to a curvature-equivalent (deg/v? no -- just report raw
plus a per-window VARIANCE check)."""
import numpy as np
from scipy import signal

FS = 50.0
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']
GOLD = ['route_b8']
ALAT_THRESH = 1.0
MIN_WIN_S = 18.0


def load(r):
    return np.load(f'explorer_st_logs/_cache_reassess/{r}.npz')


def gentle_mask(d, slo, shi):
    alat = np.abs(d['yaw'] * d['spd'])
    eng = d['latact'] == 1
    press = d['press'] > 0.5
    m = eng & (~press) & (alat < ALAT_THRESH) & (d['spd'] > slo) & (d['spd'] < shi)
    m &= np.abs(d['steer']) < 90.0
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


def collect(routes, chan, slo, shi, rate=False):
    min_len = int(MIN_WIN_S * FS)
    nper = int(16 * FS)
    psds = []
    total_s = 0.0
    nwin = 0
    for r in routes:
        d = load(r)
        m = gentle_mask(d, slo, shi)
        for a, b in contiguous(m, min_len):
            x = d['steer' if chan == 'steer' else chan][a:b].astype(float)
            if np.any(~np.isfinite(x)):
                continue
            if rate:
                x = np.diff(x) * FS
            seglen = min(nper, len(x))
            f, p = signal.welch(x, fs=FS, nperseg=seglen, noverlap=seglen // 2,
                                detrend='constant', scaling='density')
            psds.append((f, p, len(x)))
            total_s += len(x) / FS
            nwin += 1
    if not psds:
        return None, None, 0, 0
    fref = max((f for f, _, _ in psds), key=len)
    acc = np.zeros_like(fref)
    wsum = 0.0
    for f, p, w in psds:
        acc += np.interp(fref, f, p) * w
        wsum += w
    return fref, acc / wsum, total_s, nwin


def bandpow(f, p, lo, hi):
    sel = (f >= lo) & (f <= hi)
    return np.trapezoid(p[sel], f[sel])


def run(chan, lbl, rate=False, slo=48/2.237, shi=60/2.237):
    fw, pw, sw, nw = collect(WEAK, chan, slo, shi, rate)
    fg, pg, sg, ng = collect(GOLD, chan, slo, shi, rate)
    if fw is None or fg is None:
        print(f'{lbl}: insufficient windows W={sw}s G={sg}s')
        return
    fref = fw if len(fw) <= len(fg) else fg
    pwi = np.interp(fref, fw, pw)
    pgi = np.interp(fref, fg, pg)
    print(f'\n===== {lbl} (speed {slo*2.237:.0f}-{shi*2.237:.0f}mph) =====')
    print(f'WEAK {sw:.0f}s/{nw}win  GOLD {sg:.0f}s/{ng}win')
    for lo, hi, bl in [(0.05, 0.2, 'verylow'), (0.1, 0.5, 'weave'),
                       (0.2, 0.5, 'mid'), (0.5, 1.0, 'hunt-lo'),
                       (1.0, 2.0, 'hi')]:
        wp = bandpow(fref, pwi, lo, hi)
        gp = bandpow(fref, pgi, lo, hi)
        print(f'  {bl:8s} {lo:.2f}-{hi:.2f}Hz  W={wp:.3e} G={gp:.3e}  G/W={gp/wp:.2f}')
    print('  f(Hz)   WEAK         GOLD         G/W')
    for ff in np.arange(0.0625, 1.51, 0.0625):
        wv = np.interp(ff, fref, pwi)
        gv = np.interp(ff, fref, pgi)
        print(f'  {ff:5.3f} {wv:11.4e} {gv:11.4e}  {gv/wv if wv>0 else 0:5.2f}')


if __name__ == '__main__':
    run('steer', 'STEERING ANGLE (deg)')
    run('steer', 'STEERING RATE (deg/s)', rate=True)
    run('pos', 'LANE POSITION (m)')
    run('cmd', 'COMMANDED CURVATURE (1/m)')
