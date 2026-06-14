#!/usr/bin/env python3
"""Characterize the 0.167Hz peak shape + the STEER-vs-POSITION dissociation.
Key prior result: at matched speed band-RMS is flat, BUT WEAK PSD peakedness (peak/floor)
~210 vs GOLD ~140. And full-data pooled steering PSD G/W~0.5 while lane-pos G/W~1.4.
Mechanism hypothesis: WEAK steers HARDER (more wheel motion) at the weave freq to hold
the SAME or worse path -> a 'busy wheel' limit cycle. Gold's stronger PI holds path with
LESS wheel hunting. The DRIVER FELT the WHEEL weave (visible to other cars) + the wheel
motion, which is the STEERING channel, not the lane-position channel.
Test: ratio of steering-angle weave power to lane-position weave power (gain of the
controller in the weave band). A limit cycle = high steer-per-position = wheel works hard
for little/negative path benefit. Compute per-config and per-window."""
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


def pooled_psd(routes, chan, slo, shi):
    min_len = int(MIN_WIN_S * FS)
    nper = int(16 * FS)
    psds = []
    for r in routes:
        d = load(r)
        m = gentle_mask(d, slo, shi)
        for a, b in contiguous(m, min_len):
            x = d[chan][a:b].astype(float)
            if np.any(~np.isfinite(x)):
                continue
            seglen = min(nper, len(x))
            f, p = signal.welch(x, fs=FS, nperseg=seglen, noverlap=seglen // 2,
                                detrend='constant')
            psds.append((f, p, len(x)))
    fref = max((f for f, _, _ in psds), key=len)
    acc = np.zeros_like(fref); w = 0.0
    for f, p, ww in psds:
        acc += np.interp(fref, f, p) * ww; w += ww
    return fref, acc / w


def bp(f, p, lo, hi):
    sel = (f >= lo) & (f <= hi)
    return np.trapezoid(p[sel], f[sel])


print('=== Pooled PSD zoom 0.08-0.45Hz, FULL data (30-75mph) ===')
fs_st, ps_st = pooled_psd(WEAK, 'steer', 30/2.237, 75/2.237)
fg_st, pg_st = pooled_psd(GOLD, 'steer', 30/2.237, 75/2.237)
fs_po, ps_po = pooled_psd(WEAK, 'pos', 30/2.237, 75/2.237)
fg_po, pg_po = pooled_psd(GOLD, 'pos', 30/2.237, 75/2.237)
print('  f(Hz)  STEER_W   STEER_G  S_G/W |  POS_W    POS_G   P_G/W | steer/pos_W steer/pos_G')
for ff in np.arange(0.0625, 0.46, 0.03125):
    sw = np.interp(ff, fs_st, ps_st); sg = np.interp(ff, fg_st, pg_st)
    pw = np.interp(ff, fs_po, ps_po); pg = np.interp(ff, fg_po, pg_po)
    print(f'  {ff:5.3f} {sw:8.3e} {sg:8.3e} {sg/sw:5.2f} | {pw:8.3e} {pg:8.3e} {pg/pw:5.2f} | {sw/pw:9.1f} {sg/pg:9.1f}')

# steer-per-position GAIN in weave band (sqrt of power ratio = amplitude gain)
print('\n=== Weave-band STEER/POSITION amplitude gain (deg per m) ===')
for lo, hi, lbl in [(0.1, 0.3, 'core 0.1-0.3'), (0.15, 0.4, 'weave 0.15-0.4'),
                    (0.3, 0.6, 'upper 0.3-0.6')]:
    sw = bp(fs_st, ps_st, lo, hi); pw = bp(fs_po, ps_po, lo, hi)
    sg = bp(fg_st, pg_st, lo, hi); pg = bp(fg_po, pg_po, lo, hi)
    gw = np.sqrt(sw / pw); gg = np.sqrt(sg / pg)
    print(f'  {lbl}Hz: WEAK gain={gw:.1f} deg/m  GOLD gain={gg:.1f} deg/m  G/W={gg/gw:.2f}')
    print(f'      (steerP W={sw:.3e} G={sg:.3e} G/W={sg/sw:.2f}; posP W={pw:.3e} G={pg:.3e} G/W={pg/pw:.2f})')
