#!/usr/bin/env python3
"""ANGLE 2: spectral peak / limit-cycle hunt.
Per-config averaged Welch PSD of steeringAngleDeg, steering-RATE, and lane position
on ENGAGED STRAIGHT/GENTLE segments. Look for a limit-cycle PEAK in 0.1-0.8Hz that
is present in WEAK (CD210+weak PI, unusable) and reduced/absent/shifted in GOLD.
"""
import sys
import numpy as np
from scipy import signal

FS = 50.0
WEAK = ['route_b1', 'route_b2', 'route_b4', 'route_b5']
GOLD = ['route_b8']

# straight/gentle: |yawRate*vEgo| < THRESH (m/s^2 lateral accel); engaged; speed band
ALAT_THRESH = 1.0   # m/s^2
SPD_LO, SPD_HI = 30/2.237, 75/2.237   # mph -> m/s
MIN_WIN_S = 12.0    # minimum contiguous window seconds for a usable PSD chunk


def load(r):
    return np.load(f'explorer_st_logs/_cache_reassess/{r}.npz')


def gentle_mask(d):
    yaw = d['yaw']          # rad/s
    spd = d['spd']          # m/s
    alat = np.abs(yaw * spd)
    eng = d['latact'] == 1
    press = d['press'] > 0.5 if 'press' in d else np.zeros(len(spd), bool)
    m = eng & (~press) & (alat < ALAT_THRESH) & (spd > SPD_LO) & (spd < SPD_HI)
    # drop wild steering outliers (parking-lot artifacts)
    m &= np.abs(d['steer']) < 90.0
    return m


def contiguous(mask, min_len):
    """yield (start,end) index ranges where mask True for >= min_len samples."""
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


def avg_psd(routes, chan, detrend='constant'):
    """Welch PSD averaged across all gentle windows in routes, length-weighted."""
    min_len = int(MIN_WIN_S * FS)
    nper = int(8 * FS)   # 8s segments -> ~0.125Hz resolution
    psds = []
    weights = []
    total_s = 0.0
    nwin = 0
    for r in routes:
        d = load(r)
        m = gentle_mask(d)
        for a, b in contiguous(m, min_len):
            x = d[chan][a:b].astype(float)
            if chan == 'steer_rate':
                continue
            if np.any(~np.isfinite(x)):
                continue
            seglen = min(nper, len(x))
            f, p = signal.welch(x, fs=FS, nperseg=seglen,
                                noverlap=seglen // 2, detrend=detrend,
                                scaling='density')
            psds.append((f, p, len(x)))
            weights.append(len(x))
            total_s += len(x) / FS
            nwin += 1
    if not psds:
        return None, None, 0, 0
    # interpolate all onto the finest common freq grid (use the longest f)
    fref = max((f for f, _, _ in psds), key=len)
    acc = np.zeros_like(fref)
    wsum = 0.0
    for f, p, w in psds:
        pi = np.interp(fref, f, p)
        acc += pi * w
        wsum += w
    return fref, acc / wsum, total_s, nwin


def avg_psd_rate(routes):
    """PSD of steering RATE (deg/s) = diff of steer * FS."""
    min_len = int(MIN_WIN_S * FS)
    nper = int(8 * FS)
    psds = []
    total_s = 0.0
    nwin = 0
    for r in routes:
        d = load(r)
        m = gentle_mask(d)
        for a, b in contiguous(m, min_len):
            x = d['steer'][a:b].astype(float)
            if np.any(~np.isfinite(x)):
                continue
            xr = np.diff(x) * FS
            seglen = min(nper, len(xr))
            f, p = signal.welch(xr, fs=FS, nperseg=seglen,
                                noverlap=seglen // 2, detrend='constant',
                                scaling='density')
            psds.append((f, p, len(xr)))
            total_s += len(xr) / FS
            nwin += 1
    if not psds:
        return None, None, 0, 0
    fref = max((f for f, _, _ in psds), key=len)
    acc = np.zeros_like(fref)
    wsum = 0.0
    for f, p, w in psds:
        pi = np.interp(fref, f, p)
        acc += pi * w
        wsum += w
    return fref, acc / wsum, total_s, nwin


def band_metrics(f, p, lo, hi):
    sel = (f >= lo) & (f <= hi)
    fb, pb = f[sel], p[sel]
    pk_i = np.argmax(pb)
    pk_f = fb[pk_i]
    pk_h = pb[pk_i]
    band_pow = np.trapezoid(pb, fb)
    # prominence: peak height / median in band (limit-cycle = sharp peak above broadband)
    prom = pk_h / np.median(pb)
    return pk_f, pk_h, band_pow, prom


def report_chan(name, chan, rate=False):
    print(f'\n===== {name} =====')
    if rate:
        fw, pw, sw, nw = avg_psd_rate(WEAK)
        fg, pg, sg, ng = avg_psd_rate(GOLD)
    else:
        fw, pw, sw, nw = avg_psd(WEAK, chan)
        fg, pg, sg, ng = avg_psd(GOLD, chan)
    print(f'WEAK: {sw:.0f}s in {nw} windows | GOLD: {sg:.0f}s in {ng} windows')
    # common grid
    fref = fw if len(fw) <= len(fg) else fg
    pwi = np.interp(fref, fw, pw)
    pgi = np.interp(fref, fg, pg)
    for lo, hi, lbl in [(0.1, 0.5, 'WEAVE 0.1-0.5'), (0.1, 0.8, 'LC 0.1-0.8'),
                        (0.5, 1.5, 'HUNT 0.5-1.5')]:
        wf, wh, wp, wpr = band_metrics(fref, pwi, lo, hi)
        gf, gh, gp, gpr = band_metrics(fref, pgi, lo, hi)
        print(f'  [{lbl}Hz] WEAK peak@{wf:.3f}Hz h={wh:.3e} prom={wpr:.2f} bandP={wp:.3e}')
        print(f'  {" "*(len(lbl)+5)} GOLD peak@{gf:.3f}Hz h={gh:.3e} prom={gpr:.2f} bandP={gp:.3e}')
        print(f'  {" "*(len(lbl)+5)} ratio G/W: peakH={gh/wh:.2f} bandP={gp/wp:.2f} prom={gpr/wpr:.2f}')
    # dump the spectrum 0-2Hz for eyeballing
    print('  f(Hz)   WEAK_psd     GOLD_psd     G/W')
    for ff in np.arange(0.0, 2.01, 0.125):
        wv = np.interp(ff, fref, pwi)
        gv = np.interp(ff, fref, pgi)
        print(f'  {ff:5.3f} {wv:11.4e} {gv:11.4e}  {gv/wv if wv>0 else 0:6.2f}')
    return fref, pwi, pgi


if __name__ == '__main__':
    report_chan('STEERING ANGLE (deg)', 'steer')
    report_chan('LANE POSITION (m)', 'pos')
    report_chan('STEERING RATE (deg/s)', 'steer', rate=True)
