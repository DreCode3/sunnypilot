#!/usr/bin/env python3
"""Limit-cycle SIGNATURE test (not just band power).
A self-sustaining limit cycle = a SHARP, PERSISTENT, frequency-LOCKED peak. Road-driven
motion = broadband. Band-RMS is FLAT at matched speed (confirmed) -> so test the
SHAPE/PERSISTENCE instead:
  (1) Spectral PEAKEDNESS per window: max(PSD in 0.15-0.6Hz)/median(PSD in 0.1-1.5Hz).
      High = sharp peak = limit cycle.
  (2) Peak FREQUENCY consistency across windows (limit cycle locks to one f; road varies).
  (3) Temporal SINUSOIDALITY: autocorrelation of band-passed (0.15-0.6Hz) steering angle
      -> a limit cycle gives a strong, slowly-decaying oscillatory ACF (high 1st-lag
      envelope, clear periodicity); broadband gives a fast-decaying ACF.
  (4) Crest/kurtosis of the band-passed signal.
All computed PER WINDOW -> report median + distribution. Speed-matched 48-62mph + full.
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


# bandpass 0.15-0.6Hz (the weave / limit-cycle band)
SOS = signal.butter(4, [0.15, 0.6], btype='band', fs=FS, output='sos')


def osc_acf_strength(xbp):
    """Return (acf_env, dominant_period_s). acf_env = magnitude of the strongest
    oscillatory ACF lobe beyond lag 0 -> high for a sustained sinusoid."""
    x = xbp - xbp.mean()
    n = len(x)
    ac = signal.correlate(x, x, mode='full')[n - 1:]
    ac = ac / ac[0]
    # search lags 0.8-7s for the first strong positive peak (period of the oscillation)
    lo, hi = int(0.8 * FS), min(int(7 * FS), n - 1)
    if hi <= lo:
        return np.nan, np.nan
    seg = ac[lo:hi]
    pk = np.argmax(seg)
    return seg[pk], (lo + pk) / FS


def per_window(routes, slo, shi):
    min_len = int(MIN_WIN_S * FS)
    nper = int(12 * FS)
    peakedness, pkfreq, acfenv, period, crest = [], [], [], [], []
    for r in routes:
        d = load(r)
        m = gentle_mask(d, slo, shi)
        for a, b in contiguous(m, min_len):
            x = d['steer'][a:b].astype(float)
            if np.any(~np.isfinite(x)):
                continue
            # PSD peakedness
            seglen = min(nper, len(x))
            f, p = signal.welch(x, fs=FS, nperseg=seglen, noverlap=seglen // 2,
                                detrend='constant')
            band = (f >= 0.15) & (f <= 0.6)
            floor = (f >= 0.1) & (f <= 1.5)
            pk_i = np.argmax(p[band])
            peakedness.append(p[band][pk_i] / np.median(p[floor]))
            pkfreq.append(f[band][pk_i])
            # band-passed temporal metrics
            xbp = signal.sosfiltfilt(SOS, x - x.mean())
            env, per = osc_acf_strength(xbp)
            acfenv.append(env)
            period.append(per)
            crest.append(np.max(np.abs(xbp)) / (np.std(xbp) + 1e-9))
    return dict(peakedness=np.array(peakedness), pkfreq=np.array(pkfreq),
                acfenv=np.array(acfenv), period=np.array(period),
                crest=np.array(crest))


def show(tag, slo, shi):
    print(f'\n===== {tag} (speed {slo*2.237:.0f}-{shi*2.237:.0f}mph) =====')
    W = per_window(WEAK, slo, shi)
    G = per_window(GOLD, slo, shi)
    print(f'   nW={len(W["peakedness"])}  nG={len(G["peakedness"])}')

    def med(a):
        return (np.median(a), np.percentile(a, 25), np.percentile(a, 75))
    for k, lbl in [('peakedness', 'PSD peakedness (peak/floor) [limit-cycle sharpness]'),
                   ('acfenv', 'ACF osc envelope [sustained sinusoid 0=none 1=pure]'),
                   ('crest', 'band crest factor'),
                   ('pkfreq', 'peak freq Hz'),
                   ('period', 'ACF period s')]:
        wm = med(W[k])
        gm = med(G[k])
        print(f'   {lbl}')
        print(f'      WEAK med={wm[0]:.3f} IQR[{wm[1]:.3f},{wm[2]:.3f}]   GOLD med={gm[0]:.3f} IQR[{gm[1]:.3f},{gm[2]:.3f}]')
    # peak-freq lock: std of peak freq across windows (limit cycle -> low spread)
    print(f'   peak-freq SPREAD (std across windows): WEAK={np.std(W["pkfreq"]):.3f}Hz GOLD={np.std(G["pkfreq"]):.3f}Hz')


show('LIMIT-CYCLE SIGNATURE', 30/2.237, 75/2.237)
show('LIMIT-CYCLE SIGNATURE', 48/2.237, 62/2.237)
