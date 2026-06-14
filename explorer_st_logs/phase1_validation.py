#!/usr/bin/env python3
"""Phase 1 validation: does cmd_cps actually map to felt wheel oscillation?

Three analyses:

  Tier 2.1: Compare CX1 cmd_cps and aLat 0.5-3 Hz power between routes
            user marked subjectively (a0, a1 = "felt jerky"; 9b = "validated baseline").
            If cmd_cps correlates with subjective feel, jerky routes should show
            higher cmd_cps and/or higher aLat 0.5-3 Hz power.

  Tier 2.2: Compute jerk RMS per route at production tau (from CX1 aLat).
            Compare across routes.

  Tier 3.1: FFT 0.5-3 Hz band power on cmd and aLat per event across many routes.

  Tier 3.2: Correlate cmd 0.5-3 Hz power with aLat 0.5-3 Hz power.
            If they correlate strongly, simulator's predicted cmd reduction
            implies real aLat reduction (= felt benefit).
            If weak correlation, cmd is not a good proxy for felt feel.

Usage:
  ./phase1_validation.py [--routes route_91,route_9b,route_9d,route_9e,route_9f,route_a0,route_a1]
"""

import sys
import os
import glob
import argparse
import numpy as np

sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_curves_v2 import load_cx1, detect_curve_events, IDX

try:
    from scipy.fft import rfft, rfftfreq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Subjective labels from MEMORY.md and customizations.md
SUBJECTIVE_LABELS = {
    'route_91': 'PI-OFF era',
    'route_9b': 'BASELINE_OK',       # 2026-05-23 validated baseline
    'route_9c': 'ITER1_Ki=3e-4',
    'route_9d': 'ITER1_Ki=3e-4',
    'route_9e': 'ITER1_Ki=3e-4',     # iter1 era
    'route_9f': 'ITER1_Ki=3e-4',
    'route_a0': 'ITER2_JERKY',       # SR=18 — user reported regression
    'route_a1': 'ITER2_JERKY',       # SR=18 — user reported regression
}


def find_rlogs(route_prefix):
    files = sorted(glob.glob(os.path.join(route_prefix, 'rlog_*.zst')))
    if files:
        return files
    seg_dirs = sorted(glob.glob(route_prefix + '--*'),
                      key=lambda d: int(d.rsplit('--', 1)[-1]))
    out = []
    for sd in seg_dirs:
        rlog = os.path.join(sd, 'rlog.zst')
        if os.path.exists(rlog):
            out.append(rlog)
    return out


def route_prefix_for(route_id):
    """Return e.g. 'explorer_st_logs/route_9b/0000009b--b2841022fd' from 'route_9b'."""
    d = f'explorer_st_logs/{route_id}'
    seg = sorted(os.listdir(d))[0] if os.path.isdir(d) else None
    if not seg:
        return None
    parts = seg.split('--')
    return f'{d}/{parts[0]}--{parts[1]}'


def band_power(sig, fs_hz, f_lo, f_hi):
    """Power in [f_lo, f_hi] Hz band via Hann-windowed rFFT."""
    if not HAS_SCIPY or len(sig) < 10:
        return float('nan')
    s = sig - np.mean(sig)
    w = np.hanning(len(s))
    yf = rfft(s * w)
    freqs = rfftfreq(len(s), d=1.0 / fs_hz)
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not mask.any():
        return 0.0
    return float(np.sum(np.abs(yf[mask]) ** 2))


def crossings_per_sec(sig, dt, threshold=4e-5):
    """Sign-change rate of d(sig)/dt above threshold."""
    if len(sig) < 3:
        return 0.0
    rate = np.diff(sig) / dt
    signs = np.where(np.abs(rate) > threshold, np.sign(rate), 0.0)
    last = 0.0
    crossings = 0
    for s in signs:
        if s != 0:
            if last != 0 and s != last:
                crossings += 1
            last = s
    duration = len(sig) * dt
    return float(crossings / duration) if duration > 0 else 0.0


def per_event_metrics(arr, t_cx1, ev, window_sec=3.0):
    """Compute per-event cps, FFT band power, jerk RMS, lag etc on production CX1 data.

    Resamples to uniform 20 Hz to bypass CX1 bursty sampling artifact.
    """
    t_lo = ev['t_apex'] - window_sec
    t_hi = ev['t_apex'] + window_sec
    mask = (t_cx1 >= t_lo) & (t_cx1 <= t_hi)
    if mask.sum() < 5:
        return None

    t_in = t_cx1[mask]
    if (t_in[-1] - t_in[0]) <= 0:
        return None

    # Uniform 20 Hz resample
    fs = 20.0
    dt = 1.0 / fs
    n = int(np.ceil((t_in[-1] - t_in[0]) / dt)) + 1
    t_u = t_in[0] + np.arange(n) * dt

    # Pull cmd, aLat, ang (steering angle)
    cmd_in = arr[mask, IDX['cmd']]
    alat_in = arr[mask, IDX['aLat']]
    ang_in = arr[mask, IDX['ang']]
    v_in = arr[mask, IDX['v']]

    cmd_u = np.interp(t_u, t_in, cmd_in)
    alat_u = np.interp(t_u, t_in, alat_in)
    ang_u = np.interp(t_u, t_in, ang_in)

    # Filter to scoring window only (drop CX1 undersampled regions: gap > 200ms)
    gaps = np.array([np.min(np.abs(t_in - g)) for g in t_u])
    score_mask = gaps < 0.2
    if score_mask.sum() < 10:
        return None
    cmd_s = cmd_u[score_mask]
    alat_s = alat_u[score_mask]
    ang_s = ang_u[score_mask]
    t_s = t_u[score_mask]

    # Crossings
    cmd_cps = crossings_per_sec(cmd_s, dt)
    # aLat will be much noisier so use higher threshold
    alat_cps = crossings_per_sec(alat_s, dt, threshold=0.05)

    # FFT band power
    cmd_band = band_power(cmd_s, fs, 0.5, 3.0)
    alat_band = band_power(alat_s, fs, 0.5, 3.0)
    ang_band = band_power(ang_s, fs, 0.5, 3.0)

    # Jerk RMS: derivative of aLat (observed wheel-felt jerk)
    jerk = np.diff(alat_s) / dt
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))
    # cmd jerk surrogate: d²cmd/dt² RMS — what the simulator can predict from cmd alone
    d2cmd = np.diff(cmd_s, n=2) / (dt ** 2)
    cmd_jerk_surrogate = float(np.sqrt(np.mean(d2cmd ** 2))) if len(d2cmd) > 0 else float('nan')

    return {
        'cmd_cps': cmd_cps,
        'alat_cps': alat_cps,
        'cmd_band_0_5_3': cmd_band,
        'alat_band_0_5_3': alat_band,
        'ang_band_0_5_3': ang_band,
        'jerk_rms': jerk_rms,
        'cmd_jerk_surrogate': cmd_jerk_surrogate,
        'mean_speed_mph': float(np.mean(v_in)) * 2.237,
        'duration': float(ev['duration_sec']),
        'magnitude_class': ev['magnitude_class'],
    }


def analyze_route(route_id):
    """Pull CX1, detect events, compute per-event metrics."""
    prefix = route_prefix_for(route_id)
    if prefix is None:
        return None
    print(f'  [{route_id}] loading...', flush=True)
    try:
        arr, t_cx1 = load_cx1(prefix)
    except Exception as e:
        print(f'    error: {e}')
        return None
    if arr is None or len(arr) < 100:
        return None
    events = detect_curve_events(arr, t_cx1)
    rows = []
    for ev in events:
        m = per_event_metrics(arr, t_cx1, ev)
        if m is not None:
            rows.append(m)
    print(f'    {len(rows)} events analyzed')
    return rows


def fmt_med_iqr(vals):
    if not vals:
        return '—'
    return f'{np.median(vals):.3f} [{np.percentile(vals, 25):.3f}, {np.percentile(vals, 75):.3f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--routes', default='route_91,route_9b,route_9c,route_9d,'
                                        'route_9e,route_9f,route_a0,route_a1')
    args = ap.parse_args()

    route_ids = args.routes.split(',')
    all_data = {}
    for r in route_ids:
        rows = analyze_route(r)
        if rows:
            all_data[r] = rows

    if not all_data:
        print('No data.')
        return

    print()
    print('=' * 130)
    print(' TIER 2.1 — SUBJECTIVE vs OBJECTIVE: cmd_cps & aLat band power across routes')
    print('=' * 130)
    print(f'  {"Route":<10} {"Label":<18} {"N":>4} {"cmd_cps":>20} {"alat_cps":>20} {"alat_band_0.5-3":>22} {"jerk_RMS":>20}')
    print('  ' + '-' * 120)
    # Aggregate by route, then by subjective label
    for route_id, rows in all_data.items():
        label = SUBJECTIVE_LABELS.get(route_id, 'unknown')
        # Filter to moderate+sharp only (where complaint is)
        rows_f = [r for r in rows if r['magnitude_class'] in ('moderate', 'sharp')]
        if not rows_f:
            continue
        n = len(rows_f)
        print(f'  {route_id:<10} {label:<18} {n:>4} '
              f'{fmt_med_iqr([r["cmd_cps"] for r in rows_f]):>20} '
              f'{fmt_med_iqr([r["alat_cps"] for r in rows_f]):>20} '
              f'{fmt_med_iqr([r["alat_band_0_5_3"] for r in rows_f]):>22} '
              f'{fmt_med_iqr([r["jerk_rms"] for r in rows_f]):>20}')

    # Group by subjective label
    print()
    print('  Aggregated by subjective label:')
    by_label = {}
    for route_id, rows in all_data.items():
        label = SUBJECTIVE_LABELS.get(route_id, 'unknown')
        if label not in by_label:
            by_label[label] = []
        by_label[label].extend([r for r in rows if r['magnitude_class'] in ('moderate', 'sharp')])
    print(f'  {"Label":<22} {"N":>4} {"cmd_cps":>20} {"alat_band_0.5-3":>22} {"jerk_RMS":>20}')
    print('  ' + '-' * 100)
    for label, rows_f in sorted(by_label.items()):
        if not rows_f:
            continue
        print(f'  {label:<22} {len(rows_f):>4} '
              f'{fmt_med_iqr([r["cmd_cps"] for r in rows_f]):>20} '
              f'{fmt_med_iqr([r["alat_band_0_5_3"] for r in rows_f]):>22} '
              f'{fmt_med_iqr([r["jerk_rms"] for r in rows_f]):>20}')

    print()
    print('=' * 130)
    print(' TIER 3.1/3.2 — DOES cmd_cps PREDICT aLat 0.5-3 Hz POWER?')
    print('=' * 130)
    print('  Pool all events across all routes; correlate cmd_cps with aLat band power.')
    print('  If R² is high, cmd reductions translate to felt-wheel oscillation reductions.')
    print('  If R² is low, optimizing cmd_cps does NOT improve subjective feel.')
    print()

    all_rows = []
    for rows in all_data.values():
        all_rows.extend([r for r in rows if r['magnitude_class'] in ('moderate', 'sharp')])
    if len(all_rows) < 10:
        print('  Not enough data.')
        return

    cmd_cps = np.array([r['cmd_cps'] for r in all_rows])
    cmd_band = np.array([r['cmd_band_0_5_3'] for r in all_rows])
    alat_cps = np.array([r['alat_cps'] for r in all_rows])
    alat_band = np.array([r['alat_band_0_5_3'] for r in all_rows])
    ang_band = np.array([r['ang_band_0_5_3'] for r in all_rows])
    jerk = np.array([r['jerk_rms'] for r in all_rows])
    cmd_jerk = np.array([r['cmd_jerk_surrogate'] for r in all_rows])
    speeds = np.array([r['mean_speed_mph'] for r in all_rows])

    def r2(x, y):
        valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if valid.sum() < 5:
            return float('nan'), 0
        xv, yv = x[valid], y[valid]
        # Spearman rank correlation (robust to outliers and non-linearity)
        rx = np.argsort(np.argsort(xv))
        ry = np.argsort(np.argsort(yv))
        return float(np.corrcoef(rx, ry)[0, 1] ** 2), int(valid.sum())

    def spearman_r(x, y):
        """Spearman correlation (signed)."""
        valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
        if valid.sum() < 5:
            return float('nan')
        rx = np.argsort(np.argsort(x[valid]))
        ry = np.argsort(np.argsort(y[valid]))
        return float(np.corrcoef(rx, ry)[0, 1])

    def partial_r2(x, y, z):
        """Partial correlation R² of x with y, controlling for z.
        Uses Spearman ranks. Formula: r_xy.z = (r_xy - r_xz*r_yz) / sqrt((1-r_xz²)(1-r_yz²))
        """
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z) |
                  np.isinf(x) | np.isinf(y) | np.isinf(z))
        if valid.sum() < 8:
            return float('nan'), 0
        xv, yv, zv = x[valid], y[valid], z[valid]
        r_xy = spearman_r(xv, yv)
        r_xz = spearman_r(xv, zv)
        r_yz = spearman_r(yv, zv)
        denom = np.sqrt(max(1 - r_xz ** 2, 1e-12) * max(1 - r_yz ** 2, 1e-12))
        if denom < 1e-9:
            return float('nan'), int(valid.sum())
        r_partial = (r_xy - r_xz * r_yz) / denom
        return float(r_partial ** 2), int(valid.sum())

    print(f'  Pairwise R² (Spearman rank, N={len(all_rows)} events):\n')
    pairs = [
        ('cmd_cps          vs aLat_cps         ', cmd_cps, alat_cps),
        ('cmd_cps          vs aLat_band_0.5-3  ', cmd_cps, alat_band),
        ('cmd_cps          vs ang_band_0.5-3   ', cmd_cps, ang_band),
        ('cmd_cps          vs jerk_RMS         ', cmd_cps, jerk),
        ('cmd_band         vs aLat_band_0.5-3  ', cmd_band, alat_band),
        ('cmd_band         vs jerk_RMS         ', cmd_band, jerk),
        ('cmd_jerk_surr    vs jerk_RMS         ', cmd_jerk, jerk),
        ('cmd_jerk_surr    vs aLat_band_0.5-3  ', cmd_jerk, alat_band),
        ('aLat_band        vs jerk_RMS         ', alat_band, jerk),
        ('speed            vs aLat_band_0.5-3  ', speeds, alat_band),
        ('speed            vs jerk_RMS         ', speeds, jerk),
        ('speed            vs cmd_cps          ', speeds, cmd_cps),
    ]
    for label, x, y in pairs:
        r2v, n = r2(x, y)
        flag = ''
        if r2v > 0.5:
            flag = ' ★ strong'
        elif r2v > 0.25:
            flag = ' moderate'
        elif r2v < 0.05:
            flag = ' negligible'
        print(f'    {label} R²={r2v:.3f} (N={n}){flag}')

    print('\n  INTERPRETATION GUIDE:')
    print('    cmd_cps vs aLat_band_0.5-3 — if strong, cmd zero-crossings predict felt wobble')
    print('    cmd_band vs aLat_band      — if strong, simulator-predicted cmd reductions')
    print('                                  would proportionally reduce wheel-felt oscillation')
    print('    speed confounds: high R² between speed and aLat suggests speed dominates over cmd')

    # FRESH QA round: partial correlations controlling for speed
    print('\n' + '=' * 130)
    print(' PARTIAL CORRELATION: cmd metrics → aLat, CONTROLLING FOR SPEED')
    print('   If R²_partial drops near zero, the apparent cmd→aLat relationship was')
    print('   actually speed-driven (both signals rise with speed together).')
    print('=' * 130)
    partial_pairs = [
        ('cmd_cps   → aLat_band | speed     ', cmd_cps, alat_band, speeds),
        ('cmd_band  → aLat_band | speed     ', cmd_band, alat_band, speeds),
        ('cmd_cps   → jerk_RMS  | speed     ', cmd_cps, jerk, speeds),
        ('cmd_band  → jerk_RMS  | speed     ', cmd_band, jerk, speeds),
    ]
    for label, x, y, z in partial_pairs:
        r2v_raw, _ = r2(x, y)
        r2v_partial, n = partial_r2(x, y, z)
        delta = r2v_partial - r2v_raw
        flag = ''
        if r2v_partial < 0.05:
            flag = ' — collapses to noise after partialing speed'
        elif delta < -0.10:
            flag = ' — much weaker after speed-control (speed was confound)'
        elif abs(delta) < 0.05:
            flag = ' — robust to speed (not a confound)'
        print(f'    {label} R²_raw={r2v_raw:.3f}, R²_partial={r2v_partial:.3f} '
              f'(N={n}, Δ={delta:+.3f}){flag}')

    # Speed-binned baseline-vs-iter2 comparison
    print('\n' + '=' * 130)
    print(' SPEED-BINNED: BASELINE_OK (9b) vs ITER2_JERKY (a0, a1) at matched speed')
    print('   If iter2 aLat > baseline aLat AT THE SAME SPEED, controller config matters.')
    print('   If equal at matched speed, the apparent regression was just speed-distribution.')
    print('=' * 130)
    baseline_rows = []
    iter2_rows = []
    for route_id, rows in all_data.items():
        label = SUBJECTIVE_LABELS.get(route_id, 'unknown')
        rows_f = [r for r in rows if r['magnitude_class'] in ('moderate', 'sharp')]
        if label == 'BASELINE_OK':
            baseline_rows.extend(rows_f)
        elif label == 'ITER2_JERKY':
            iter2_rows.extend(rows_f)

    speed_bins = [(20, 35), (35, 45), (45, 55), (55, 65)]
    print(f'  {"Speed bin":<14} {"BASELINE_OK (9b)":<35} {"ITER2_JERKY (a0+a1)":<35} {"Δ aLat_band":>14}')
    print('  ' + '-' * 100)
    for lo, hi in speed_bins:
        b_in = [r['alat_band_0_5_3'] for r in baseline_rows if lo <= r['mean_speed_mph'] < hi]
        i_in = [r['alat_band_0_5_3'] for r in iter2_rows if lo <= r['mean_speed_mph'] < hi]
        b_str = f'N={len(b_in)} med={np.median(b_in):.1f}' if b_in else 'no events'
        i_str = f'N={len(i_in)} med={np.median(i_in):.1f}' if i_in else 'no events'
        if b_in and i_in:
            delta = f'+{np.median(i_in) - np.median(b_in):+.1f}'
        else:
            delta = '—'
        print(f'  {lo}-{hi} mph     {b_str:<35} {i_str:<35} {delta:>14}')

    # Speed distribution by label
    print()
    print('  SPEED DISTRIBUTIONS (mph) per label — were iter2 routes driven faster?')
    print(f'  {"Label":<22} {"N":>4} {"med":>6} {"P25":>6} {"P75":>6} {"min":>6} {"max":>6}')
    for label, rows_f in sorted(by_label.items()):
        if not rows_f:
            continue
        sp = [r['mean_speed_mph'] for r in rows_f]
        print(f'  {label:<22} {len(sp):>4} {np.median(sp):>6.1f} '
              f'{np.percentile(sp, 25):>6.1f} {np.percentile(sp, 75):>6.1f} '
              f'{np.min(sp):>6.1f} {np.max(sp):>6.1f}')

    # Regression cmd_band → aLat_band for translating simulator predictions
    print('\n' + '=' * 130)
    print(' REGRESSION: aLat_band = a + b × cmd_band — for translating sweep predictions to aLat')
    print('=' * 130)
    valid = ~(np.isnan(cmd_band) | np.isnan(alat_band) | np.isinf(cmd_band) | np.isinf(alat_band))
    cb = cmd_band[valid]
    ab = alat_band[valid]
    # Linear fit
    a, b = np.polyfit(cb, ab, 1)  # ab = b*cb + a (note np.polyfit returns highest first)
    # Slope sign and meaning
    print(f'    Linear fit: aLat_band = {b:.3f} + ({a:.3e}) × cmd_band  (N={len(cb)})')
    # Log-log fit (often more meaningful for power signals)
    valid2 = (cb > 0) & (ab > 0)
    if valid2.sum() > 5:
        lc, la = np.polyfit(np.log(cb[valid2]), np.log(ab[valid2]), 1)
        print(f'    Log-log fit: log(aLat_band) = {la:.2f} + {lc:.3f} × log(cmd_band)  (N={valid2.sum()})')
        print(f'    Power-law: aLat_band ∝ cmd_band^{lc:.2f}')
        print(f'    → A {{p}}% reduction in cmd_band predicts ~{{p*lc:.2f}}% reduction in aLat_band')
        # Example translations
        print('\n    Translation table (predicted aLat_band % reduction given cmd_band % reduction):')
        for pct in [10, 20, 30, 50]:
            # If cmd_band drops by pct%, new cmd_band = cb_old * (1 - pct/100)
            # New aLat_band = aLat_band * (1 - pct/100)^lc
            ratio_drop = 1 - (1 - pct / 100) ** lc
            print(f'      cmd_band -{pct}% → aLat_band -{ratio_drop * 100:.1f}%')


if __name__ == '__main__':
    main()
