#!/usr/bin/env python3
"""Trace zero-crossings through the curvature pipeline, per event.

Stages (in pipeline order):
  des    — planner output (controlsd actuators.curvature)
  pred   — model predicted curvature (model.orientationRate.z / vEgo, time-shifted)
  ema    — post-blend, post-EMA stabilizer (smooth_curvature_last)
  preRL  — post-PI (ema + Kp*lane_offset + Ki*integral)
  cmd    — post-rate-limit (apply_curv_send to CAN)

If the user's "hunting" complaint maps to dCmd/dt zero-crossings, this script
shows where in the pipeline those crossings ORIGINATE. Each stage either:
  - introduces new crossings (delta > 0 vs prior stage)
  - filters them out (delta < 0)
  - passes them through (delta ≈ 0)

Usage:
  ./planner_trace.py <route_prefix> [--top N] [--mag {sharp,moderate,gentle}]
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_curves_v2 import (load_cx1, detect_curve_events, extract_waveform,
                                compute_signal_fft, IDX)


# pmd is logged in CX1 as pred_minus_des — direct planner/model disagreement signal
PIPELINE_STAGES = ['des', 'pred', 'ema', 'preRL', 'cmd', 'pmd']

# QA round 3 fix: log precision is %+.6f (1e-6 1/m) for ALL pipeline stages, not
# 2e-5 (which was cmd's pre-CAN quantization). Threshold scaled to log precision
# so PI contribution (rms ~6e-5, p95 ~2e-4 1/m per cycle → 1e-3 to 4e-3 1/m/s rate
# at 50Hz) is no longer invisible.
LOG_PRECISION = 1e-6     # 1/m, from carcontroller.py:536 format "%+.6f"
RATE_THRESHOLD_K = 2.0   # K * LOG_PRECISION / dt = noise-floor multiple


def signal_crossings_per_sec(waveform, signal_key, sample_rate_hz=20.0,
                             smoothing_window=0):
    """Crossings/sec for a single pipeline signal — QA round 3 corrected.

    Fixes vs prior version:
    - Drop default boxcar smoothing (was 5-sample = ~4 Hz LP applied uniformly
      regardless of how filtered the stage already was; biased downstream stages
      to look smoother). Caller can opt back in via smoothing_window>0.
    - Threshold derived from log precision (1e-6 1/m at %+.6f), not from cmd
      quantum (2e-5). Detects PI-scale contribution (>1e-3 1/m/s rate) instead
      of hiding it under the floor.

    Returns crossings/sec, raw count, peak/std rate, sample counts.
    """
    if waveform is None or signal_key not in waveform:
        return None
    sig_raw = np.array(waveform[signal_key])
    t_raw = np.array(waveform['t_rel'])
    if len(sig_raw) != len(t_raw) or len(sig_raw) < max(smoothing_window, 1) + 5:
        return None
    if (t_raw[-1] - t_raw[0]) <= 0:
        return None

    dt_uniform = 1.0 / sample_rate_hz
    n_uni = int(np.ceil((t_raw[-1] - t_raw[0]) / dt_uniform)) + 1
    t_uni = t_raw[0] + np.arange(n_uni) * dt_uniform
    sig_uni = np.interp(t_uni, t_raw, sig_raw)

    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        sig_s = np.convolve(sig_uni, kernel, mode='same')
    else:
        sig_s = sig_uni
    rate = np.diff(sig_s) / dt_uniform
    threshold = RATE_THRESHOLD_K * LOG_PRECISION / dt_uniform  # ~4e-5 1/m/s

    valid = np.abs(rate) > threshold
    nonzero = rate[valid]
    if len(nonzero) < 2:
        cps = 0.0
        crossings = 0
    else:
        signs = np.sign(nonzero)
        crossings = int(np.sum(np.diff(signs) != 0))
        duration = float(t_uni[-1] - t_uni[0])
        cps = float(crossings / duration) if duration > 0 else 0.0
    return {
        'crossings_per_sec': cps,
        'zero_crossings': crossings,
        'peak_rate_per_s': float(np.max(np.abs(rate))) if len(rate) else 0.0,
        'std_rate_per_s': float(np.std(rate)) if len(rate) else 0.0,
        'n_samples': int(n_uni),
        'signal_std': float(np.std(sig_uni)),
        'threshold_used': float(threshold),
    }


def detect_pred_fallback(waveform):
    """Detect samples where pred==des exactly (the v<=1.0 m/s fallback in
    carcontroller.py:258-260). Returns fraction of samples where the fallback
    was active; if >25%, pred values are largely just des and shouldn't be
    counted as an independent jitter source for that event.
    """
    if waveform is None or 'pred' not in waveform or 'des' not in waveform:
        return None
    pred = np.array(waveform['pred'])
    des = np.array(waveform['des'])
    if len(pred) != len(des) or len(pred) == 0:
        return None
    same = np.isclose(pred, des, atol=1e-9)
    return {
        'fallback_frac': float(same.mean()),
        'fallback_samples': int(same.sum()),
        'total_samples': int(len(pred)),
    }


def trace_event(arr, t_log, event, prev_event=None, next_event=None):
    """Trace one event through every pipeline stage."""
    wf = extract_waveform(arr, t_log, event, prev_event=prev_event, next_event=next_event)
    if wf is None:
        return None

    stages_raw = {}     # no boxcar — raw crossings
    stages_smooth = {}  # 5-sample boxcar — smoothed crossings (for comparison)
    for sig in PIPELINE_STAGES:
        stages_raw[sig] = signal_crossings_per_sec(wf, sig, smoothing_window=0)
        stages_smooth[sig] = signal_crossings_per_sec(wf, sig, smoothing_window=5)

    fft_pcts = {}
    for sig in PIPELINE_STAGES:
        fft = compute_signal_fft(wf, sig)
        if fft is not None:
            fft_pcts[sig] = {
                'pct_1_3': fft.get('pct_1_3', 0),
                'dom_hz': fft.get('dom_hz', 0),
                'total_power': fft.get('total_power', 0),
            }
        else:
            fft_pcts[sig] = None

    fallback = detect_pred_fallback(wf)

    return {
        'metadata': event,
        'stages': stages_raw,           # primary view
        'stages_smoothed': stages_smooth,  # boxcar-filtered for comparison
        'fft_pcts': fft_pcts,
        'pred_fallback': fallback,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route_prefix', help='Route prefix path (same format as analyze_curves_v2)')
    ap.add_argument('--top', type=int, default=5, help='Number of top-crossings events to show in detail')
    ap.add_argument('--mag', default='moderate', choices=['sharp', 'moderate', 'gentle'],
                    help='Magnitude class to filter on (default: moderate)')
    args = ap.parse_args()

    print(f'Loading CX1 from {args.route_prefix} ...', flush=True)
    arr, t_log = load_cx1(args.route_prefix)
    print(f'  {len(arr)} rows, {t_log[-1] - t_log[0]:.1f}s')

    events = detect_curve_events(arr, t_log)
    print(f'  {len(events)} curve events detected')
    events = [e for e in events if e['magnitude_class'] == args.mag]
    print(f'  {len(events)} {args.mag} events after magnitude filter')

    # Trace all
    traces = []
    for i, ev in enumerate(events):
        prev_ev = events[i - 1] if i > 0 else None
        next_ev = events[i + 1] if i + 1 < len(events) else None
        tr = trace_event(arr, t_log, ev, prev_event=prev_ev, next_event=next_ev)
        if tr is not None:
            tr['idx'] = i
            traces.append(tr)

    if not traces:
        print('No traceable events.')
        return

    # Flag pred-fallback events (where pred ≈ des on >25% of samples — low speed)
    fb_flagged = [tr for tr in traces if tr.get('pred_fallback') and tr['pred_fallback']['fallback_frac'] > 0.25]
    if fb_flagged:
        print(f'\n  NOTE: {len(fb_flagged)}/{len(traces)} events have >25% pred=des fallback (low-speed segment). '
              f'pred values on those events are not an independent jitter source.')

    print(f'\n{"="*100}')
    print(f'PIPELINE-STAGE CROSSINGS/SEC — RAW (no smoothing) — median across {len(traces)} {args.mag} events')
    print(f'  threshold = {2.0 * 1e-6 / 0.05:.2e} 1/m/s (= 2 × log precision / dt). Boxcar smoothing OFF.')
    print(f'{"="*100}')
    print(f'  {"Stage":<8} {"cps_med":>9} {"cps_p25":>9} {"cps_p75":>9} {"cps_p90":>9} '
          f'{"std_rate_med":>14} {"pct_1_3_med":>13}')
    print('  ' + '-' * 90)
    for stage in PIPELINE_STAGES:
        cps_vals = [tr['stages'][stage]['crossings_per_sec'] for tr in traces if tr['stages'][stage]]
        std_vals = [tr['stages'][stage]['std_rate_per_s'] for tr in traces if tr['stages'][stage]]
        pct13_vals = [tr['fft_pcts'][stage]['pct_1_3'] for tr in traces if tr['fft_pcts'][stage]]
        if not cps_vals:
            print(f'  {stage:<8} {"—":>9}')
            continue
        print(f'  {stage:<8} {np.median(cps_vals):>9.3f} {np.percentile(cps_vals, 25):>9.3f} '
              f'{np.percentile(cps_vals, 75):>9.3f} {np.percentile(cps_vals, 90):>9.3f} '
              f'{np.median(std_vals):>14.5f} '
              f'{(np.median(pct13_vals) if pct13_vals else 0):>13.2%}')

    print(f'\n{"="*100}')
    print(f'PIPELINE-STAGE CROSSINGS/SEC — SMOOTHED (5-sample boxcar) — for comparison')
    print(f'  Boxcar is applied uniformly but downstream stages already filtered, so smoothing')
    print(f'  affects raw stages more than smoothed ones. Use RAW above for fair comparison.')
    print(f'{"="*100}')
    print(f'  {"Stage":<8} {"cps_med":>9} {"cps_p25":>9} {"cps_p75":>9}')
    for stage in PIPELINE_STAGES:
        cps_vals = [tr['stages_smoothed'][stage]['crossings_per_sec'] for tr in traces if tr['stages_smoothed'][stage]]
        if not cps_vals:
            print(f'  {stage:<8} {"—":>9}')
            continue
        print(f'  {stage:<8} {np.median(cps_vals):>9.3f} {np.percentile(cps_vals, 25):>9.3f} '
              f'{np.percentile(cps_vals, 75):>9.3f}')

    print(f'\n{"="*100}')
    print(f'STAGE-TO-STAGE DELTAS — raw crossings/sec added (>0) or removed (<0)')
    print(f'{"="*100}')
    print(f'  {"Transition":<20} {"Δ_cps_med":>11} {"Δ_cps_p25":>11} {"Δ_cps_p75":>11} '
          f'{"n_passthru":>11} {"interpretation":<40}')
    print('  ' + '-' * 110)
    transitions = [('des → ema', 'des', 'ema', 'blend + deadband + EMA'),
                   ('pred → ema', 'pred', 'ema', 'blend + deadband + EMA'),
                   ('ema → preRL', 'ema', 'preRL', 'PI (Kp·offset + Ki·integral)'),
                   ('preRL → cmd', 'preRL', 'cmd', 'rate limit + curvature safety')]
    for label, a, b, interp in transitions:
        deltas = []
        n_passthru = 0  # events where the stage adds/removes nothing
        for tr in traces:
            if tr['stages'][a] and tr['stages'][b]:
                d = tr['stages'][b]['crossings_per_sec'] - tr['stages'][a]['crossings_per_sec']
                deltas.append(d)
                if abs(d) < 1e-9:
                    n_passthru += 1
        if not deltas:
            continue
        deltas = np.array(deltas)
        print(f'  {label:<20} {np.median(deltas):>+11.3f} {np.percentile(deltas, 25):>+11.3f} '
              f'{np.percentile(deltas, 75):>+11.3f} {n_passthru:>3}/{len(deltas):<7} {interp:<40}')

    print(f'\n{"="*100}')
    print(f'PMD (pred - des) ANALYSIS — direct planner/model disagreement signal')
    print(f'{"="*100}')
    pmd_cps = [tr['stages']['pmd']['crossings_per_sec'] for tr in traces if tr['stages'].get('pmd')]
    pmd_std = [tr['stages']['pmd']['signal_std'] for tr in traces if tr['stages'].get('pmd')]
    pmd_peak = [abs(tr['stages']['pmd']['peak_rate_per_s']) for tr in traces if tr['stages'].get('pmd')]
    if pmd_cps:
        print(f'  Events with pmd data: {len(pmd_cps)}')
        print(f'  pmd_cps (crossings/sec of pred-des):  median {np.median(pmd_cps):.3f}, '
              f'IQR [{np.percentile(pmd_cps, 25):.3f}, {np.percentile(pmd_cps, 75):.3f}]')
        print(f'  pmd magnitude (|pred-des| std):       median {np.median(pmd_std):.5f}, '
              f'IQR [{np.percentile(pmd_std, 25):.5f}, {np.percentile(pmd_std, 75):.5f}]')
        print(f'  pmd peak rate (max d/dt|pred-des|):   median {np.median(pmd_peak):.5f}')
    else:
        print('  No pmd data available.')

    print(f'\n{"="*100}')
    print(f'TOP {args.top} EVENTS BY DES CROSSINGS/SEC — RAW (no smoothing)')
    print(f'{"="*100}')
    sorted_by_des = sorted(traces, key=lambda t: -t['stages']['des']['crossings_per_sec'] if t['stages']['des'] else 0)
    print(f'  {"Idx":>4} {"v_mph":>6} {"des":>7} {"pred":>7} {"ema":>7} {"preRL":>7} {"cmd":>7} {"pmd":>7} {"fb%":>5}')
    for tr in sorted_by_des[:args.top]:
        s = tr['stages']
        def cps(k):
            return f"{s[k]['crossings_per_sec']:.2f}" if s.get(k) else "—"
        fb = tr.get('pred_fallback') or {}
        fb_str = f'{fb.get("fallback_frac", 0) * 100:.0f}%' if fb else '—'
        print(f'  {tr["idx"]:>4} {tr["metadata"]["mean_v_mph"]:>6.1f} '
              f'{cps("des"):>7} {cps("pred"):>7} {cps("ema"):>7} {cps("preRL"):>7} '
              f'{cps("cmd"):>7} {cps("pmd"):>7} {fb_str:>5}')

    print(f'\n{"="*100}')
    print(f'SIGNAL DYNAMICS — std of rate (1/m/s) per stage; raw (no boxcar)')
    print(f'{"="*100}')
    print(f'  {"Stage":<8} {"std_rate_med":>14} {"std_rate_p90":>14} {"peak_rate_p90":>15}')
    for stage in PIPELINE_STAGES:
        stds = [tr['stages'][stage]['std_rate_per_s'] for tr in traces if tr['stages'][stage]]
        peaks = [tr['stages'][stage]['peak_rate_per_s'] for tr in traces if tr['stages'][stage]]
        if not stds:
            continue
        print(f'  {stage:<8} {np.median(stds):>14.5f} {np.percentile(stds, 90):>14.5f} '
              f'{np.percentile(peaks, 90):>15.5f}')


if __name__ == '__main__':
    main()
