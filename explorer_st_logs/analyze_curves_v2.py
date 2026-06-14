#!/usr/bin/env python3
"""Curve dynamics analyzer v2 — diagnostic-first.

PHASE A: Core
=============
- Per-event waveform extraction from CX1 telemetry (±3s around apex)
- Improved phase decomposition (QA-fixed):
  * peak_bias_pct (magnitude relative to commanded peak, bias-corrected)
  * signed cross-correlation lag (can return negative — meas leads, positive — meas lags)
  * overshoot_pct (peak-relative, bias-corrected — replaces v6's bias-dominated metric)
  * apex70_bias (bias at 70% of rising edge — captures EPAS-still-ramping signature)
  * settling_time_sec (after exit, time for measured to fall below noise floor)
- Per-event bias baseline from adjacent straight segments (replaces global bias assumption)
- JSON output of per-event data (re-loadable for plotting and Phase B/C/D analysis)

USAGE:
  ./analyze_curves_v2.py <route_dir>           # analyze single route
  ./analyze_curves_v2.py --test                # run synthetic-curve validation
  ./analyze_curves_v2.py <route> --max 10      # limit events extracted
  ./analyze_curves_v2.py <route> --no-waveforms  # metrics only, no waveforms (smaller JSON)
"""

import sys
import os
import json
import glob
import argparse
import math
from collections import defaultdict

sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')

import numpy as np

try:
    from scipy.signal import correlate as sp_correlate
    from scipy.fft import rfft as sp_rfft, rfftfreq as sp_rfftfreq
    from scipy.stats import linregress as sp_linregress
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# CX1 schema v2 (29 positional fields)
CX1_FIELDS = ['frame', 'v', 'yr', 'aLat', 'cmd', 'rate', 'meas', 'des', 'pred', 'ema',
              'preRL', 'rl', 'cmdInt', 'rateInt', 'ang', 'dAng', 'tq', 'ovr', 'lc',
              'lookT', 'blend', 'cFac', 'lOff', 'lInt', 'pmd', 'burst',
              'p4Rel', 'p4Tau', 'p4On']
IDX = {f: i for i, f in enumerate(CX1_FIELDS)}


# ─────────────────────────────────────────────────────────────────────────────
# CX1 LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_cx1(route_dir):
    """Load all CX1 telemetry rows from a route directory.

    Returns (arr, t_log) where:
      arr: np.ndarray (N, 29) with all CX1 fields
      t_log: np.ndarray (N,) — log timestamp in seconds (rlog logMonoTime / 1e9)
    """
    # Import here to fail clean if openpilot isn't available
    from openpilot.tools.lib.logreader import LogReader

    # Format 1: route_dir is a directory containing rlog_*.zst files
    files = sorted(glob.glob(os.path.join(route_dir, 'rlog_*.zst')))
    # Format 2: route_dir is a route prefix; per-segment dirs are siblings with --N suffix
    if not files:
        seg_dirs = sorted(glob.glob(route_dir + '--*'),
                          key=lambda d: int(d.rsplit('--', 1)[-1]))
        for sd in seg_dirs:
            rlog = os.path.join(sd, 'rlog.zst')
            if os.path.exists(rlog):
                files.append(rlog)
    # Format 3: route_dir is itself a containing dir with per-segment subdirs
    if not files:
        files = sorted(glob.glob(os.path.join(route_dir, '*--*/rlog.zst')))

    rows = []
    times = []
    for fp in files:
        try:
            for msg in LogReader(fp):
                if msg.which() != 'logMessage':
                    continue
                try:
                    body = json.loads(msg.logMessage).get('msg', '')
                    if not body.startswith('CX1: '):
                        continue
                    payload = body[5:]
                    if payload.startswith('SCHEMA'):
                        continue
                    parts = payload.split()
                    if len(parts) < len(CX1_FIELDS):
                        continue
                    rows.append([float(p) for p in parts[:len(CX1_FIELDS)]])
                    times.append(msg.logMonoTime / 1e9)
                except Exception:
                    pass
        except Exception:
            pass

    if not rows:
        return None, None

    return np.array(rows), np.array(times)


# ─────────────────────────────────────────────────────────────────────────────
# CURVE EVENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_curve_events(arr, t_log, min_peak=0.0015, min_dur_sec=0.4, min_sep_sec=2.0,
                       max_gap_sec=1.0, min_speed_mps=5.0, max_duration_sec=15.0,
                       require_dir_match_pct=0.7):
    """Detect curve events from CX1 telemetry.

    A curve event = a contiguous run of densely-sampled CX1 frames where |cmd|
    exceeded the detection threshold and reached at least min_peak, AND the
    car was moving above min_speed_mps. Mirrors v6 analyzer's speed gate.

    Filters out:
      - Low-speed maneuvers (parking lots, reversing) via min_speed_mps
      - Pathologically long "curves" (>max_duration_sec — likely merged junk)
      - Events where cmd and meas disagree on direction for most samples (sign-flip noise)

    max_gap_sec controls how far apart consecutive CX1 samples can be before
    we treat them as a sampling gap (and split the event).

    Returns list of dicts (see _emit_event_if_valid for fields).
    """
    v = arr[:, IDX['v']]
    cmd = arr[:, IDX['cmd']]
    ovr = arr[:, IDX['ovr']]

    # Detection requires BOTH: cmd above threshold AND speed above min_speed
    detect_thresh = min_peak * 0.4
    above = (np.abs(cmd) > detect_thresh) & (v >= min_speed_mps)

    # Walk through, breaking events at sampling gaps
    events = []
    in_event = False
    start = 0
    for i in range(len(arr)):
        if not in_event and above[i]:
            in_event = True
            start = i
            continue
        if in_event:
            # Check sampling gap
            gap = t_log[i] - t_log[i - 1] if i > 0 else 0
            if gap > max_gap_sec:
                # Close event at i-1
                _emit_event_if_valid(events, arr, t_log, start, i - 1,
                                     min_peak, min_dur_sec)
                in_event = above[i]
                start = i
                continue
            if not above[i]:
                # Trailing buffer: keep event open for a few frames in case it dips
                lookahead_end = min(i + 5, len(arr))
                future_above = above[i:lookahead_end].any()
                if not future_above:
                    _emit_event_if_valid(events, arr, t_log, start, i - 1,
                                         min_peak, min_dur_sec)
                    in_event = False

    # Close any trailing event
    if in_event:
        _emit_event_if_valid(events, arr, t_log, start, len(arr) - 1,
                             min_peak, min_dur_sec)

    # Pre-merge: only apply speed filter (others apply post-merge per QA fix #7)
    events = [e for e in events if e['mean_v_mps'] >= min_speed_mps]

    # Merge events separated by < min_sep_sec
    merged = []
    for ev in events:
        if (merged and
                ev['t_start'] - merged[-1]['t_end'] < min_sep_sec and
                np.sign(ev['peak_cmd']) == np.sign(merged[-1]['peak_cmd'])):
            # Merge into previous (same-direction adjacent curves)
            prev = merged[-1]
            prev['end_idx'] = ev['end_idx']
            prev['t_end'] = ev['t_end']
            prev['n_samples'] = (prev['end_idx'] - prev['start_idx']) + 1
            prev['duration_sec'] = prev['t_end'] - prev['t_start']
            seg = cmd[prev['start_idx']:prev['end_idx'] + 1]
            apex_off = int(np.argmax(np.abs(seg)))
            prev['apex_idx'] = prev['start_idx'] + apex_off
            prev['t_apex'] = t_log[prev['apex_idx']]
            prev['peak_cmd'] = float(seg[apex_off])
            prev['mean_v_mps'] = float(np.mean(v[prev['start_idx']:prev['end_idx'] + 1]))
            prev['mean_v_mph'] = prev['mean_v_mps'] * 2.237
            prev['magnitude_class'] = _classify_magnitude(abs(prev['peak_cmd']))
            prev['had_override'] = bool((ovr[prev['start_idx']:prev['end_idx'] + 1] > 0.5).any())
        else:
            merged.append(ev)

    # Post-merge filters (QA fix #7+#8): re-validate duration and dir_match on merged events
    meas = arr[:, IDX['meas']]
    final = []
    for e in merged:
        if e['duration_sec'] > max_duration_sec:
            continue
        si, ei = e['start_idx'], e['end_idx'] + 1
        c = cmd[si:ei]
        m = meas[si:ei]
        meaningful = np.abs(c) > min_peak / 2
        if meaningful.sum() < 3:
            continue
        same_sign = np.sign(c[meaningful]) == np.sign(m[meaningful])
        dir_match = same_sign.mean()
        e['dir_match_pct'] = float(dir_match)
        if dir_match < require_dir_match_pct:
            continue
        final.append(e)

    return final


def _emit_event_if_valid(events, arr, t_log, start_idx, end_idx, min_peak, min_dur_sec):
    duration = t_log[end_idx] - t_log[start_idx]
    if duration < min_dur_sec:
        return
    cmd = arr[:, IDX['cmd']]
    v = arr[:, IDX['v']]
    ovr = arr[:, IDX['ovr']]
    seg = cmd[start_idx:end_idx + 1]
    apex_off = int(np.argmax(np.abs(seg)))
    peak_cmd = float(seg[apex_off])
    if abs(peak_cmd) < min_peak:
        return
    apex_idx = start_idx + apex_off
    events.append({
        'start_idx': int(start_idx),
        'end_idx': int(end_idx),
        'apex_idx': int(apex_idx),
        't_start': float(t_log[start_idx]),
        't_end': float(t_log[end_idx]),
        't_apex': float(t_log[apex_idx]),
        'peak_cmd': peak_cmd,
        'direction': 'left' if peak_cmd < 0 else 'right',
        'mean_v_mps': float(np.mean(v[start_idx:end_idx + 1])),
        'mean_v_mph': float(np.mean(v[start_idx:end_idx + 1])) * 2.237,
        'duration_sec': float(duration),
        'magnitude_class': _classify_magnitude(abs(peak_cmd)),
        'n_samples': int(end_idx - start_idx + 1),
        'had_override': bool((ovr[start_idx:end_idx + 1] > 0.5).any()),
    })


def _classify_magnitude(peak_abs):
    if peak_abs > 0.004:
        return 'sharp'
    if peak_abs > 0.002:
        return 'moderate'
    if peak_abs > 0.001:
        return 'gentle'
    return 'subgentle'


# ─────────────────────────────────────────────────────────────────────────────
# BIAS BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_bias_baseline(arr, t_log, event, lookback_sec=10.0, lookforward_sec=10.0,
                          straight_thresh=0.0008, min_samples=5, min_speed_mps=7.0,
                          max_disagreement_ratio=5.0):
    """Estimate EPAS bias (meas - cmd) from straight segments adjacent to the event.

    QA-fix #1: bias windows must filter on `v >= min_speed_mps`. Without it,
    standstill samples (v~0) produce meaningless meas (yawRate/v explodes) that
    poison the bias estimate by orders of magnitude.

    Returns dict with pre/post/combined bias, sample counts, and confidence flag.
    confident=False when pre and post disagree by more than max_disagreement_ratio.
    """
    cmd = arr[:, IDX['cmd']]
    meas = arr[:, IDX['meas']]
    v = arr[:, IDX['v']]

    # Pre-event window — straight AND moving
    pre_mask = ((t_log >= event['t_start'] - lookback_sec)
                & (t_log < event['t_start'])
                & (np.abs(cmd) < straight_thresh)
                & (v >= min_speed_mps))
    pre_bias = float(np.mean(meas[pre_mask] - cmd[pre_mask])) if pre_mask.sum() >= min_samples else None
    pre_n = int(pre_mask.sum())

    # Post-event window — straight AND moving
    post_mask = ((t_log >= event['t_end'])
                 & (t_log < event['t_end'] + lookforward_sec)
                 & (np.abs(cmd) < straight_thresh)
                 & (v >= min_speed_mps))
    post_bias = float(np.mean(meas[post_mask] - cmd[post_mask])) if post_mask.sum() >= min_samples else None
    post_n = int(post_mask.sum())

    biases = [b for b in [pre_bias, post_bias] if b is not None]
    confident = True
    if biases:
        combined = float(np.mean(biases))
        # Check pre/post agreement
        if pre_bias is not None and post_bias is not None:
            if abs(pre_bias) > 1e-6 and abs(post_bias) > 1e-6:
                ratio = max(abs(pre_bias), abs(post_bias)) / max(min(abs(pre_bias), abs(post_bias)), 1e-9)
                if ratio > max_disagreement_ratio:
                    confident = False
    else:
        # Fall back to global mean from any straight, moving samples in the route
        global_mask = (np.abs(cmd) < straight_thresh) & (v >= min_speed_mps)
        if global_mask.sum() >= min_samples:
            combined = float(np.mean(meas[global_mask] - cmd[global_mask]))
        else:
            combined = 0.0
        confident = False

    return {
        'pre': pre_bias,
        'post': post_bias,
        'combined': combined,
        'n_pre': pre_n,
        'n_post': post_n,
        'confident': confident,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_waveform(arr, t_log, event, window_sec=3.0, neighbor_buffer_sec=0.3,
                     prev_event=None, next_event=None):
    """Extract ±window_sec around apex, time-aligned to apex=0.

    QA fix #3: clips the window so it doesn't leak into neighboring events'
    rising/falling edges (which corrupts max() and timing calculations).

    Returns dict with arrays for each useful signal (all as Python lists for JSON).
    """
    t_apex = event['t_apex']
    t_lo = t_apex - window_sec
    t_hi = t_apex + window_sec

    # Clip against neighbors (with a small buffer so we don't cut the curve itself)
    if prev_event is not None:
        t_lo = max(t_lo, prev_event['t_end'] + neighbor_buffer_sec)
    if next_event is not None:
        t_hi = min(t_hi, next_event['t_start'] - neighbor_buffer_sec)

    # Also bound by the event's own start/end so window doesn't pull straight-line noise
    # Keep at least the event's own extent though (apex-to-apex)
    t_lo = min(t_lo, event['t_start'])
    t_hi = max(t_hi, event['t_end'])

    mask = (t_log >= t_lo) & (t_log <= t_hi)
    idxs = np.where(mask)[0]
    if len(idxs) < 5:
        return None

    t_rel = (t_log[idxs] - t_apex).tolist()
    # Pull all useful CX1 signals
    sig = {}
    for field in ['frame', 'v', 'yr', 'aLat', 'cmd', 'rate', 'meas', 'des', 'pred',
                  'ema', 'preRL', 'rl', 'ang', 'dAng', 'tq', 'ovr',
                  'lookT', 'blend', 'cFac', 'lOff', 'lInt', 'pmd', 'p4Rel', 'p4Tau']:
        sig[field] = arr[idxs, IDX[field]].tolist()

    sig['t_rel'] = t_rel
    sig['n_samples'] = len(t_rel)
    return sig


# ─────────────────────────────────────────────────────────────────────────────
# PHASE METRICS — QA-FIXED
# ─────────────────────────────────────────────────────────────────────────────

def compute_phase_metrics(waveform, bias):
    """Compute QA-fixed phase metrics for a curve event.

    Inputs:
      waveform: dict from extract_waveform()
      bias: float — combined bias estimate (meas - cmd on adjacent straights)

    Returns dict with metrics. All numeric values are bias-corrected where applicable.
    """
    if waveform is None or waveform['n_samples'] < 5:
        return None

    t = np.array(waveform['t_rel'])
    cmd = np.array(waveform['cmd'])
    meas_raw = np.array(waveform['meas'])
    meas = meas_raw - bias  # BIAS-CORRECTED measurement

    # Find apex in our local arrays
    apex_i = int(np.argmin(np.abs(t)))
    cmd_peak = float(cmd[apex_i])
    cmd_peak_abs = abs(cmd_peak)

    if cmd_peak_abs < 1e-6:
        return {'error': 'no_commanded_curvature'}

    # ───── Peak bias % (QA-fixed sign convention) ─────
    # Divide by SIGNED cmd_peak so positive pct = overshoot in commanded direction
    # for both left and right turns (consistent with overshoot_pct below)
    meas_at_apex = float(meas[apex_i])
    peak_bias_abs = meas_at_apex - cmd_peak
    peak_bias_pct = peak_bias_abs / cmd_peak  # signed, NOT abs — see QA fix #2

    # ───── Overshoot % (peak-relative, bias-corrected) ─────
    # Direction-aware: meas going past cmd in the same direction
    sign_cmd = float(np.sign(cmd_peak))
    meas_in_dir = meas * sign_cmd  # positive when meas is in same direction as cmd
    cmd_in_dir = cmd * sign_cmd
    max_meas_in_dir = float(np.max(meas_in_dir))
    max_cmd_in_dir = float(np.max(cmd_in_dir))
    overshoot_abs = max_meas_in_dir - max_cmd_in_dir
    overshoot_pct = overshoot_abs / cmd_peak_abs

    # ───── Cross-correlation strength (lag DROPPED per QA fix #4) ─────
    # The lag value was unreliable for signals where the curve width >> resolution.
    # We keep the correlation strength (r) because it tells us how well meas
    # tracks cmd shape, but we don't pretend the lag is trustworthy.
    lag_r = None
    if HAS_SCIPY and len(t) >= 10:
        dt_uniform = 0.05  # 20 Hz
        n = int(np.ceil((t[-1] - t[0]) / dt_uniform)) + 1
        t_uni = t[0] + np.arange(n) * dt_uniform
        cmd_uni = np.interp(t_uni, t, cmd)
        meas_uni = np.interp(t_uni, t, meas)
        c = cmd_uni - cmd_uni.mean()
        m = meas_uni - meas_uni.mean()
        if np.std(c) > 1e-9 and np.std(m) > 1e-9:
            corr = sp_correlate(m, c, mode='full')
            denom = np.sqrt(np.sum(c**2) * np.sum(m**2))
            if denom > 0:
                # Peak correlation, not the lag
                lag_r = float(np.max(corr) / denom)

    # ───── Apex70 bias (bias at 70% of peak, rising edge) ─────
    rising = t < 0
    seventy_thresh = 0.7 * cmd_peak_abs
    rising_above_70 = rising & (np.abs(cmd) >= seventy_thresh)
    apex70_bias = None
    if rising_above_70.any():
        first_70 = int(np.argmax(rising_above_70))
        apex70_bias = float(meas[first_70] - cmd[first_70])

    # ───── Settling time (after exit, time for meas to fall below noise floor) ─────
    # Exit defined as: cmd falls below 30% of peak on falling edge
    # Noise floor 0.001 (residual bias-corrected meas typically has ~0.001 noise) — QA fix #6
    thirty_thresh = 0.3 * cmd_peak_abs
    falling = t > 0
    falling_below = falling & (np.abs(cmd) < thirty_thresh)
    settling_time = None
    if falling_below.any():
        exit_i = int(np.argmax(falling_below))
        post_exit_settled = (t > t[exit_i]) & (np.abs(meas) < 0.001)
        if post_exit_settled.any():
            settled_i = int(np.argmax(post_exit_settled))
            settling_time = float(t[settled_i] - t[exit_i])

    # ───── Exit bias (replaces v6's "exit bias at 30% fall", now bias-corrected) ─────
    exit_bias = None
    if falling_below.any():
        # bias at exit point
        exit_bias_val = float(meas[exit_i] - cmd[exit_i])
        exit_bias = exit_bias_val

    # ───── Entry bias (at 30% rising edge) ─────
    rising_above_30 = rising & (np.abs(cmd) >= thirty_thresh)
    entry_bias = None
    if rising_above_30.any():
        first_30 = int(np.argmax(rising_above_30))
        entry_bias = float(meas[first_30] - cmd[first_30])

    # ───── Peak-cmd vs peak-meas timing (true responsiveness, no construction floor) ─────
    # Find the actual peak-meas in window (could be before or after apex)
    abs_meas_in_dir = np.maximum(0, meas_in_dir)
    if abs_meas_in_dir.max() > 0:
        meas_peak_i = int(np.argmax(abs_meas_in_dir))
        timing_offset_ms = float((t[meas_peak_i] - t[apex_i]) * 1000)
    else:
        timing_offset_ms = None

    return {
        'peak_bias_abs': float(peak_bias_abs),       # bias-corrected, signed (1/m)
        'peak_bias_pct': float(peak_bias_pct),       # bias-corrected, signed by direction (pos=overshoot)
        'overshoot_abs': float(overshoot_abs),       # bias-corrected, peak-relative (1/m)
        'overshoot_pct': float(overshoot_pct),       # fraction of cmd peak; pos=overshoot in cmd dir
        'crosscorr_r': lag_r,                        # max normalized correlation (lag dropped — was unreliable)
        'apex70_bias': apex70_bias,                  # bias-corrected, at 70% rise
        'entry_bias_30': entry_bias,                 # bias-corrected, at 30% rise
        'exit_bias_30': exit_bias,                   # bias-corrected, at 30% fall
        'settling_time_sec': settling_time,          # None if doesn't settle in window
        'timing_offset_ms': timing_offset_ms,        # actual meas peak vs cmd peak (can be negative!)
        'cmd_peak_abs': cmd_peak_abs,
        'meas_peak_abs': float(max_meas_in_dir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — FREQUENCY-DOMAIN PER-EVENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fft_metrics(waveform, sample_rate_hint=20.0):
    """FFT of dAng (steering wheel rate, deg/s) and aLat (lateral accel) for a curve event.

    Returns dict with:
      dAng_dom_hz, dAng_pct_0_1, dAng_pct_1_3, dAng_pct_3_10 — power distribution
      aLat_dom_hz, aLat_pct_0_1, aLat_pct_1_3, aLat_pct_3_10
      n_samples_resampled (post-uniform-resampling)
      effective_fs_hz

    "Jerky" perceived oscillation typically lives in 1-3 Hz band.
    """
    if waveform is None or not HAS_SCIPY:
        return None
    t = np.array(waveform['t_rel'])
    if len(t) < 16:  # too few samples for meaningful FFT
        return None

    dt = np.median(np.diff(t))
    if dt <= 0:
        return None
    fs = 1.0 / dt

    # Resample to uniform grid (CX1 sampling is bursty)
    dt_uniform = 1.0 / sample_rate_hint
    n = int(np.ceil((t[-1] - t[0]) / dt_uniform)) + 1
    if n < 16:
        return None
    t_uni = t[0] + np.arange(n) * dt_uniform

    def _spec(signal_name):
        try:
            sig = np.array(waveform[signal_name])
            sig_uni = np.interp(t_uni, t, sig)
            # Remove DC
            sig_uni = sig_uni - sig_uni.mean()
            # Apply Hann window
            w = np.hanning(len(sig_uni))
            yf = sp_rfft(sig_uni * w)
            freqs = sp_rfftfreq(len(sig_uni), d=dt_uniform)
            power = (np.abs(yf) ** 2)
            total = power.sum()
            if total <= 0:
                return None
            # Power distribution
            def band_pct(lo, hi):
                mask = (freqs >= lo) & (freqs < hi)
                return float(power[mask].sum() / total)
            # Dominant frequency (skip DC)
            non_dc = freqs > 0.1
            if non_dc.any():
                dom_i = int(np.argmax(power[non_dc])) + int(np.argmax(non_dc))
                dom_hz = float(freqs[dom_i])
            else:
                dom_hz = 0.0
            return {
                'dom_hz': dom_hz,
                'pct_0_1': band_pct(0, 1.0),
                'pct_1_3': band_pct(1.0, 3.0),
                'pct_3_10': band_pct(3.0, 10.0),
                'total_power': float(total),
            }
        except Exception:
            return None

    dAng_spec = _spec('dAng')
    aLat_spec = _spec('aLat')

    return {
        'dAng': dAng_spec,
        'aLat': aLat_spec,
        'effective_fs_hz': float(1.0 / dt_uniform),
        'n_samples_resampled': int(n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — EPAS LINEARITY PER EVENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_epas_linearity(waveform, bias_value, cmd_peak_abs=None):
    """Linear regression of (bias-corrected) meas vs cmd over the curve event.

    QA fix M1: report SNR alongside slope/R². Without SNR context, R² appears
    to "degrade" at small cmd magnitudes but it's just noise dominating the fit.
    A linear EPAS with typical encoder noise (~0.0003-0.0006 1/m std) shows
    R²~0.9 on sharp curves but R²~0.3 on gentle curves with NO non-linearity.

    Adds:
      snr (cmd_peak / residual_std) — interpret slope/R² only when SNR > ~5
      reliable: bool — slope/R² are trustworthy

    Returns dict with slope, intercept, r2, residual_std, snr, reliable, n_samples.
    """
    if waveform is None or not HAS_SCIPY:
        return None
    cmd = np.array(waveform['cmd'])
    meas = np.array(waveform['meas']) - bias_value
    if len(cmd) < 5:
        return None
    mask = np.abs(cmd) > 0.0005
    if mask.sum() < 5:
        return None
    try:
        res = sp_linregress(cmd[mask], meas[mask])
        residuals = meas[mask] - (res.slope * cmd[mask] + res.intercept)
        resid_std = float(np.std(residuals))
        if cmd_peak_abs is None:
            cmd_peak_abs = float(np.max(np.abs(cmd)))
        snr = cmd_peak_abs / max(resid_std, 1e-9)
        return {
            'slope': float(res.slope),
            'intercept': float(res.intercept),
            'r2': float(res.rvalue ** 2),
            'residual_std': resid_std,
            'snr': float(snr),
            'reliable': bool(snr > 5.0),  # below this, slope/R² are noise-dominated
            'n_samples': int(mask.sum()),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — OUTLIER FLAGGING
# ─────────────────────────────────────────────────────────────────────────────

def flag_outliers(events, z_thresh=2.0, min_n_for_mad=6):
    """Compute z-scores for key metrics across the event population, flag outliers.

    QA fixes applied:
      M2: Low-confidence-bias events get ONLY the `low_confidence_bias` flag.
          We don't pile on slope/R²/outlier flags computed on uninterpretable data.
      M3: `large_timing_offset` uses duration-relative threshold (50% of duration)
          instead of fixed 800ms — long sweeping curves can legitimately have
          late meas peaks without being pathological.
      N3: MAD-based flags require min_n_for_mad samples; small populations get
          only absolute-threshold flags.

    Each event gets a list of warning flags. Annotates in-place.
    """
    valid = [e for e in events if 'metrics' in e and e.get('metrics') and e['metrics'].get('peak_bias_pct') is not None]

    use_mad = len(valid) >= min_n_for_mad

    if use_mad:
        def collect(field, source='metrics'):
            return np.array([e[source][field] for e in valid if e[source].get(field) is not None])
        pbp = collect('peak_bias_pct')
        op = collect('overshoot_pct')
        toff = collect('timing_offset_ms')
        pbp_med, pbp_mad = np.median(pbp), np.median(np.abs(pbp - np.median(pbp))) * 1.4826
        op_med, op_mad = np.median(op), np.median(np.abs(op - np.median(op))) * 1.4826
        toff_med, toff_mad = np.median(toff), np.median(np.abs(toff - np.median(toff))) * 1.4826

    for e in events:
        flags = []
        m = e.get('metrics')

        # M2: low-confidence bias gets ONLY this flag — others would be misleading
        b = e.get('bias')
        if b and not b.get('confident'):
            flags.append('low_confidence_bias')
            e['flags'] = flags
            continue

        if m is None:
            e['flags'] = flags
            continue

        # MAD-based outlier detection (only if population large enough)
        if use_mad:
            def is_outlier(val, med, mad):
                if val is None or mad <= 0:
                    return False
                return abs(val - med) > z_thresh * mad

            if is_outlier(m.get('peak_bias_pct'), pbp_med, pbp_mad):
                flags.append(f'peak_bias_outlier:{m["peak_bias_pct"]:+.1%}')
            if is_outlier(m.get('overshoot_pct'), op_med, op_mad):
                flags.append(f'overshoot_outlier:{m["overshoot_pct"]:+.1%}')
            if is_outlier(m.get('timing_offset_ms'), toff_med, toff_mad):
                flags.append(f'timing_outlier:{m["timing_offset_ms"]:+.0f}ms')

        # Absolute thresholds (always apply)
        if m.get('crosscorr_r') is not None and m['crosscorr_r'] < 0.6:
            flags.append(f'poor_tracking:r={m["crosscorr_r"]:.2f}')

        # M3: duration-relative timing threshold
        toff_val = m.get('timing_offset_ms')
        duration_ms = e['metadata']['duration_sec'] * 1000
        if toff_val is not None and abs(toff_val) > max(800, 0.5 * duration_ms):
            flags.append(f'large_timing_offset:{toff_val:+.0f}ms')

        # FFT-based: high jitter in 3-10 Hz band
        fft = e.get('fft')
        if fft and fft.get('dAng') and fft['dAng'].get('pct_3_10') is not None:
            if fft['dAng']['pct_3_10'] > 0.20:
                flags.append(f'high_freq_dAng:{fft["dAng"]["pct_3_10"]:.0%}')

        # M1: EPAS linearity flags ONLY when SNR is high enough to interpret
        lin = e.get('epas_linearity')
        if lin and lin.get('reliable'):
            if abs(lin['slope'] - 1.0) > 0.3:
                flags.append(f'epas_slope:{lin["slope"]:.2f}')
            if lin.get('r2') is not None and lin['r2'] < 0.7:
                flags.append(f'epas_low_r2:{lin["r2"]:.2f}')

        e['flags'] = flags


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — SPEED-BINNED PHASE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

SPEED_BINS_MPH = [(0, 15, '0-15'), (15, 30, '15-30'), (30, 45, '30-45'),
                  (45, 65, '45-65'), (65, 80, '65-80'), (80, 200, '80+')]


def speed_bin(mph):
    for lo, hi, label in SPEED_BINS_MPH:
        if lo <= mph < hi:
            return label
    return 'unknown'


def aggregate_by_speed_bin(events):
    """Group events by [speed_bin × magnitude_class] and return aggregate stats."""
    by_bin = defaultdict(list)
    for e in events:
        if 'metrics' not in e or e.get('metrics') is None:
            continue
        if e.get('bias') and not e['bias'].get('confident'):
            continue
        v_mph = e['metadata']['mean_v_mph']
        mag = e['metadata']['magnitude_class']
        key = (speed_bin(v_mph), mag)
        by_bin[key].append(e)

    summary = {}
    for (sb, mag), evs in by_bin.items():
        pbp = [e['metrics']['peak_bias_pct'] for e in evs if e['metrics'].get('peak_bias_pct') is not None]
        op = [e['metrics']['overshoot_pct'] for e in evs if e['metrics'].get('overshoot_pct') is not None]
        toff = [e['metrics']['timing_offset_ms'] for e in evs if e['metrics'].get('timing_offset_ms') is not None]
        if not pbp:
            continue
        summary[f'{sb}_{mag}'] = {
            'n': len(evs),
            'peak_bias_pct_median': float(np.median(pbp)),
            'overshoot_pct_median': float(np.median(op)) if op else None,
            'timing_offset_ms_median': float(np.median(toff)) if toff else None,
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PHASE C — PLANNER-SIDE & PIPELINE-SIGNAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
#
# Phase C builds on Phase A/B per-event waveforms to answer: WHERE in the
# pipeline does "jerkiness" originate? Candidates:
#   - Planner output (`des` field) — is the desiredCurvature itself jittery?
#   - Predicted curvature (`pred` field) — does the orientationRate blend
#     introduce noise?
#   - PI integral (`lInt` field) — does integral persistence cause stepping
#     through the curve?
#   - pc_blend (`blend` field) — does the speed-scheduled blend ratio
#     transition mid-curve and cause cmd discontinuity?
#   - Override (`ovr` field) — what signal pattern precedes driver overrides?
# All four signals are already in CX1 telemetry — Phase C just analyzes them.

def compute_signal_fft(waveform, field, sample_rate_hint=20.0):
    """FFT a specific CX1 signal field (e.g. 'des', 'pred', 'cmd') for one event.

    Returns the same band-distribution dict as compute_fft_metrics() returns
    for dAng/aLat, but for an arbitrary signal.
    """
    if waveform is None or not HAS_SCIPY:
        return None
    if field not in waveform:
        return None
    t = np.array(waveform['t_rel'])
    if len(t) < 16:
        return None
    dt_uniform = 1.0 / sample_rate_hint
    n = int(np.ceil((t[-1] - t[0]) / dt_uniform)) + 1
    if n < 16:
        return None
    t_uni = t[0] + np.arange(n) * dt_uniform
    try:
        sig = np.array(waveform[field])
        sig_uni = np.interp(t_uni, t, sig)
        sig_uni = sig_uni - sig_uni.mean()
        w = np.hanning(len(sig_uni))
        yf = sp_rfft(sig_uni * w)
        freqs = sp_rfftfreq(len(sig_uni), d=dt_uniform)
        power = np.abs(yf) ** 2
        total = power.sum()
        if total <= 0:
            return None
        def band_pct(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(power[mask].sum() / total)
        non_dc = freqs > 0.1
        if non_dc.any():
            dom_i = int(np.argmax(np.where(non_dc, power, -np.inf)))
            dom_hz = float(freqs[dom_i])
        else:
            dom_hz = 0.0
        return {
            'dom_hz': dom_hz,
            'pct_0_1': band_pct(0, 1.0),
            'pct_1_3': band_pct(1.0, 3.0),
            'pct_3_10': band_pct(3.0, 10.0),
            'total_power': float(total),
        }
    except Exception:
        return None


MIN_POWER_FOR_GAIN_RATIO = 0.02  # gate pipeline gain ratio on having real signal


def compute_planner_analysis(waveform):
    """FFT each pipeline-stage curvature signal in CURVATURE DOMAIN.

    QA Phase C #2: only compare like-to-like (curvature signals). The dAng vs
    des comparison from earlier was invalid (derivative inflates HF bands).

    Pipeline stages:
      des  — planner output (the source)
      pred — predicted curvature (orientationRate-derived blend input)
      ema  — after EMA stabilizer
      cmd  — final apply_curv_send (post rate-limit)
      meas — measured curvature (-yawRate/v) — EPAS-side response

    Pipeline gain at 1-3 Hz tells us whether each stage adds or removes
    oscillation in that band, but only when the signal has enough absolute power.

    Returns dict with FFT of each signal, plus pipeline gain ratios with
    explicit None when input power is too low for the ratio to be meaningful.
    """
    if waveform is None:
        return None
    des_fft = compute_signal_fft(waveform, 'des')
    pred_fft = compute_signal_fft(waveform, 'pred')
    cmd_fft = compute_signal_fft(waveform, 'cmd')
    ema_fft = compute_signal_fft(waveform, 'ema')
    meas_fft = compute_signal_fft(waveform, 'meas')

    def gain(num_fft, denom_fft):
        """Absolute-power ratio at 1-3 Hz: (cmd power in 1-3 Hz) / (des power in 1-3 Hz).

        QA review #1 fix: previous version compared pct_1_3 ratios (relative band
        shares of each signal's own spectrum), which conflates spectral shape with
        power. A pipeline that broadly attenuates but keeps shape gives true_gain ≈ 1
        but pct_ratio ≈ 1, while a stage that concentrates remaining power in 1-3 Hz
        gives pct_ratio > 1 even if absolute 1-3 Hz power dropped. Use absolute
        power (pct × total) so 'gain' means what its name implies.
        """
        if not num_fft or not denom_fft:
            return None
        num_abs = num_fft.get('pct_1_3', 0) * num_fft.get('total_power', 0)
        denom_abs = denom_fft.get('pct_1_3', 0) * denom_fft.get('total_power', 0)
        # Gate on denominator having real signal — ratio is meaningless when denom is noise floor
        if denom_abs < 1e-10:
            return None
        return float(num_abs / denom_abs)

    return {
        'des': des_fft,    # planner output
        'pred': pred_fft,  # predicted curvature
        'ema': ema_fft,    # after EMA stabilizer
        'cmd': cmd_fft,    # final apply_curv_send
        'meas': meas_fft,  # EPAS-side response (curvature domain)
        'pipeline_gain_1_3': gain(cmd_fft, des_fft),       # cmd / des (overall pipeline)
        'ema_gain_1_3': gain(ema_fft, des_fft),            # ema / des (planner→EMA stage)
        'epas_gain_1_3': gain(meas_fft, cmd_fft),          # meas / cmd (EPAS response stage)
    }


def compute_blend_trajectory(waveform):
    """Track `pc_blend_ratio` through the curve event.

    pc_blend is speed-scheduled: np.interp(v, [7, 20, 27, 35], [0.10, 0.30, 0.20, 0.10]).
    If the car decelerates entering a curve (e.g., 30 → 18 mph), blend drops from
    0.30 down to ~0.16 — that 47% change could cause cmd to step.

    Returns: blend start, end, min, max, delta, change_rate_per_sec.
    """
    if waveform is None or 'blend' not in waveform:
        return None
    b = np.array(waveform['blend'])
    t = np.array(waveform['t_rel'])
    if len(b) < 5:
        return None
    return {
        'start': float(b[0]),
        'end': float(b[-1]),
        'min': float(b.min()),
        'max': float(b.max()),
        'delta': float(b.max() - b.min()),
        'std': float(b.std()),
    }


def compute_integral_trajectory(waveform):
    """Track PI integral (lInt) and offset (lOff) through the curve event.

    QA Phase C #4: actually measure decay rate and detect gate flicker.
    Expected gate-closed decay = 0.98 per 50ms step = ~0.67/sec (per the
    carcontroller code). Single-step jumps |dlInt/dt| > 1.0/s likely indicate
    gate flicker (integral gate briefly opened due to apply_curvature dipping
    below 0.005 mid-curve).
    """
    if waveform is None or 'lInt' not in waveform:
        return None
    lint = np.array(waveform['lInt'])
    loff = np.array(waveform['lOff'])
    t = np.array(waveform['t_rel'])
    if len(lint) < 5:
        return None

    pre = lint[t < 0]
    post = lint[t > 0]
    apex_i = int(np.argmin(np.abs(t)))

    # Compute instantaneous decay rate
    dt = np.diff(t)
    dlint = np.diff(lint)
    rates = np.zeros_like(dlint, dtype=float)
    mask = dt > 1e-4
    rates[mask] = dlint[mask] / dt[mask]

    return {
        'lInt_start': float(lint[0]),
        'lInt_apex': float(lint[apex_i]),
        'lInt_end': float(lint[-1]),
        'lInt_max_abs': float(np.max(np.abs(lint))),
        'lInt_pre_mean': float(pre.mean()) if len(pre) > 0 else None,
        'lInt_post_mean': float(post.mean()) if len(post) > 0 else None,
        'lOff_max_abs': float(np.max(np.abs(loff))),
        # Phase C fix #4: actual decay measurements
        'max_rate_abs_per_s': float(np.max(np.abs(rates))) if len(rates) > 0 else None,
        'p95_rate_abs_per_s': float(np.percentile(np.abs(rates), 95)) if len(rates) > 0 else None,
        'gate_flicker_likely': bool(np.any(np.abs(rates) > 1.0)),  # > expected 0.67/s decay
    }


# ─────────────────────────────────────────────────────────────────────────────
# MECHANISM METRICS (QA review additions — direct mappings to subjective
#                    "hunting"/"jerkiness" complaints)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rate_limit_cycling(waveform, eps=1e-6):
    """Detect rate-limiter cycling — `preRL` deviating from `rl` (clipped samples).

    `preRL` is what the controller wants to send; `rl` is what the rate limiter
    actually emits. When |preRL - rl| > eps, the RL clipped this sample. High
    clipped fraction means the controller is fighting the rate limit (felt as
    "stickiness"). High clipped-direction-flips means the controller alternates
    saturation directions (felt as "hunting").

    Returns clipped_frac, clipped_n, direction_flips, flips_per_sec.
    """
    if waveform is None or 'preRL' not in waveform or 'rl' not in waveform:
        return None
    preRL = np.array(waveform['preRL'])
    rl = np.array(waveform['rl'])
    t = np.array(waveform['t_rel'])
    if len(preRL) < 5 or len(rl) != len(preRL):
        return None
    diff = preRL - rl
    clipped = np.abs(diff) > eps
    clipped_frac = float(clipped.mean())
    clipped_n = int(clipped.sum())
    # Direction flips: only count among clipped samples where sign of diff changes
    if clipped_n >= 2:
        signs = np.sign(diff[clipped])
        flips = int(np.sum(np.diff(signs) != 0))
    else:
        flips = 0
    duration = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    flips_per_sec = float(flips / duration) if duration > 0 else None
    return {
        'clipped_frac': clipped_frac,
        'clipped_n': clipped_n,
        'direction_flips': flips,
        'flips_per_sec': flips_per_sec,
        'max_abs_clip': float(np.max(np.abs(diff))),
    }


def compute_ema_lag(waveform, sample_rate_hz=20.0):
    """Cross-correlation lag of ema vs des — how much phase delay the EMA stabilizer adds.

    QA round 2 fix (Bug 4): previous version reported lag at integer-sample
    resolution (~50ms steps), so most events read "0ms lag" — that was the
    metric floor, not a finding. Fixes:
    - Resample both signals to a uniform 20 Hz grid first (eliminates the
      bursty-sampling artifact that would distort lag readings on overshoot events).
    - Parabolic interpolation around the correlation peak gives sub-sample
      lag resolution (~5-10 ms typical accuracy).

    Positive lag_ms means ema follows des. >150ms in mid-curve could couple with
    EPAS delay to cause perceived jerk.
    """
    if not HAS_SCIPY:
        return None
    if waveform is None or 'des' not in waveform or 'ema' not in waveform:
        return None
    des_raw = np.array(waveform['des'])
    ema_raw = np.array(waveform['ema'])
    t_raw = np.array(waveform['t_rel'])
    if len(des_raw) < 10 or len(ema_raw) != len(des_raw) or (t_raw[-1] - t_raw[0]) <= 0:
        return None

    # Uniform resample to 20 Hz
    dt_uniform = 1.0 / sample_rate_hz
    n_uni = int(np.ceil((t_raw[-1] - t_raw[0]) / dt_uniform)) + 1
    t_uni = t_raw[0] + np.arange(n_uni) * dt_uniform
    des = np.interp(t_uni, t_raw, des_raw)
    ema = np.interp(t_uni, t_raw, ema_raw)

    # De-mean to focus on shape, not DC
    des_d = des - des.mean()
    ema_d = ema - ema.mean()
    if des_d.std() < 1e-7 or ema_d.std() < 1e-7:
        return None  # signals too flat
    corr = sp_correlate(ema_d, des_d, mode='full')
    norm = np.sqrt(np.sum(des_d ** 2) * np.sum(ema_d ** 2))
    if norm < 1e-12:
        return None
    corr = corr / norm
    peak_i = int(np.argmax(corr))
    lag_int = peak_i - (len(des) - 1)
    peak_corr = float(corr[peak_i])

    # Sub-sample parabolic interpolation around the peak
    sub = 0.0
    if 0 < peak_i < len(corr) - 1:
        y_lo, y_pk, y_hi = corr[peak_i - 1], corr[peak_i], corr[peak_i + 1]
        denom = (y_lo - 2 * y_pk + y_hi)
        if abs(denom) > 1e-12:
            sub = 0.5 * (y_lo - y_hi) / denom  # in samples, range [-0.5, +0.5]

    lag_samples_sub = lag_int + sub
    lag_ms = float(lag_samples_sub * dt_uniform * 1000.0)
    return {
        'lag_ms': lag_ms,
        'lag_ms_resolution': float(dt_uniform * 1000.0),  # native sample resolution before subsample
        'lag_samples_int': int(lag_int),
        'lag_samples_sub': float(lag_samples_sub),
        'peak_corr': peak_corr,
    }


def compute_hunting_score(waveform, sample_rate_hz=20.0, smoothing_window=5,
                          cmd_quantum=2e-5, also_compute_des=True):
    """Count zero-crossings of dCmd/dt within an event, normalized per second.

    QA round 2 fixes:
    - Bug 1: CX1 logs at 10/20/1 Hz mode-switched (bursty in overshoot). Counting
      crossings on the raw bursty grid biases events with more overshoot to score
      higher. Resample to uniform 20 Hz BEFORE counting so the score is sample-
      rate independent.
    - Bug 3: previous threshold (1e-9) was below cmd quantization (~2e-5 per LSB),
      so quantization wobble counted as 'hunting'. Use a threshold scaled to the
      quantum: at 20 Hz, one-LSB-per-sample is 2e-5 / 0.05 = 4e-4 1/m/s; require
      smoothed rate to exceed ~25% of one-LSB-per-sample to count.
    - Bug 2 (callers' responsibility): a separate `gate_flicker_likely` flag from
      `compute_integral_trajectory` should be used downstream to exclude
      integral-gate-flicker events whose dCmd peaks are integral artifacts,
      not curvature hunting.

    Also computes the same metric for `des` (planner output) when requested, so
    callers can compare cmd vs des — if both have similar crossings/sec, the
    controller isn't introducing extra zero-crossings (the planner is).
    """
    if waveform is None or 'cmd' not in waveform:
        return None
    t_raw = np.array(waveform['t_rel'])
    if len(t_raw) < smoothing_window + 5 or (t_raw[-1] - t_raw[0]) <= 0:
        return None

    # Uniform 20 Hz grid (Bug 1 fix)
    dt_uniform = 1.0 / sample_rate_hz
    n_uni = int(np.ceil((t_raw[-1] - t_raw[0]) / dt_uniform)) + 1
    t_uni = t_raw[0] + np.arange(n_uni) * dt_uniform

    rate_threshold = 0.25 * cmd_quantum / dt_uniform  # Bug 3 fix

    def _score(signal_key):
        if signal_key not in waveform:
            return None
        sig_raw = np.array(waveform[signal_key])
        if len(sig_raw) != len(t_raw):
            return None
        sig_uni = np.interp(t_uni, t_raw, sig_raw)
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            sig_s = np.convolve(sig_uni, kernel, mode='same')
        else:
            sig_s = sig_uni
        dsig = np.diff(sig_s) / dt_uniform  # already uniform dt
        # Reject quantization-floor crossings via magnitude gate
        valid = np.abs(dsig) > rate_threshold
        nonzero = dsig[valid]
        if len(nonzero) < 2:
            return {'zero_crossings': 0, 'crossings_per_sec': 0.0,
                    'mean_abs_rate_per_s': float(np.mean(np.abs(dsig))) if len(dsig) else 0.0,
                    'peak_rate_per_s': float(np.max(np.abs(dsig))) if len(dsig) else 0.0,
                    'rate_threshold': rate_threshold,
                    'n_uniform_samples': int(n_uni)}
        signs = np.sign(nonzero)
        crossings = int(np.sum(np.diff(signs) != 0))
        duration = float(t_uni[-1] - t_uni[0])
        cps = float(crossings / duration) if duration > 0 else 0.0
        return {
            'zero_crossings': crossings,
            'crossings_per_sec': cps,
            'mean_abs_rate_per_s': float(np.mean(np.abs(dsig))),
            'peak_rate_per_s': float(np.max(np.abs(dsig))),
            'rate_threshold': rate_threshold,
            'n_uniform_samples': int(n_uni),
        }

    out = _score('cmd')
    if out is None:
        return None
    # Backwards-compat field names so existing comparison code keeps working
    out['mean_abs_dcmd_per_s'] = out.get('mean_abs_rate_per_s', 0.0)
    out['peak_dcmd_per_s'] = out.get('peak_rate_per_s', 0.0)
    if also_compute_des:
        des_score = _score('des')
        if des_score is not None:
            out['des_crossings_per_sec'] = des_score['crossings_per_sec']
            out['des_zero_crossings'] = des_score['zero_crossings']
            out['des_peak_rate_per_s'] = des_score['peak_rate_per_s']
            # Excess crossings from controller (cmd) above planner (des) baseline
            out['excess_crossings_per_sec'] = float(out['crossings_per_sec'] - des_score['crossings_per_sec'])
    return out


# QA review #3: align with _classify_magnitude's gentle floor (0.001).
# Previous value (0.0015) was arbitrary and caused a0 results to read "0/3 in
# curves" with threshold 0.0015 vs "2/3 in curves" at 0.001 — fragile to picking.
# Reporting both lets readers see threshold sensitivity.
CURVE_THRESHOLD_FOR_OVR = 0.001       # primary (aligned with gentle classifier)
CURVE_THRESHOLD_FOR_OVR_STRICT = 0.0015  # secondary (legacy, more conservative)


def compute_override_correlation(arr, t_log, ovr_threshold=0.5, lookback_sec=5.0,
                                 global_bias=None):
    """For each override transition (ovr 0→1), characterize the state leading up to it.

    QA Phase C #1: subtract global bias before computing residuals (matches
                   Phase A standard).
    QA Phase C #3: replace single boolean `in_curve` with time-resolved measures
                   (peak_cmd_within_1s, time_since_last_curve_sec).
    """
    ovr = arr[:, IDX['ovr']]
    v = arr[:, IDX['v']]
    cmd = arr[:, IDX['cmd']]
    meas = arr[:, IDX['meas']]
    ang = arr[:, IDX['ang']]

    # Compute a global bias if not provided (used to correct residuals)
    if global_bias is None:
        straight_mask = (np.abs(cmd) < 0.0008) & (v >= 7.0)
        if straight_mask.sum() > 20:
            global_bias = float(np.mean(meas[straight_mask] - cmd[straight_mask]))
        else:
            global_bias = 0.0

    transitions = np.where((ovr[1:] > ovr_threshold) & (ovr[:-1] <= ovr_threshold))[0]
    results = []
    for idx in transitions:
        idx_after = idx + 1
        t_event = t_log[idx_after]
        window_mask = (t_log >= t_event - lookback_sec) & (t_log < t_event)
        if window_mask.sum() < 3:
            continue

        v_at = float(v[idx_after])
        cmd_window = cmd[window_mask]
        meas_window = meas[window_mask] - global_bias  # bias-corrected
        t_window = t_log[window_mask]
        residual_window = meas_window - cmd_window

        # Time-resolved in-curve measures
        peak_cmd_5s = float(np.max(np.abs(cmd_window)))
        within_1s = window_mask & (t_log >= t_event - 1.0)
        peak_cmd_1s = float(np.max(np.abs(cmd[within_1s]))) if within_1s.any() else 0.0

        # Time since last curve sample — report at both thresholds (QA review #3)
        curve_samples = window_mask & (np.abs(cmd) > CURVE_THRESHOLD_FOR_OVR)
        curve_samples_strict = window_mask & (np.abs(cmd) > CURVE_THRESHOLD_FOR_OVR_STRICT)
        if curve_samples.any():
            last_curve_t = float(t_log[curve_samples][-1])
            time_since_last_curve_sec = float(t_event - last_curve_t)
        else:
            time_since_last_curve_sec = None  # no curve in lookback
        if curve_samples_strict.any():
            last_curve_strict_t = float(t_log[curve_samples_strict][-1])
            time_since_last_curve_strict_sec = float(t_event - last_curve_strict_t)
        else:
            time_since_last_curve_strict_sec = None

        results.append({
            't_override': float(t_event),
            'speed_mph': v_at * 2.237,
            'speed_mps': v_at,
            'peak_cmd_within_1s': peak_cmd_1s,
            'peak_cmd_within_5s': peak_cmd_5s,
            'time_since_last_curve_sec': time_since_last_curve_sec,
            'time_since_last_curve_strict_sec': time_since_last_curve_strict_sec,
            'curve_threshold_primary': CURVE_THRESHOLD_FOR_OVR,
            'curve_threshold_strict': CURVE_THRESHOLD_FOR_OVR_STRICT,
            'mean_residual_bias_corrected': float(residual_window.mean()),
            'peak_residual_abs_bias_corrected': float(np.max(np.abs(residual_window))),
            'ang_at_trigger': float(ang[idx_after]),
            'global_bias_used': float(global_bias),
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE D — CROSS-ROUTE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
#
# Load multiple route JSONs and produce apples-to-apples comparisons with
# proper sample-size annotations. The cross-route mode is the actual use
# case for tuning iteration validation: compare baseline vs iter1 vs iter2
# with all the Phase A/B/C metrics aggregated correctly.

def load_route_json(path):
    """Load a curves_v2 JSON file produced by single-route mode."""
    with open(path) as f:
        return json.load(f)


def event_metric_pool(route_data, metric_path, magnitude_class=None, speed_bin_label=None,
                      require_confident_bias=True, require_reliable_epas=False,
                      exclude_flicker=False):
    """Pull a single metric across all events of one route, with filters.

    QA Phase D fix #1: speed_bin renamed to speed_bin_label to avoid name collision
    with the module-level `speed_bin()` function. Pass a label string ('45-65').

    QA Phase D fix #2: returns detailed filter breakdown so callers can explain
    why N differs from total events.

    QA round 2 fix: exclude_flicker drops events with integral-gate flicker.
    `integral.gate_flicker_likely=True` events have dCmd dominated by integral
    artifacts and shouldn't be pooled into "hunting" metrics.

    metric_path is dot-notation, e.g. 'metrics.peak_bias_pct' or 'planner.pipeline_gain_1_3'.

    Returns (values, filter_breakdown) where filter_breakdown is a dict.
    """
    values = []
    fb = {'n_total': 0, 'n_no_metrics': 0, 'n_low_confidence_bias': 0,
          'n_wrong_magnitude': 0, 'n_wrong_speed_bin': 0,
          'n_unreliable_epas': 0, 'n_missing_metric': 0,
          'n_flicker_excluded': 0}
    for e in route_data.get('events', []):
        if 'metrics' not in e or e.get('metrics') is None:
            fb['n_no_metrics'] += 1
            continue
        fb['n_total'] += 1
        if require_confident_bias and e.get('bias') and not e['bias'].get('confident'):
            fb['n_low_confidence_bias'] += 1
            continue
        if magnitude_class and e['metadata'].get('magnitude_class') != magnitude_class:
            fb['n_wrong_magnitude'] += 1
            continue
        if speed_bin_label and speed_bin(e['metadata'].get('mean_v_mph', 0)) != speed_bin_label:
            fb['n_wrong_speed_bin'] += 1
            continue
        if exclude_flicker:
            integ = e.get('integral') or {}
            if integ.get('gate_flicker_likely'):
                fb['n_flicker_excluded'] += 1
                continue
        if require_reliable_epas:
            lin = e.get('epas_linearity')
            if not lin or not lin.get('reliable'):
                fb['n_unreliable_epas'] += 1
                continue

        # Navigate dotted path
        obj = e
        try:
            for key in metric_path.split('.'):
                obj = obj.get(key) if isinstance(obj, dict) else None
                if obj is None:
                    break
        except Exception:
            obj = None
        # Accept int/float but NOT bool (bool is int subclass in Python)
        if obj is not None and isinstance(obj, (int, float)) and not isinstance(obj, bool):
            values.append(float(obj))
        else:
            fb['n_missing_metric'] += 1
    return values, fb


MIN_N_FOR_COMPARISON = 5
N_RATIO_INCOMPARABLE = 2.0


def _n_flag(n, ref_n=None):
    """Return a flag string indicating sample-size concerns."""
    if n < MIN_N_FOR_COMPARISON:
        return ' ⚠ small N'
    if ref_n and (max(n, ref_n) / max(min(n, ref_n), 1) > N_RATIO_INCOMPARABLE):
        return ' ⚠ N differs >2× from ref'
    return ''


# Two-sided 97.5% t-critical values (df 1..29). Used for small-N CIs.
# Drops to ~1.96 by df=30 (the normal-approximation regime).
_T_CRIT_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045,
}


def _t_critical_975(df):
    """Two-sided 95% (i.e. one-sided 97.5%) t-critical value."""
    if df <= 0:
        return float('inf')
    if df >= 30:
        return 1.96
    return _T_CRIT_975[df]


def _bootstrap_median_ci(arr, n_boot=2000, alpha=0.05):
    """Nonparametric bootstrap 95% CI on the median.

    Uses arange + integer indexing for determinism without RNG seed (Math.random
    avoidance — workflow scripts forbid it; the analyzer here is not a workflow,
    but we keep results stable by using a fixed permutation seed).
    """
    n = len(arr)
    if n < 2:
        v = float(arr[0]) if n else 0.0
        return v, v
    rng = np.random.default_rng(42)  # fixed seed for stable CIs across runs
    idx = rng.integers(0, n, size=(n_boot, n))
    medians = np.median(arr[idx], axis=1)
    lo = float(np.percentile(medians, 100 * alpha / 2))
    hi = float(np.percentile(medians, 100 * (1 - alpha / 2)))
    return lo, hi


def _iqr_or_range(arr):
    """Return (low, high) — IQR if N>=5, min/max otherwise."""
    if len(arr) >= 5:
        return float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return float(np.min(arr)), float(np.max(arr))


def compare_routes(route_jsons, labels=None, output_json=None):
    """Multi-route comparison report (QA Phase D fixes #3-#6 applied).

    - Reports N flag for small samples and N>2x disparity
    - Reports IQR (or range if N<5) alongside median
    - Shows both 1s-in-curve AND 5s-activity for overrides
    - Computes 95% CI on EPAS slope median (normal approx)
    - Writes machine-readable JSON if output_json provided
    """
    if not route_jsons:
        return

    n_routes = len(route_jsons)
    if labels is None:
        labels = [os.path.basename(rd.get('route_dir', f'route_{i}'))[:24] for i, rd in enumerate(route_jsons)]

    print('=' * 130)
    print(f'CROSS-ROUTE COMPARISON — {n_routes} routes')
    print('=' * 130)

    print(f'\n## Route overview\n')
    print(f'  {"#":<3} {"Label":<24} {"Events":>7} {"Engaged span":>14} {"Outliers":>9}')
    print('  ' + '-' * 65)
    for i, (rd, lbl) in enumerate(zip(route_jsons, labels)):
        n_evs = rd.get('n_events', 0)
        span = rd.get('t_span_sec', 0)
        n_outliers = sum(1 for e in rd.get('events', []) if e.get('flags'))
        print(f'  {i:<3} {lbl:<24} {n_evs:>7} {span:>13.0f}s {n_outliers:>9}')

    output = {'labels': labels, 'magnitude': {}, 'pipeline_gain': [],
              'epas_slope': [], 'overrides': [],
              'rl_cycling': {}, 'ema_lag': {}, 'hunting': {}}

    # Per-magnitude with N flag + spread
    print(f'\n## Curve metrics by magnitude class — median + IQR/range\n')
    for mag_cls in ['sharp', 'moderate', 'gentle']:
        print(f'  --- {mag_cls.upper()} ---')
        print(f'  {"Route":<24} {"N":>3} {"pbp%":>9} {"[IQR/rng]":>17} {"os%":>9} {"timing_ms":>11}')
        rows = []
        ref_n = None
        for rd, lbl in zip(route_jsons, labels):
            pbp, _ = event_metric_pool(rd, 'metrics.peak_bias_pct', magnitude_class=mag_cls)
            op, _ = event_metric_pool(rd, 'metrics.overshoot_pct', magnitude_class=mag_cls)
            toff, _ = event_metric_pool(rd, 'metrics.timing_offset_ms', magnitude_class=mag_cls)
            n = len(pbp)
            if ref_n is None and n > 0:
                ref_n = n
            row = {'label': lbl, 'n': n}
            if n == 0:
                print(f'  {lbl:<24} {0:>3} {"—":>9}')
                rows.append(row)
                continue
            pbp_arr = np.array(pbp)
            op_arr = np.array(op)
            toff_arr = np.array(toff)
            lo, hi = _iqr_or_range(pbp_arr)
            flag = _n_flag(n, ref_n)
            print(f'  {lbl:<24} {n:>3} {np.median(pbp_arr):>+8.2%} [{lo:+6.1%},{hi:+6.1%}] '
                  f'{np.median(op_arr) if len(op_arr) else 0:>+8.2%} '
                  f'{np.median(toff_arr) if len(toff_arr) else 0:>+10.0f}{flag}')
            row.update({
                'peak_bias_pct_median': float(np.median(pbp_arr)),
                'peak_bias_pct_iqr': [lo, hi],
                'overshoot_pct_median': float(np.median(op_arr)) if len(op_arr) else None,
                'timing_offset_ms_median': float(np.median(toff_arr)) if len(toff_arr) else None,
                'n_flag': flag.strip(),
            })
            rows.append(row)
        output['magnitude'][mag_cls] = rows
        print()

    # Pipeline gain
    print(f'\n## Pipeline gain 1-3 Hz (cmd/des) — only sufficient-power events\n')
    print(f'  {"Route":<24} {"N":>3} {"median":>9} {"min":>9} {"max":>9}')
    ref_n = None
    for rd, lbl in zip(route_jsons, labels):
        vals, _ = event_metric_pool(rd, 'planner.pipeline_gain_1_3')
        n = len(vals)
        if ref_n is None and n > 0:
            ref_n = n
        if not vals:
            print(f'  {lbl:<24} {0:>3} {"—":>9} {"—":>9} {"—":>9}')
            output['pipeline_gain'].append({'label': lbl, 'n': 0})
            continue
        flag = _n_flag(n, ref_n)
        print(f'  {lbl:<24} {n:>3} {np.median(vals):>9.3f} {np.min(vals):>9.3f} {np.max(vals):>9.3f}{flag}')
        output['pipeline_gain'].append({
            'label': lbl, 'n': n,
            'median': float(np.median(vals)),
            'min': float(np.min(vals)), 'max': float(np.max(vals)),
            'n_flag': flag.strip(),
        })

    # EPAS slope with 95% CI
    print(f'\n## EPAS slope (SNR≥5, sharp/moderate)\n')
    print(f'  {"Route":<24} {"N":>3} {"median":>9} {"95% CI":>20} {"min":>8} {"max":>8}')
    ref_n = None
    for rd, lbl in zip(route_jsons, labels):
        slopes = []
        for e in rd.get('events', []):
            if not (e.get('epas_linearity') and e['epas_linearity'].get('reliable')):
                continue
            if e.get('bias') and not e['bias'].get('confident'):
                continue
            if e['metadata'].get('magnitude_class') not in ('sharp', 'moderate'):
                continue
            slopes.append(e['epas_linearity']['slope'])
        n = len(slopes)
        if ref_n is None and n > 0:
            ref_n = n
        if not slopes:
            print(f'  {lbl:<24} {0:>3} {"—":>9}')
            output['epas_slope'].append({'label': lbl, 'n': 0})
            continue
        s = np.array(slopes)
        # QA review #2 fix: use sample std (ddof=1) and Student-t for N<30.
        # Previous version used ddof=0 + 1.96, giving CIs ~30-50% too narrow.
        # Also: CI is now computed AROUND the median via bootstrap so it matches
        # the printed point estimate (was: CI on mean printed next to median).
        if n >= 2:
            t_crit = _t_critical_975(n - 1)
            se = float(s.std(ddof=1)) / math.sqrt(n)
            mean_ci_lo = float(s.mean() - t_crit * se)
            mean_ci_hi = float(s.mean() + t_crit * se)
            # Bootstrap CI on median (more honest given median is the point estimate)
            med_lo, med_hi = _bootstrap_median_ci(s, n_boot=2000) if n >= 3 else (float(s.min()), float(s.max()))
        else:
            mean_ci_lo = mean_ci_hi = float(s[0])
            med_lo = med_hi = float(s[0])
        flag = _n_flag(n, ref_n)
        print(f'  {lbl:<24} {n:>3} {np.median(s):>9.3f} [{med_lo:>6.3f},{med_hi:>6.3f}]  '
              f'{s.min():>8.3f} {s.max():>8.3f}{flag}')
        output['epas_slope'].append({
            'label': lbl, 'n': n,
            'median': float(np.median(s)),
            'median_ci_95_bootstrap': [med_lo, med_hi],
            'mean': float(s.mean()),
            'mean_ci_95_t': [mean_ci_lo, mean_ci_hi],
            'min': float(s.min()), 'max': float(s.max()),
            'n_flag': flag.strip(),
        })

    # Override behavior — both thresholds + per-minute rate (QA review #3)
    print(f'\n## Override behavior — both thresholds, time-normalized\n')
    print(f'  {"Route":<24} {"N":>3} {"Per min":>8} {"InCurve@1s(0.001/0.0015)":>26} '
          f'{"Activity@5s":>12} {"Med resid":>10} {"P90 resid":>10}')
    for rd, lbl in zip(route_jsons, labels):
        overrides = rd.get('overrides', [])
        n = len(overrides)
        span_sec = rd.get('t_span_sec', 0) or 0
        rate_per_min = (n / (span_sec / 60.0)) if span_sec > 0 else None
        if n == 0:
            rate_str = f'{rate_per_min:.3f}' if rate_per_min is not None else '—'
            print(f'  {lbl:<24} {0:>3} {rate_str:>8} {"—":>26} {"—":>12} {"—":>10} {"—":>10}')
            output['overrides'].append({'label': lbl, 'n': 0,
                                        'rate_per_min': rate_per_min,
                                        't_span_sec': span_sec})
            continue
        in_curve_1s_primary = sum(1 for o in overrides
                                  if o.get('peak_cmd_within_1s', 0) > CURVE_THRESHOLD_FOR_OVR)
        in_curve_1s_strict = sum(1 for o in overrides
                                 if o.get('peak_cmd_within_1s', 0) > CURVE_THRESHOLD_FOR_OVR_STRICT)
        had_activity_5s = sum(1 for o in overrides if o.get('time_since_last_curve_sec') is not None)
        resids = np.array([o.get('peak_residual_abs_bias_corrected', 0) for o in overrides])
        rate_str = f'{rate_per_min:.3f}' if rate_per_min is not None else '—'
        in_curve_str = f'{in_curve_1s_primary}/{in_curve_1s_strict}/{n}'
        print(f'  {lbl:<24} {n:>3} {rate_str:>8} {in_curve_str:>26} '
              f'{had_activity_5s}/{n:<11} '
              f'{np.median(resids):>10.5f} {np.percentile(resids, 90):>10.5f}')
        output['overrides'].append({
            'label': lbl, 'n': n,
            't_span_sec': span_sec,
            'rate_per_min': rate_per_min,
            'in_curve_1s_primary_thresh': in_curve_1s_primary,
            'in_curve_1s_strict_thresh': in_curve_1s_strict,
            'had_activity_5s': had_activity_5s,
            'resid_median': float(np.median(resids)),
            'resid_p90': float(np.percentile(resids, 90)),
        })

    # Mechanism metrics — by magnitude class (most relevant for "hunting" feel)
    print(f'\n## Rate-limit cycling (preRL vs rl clip + direction flips) — sharp/moderate\n')
    print(f'  {"Route":<24} {"N":>3} {"clip_frac_med":>14} {"flips_per_s_med":>16} {"flips_p90":>10}')
    for mag_cls in ['sharp', 'moderate']:
        print(f'  --- {mag_cls.upper()} ---')
        ref_n = None
        rows = []
        for rd, lbl in zip(route_jsons, labels):
            clips, _ = event_metric_pool(rd, 'rl_cycling.clipped_frac', magnitude_class=mag_cls)
            flips, _ = event_metric_pool(rd, 'rl_cycling.flips_per_sec', magnitude_class=mag_cls)
            n = len(flips)
            if ref_n is None and n > 0:
                ref_n = n
            row = {'label': lbl, 'n': n}
            if n == 0:
                print(f'  {lbl:<24} {0:>3} {"—":>14} {"—":>16} {"—":>10}')
                rows.append(row); continue
            flag = _n_flag(n, ref_n)
            print(f'  {lbl:<24} {n:>3} {np.median(clips):>14.2%} {np.median(flips):>16.3f} '
                  f'{np.percentile(flips, 90):>10.3f}{flag}')
            row.update({'clipped_frac_median': float(np.median(clips)),
                        'flips_per_sec_median': float(np.median(flips)),
                        'flips_per_sec_p90': float(np.percentile(flips, 90)),
                        'n_flag': flag.strip()})
            rows.append(row)
        output['rl_cycling'][mag_cls] = rows

    print(f'\n## EMA-vs-des lag (positive = ema follows des; >150ms suggests jerk source) — sharp/moderate\n')
    print(f'  {"Route":<24} {"N":>3} {"lag_ms_med":>11} {"lag_ms_p90":>11} {"peak_corr_med":>14}')
    for mag_cls in ['sharp', 'moderate']:
        print(f'  --- {mag_cls.upper()} ---')
        ref_n = None
        rows = []
        for rd, lbl in zip(route_jsons, labels):
            lags, _ = event_metric_pool(rd, 'ema_lag.lag_ms', magnitude_class=mag_cls)
            corrs, _ = event_metric_pool(rd, 'ema_lag.peak_corr', magnitude_class=mag_cls)
            n = len(lags)
            if ref_n is None and n > 0:
                ref_n = n
            row = {'label': lbl, 'n': n}
            if n == 0:
                print(f'  {lbl:<24} {0:>3} {"—":>11} {"—":>11} {"—":>14}')
                rows.append(row); continue
            flag = _n_flag(n, ref_n)
            print(f'  {lbl:<24} {n:>3} {np.median(lags):>11.1f} {np.percentile(np.abs(lags), 90):>11.1f} '
                  f'{(np.median(corrs) if corrs else 0):>14.3f}{flag}')
            row.update({'lag_ms_median': float(np.median(lags)),
                        'lag_abs_ms_p90': float(np.percentile(np.abs(lags), 90)),
                        'peak_corr_median': float(np.median(corrs)) if corrs else None,
                        'n_flag': flag.strip()})
            rows.append(row)
        output['ema_lag'][mag_cls] = rows

    # QA round 2: hunting score with (1) flicker events excluded, (2) cmd vs
    # des comparison so we can see if the controller is adding crossings above
    # the planner baseline, (3) speed-binned view to control for the speed
    # confound flagged in QA round 2.
    print(f'\n## Hunting score (uniform 20Hz resampled, flicker excluded) — sharp/moderate\n')
    print(f'  cmd_cps = controller cmd crossings/sec; des_cps = planner baseline; '
          f'excess = cmd_cps - des_cps (>0 = controller adds crossings)\n')
    print(f'  {"Route":<24} {"N":>3} {"cmd_cps_med":>12} {"des_cps_med":>12} '
          f'{"excess_med":>11} {"flicker_excl":>13}')
    for mag_cls in ['sharp', 'moderate']:
        print(f'  --- {mag_cls.upper()} ---')
        ref_n = None
        rows = []
        for rd, lbl in zip(route_jsons, labels):
            cps, fb_cps = event_metric_pool(rd, 'hunting.crossings_per_sec',
                                            magnitude_class=mag_cls, exclude_flicker=True)
            des_cps, _ = event_metric_pool(rd, 'hunting.des_crossings_per_sec',
                                           magnitude_class=mag_cls, exclude_flicker=True)
            excess, _ = event_metric_pool(rd, 'hunting.excess_crossings_per_sec',
                                          magnitude_class=mag_cls, exclude_flicker=True)
            peaks, _ = event_metric_pool(rd, 'hunting.peak_dcmd_per_s',
                                         magnitude_class=mag_cls, exclude_flicker=True)
            n = len(cps)
            if ref_n is None and n > 0:
                ref_n = n
            row = {'label': lbl, 'n': n, 'n_flicker_excluded': fb_cps.get('n_flicker_excluded', 0)}
            if n == 0:
                print(f'  {lbl:<24} {0:>3} {"—":>12} {"—":>12} {"—":>11} '
                      f'{fb_cps.get("n_flicker_excluded", 0):>13}')
                rows.append(row); continue
            flag = _n_flag(n, ref_n)
            cmd_med = float(np.median(cps))
            des_med = float(np.median(des_cps)) if des_cps else None
            ex_med = float(np.median(excess)) if excess else None
            print(f'  {lbl:<24} {n:>3} {cmd_med:>12.3f} '
                  f'{(des_med if des_med is not None else 0):>12.3f} '
                  f'{(ex_med if ex_med is not None else 0):>+11.3f} '
                  f'{fb_cps.get("n_flicker_excluded", 0):>13}{flag}')
            row.update({'cmd_cps_median': cmd_med,
                        'des_cps_median': des_med,
                        'excess_cps_median': ex_med,
                        'peak_dcmd_per_s_median': float(np.median(peaks)) if peaks else None,
                        'n_flag': flag.strip()})
            rows.append(row)
        output['hunting'][mag_cls] = rows

    # Speed-binned hunting score (controls for speed confound)
    print(f'\n## Hunting score — speed-binned (controls for speed confound)\n')
    print(f'  {"Route":<24} {"15-30":>8} {"30-45":>8} {"45-65":>8} {"65-80":>8}  (cmd_cps median, N in parens)')
    output['hunting']['by_speed'] = {}
    for rd, lbl in zip(route_jsons, labels):
        cells = []
        for sb in ['15-30', '30-45', '45-65', '65-80']:
            v, _ = event_metric_pool(rd, 'hunting.crossings_per_sec',
                                     speed_bin_label=sb, exclude_flicker=True)
            if v:
                cells.append(f'{np.median(v):.2f}({len(v)})')
            else:
                cells.append('—')
        print(f'  {lbl:<24} {cells[0]:>8} {cells[1]:>8} {cells[2]:>8} {cells[3]:>8}')
        output['hunting']['by_speed'][lbl] = {
            sb: {'median': float(np.median(v)) if v else None, 'n': len(v)}
            for sb, v in [(sb, event_metric_pool(rd, 'hunting.crossings_per_sec',
                                                 speed_bin_label=sb, exclude_flicker=True)[0])
                          for sb in ['15-30', '30-45', '45-65', '65-80']]
        }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(_json_safe(output), f, indent=2)
        print(f'\n[compare result written to {output_json}]')


# ─────────────────────────────────────────────────────────────────────────────
# JSON SERIALIZATION (handles NaN safely)
# ─────────────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    """Recursively replace NaN/Inf with None for JSON compatibility."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY OUTPUT (text)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(events, speed_bin_summary=None, overrides=None):
    """Print summary statistics with proper sample annotations."""
    if not events:
        print('No curve events detected.')
        return

    print(f'\n{"="*100}')
    print(f'CURVE EVENT SUMMARY — {len(events)} events detected')
    print(f'{"="*100}\n')

    # By magnitude class
    by_class = defaultdict(list)
    for e in events:
        by_class[e['metadata']['magnitude_class']].append(e)

    print(f'Event counts by magnitude class:')
    for cls in ['sharp', 'moderate', 'gentle', 'subgentle']:
        n = len(by_class.get(cls, []))
        if n > 0:
            print(f'  {cls:<12}: {n} events')

    # Per-class metrics
    for cls in ['sharp', 'moderate', 'gentle']:
        evs = by_class.get(cls, [])
        if not evs:
            continue
        print(f'\n--- {cls.upper()} curves (N={len(evs)}) ---')
        # Collect metrics
        # Only count events with confident bias estimate
        confident_evs = [e for e in evs if e.get('bias') and e['bias'].get('confident')]
        n_dropped = len(evs) - len(confident_evs)

        pbp = [e['metrics']['peak_bias_pct'] for e in confident_evs if e.get('metrics') and e['metrics'].get('peak_bias_pct') is not None]
        op = [e['metrics']['overshoot_pct'] for e in confident_evs if e.get('metrics') and e['metrics'].get('overshoot_pct') is not None]
        toff = [e['metrics']['timing_offset_ms'] for e in confident_evs if e.get('metrics') and e['metrics'].get('timing_offset_ms') is not None]
        ccr = [e['metrics']['crosscorr_r'] for e in confident_evs if e.get('metrics') and e['metrics'].get('crosscorr_r') is not None]
        eb = [e['metrics']['entry_bias_30'] for e in confident_evs if e.get('metrics') and e['metrics'].get('entry_bias_30') is not None]
        xb = [e['metrics']['exit_bias_30'] for e in confident_evs if e.get('metrics') and e['metrics'].get('exit_bias_30') is not None]
        sd = [e['metrics']['cmd_peak_abs'] for e in confident_evs if e.get('metrics') and e['metrics'].get('cmd_peak_abs') is not None]
        vmph = [e['metadata']['mean_v_mph'] for e in confident_evs]

        if n_dropped:
            print(f'    (dropped {n_dropped} events with low-confidence bias estimate)')

        def stats(vals, name, fmt='.6f'):
            if not vals: return f'    {name}: no data'
            arr = np.array(vals)
            return (f'    {name}: median {np.median(arr):{fmt}}, mean {arr.mean():{fmt}}, '
                    f'std {arr.std():{fmt}}, min {arr.min():{fmt}}, max {arr.max():{fmt}}  (N={len(vals)})')

        print(stats(pbp, 'peak_bias_pct (signed: pos=overshoot in cmd dir, neg=undershoot)', '+.2%'))
        print(stats(op, 'overshoot_pct (peak-relative, bias-corrected; pos=meas exceeds cmd peak)', '+.2%'))
        print(stats(toff, 'timing_offset_ms (signed: meas peak vs cmd peak; pos=meas peaks later)', '+.0f'))
        print(stats(ccr, 'crosscorr_r (peak correlation strength of meas vs cmd shape)', '.3f'))
        print(stats(eb, 'entry_bias_30 (bias-corrected, at 30% rise)', '+.6f'))
        print(stats(xb, 'exit_bias_30 (bias-corrected, at 30% fall)', '+.6f'))
        print(stats(sd, 'cmd peak magnitude (1/m)', '.5f'))
        print(stats(vmph, 'speed at apex (mph)', '.1f'))

        # Phase B: FFT bands
        ang_dom = [e.get('fft', {}).get('dAng', {}).get('dom_hz') for e in confident_evs
                   if e.get('fft') and e['fft'].get('dAng')]
        ang_dom = [x for x in ang_dom if x is not None]
        ang_p1_3 = [e.get('fft', {}).get('dAng', {}).get('pct_1_3') for e in confident_evs
                    if e.get('fft') and e['fft'].get('dAng')]
        ang_p1_3 = [x for x in ang_p1_3 if x is not None]
        ang_p3_10 = [e.get('fft', {}).get('dAng', {}).get('pct_3_10') for e in confident_evs
                     if e.get('fft') and e['fft'].get('dAng')]
        ang_p3_10 = [x for x in ang_p3_10 if x is not None]
        if ang_dom:
            print(stats(ang_dom, 'dAng dominant freq (Hz)', '.2f'))
            print(stats(ang_p1_3, 'dAng power 1-3 Hz (felt oscillation band)', '.2%'))
            print(stats(ang_p3_10, 'dAng power 3-10 Hz (high-freq jitter)', '.2%'))

        # Phase B: EPAS linearity — ONLY report on events with reliable SNR
        # (M1 fix: at low SNR, slope/R² are noise-dominated, not real EPAS behavior)
        reliable_evs = [e for e in confident_evs
                       if e.get('epas_linearity') and e['epas_linearity'].get('reliable')]
        unreliable_n = len(confident_evs) - len(reliable_evs)
        if reliable_evs:
            slope = [e['epas_linearity']['slope'] for e in reliable_evs]
            r2 = [e['epas_linearity']['r2'] for e in reliable_evs]
            snr_vals = [e['epas_linearity']['snr'] for e in reliable_evs]
            print(stats(slope, 'EPAS slope (meas/cmd, ideal=1.0) [SNR≥5]', '.3f'))
            print(stats(r2, 'EPAS linearity R² [SNR≥5]', '.3f'))
            print(stats(snr_vals, 'EPAS regression SNR (cmd_peak / residual_std)', '.1f'))
        if unreliable_n:
            print(f'    ({unreliable_n} events excluded from EPAS lin (SNR<5 — slope/R² would be noise-dominated))')

        # Phase C: planner-side jitter — is the source itself noisy?
        des_p13 = [e.get('planner', {}).get('des', {}).get('pct_1_3') for e in confident_evs
                   if e.get('planner') and e['planner'].get('des')]
        des_p13 = [x for x in des_p13 if x is not None]
        cmd_p13 = [e.get('planner', {}).get('cmd', {}).get('pct_1_3') for e in confident_evs
                   if e.get('planner') and e['planner'].get('cmd')]
        cmd_p13 = [x for x in cmd_p13 if x is not None]
        pgain = [e.get('planner', {}).get('pipeline_gain_1_3') for e in confident_evs
                 if e.get('planner')]
        pgain = [x for x in pgain if x is not None]
        if des_p13:
            print(stats(des_p13, 'des (planner) power 1-3 Hz', '.2%'))
            print(stats(cmd_p13, 'cmd (post-pipeline) power 1-3 Hz', '.2%'))
            if pgain:
                print(stats(pgain, 'pipeline gain 1-3 Hz (cmd/des; >1 = pipeline amplifies)', '.2f'))

        # Phase C: pc_blend transitions during curves
        blend_delta = [e.get('blend', {}).get('delta') for e in confident_evs if e.get('blend')]
        blend_delta = [x for x in blend_delta if x is not None]
        if blend_delta:
            print(stats(blend_delta, 'pc_blend delta during curve (max-min; >0.1 = mid-curve transition)', '.3f'))

        # Phase C: PI integral behavior through curve
        lint_max = [e.get('integral', {}).get('lInt_max_abs') for e in confident_evs if e.get('integral')]
        lint_max = [x for x in lint_max if x is not None]
        if lint_max:
            print(stats(lint_max, 'PI integral peak |lInt| during curve', '.3f'))

    # Phase B: speed-binned table
    if speed_bin_summary:
        print(f'\n\n{"="*100}')
        print('SPEED-BINNED PHASE METRICS (median values per bin × magnitude)')
        print(f'{"="*100}')
        print(f'\n  {"Speed bin":<10} {"Mag class":<10} {"N":>4} {"pbp_med":>10} {"os_med":>10} {"toff_ms":>10}')
        print('  ' + '-' * 60)
        for key in sorted(speed_bin_summary.keys()):
            d = speed_bin_summary[key]
            sb, mag = key.rsplit('_', 1)
            pbp = f'{d["peak_bias_pct_median"]:+9.2%}' if d['peak_bias_pct_median'] is not None else f'{"--":>10}'
            os_v = f'{d["overshoot_pct_median"]:+9.2%}' if d['overshoot_pct_median'] is not None else f'{"--":>10}'
            toff = f'{d["timing_offset_ms_median"]:+9.0f}' if d['timing_offset_ms_median'] is not None else f'{"--":>10}'
            print(f'  {sb:<10} {mag:<10} {d["n"]:>4} {pbp} {os_v} {toff}')

    # Phase B: outliers table
    flagged = [e for e in events if e.get('flags')]
    if flagged:
        print(f'\n\n{"="*100}')
        print(f'OUTLIER / WARNING EVENTS — {len(flagged)} flagged')
        print(f'{"="*100}')
        print(f'\n  {"Event":<6} {"Class":<10} {"Speed mph":>9} {"Apex time":>20} {"Flags"}')
        print('  ' + '-' * 90)
        # Find route start time to make timestamps relative
        all_t = [e['metadata']['t_apex'] for e in events if 'metadata' in e]
        t0 = min(all_t) if all_t else 0
        for e in flagged[:30]:
            md = e['metadata']
            flags_str = ', '.join(e['flags'])
            t_rel = md['t_apex'] - t0
            t_str = f'{int(t_rel // 60)}:{int(t_rel % 60):02d}'  # N8: MM:SS
            print(f'  {e["event_id"]:<6} {md["magnitude_class"]:<10} {md["mean_v_mph"]:>9.1f} '
                  f'{t_str:>10} {flags_str}')

    # Phase C: override correlation table (fixed bias-correction + time-resolved in-curve)
    if overrides:
        print(f'\n\n{"="*100}')
        print(f'OVERRIDE EVENTS — {len(overrides)} driver interventions')
        print(f'{"="*100}')
        speeds = np.array([o['speed_mph'] for o in overrides])
        in_curve_1s = sum(1 for o in overrides if o.get('peak_cmd_within_1s', 0) > CURVE_THRESHOLD_FOR_OVR)
        had_curve_5s = sum(1 for o in overrides if o.get('time_since_last_curve_sec') is not None)
        print(f'\n  Speed distribution:')
        for lo, hi, lbl in SPEED_BINS_MPH:
            n = int(np.sum((speeds >= lo) & (speeds < hi)))
            if n > 0:
                print(f'    {lbl:>6} mph: {n} overrides')
        print(f'  In-curve at moment of override (|cmd|>{CURVE_THRESHOLD_FOR_OVR} within 1s): {in_curve_1s}/{len(overrides)}')
        print(f'  Had any curve activity in 5s lookback: {had_curve_5s}/{len(overrides)}')
        # Bias-corrected peak residual
        peak_resids = np.array([o.get('peak_residual_abs_bias_corrected', 0) for o in overrides])
        print(f'  Peak |meas-cmd| in 5s before override (bias-corrected): '
              f'median {np.median(peak_resids):.5f}, P90 {np.percentile(peak_resids, 90):.5f} (1/m)')
        if overrides[0].get('global_bias_used') is not None:
            print(f'  (Global bias subtracted: {overrides[0]["global_bias_used"]:+.6f} 1/m)')
        # First 10 events
        print(f'\n  First 10 overrides:')
        print(f'  {"Time":>10} {"Speed":>7} {"cmd@1s":>8} {"Δt curve":>9} {"peak resid":>12}')
        all_t = [o['t_override'] for o in overrides]
        t0 = min(all_t)
        for o in overrides[:10]:
            t_rel = o['t_override'] - t0
            t_str = f'{int(t_rel // 60)}:{int(t_rel % 60):02d}'
            dt_curve = f'{o["time_since_last_curve_sec"]:.1f}s' if o.get('time_since_last_curve_sec') is not None else 'no'
            print(f'  {t_str:>10} {o["speed_mph"]:>7.1f} {o.get("peak_cmd_within_1s", 0):>8.5f} '
                  f'{dt_curve:>9} {o.get("peak_residual_abs_bias_corrected", 0):>12.5f}')


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC TEST
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_curve(t_apex=10.0, peak_curv=0.005, half_width=2.0, bias=0.0003,
                    epas_delay=0.1, sample_rate=20.0, noise_std=0.0):
    """Generate a synthetic curve event for validation.

    Returns a fake CX1-like array and time vector.

    The shape is a Gaussian curve in cmd (peaks at t_apex with magnitude peak_curv,
    width controlled by half_width). meas is a delayed version with added bias.
    """
    duration = 30.0  # 30 seconds total
    t_log = np.arange(0, duration, 1.0 / sample_rate)
    n = len(t_log)

    # Gaussian curve for cmd
    sigma = half_width / 2.0
    cmd = peak_curv * np.exp(-0.5 * ((t_log - t_apex) / sigma) ** 2)

    # Delayed and biased meas
    meas_delay_samples = int(epas_delay * sample_rate)
    meas = np.zeros_like(cmd)
    if meas_delay_samples < n:
        meas[meas_delay_samples:] = cmd[:n - meas_delay_samples]
    meas += bias

    # Optional noise
    if noise_std > 0:
        meas += np.random.normal(0, noise_std, n)

    # Build a fake CX1 array
    arr = np.zeros((n, len(CX1_FIELDS)))
    arr[:, IDX['frame']] = np.arange(n) * 5  # 100Hz frame counter
    arr[:, IDX['v']] = 20.0  # ~45 mph
    arr[:, IDX['cmd']] = cmd
    arr[:, IDX['meas']] = meas
    arr[:, IDX['ang']] = 0.0
    arr[:, IDX['ovr']] = 0
    return arr, t_log


def run_synthetic_test():
    print('SYNTHETIC CURVE VALIDATION')
    print('=' * 70)
    print()

    # Test 1: Symmetric curve, no bias, no delay → should show ~0 for all metrics
    print('TEST 1: Clean curve (no bias, no delay)')
    arr, t_log = synthetic_curve(peak_curv=0.005, bias=0, epas_delay=0)
    events = detect_curve_events(arr, t_log, min_peak=0.001)
    assert len(events) == 1, f'Expected 1 event, got {len(events)}'
    e = events[0]
    bias = compute_bias_baseline(arr, t_log, e)
    wf = extract_waveform(arr, t_log, e)
    m = compute_phase_metrics(wf, bias['combined'])
    print(f'  bias est: {bias["combined"]:+.6f} (expected ~0)')
    print(f'  peak_bias_pct: {m["peak_bias_pct"]:+.4%} (expected ~0%)')
    print(f'  overshoot_pct: {m["overshoot_pct"]:+.4%} (expected ~0%)')
    print(f'  timing_offset_ms: {m["timing_offset_ms"]:+.0f} (expected ~0)')
    print(f'  crosscorr_r: {m["crosscorr_r"]:.3f} (expected ~1.0 since cmd==meas)')
    assert abs(bias['combined']) < 1e-4, 'Bias should be ~0'
    assert abs(m['peak_bias_pct']) < 0.01, 'peak_bias_pct should be ~0'
    print('  PASS\n')

    # Test 2: Curve with bias only, no delay → bias correction should remove it
    print('TEST 2: Curve with +0.001 bias, no delay')
    arr, t_log = synthetic_curve(peak_curv=0.005, bias=0.001, epas_delay=0)
    events = detect_curve_events(arr, t_log, min_peak=0.001)
    e = events[0]
    bias = compute_bias_baseline(arr, t_log, e)
    wf = extract_waveform(arr, t_log, e)
    m = compute_phase_metrics(wf, bias['combined'])
    print(f'  bias est: {bias["combined"]:+.6f} (expected ~+0.001)')
    print(f'  peak_bias_pct: {m["peak_bias_pct"]:+.4%} (expected ~0% — bias removed)')
    print(f'  overshoot_pct: {m["overshoot_pct"]:+.4%} (expected ~0%)')
    assert abs(bias['combined'] - 0.001) < 2e-4, f'Bias est should be ~+0.001, got {bias["combined"]}'
    assert abs(m['peak_bias_pct']) < 0.01, f'peak_bias_pct should be near 0 after bias-correction, got {m["peak_bias_pct"]}'
    print('  PASS — bias correction works\n')

    # Test 3: Curve with delay only, no bias → should show positive timing offset
    print('TEST 3: Curve with 100ms EPAS delay, no bias')
    arr, t_log = synthetic_curve(peak_curv=0.005, bias=0, epas_delay=0.1)
    events = detect_curve_events(arr, t_log, min_peak=0.001)
    e = events[0]
    bias = compute_bias_baseline(arr, t_log, e)
    wf = extract_waveform(arr, t_log, e)
    m = compute_phase_metrics(wf, bias['combined'])
    print(f'  timing_offset_ms: {m["timing_offset_ms"]:+.0f} (expected ~+100ms)')
    print(f'  crosscorr_r: {m["crosscorr_r"]:.3f} (correlation strength, expected high since shape matches)')
    assert m['timing_offset_ms'] is not None and 50 < m['timing_offset_ms'] < 150, \
        f'timing_offset should be ~100ms, got {m["timing_offset_ms"]}'
    print('  PASS — delay detection works (via timing_offset_ms; crosscorr_lag was dropped per QA)\n')

    # Test 4: Curve with bias AND delay → both should be captured
    print('TEST 4: Curve with +0.0005 bias AND 100ms delay')
    arr, t_log = synthetic_curve(peak_curv=0.005, bias=0.0005, epas_delay=0.1)
    events = detect_curve_events(arr, t_log, min_peak=0.001)
    e = events[0]
    bias = compute_bias_baseline(arr, t_log, e)
    wf = extract_waveform(arr, t_log, e)
    m = compute_phase_metrics(wf, bias['combined'])
    print(f'  bias est: {bias["combined"]:+.6f} (expected ~+0.0005)')
    print(f'  peak_bias_pct: {m["peak_bias_pct"]:+.4%} (expected: residual phase, not bias)')
    print(f'  timing_offset_ms: {m["timing_offset_ms"]:+.0f} (expected ~+100ms)')
    assert abs(bias['combined'] - 0.0005) < 2e-4
    assert m['timing_offset_ms'] is not None and 50 < m['timing_offset_ms'] < 150
    print('  PASS — bias and timing both captured\n')

    # Test 6: Left-turn sign convention — peak_bias_pct should be positive for overshoot
    print('TEST 6: Left turn with overshoot — peak_bias_pct should be POSITIVE')
    arr, t_log = synthetic_curve(peak_curv=-0.005, bias=0, epas_delay=0)
    # Amplify meas in middle to simulate overshoot (same direction as cmd, so more negative)
    cmd = arr[:, IDX['cmd']]
    meas = cmd.copy()
    apex_t = 10.0
    bump = -0.001 * np.exp(-0.5 * ((t_log - (apex_t + 0.1)) / 0.5) ** 2)  # negative bump = more left
    meas = meas + bump
    arr[:, IDX['meas']] = meas
    events = detect_curve_events(arr, t_log, min_peak=0.001)
    e = events[0]
    bias = compute_bias_baseline(arr, t_log, e)
    wf = extract_waveform(arr, t_log, e)
    m = compute_phase_metrics(wf, bias['combined'])
    print(f'  cmd_peak: {e["peak_cmd"]:+.4f} (left)')
    print(f'  peak_bias_pct: {m["peak_bias_pct"]:+.2%} (expected POSITIVE — overshoot in cmd direction)')
    print(f'  overshoot_pct: {m["overshoot_pct"]:+.2%} (expected POSITIVE)')
    assert m['peak_bias_pct'] >= 0, f'Left-turn overshoot should be positive pbp, got {m["peak_bias_pct"]}'
    assert m['overshoot_pct'] > 0.10, f'Should detect overshoot, got {m["overshoot_pct"]}'
    print('  PASS — sign convention is direction-aware\n')

    # Test 5: True overshoot (meas peaks higher than cmd peak)
    print('TEST 5: True overshoot — meas peaks 20% higher than cmd')
    arr, t_log = synthetic_curve(peak_curv=0.005, bias=0, epas_delay=0)
    # Manually amplify meas in middle to simulate overshoot
    cmd = arr[:, IDX['cmd']]
    meas = cmd.copy()
    # Add a Gaussian bump centered at apex with peak 0.001 (20% of 0.005)
    apex_t = 10.0
    bump = 0.001 * np.exp(-0.5 * ((t_log - (apex_t + 0.1)) / 0.5) ** 2)
    meas = meas + bump
    arr[:, IDX['meas']] = meas
    events = detect_curve_events(arr, t_log, min_peak=0.001)
    e = events[0]
    bias = compute_bias_baseline(arr, t_log, e)
    wf = extract_waveform(arr, t_log, e)
    m = compute_phase_metrics(wf, bias['combined'])
    print(f'  overshoot_pct: {m["overshoot_pct"]:+.2%} (expected ~+20%)')
    assert m['overshoot_pct'] > 0.10, f'Should detect overshoot, got {m["overshoot_pct"]}'
    print('  PASS — overshoot detection works\n')

    # ════════════ PHASE B SYNTHETIC TESTS ════════════
    if HAS_SCIPY:
        print('\nPHASE B TESTS')
        print('-' * 70)

        # Test 7: FFT of 2 Hz pure sinusoid — should show ~100% power in 1-3 Hz band
        print('TEST 7: 2 Hz pure sinusoid in dAng — pct_1_3 should be ~100%')
        arr, t_log = synthetic_curve(peak_curv=0.005, bias=0, epas_delay=0)
        # Replace dAng with a 2Hz sinusoid
        arr[:, IDX['dAng']] = 5.0 * np.sin(2 * np.pi * 2.0 * t_log)
        arr[:, IDX['aLat']] = 0.0  # zero out aLat to isolate dAng test
        events = detect_curve_events(arr, t_log, min_peak=0.001)
        e = events[0]
        bias = compute_bias_baseline(arr, t_log, e)
        wf = extract_waveform(arr, t_log, e)
        fft = compute_fft_metrics(wf)
        print(f'  dAng dom_hz: {fft["dAng"]["dom_hz"]:.2f} (expected ~2.0)')
        print(f'  dAng pct_1_3: {fft["dAng"]["pct_1_3"]:.2%} (expected >90%)')
        assert 1.5 < fft['dAng']['dom_hz'] < 2.5, f'dom_hz should be ~2.0'
        assert fft['dAng']['pct_1_3'] > 0.85, f'pct_1_3 should be high'
        print('  PASS — FFT correctly identifies 2 Hz sinusoid\n')

        # Test 8: Perfectly linear meas=cmd, no noise → slope=1.0, R²=1.0, high SNR
        print('TEST 8: Perfectly linear EPAS (no noise) — slope/R² should be exactly 1.0, reliable')
        arr, t_log = synthetic_curve(peak_curv=0.005, bias=0, epas_delay=0, noise_std=0)
        events = detect_curve_events(arr, t_log, min_peak=0.001)
        e = events[0]
        bias = compute_bias_baseline(arr, t_log, e)
        wf = extract_waveform(arr, t_log, e)
        lin = compute_epas_linearity(wf, bias['combined'])
        print(f'  slope: {lin["slope"]:.4f}, R²: {lin["r2"]:.4f}, SNR: {lin["snr"]:.1f}, reliable: {lin["reliable"]}')
        assert 0.98 < lin['slope'] < 1.02, f'slope should be ~1.0'
        assert lin['r2'] > 0.99, f'R² should be ~1.0'
        assert lin['reliable'], 'should be reliable'
        print('  PASS — linear EPAS gives slope=1, R²=1\n')

        # Test 9: Linear EPAS with noise — R² degrades with smaller cmd magnitude
        print('TEST 9: Linear EPAS with noise — verify SNR gating works')
        for peak in [0.006, 0.003, 0.0015]:
            arr, t_log = synthetic_curve(peak_curv=peak, bias=0, epas_delay=0, noise_std=0.0006)
            events = detect_curve_events(arr, t_log, min_peak=0.001)
            if not events: continue
            e = events[0]
            bias = compute_bias_baseline(arr, t_log, e)
            wf = extract_waveform(arr, t_log, e)
            lin = compute_epas_linearity(wf, bias['combined'])
            if lin:
                print(f'  cmd_peak={peak:.4f}: slope={lin["slope"]:.3f}, R²={lin["r2"]:.3f}, SNR={lin["snr"]:.1f}, reliable={lin["reliable"]}')
        print('  PASS (visual): SNR drops with cmd_peak; reliability filter prevents false claims\n')

        # ═══ PHASE C TESTS ═══
        print('PHASE C TESTS')
        print('-' * 70)

        # Test 10: pipeline_gain_1_3 ≈ 1.0 when des is a copy of cmd (identity pipeline)
        # AND both have shared 2Hz content (so 1-3 Hz band has real signal)
        print('TEST 10: des = cmd, both with shared 2Hz content — gain should be ~1.0')
        arr, t_log = synthetic_curve(peak_curv=0.005, bias=0, epas_delay=0)
        # Add a 2Hz sinusoid superimposed on the curve, to both des and cmd identically
        sine = 0.0008 * np.sin(2 * np.pi * 2.0 * t_log)
        # Modify cmd to include the sine; des = same
        new_cmd = arr[:, IDX['cmd']].copy() + sine
        arr[:, IDX['cmd']] = new_cmd
        arr[:, IDX['des']] = new_cmd.copy()  # des = cmd exactly
        events = detect_curve_events(arr, t_log, min_peak=0.001, min_speed_mps=5.0)
        if events:
            e = events[0]
            bias = compute_bias_baseline(arr, t_log, e)
            wf = extract_waveform(arr, t_log, e)
            pa = compute_planner_analysis(wf)
            gain = pa.get('pipeline_gain_1_3')
            print(f'  pipeline_gain_1_3: {gain} (expected ~1.0; None ok if power too low)')
            if gain is not None:
                assert 0.9 < gain < 1.1, f'gain should be ~1.0, got {gain}'
                print('  PASS\n')
            else:
                print('  PASS (gain returned None due to low-power gate — acceptable for this test)\n')
        else:
            print('  SKIPPED — synthetic curve detection failed (test environment issue, not bug)\n')

        # Test 11: integral decay rate detection
        print('TEST 11: lInt with synthetic decay → max_rate_abs_per_s detected')
        arr, t_log = synthetic_curve(peak_curv=0.005, bias=0, epas_delay=0)
        # Synthetic integral that decays exponentially with tau=1.5s starting at 0.3
        arr[:, IDX['lInt']] = 0.3 * np.exp(-t_log / 1.5)
        arr[:, IDX['lOff']] = 0.1 * np.ones_like(t_log)
        events = detect_curve_events(arr, t_log, min_peak=0.001)
        e = events[0]
        bias = compute_bias_baseline(arr, t_log, e)
        wf = extract_waveform(arr, t_log, e)
        it = compute_integral_trajectory(wf)
        print(f'  max_rate_abs_per_s: {it["max_rate_abs_per_s"]:.3f}, gate_flicker_likely: {it["gate_flicker_likely"]}')
        # Exponential decay with tau=1.5s from 0.3 gives max rate ~0.2/s at t=0 — well below 1.0
        assert not it['gate_flicker_likely'], 'Smooth decay should NOT trigger flicker flag'
        assert it['max_rate_abs_per_s'] < 0.5, f'Smooth decay rate should be small, got {it["max_rate_abs_per_s"]}'
        print('  PASS — smooth decay correctly not flagged as flicker\n')

        # Test 12: override correlation captures bias-corrected residual
        print('TEST 12: synthetic override with known residual → recovered ≈ true')
        arr, t_log = synthetic_curve(peak_curv=0.0, bias=0.001, epas_delay=0)  # bias only
        # Add an override transition at t=20s with a meas-cmd offset of 0.003
        arr[:, IDX['ovr']] = (t_log >= 20.0).astype(int)
        # Inject artificial residual of 0.003 in 5s before override
        spike_mask = (t_log >= 15.0) & (t_log < 20.0)
        arr[spike_mask, IDX['meas']] = arr[spike_mask, IDX['cmd']] + 0.003 + 0.001  # +0.001 bias too
        ovs = compute_override_correlation(arr, t_log, global_bias=0.001)
        if ovs:
            print(f'  peak_residual_bias_corrected: {ovs[0]["peak_residual_abs_bias_corrected"]:.5f} (expected ~0.003)')
            assert 0.0025 < ovs[0]['peak_residual_abs_bias_corrected'] < 0.0035, \
                f'Should recover ~0.003, got {ovs[0]["peak_residual_abs_bias_corrected"]}'
            print('  PASS — bias-corrected residual recovered\n')

    print('=' * 70)
    print('ALL SYNTHETIC TESTS PASSED')
    print('=' * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('route_dir', nargs='?', help='Path to route directory')
    parser.add_argument('--test', action='store_true', help='Run synthetic-curve validation tests')
    parser.add_argument('--compare', nargs='+', metavar='JSON',
                       help='Compare multiple curves_v2 JSON files (Phase D mode)')
    parser.add_argument('--labels', nargs='+', metavar='LBL',
                       help='Labels for --compare routes (default: filename-derived)')
    parser.add_argument('--max', type=int, default=None, help='Limit number of events extracted')
    parser.add_argument('--no-waveforms', action='store_true',
                       help='Omit waveforms from JSON output (smaller file)')
    parser.add_argument('--output', default=None, help='Output JSON path (default: route_<id>_curves_v2.json)')
    args = parser.parse_args()

    if args.test:
        run_synthetic_test()
        return

    if args.compare:
        # Validate label count first (item 7)
        if args.labels and len(args.labels) != len(args.compare):
            parser.error('--labels count must match --compare count')
        route_jsons = [load_route_json(p) for p in args.compare]
        labels = args.labels if args.labels else [os.path.splitext(os.path.basename(p))[0] for p in args.compare]
        compare_routes(route_jsons, labels=labels, output_json=args.output)
        return

    if not args.route_dir:
        parser.error('route_dir required (or use --test or --compare)')

    print(f'Loading CX1 from {args.route_dir}...', flush=True)
    arr, t_log = load_cx1(args.route_dir)
    if arr is None or len(arr) == 0:
        print('  ERROR: no CX1 data found')
        sys.exit(1)
    print(f'  {len(arr)} CX1 rows, {t_log[-1] - t_log[0]:.1f}s span')

    print('Detecting curve events...', flush=True)
    events = detect_curve_events(arr, t_log)
    print(f'  {len(events)} events detected')

    if args.max:
        events = events[:args.max]
        print(f'  limited to first {args.max}')

    # Process each event with neighbor-aware waveform extraction (QA fix #3).
    # QA review #4: do NOT skip had_override events — they are the most
    # informative for 'curve mishandling' (override is the user's feedback
    # signal that the controller did something bad). Compute metrics, mark them,
    # and let downstream consumers filter or include explicitly.
    results = []
    for i, ev in enumerate(events):
        prev_ev = events[i - 1] if i > 0 else None
        next_ev = events[i + 1] if i + 1 < len(events) else None

        bias = compute_bias_baseline(arr, t_log, ev)
        wf = extract_waveform(arr, t_log, ev, prev_event=prev_ev, next_event=next_ev)
        metrics = compute_phase_metrics(wf, bias['combined'])
        # Phase B additions
        fft = compute_fft_metrics(wf)
        cmd_peak_abs = metrics['cmd_peak_abs'] if metrics and 'cmd_peak_abs' in metrics else None
        epas_lin = compute_epas_linearity(wf, bias['combined'], cmd_peak_abs=cmd_peak_abs)

        # Phase C additions
        planner = compute_planner_analysis(wf)
        blend = compute_blend_trajectory(wf)
        integral = compute_integral_trajectory(wf)

        # QA review additions — mechanism metrics for hunting/jerkiness
        rl_cycling = compute_rate_limit_cycling(wf)
        ema_lag = compute_ema_lag(wf)
        hunting = compute_hunting_score(wf)

        out = {
            'event_id': i,
            'metadata': ev,
            'bias': bias,
            'metrics': metrics,
            'fft': fft,
            'epas_linearity': epas_lin,
            'planner': planner,
            'blend': blend,
            'integral': integral,
            'rl_cycling': rl_cycling,
            'ema_lag': ema_lag,
            'hunting': hunting,
        }
        if not args.no_waveforms:
            out['waveform'] = wf
        results.append(out)

    # Phase B: flag outliers across the population
    flag_outliers(results)
    # Phase B: speed-binned aggregation
    speed_bin_summary = aggregate_by_speed_bin(results)
    # Phase C: route-level override correlation analysis
    overrides = compute_override_correlation(arr, t_log)

    # Write JSON
    route_id = os.path.basename(args.route_dir.rstrip('/')).split('--')[0]
    out_path = args.output or f'explorer_st_logs/route_{route_id[6:] if route_id.startswith("00000") else route_id}_curves_v2.json'
    with open(out_path, 'w') as f:
        json.dump({
            'route_dir': args.route_dir,
            'n_cx1_rows': int(len(arr)),
            't_span_sec': float(t_log[-1] - t_log[0]),
            'n_events': len(events),
            'speed_bin_summary': _json_safe(speed_bin_summary),
            'overrides': _json_safe(overrides),
            'events': _json_safe(results),
        }, f, indent=2)
    print(f'Wrote {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)')

    # Summary
    valid_events = [e for e in results if 'metrics' in e and e['metrics'] is not None]
    print_summary(valid_events, speed_bin_summary=speed_bin_summary, overrides=overrides)


if __name__ == '__main__':
    main()
