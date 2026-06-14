#!/usr/bin/env python3
"""Pull modelV2.action.desiredCurvature from rlog and compare to des (in CX1).

QA round 4 rewrite: previous version compared modelV2 (20 Hz native) to CX1
(2-10 Hz mode-switched) by independently resampling each to 20 Hz, which
created phantom crossings on the sparse CX1 side and made it look like
controlsd's clip_curvature was filtering ~50% of crossings. It was actually
sampling aliasing.

Fix:
- Build a single uniform 100 Hz grid per event window.
- Interpolate both signals onto it.
- MASK any grid point whose nearest CX1 sample is >50 ms away (CX1 under-
  sampled in that region; interp values are fiction). Don't count crossings
  in masked regions.
- Recompute crossings on the masked grid.

If model_cps ≈ des_cps after masking, the model output reaches actuators.curvature
essentially unchanged — clip_curvature/controlsd do nothing measurable.

Usage:
  ./des_source_trace.py <route_prefix> [--mag {sharp,moderate,gentle}]
"""

import sys
import os
import glob
import argparse
import numpy as np

sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_curves_v2 import load_cx1, detect_curve_events, IDX
from planner_trace import LOG_PRECISION, RATE_THRESHOLD_K


GRID_RATE_HZ = 100.0       # uniform comparison grid
MAX_CX1_GAP_S = 0.05       # 50 ms — drop crossings in regions where CX1 is undersampled


def pull_model_curvature(route_prefix):
    """Extract (t, desiredCurvature) from modelV2.action across the route."""
    from openpilot.tools.lib.logreader import LogReader

    files = sorted(glob.glob(os.path.join(route_prefix, 'rlog_*.zst')))
    if not files:
        seg_dirs = sorted(glob.glob(route_prefix + '--*'),
                          key=lambda d: int(d.rsplit('--', 1)[-1]))
        for sd in seg_dirs:
            rlog = os.path.join(sd, 'rlog.zst')
            if os.path.exists(rlog):
                files.append(rlog)

    t_list, curv_list = [], []
    for fp in files:
        try:
            for msg in LogReader(fp):
                if msg.which() != 'modelV2':
                    continue
                mv2 = msg.modelV2
                if not hasattr(mv2, 'action'):
                    continue
                t_list.append(msg.logMonoTime / 1e9)
                curv_list.append(float(mv2.action.desiredCurvature))
        except Exception:
            pass
    return np.array(t_list), np.array(curv_list)


def crossings_masked(t_grid, sig_grid, valid_mask):
    """Count rate-of-change zero-crossings on a uniform grid, ignoring transitions
    that begin or end in a masked (undersampled) region.

    Crossing accounting:
      - A crossing is a sign change in rate between consecutive valid samples.
      - If either neighbor of a transition is masked, skip — we cannot tell.
      - Threshold (RATE_THRESHOLD_K * LOG_PRECISION / dt) excludes quantization
        floor wobble.
    """
    if len(t_grid) < 3:
        return None
    dt = float(t_grid[1] - t_grid[0])
    rate = np.diff(sig_grid) / dt                              # length n-1
    threshold = RATE_THRESHOLD_K * LOG_PRECISION / dt
    rate_valid = valid_mask[:-1] & valid_mask[1:]              # both endpoints valid
    duration_valid = float(rate_valid.sum() * dt)
    if duration_valid <= 0:
        return None

    rate_use = np.where(np.abs(rate) > threshold, np.sign(rate), 0.0)
    # Sign transitions between consecutive samples where BOTH are nonzero AND valid
    nonzero_now = rate_use != 0
    nonzero_prev = np.concatenate([[False], nonzero_now[:-1]])
    sign_change = np.concatenate([[False], np.sign(rate_use[:-1]) != np.sign(rate_use[1:])])
    # Transition between sample i-1 and i requires both rates valid (no mask gap)
    transition_valid = np.concatenate([[False], rate_valid[:-1] & rate_valid[1:]])
    crossings = int(np.sum(sign_change & nonzero_prev & nonzero_now & transition_valid))

    cps = float(crossings / duration_valid) if duration_valid > 0 else 0.0
    return {
        'crossings_per_sec': cps,
        'zero_crossings': crossings,
        'duration_valid_s': duration_valid,
        'duration_total_s': float(len(t_grid) * dt),
        'mask_drop_frac': float(1 - rate_valid.mean()),
        'std_signal': float(np.std(sig_grid[valid_mask])) if valid_mask.any() else 0.0,
    }


def build_grid_and_compare(t_model, model_curv, t_cx1, cx1_signal, t_lo, t_hi):
    """Build a 100Hz uniform grid for [t_lo, t_hi] and return per-signal masked metrics."""
    dt = 1.0 / GRID_RATE_HZ
    n = int(np.ceil((t_hi - t_lo) / dt)) + 1
    t_grid = t_lo + np.arange(n) * dt

    # Model interpolation: model is 20 Hz native — mask grid points >25 ms from nearest model sample
    if len(t_model) == 0:
        return None
    mmask_window = (t_model >= t_lo - 0.5) & (t_model <= t_hi + 0.5)
    if mmask_window.sum() < 3:
        return None
    tm = t_model[mmask_window]
    cm = model_curv[mmask_window]
    model_grid = np.interp(t_grid, tm, cm)
    model_gap = np.array([np.min(np.abs(tm - g)) for g in t_grid])
    model_valid = model_gap < 0.075  # 75 ms — 1.5x the native 50 ms interval

    # CX1 interpolation: undersampled — mask grid points >50ms from nearest CX1 sample
    cmask_window = (t_cx1 >= t_lo - 0.5) & (t_cx1 <= t_hi + 0.5)
    if cmask_window.sum() < 3:
        return None
    tc = t_cx1[cmask_window]
    sc = cx1_signal[cmask_window]
    cx1_grid = np.interp(t_grid, tc, sc)
    cx1_gap = np.array([np.min(np.abs(tc - g)) for g in t_grid])
    cx1_valid = cx1_gap < MAX_CX1_GAP_S

    # Joint mask: only compare where BOTH signals have nearby native samples
    joint_valid = model_valid & cx1_valid

    return {
        'model': crossings_masked(t_grid, model_grid, joint_valid),
        'cx1': crossings_masked(t_grid, cx1_grid, joint_valid),
        'model_only': crossings_masked(t_grid, model_grid, model_valid),  # diagnostic
        'mask_drop_frac': float(1 - joint_valid.mean()),
        'cx1_samples_in_window': int(cmask_window.sum()),
        'model_samples_in_window': int(mmask_window.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route_prefix')
    ap.add_argument('--mag', default='moderate', choices=['sharp', 'moderate', 'gentle'])
    args = ap.parse_args()

    print(f'[1/3] Loading CX1...', flush=True)
    arr, t_cx1 = load_cx1(args.route_prefix)
    print(f'  {len(arr)} CX1 rows, {t_cx1[-1] - t_cx1[0]:.1f}s '
          f'(avg {len(arr) / (t_cx1[-1] - t_cx1[0]):.1f} Hz)')

    print(f'[2/3] Pulling modelV2.action.desiredCurvature...', flush=True)
    t_model, model_curv = pull_model_curvature(args.route_prefix)
    if len(t_model) == 0:
        print('  No modelV2 data. Exiting.')
        return
    print(f'  {len(t_model)} modelV2 messages '
          f'(avg {len(t_model) / (t_model[-1] - t_model[0]):.1f} Hz)')

    events = detect_curve_events(arr, t_cx1)
    events = [e for e in events if e['magnitude_class'] == args.mag]
    print(f'[3/3] {len(events)} {args.mag} events from CX1\n')

    cmd_cx1 = arr[:, IDX['cmd']]
    des_cx1 = arr[:, IDX['des']]

    print(f'{"="*100}')
    print(f'MODEL vs DES vs CMD — uniform 100Hz grid, CX1-undersampled regions masked '
          f'(gap > {MAX_CX1_GAP_S*1000:.0f}ms)')
    print(f'{"="*100}')
    print(f'  {"Idx":>4} {"v_mph":>6} {"dur_s":>6} {"model_cps":>10} {"des_cps":>9} {"cmd_cps":>9} '
          f'{"m→d":>7} {"d→c":>7} {"mask%":>7}')
    print('  ' + '-' * 95)

    model_des_deltas = []
    des_cmd_deltas = []
    model_cps_list = []
    des_cps_list = []
    cmd_cps_list = []
    mask_drop_list = []

    for i, ev in enumerate(events):
        t_lo = ev['t_apex'] - 3.0
        t_hi = ev['t_apex'] + 3.0

        des_res = build_grid_and_compare(t_model, model_curv, t_cx1, des_cx1, t_lo, t_hi)
        cmd_res = build_grid_and_compare(t_model, model_curv, t_cx1, cmd_cx1, t_lo, t_hi)
        if not (des_res and cmd_res and des_res['model'] and des_res['cx1'] and cmd_res['cx1']):
            continue

        m_cps = des_res['model']['crossings_per_sec']
        d_cps = des_res['cx1']['crossings_per_sec']
        c_cps = cmd_res['cx1']['crossings_per_sec']
        md = d_cps - m_cps
        dc = c_cps - d_cps
        mask = des_res['mask_drop_frac']

        model_des_deltas.append(md)
        des_cmd_deltas.append(dc)
        model_cps_list.append(m_cps)
        des_cps_list.append(d_cps)
        cmd_cps_list.append(c_cps)
        mask_drop_list.append(mask)

        print(f'  {i:>4} {ev["mean_v_mph"]:>6.1f} {ev["duration_sec"]:>6.1f} '
              f'{m_cps:>10.3f} {d_cps:>9.3f} {c_cps:>9.3f} '
              f'{md:>+7.3f} {dc:>+7.3f} {mask*100:>6.1f}%')

    if model_des_deltas:
        print(f'\n  SUMMARY across {len(model_des_deltas)} events (masked):')
        print(f'    model_cps:          median {np.median(model_cps_list):.3f}, '
              f'IQR [{np.percentile(model_cps_list, 25):.3f}, {np.percentile(model_cps_list, 75):.3f}]')
        print(f'    des_cps:            median {np.median(des_cps_list):.3f}, '
              f'IQR [{np.percentile(des_cps_list, 25):.3f}, {np.percentile(des_cps_list, 75):.3f}]')
        print(f'    cmd_cps:            median {np.median(cmd_cps_list):.3f}, '
              f'IQR [{np.percentile(cmd_cps_list, 25):.3f}, {np.percentile(cmd_cps_list, 75):.3f}]')
        print(f'    model→des delta:    median {np.median(model_des_deltas):+.3f}, '
              f'IQR [{np.percentile(model_des_deltas, 25):+.3f}, {np.percentile(model_des_deltas, 75):+.3f}]')
        print(f'    des→cmd delta:      median {np.median(des_cmd_deltas):+.3f}, '
              f'IQR [{np.percentile(des_cmd_deltas, 25):+.3f}, {np.percentile(des_cmd_deltas, 75):+.3f}]')
        print(f'    mask drop frac:     median {np.median(mask_drop_list)*100:.1f}%, '
              f'p90 {np.percentile(mask_drop_list, 90)*100:.1f}%')
        print(f'\n  INTERPRETATION:')
        if abs(np.median(model_des_deltas)) < 0.1:
            print(f'    model_cps ≈ des_cps  → clip_curvature/controlsd does not change crossings.')
            print(f'                            The model output reaches actuators.curvature unchanged.')
        else:
            print(f'    model_cps != des_cps → controlsd path IS contributing (median Δ {np.median(model_des_deltas):+.3f}).')
        if abs(np.median(des_cmd_deltas)) > 0.1:
            print(f'    des_cps != cmd_cps   → carcontroller is reducing crossings by '
                  f'{abs(np.median(des_cmd_deltas)):.2f} cps median.')
        else:
            print(f'    des_cps ≈ cmd_cps    → carcontroller pipeline is not measurably reducing crossings either.')


if __name__ == '__main__':
    main()
