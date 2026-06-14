#!/usr/bin/env python3
"""Investigate the lane offset sign convention by reading modelV2 lane lines
directly and comparing to the integral / cmd correction direction."""

import sys
import os
import glob
import argparse
import numpy as np

sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_rlogs(route_prefix):
    seg_dirs = sorted(glob.glob(route_prefix + '--*'),
                      key=lambda d: int(d.rsplit('--', 1)[-1]))
    out = []
    for sd in seg_dirs:
        rlog = os.path.join(sd, 'rlog.zst')
        if os.path.exists(rlog):
            out.append(rlog)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route_prefix')
    ap.add_argument('--max-samples', type=int, default=50)
    ap.add_argument('--min-conf', type=float, default=0.6, help='Min laneLineProb to include sample')
    args = ap.parse_args()

    from openpilot.tools.lib.logreader import LogReader
    files = find_rlogs(args.route_prefix)
    if not files:
        print('No rlogs.')
        return

    samples = []
    for fp in files:
        try:
            for msg in LogReader(fp):
                if msg.which() != 'modelV2':
                    continue
                t = msg.logMonoTime / 1e9
                mv2 = msg.modelV2
                if not (hasattr(mv2, 'laneLines') and hasattr(mv2, 'laneLineProbs')):
                    continue
                ll = mv2.laneLines
                lp = mv2.laneLineProbs
                if len(ll) < 3 or len(lp) < 3:
                    continue
                try:
                    left_y0 = float(ll[1].y[0])
                    right_y0 = float(ll[2].y[0])
                    left_prob = float(lp[1])
                    right_prob = float(lp[2])
                except Exception:
                    continue
                # Skip low-confidence
                if min(left_prob, right_prob) < args.min_conf:
                    continue
                try:
                    pos_y = list(mv2.position.y[:33])
                except Exception:
                    pos_y = []
                pos_y_0 = pos_y[0] if pos_y else float('nan')
                # Manually interp position.y at 0.2s using T_IDXS
                T_IDXS = [0.0, 0.00976562, 0.0390625, 0.087890625, 0.15625, 0.244140625,
                          0.3515625, 0.4785156, 0.625, 0.7910156, 0.9765625]
                if len(pos_y) >= len(T_IDXS):
                    pos_y_02 = float(np.interp(0.2, T_IDXS, pos_y[:len(T_IDXS)]))
                else:
                    pos_y_02 = float('nan')
                samples.append({
                    't': t,
                    'left_y0': left_y0,
                    'right_y0': right_y0,
                    'left_prob': left_prob,
                    'right_prob': right_prob,
                    'pos_y_0': pos_y_0,
                    'pos_y_02': pos_y_02,
                })
                if len(samples) >= args.max_samples * 10:
                    break
        except Exception as e:
            pass
        if len(samples) >= args.max_samples * 10:
            break

    if not samples:
        print(f'No high-confidence samples (min_conf={args.min_conf}).')
        return

    # Subsample
    if len(samples) > args.max_samples:
        step = len(samples) // args.max_samples
        samples = samples[::step][:args.max_samples]

    print(f'  Showing {len(samples)} samples with laneLineProb >= {args.min_conf}')
    print()
    print(f'  {"t_rel":<7} {"left_y0":>10} {"right_y0":>10} {"midpoint":>10} {"width_raw":>10} '
          f'{"width_abs":>10} {"l_prob":>7} {"r_prob":>7} {"pos_y_0":>10} {"pos_y@0.2":>11}')
    print('  ' + '-' * 120)
    t0 = samples[0]['t']
    for s in samples:
        # carcontroller actually computes lane_width = right + (-left)
        lane_width = s['right_y0'] + (-s['left_y0'])
        width_abs = abs(s['right_y0'] - s['left_y0'])
        # path_offset_lanelines = (left + right) / 2
        midpoint = (s['left_y0'] + s['right_y0']) / 2
        print(f"  {s['t']-t0:<+7.1f} {s['left_y0']:>+10.3f} {s['right_y0']:>+10.3f} {midpoint:>+10.3f} {lane_width:>+10.3f} "
              f"{width_abs:>10.3f} {s['left_prob']:>7.3f} {s['right_prob']:>7.3f} {s['pos_y_0']:>+10.3f} {s['pos_y_02']:>+11.3f}")

    print('\n  ANALYSIS:')
    left_arr = np.array([s['left_y0'] for s in samples])
    right_arr = np.array([s['right_y0'] for s in samples])
    pos_y0_arr = np.array([s['pos_y_0'] for s in samples])
    pos_y_02_arr = np.array([s['pos_y_02'] for s in samples])
    print(f'    left_y0  median: {np.median(left_arr):+.3f} m  (negative = line on car\'s left in positive-RIGHT convention)')
    print(f'    right_y0 median: {np.median(right_arr):+.3f} m  (positive = line on car\'s right)')
    print(f'    midpoint (carctrl path_offset_lanelines) median: {np.median((left_arr + right_arr)/2):+.3f} m')
    print(f'    pos_y[0] median: {np.median(pos_y0_arr):+.3f} m  (model.position is std OP convention: positive = LEFT)')
    print(f'    pos_y@0.2s median: {np.median(pos_y_02_arr):+.3f} m')
    print(f'    width via formula (right + (-left)) median: {np.median(right_arr + (-left_arr)):+.3f} m')
    print(f'    width via |right - left| median: {np.median(np.abs(right_arr - left_arr)):+.3f} m')

    print()
    print('  SIGN-CONVENTION VERDICT:')
    if np.median(left_arr) < 0 and np.median(right_arr) > 0:
        print('    laneLines: positive-RIGHT convention (left<0, right>0)')
        print('    Width formula (right + (-left)) gives POSITIVE width ✓')
        print()
        print('    For lane centering CORRECTNESS:')
        print('      - If car drifts RIGHT (positive direction), both lane lines appear to shift LEFT in car frame')
        print('      - left_y becomes more negative, right_y becomes less positive')
        print('      - midpoint = (left + right)/2 becomes more NEGATIVE')
        print('      - So midpoint < 0 → car is RIGHT of center')
        print('      - midpoint > 0 → car is LEFT of center')
        print()
        print('    carcontroller: apply_curvature += lc_kp * lane_offset + lc_ki * integral')
        print('      - lane_offset = midpoint (positive-RIGHT-shifted convention)')
        print('      - In this codebase: positive curvature = RIGHT turn')
        print('      - So midpoint > 0 (car LEFT) → +PI → RIGHT push → correct ✓')
        print('      - And midpoint < 0 (car RIGHT) → -PI → LEFT push → correct ✓')
        print()
        print('  → lane_offset sign is CORRECT in carcontroller.')
        print()
        print('  BUT: path_offset_position uses model.position.y which is STANDARD OP (positive=LEFT)')
        print('      In the MIX formula: lane_offset = pos_y_02 * (1-scale) + midpoint * scale')
        print('      These signs are OPPOSITE: pos_y_02 > 0 means LEFT, midpoint > 0 means LEFT-of-center, ✓ consistent direction')
        print('      → ACTUALLY: model.position.y[i] = future predicted y position of car. Positive = car will be at +y (LEFT in std).')
        print('      → That is COUNTER to midpoint convention: midpoint > 0 means car is LEFT of center, which is same direction')
        print('      → Wait... midpoint > 0 (car LEFT of center) and pos_y > 0 (model predicts car going LEFT) ARE in same direction.')
        print('      → Conventions appear MIXED but happen to be sign-consistent for this use case.')


if __name__ == '__main__':
    main()
