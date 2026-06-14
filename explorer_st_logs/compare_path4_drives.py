#!/usr/bin/env python3
"""Path 4 round-trip comparison: OUT (Path 4 OFF) vs BACK (Path 4 ON).

Same physical route in reverse — left curves become right and vice versa.
Path 4 is symmetric (acts on |cmd| shrinking), so it should help both
directions equally. Direction-segmented analysis isolates EPAS bias
interaction from the Path 4 effect itself.

Usage:
  compare_path4_drives.py <route_OUT_dir> <route_BACK_dir>
"""
import sys, os, json, glob
sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
from openpilot.tools.lib.logreader import LogReader
import numpy as np
import warnings
warnings.filterwarnings('ignore')

CX1_FIELDS = ['frame', 'v', 'yr', 'aLat', 'cmd', 'rate', 'meas', 'des', 'pred', 'ema',
              'preRL', 'rl', 'cmdInt', 'rateInt', 'ang', 'dAng', 'tq', 'ovr', 'lc',
              'lookT', 'blend', 'cFac', 'lOff', 'lInt', 'pmd', 'burst',
              'p4Rel', 'p4Tau', 'p4On']
IDX = {f: i for i, f in enumerate(CX1_FIELDS)}


def load_cx1(route_dir):
    """Load all CX1 rows from a route directory."""
    files = sorted(glob.glob(os.path.join(route_dir, '*--*/rlog.zst')))
    if not files:
        files = sorted(glob.glob(os.path.join(route_dir, 'rlog_*.zst')))
    rows = []
    for fp in files:
        try:
            for msg in LogReader(fp):
                if msg.which() != 'logMessage':
                    continue
                try:
                    body = json.loads(msg.logMessage).get('msg', '')
                    if not body.startswith('CX1: ') or body[5:].startswith('SCHEMA'):
                        continue
                    parts = body[5:].split()
                    if len(parts) < 29:
                        continue
                    rows.append([float(p) for p in parts[:29]])
                except Exception:
                    pass
        except Exception:
            pass
    if not rows:
        return None
    return np.array(rows)


def summarize(arr, label):
    """Compute per-drive metrics."""
    if arr is None or len(arr) == 0:
        return None
    v       = arr[:, IDX['v']]
    cmd     = arr[:, IDX['cmd']]
    meas    = arr[:, IDX['meas']]
    p4Rel   = arr[:, IDX['p4Rel']]
    p4On    = arr[:, IDX['p4On']]
    p4Tau   = arr[:, IDX['p4Tau']]
    ovr     = arr[:, IDX['ovr']]

    engaged = (v >= 4.0) & (ovr < 0.5)
    curve   = engaged & (np.abs(cmd) > 0.001)
    left_curve  = engaged & (cmd < -0.001)  # in carcontroller frame, negative = left
    right_curve = engaged & (cmd >  0.001)

    # Overshoot per analyzer's definition: (meas-cmd)*sign(cmd) > 0
    sub_cmd = cmd[curve]
    sub_meas = meas[curve]
    sub_p4rel = p4Rel[curve]
    os_signed = (sub_meas - sub_cmd) * np.sign(sub_cmd)
    os_mask = os_signed > 0  # any positive same-direction overshoot
    os_vals = os_signed[os_mask]

    # By direction
    def dir_stats(dir_mask):
        sub_c = cmd[dir_mask]
        sub_m = meas[dir_mask]
        sub_p = p4Rel[dir_mask]
        if len(sub_c) < 10:
            return None
        os = (sub_m - sub_c) * np.sign(sub_c)
        os_pos = os[os > 0]
        return {
            'n_curve_frames': int(dir_mask.sum()),
            'n_overshoot': int((os > 0).sum()),
            'mean_os': float(np.mean(os_pos)) if len(os_pos) else 0.0,
            'p95_os':  float(np.percentile(os_pos, 95)) if len(os_pos) else 0.0,
            'p99_os':  float(np.percentile(os_pos, 99)) if len(os_pos) else 0.0,
            'trip_rate': float(sub_p.mean()),
            'coincidence': float(sub_p[os > 0].mean()) if (os > 0).any() else 0.0,
        }

    # By curve magnitude bin
    def bin_stats(mag_lo, mag_hi):
        m = engaged & (np.abs(cmd) > mag_lo) & (np.abs(cmd) <= mag_hi)
        if m.sum() < 5:
            return None
        sub_c = cmd[m]; sub_m = meas[m]; sub_p = p4Rel[m]
        os = (sub_m - sub_c) * np.sign(sub_c)
        os_pos = os[os > 0]
        return {
            'n': int(m.sum()),
            'n_os': int((os > 0).sum()),
            'mean_os': float(np.mean(os_pos)) if len(os_pos) else 0.0,
            'p95_os':  float(np.percentile(os_pos, 95)) if len(os_pos) else 0.0,
            'trip_rate': float(sub_p.mean()),
        }

    return {
        'label': label,
        'total_rows': int(len(arr)),
        'engaged_frames': int(engaged.sum()),
        'curve_frames': int(curve.sum()),
        'p4on_unique': sorted(set(p4On.tolist())),
        'p4on_majority': float(p4On.mean()),
        'p4Tau_mean': float(p4Tau[curve].mean()) if curve.any() else 0.0,
        'overall_trip_rate': float(p4Rel[curve].mean()) if curve.any() else 0.0,
        'aggregate': {
            'n_overshoot': int(os_mask.sum()),
            'mean_os': float(os_vals.mean()) if len(os_vals) else 0.0,
            'p95_os':  float(np.percentile(os_vals, 95)) if len(os_vals) else 0.0,
            'p99_os':  float(np.percentile(os_vals, 99)) if len(os_vals) else 0.0,
            'coincidence_with_p4rel': float(sub_p4rel[os_mask].mean()) if os_mask.any() else 0.0,
        },
        'left': dir_stats(left_curve),
        'right': dir_stats(right_curve),
        'gentle':   bin_stats(0.001, 0.002),
        'moderate': bin_stats(0.002, 0.004),
        'sharp':    bin_stats(0.004, 100.0),
    }


def pct(old, new):
    if old == 0:
        return float('nan')
    return (new - old) / abs(old) * 100.0


def main():
    if len(sys.argv) < 3:
        print("Usage: compare_path4_drives.py <route_OUT_dir> <route_BACK_dir>")
        sys.exit(1)

    out_dir, back_dir = sys.argv[1], sys.argv[2]
    print(f'Loading OUT: {out_dir}', flush=True)
    out_arr = load_cx1(out_dir)
    print(f'  {len(out_arr) if out_arr is not None else 0} CX1 rows')
    print(f'Loading BACK: {back_dir}', flush=True)
    back_arr = load_cx1(back_dir)
    print(f'  {len(back_arr) if back_arr is not None else 0} CX1 rows')

    out_s = summarize(out_arr, 'OUT')
    back_s = summarize(back_arr, 'BACK')

    if not out_s or not back_s:
        print("ERROR: could not summarize one or both drives")
        sys.exit(1)

    print('\n' + '=' * 96)
    print(f'PATH 4 ROUND-TRIP COMPARISON')
    print('=' * 96)

    print(f'\n## Drive overview\n')
    print(f'  {"Field":<28} {"OUT":>18} {"BACK":>18}')
    print('  ' + '-' * 70)
    print(f'  {"Total CX1 rows":<28} {out_s["total_rows"]:>18} {back_s["total_rows"]:>18}')
    print(f'  {"Engaged frames":<28} {out_s["engaged_frames"]:>18} {back_s["engaged_frames"]:>18}')
    print(f'  {"Curve frames (|cmd|>0.001)":<28} {out_s["curve_frames"]:>18} {back_s["curve_frames"]:>18}')
    print(f'  {"p4On values":<28} {str(out_s["p4on_unique"]):>18} {str(back_s["p4on_unique"]):>18}')
    print(f'  {"p4On majority":<28} {out_s["p4on_majority"]:>18.2f} {back_s["p4on_majority"]:>18.2f}')
    print(f'  {"Detector trip % (curves)":<28} {100*out_s["overall_trip_rate"]:>17.1f}% {100*back_s["overall_trip_rate"]:>17.1f}%')
    print(f'  {"Mean p4Tau in curves":<28} {out_s["p4Tau_mean"]:>18.4f} {back_s["p4Tau_mean"]:>18.4f}')

    # Sanity check on toggle
    if out_s['p4on_majority'] > 0.1:
        print(f'\n  WARN: OUT drive p4On is {out_s["p4on_majority"]:.2f} — expected 0. Path 4 may have been ON outbound?')
    if back_s['p4on_majority'] < 0.9:
        print(f'\n  WARN: BACK drive p4On is {back_s["p4on_majority"]:.2f} — expected 1. Toggle may not have flipped?')

    print(f'\n## Aggregate overshoot (all curves)\n')
    o = out_s['aggregate']; b = back_s['aggregate']
    print(f'  {"Metric":<32} {"OUT":>14} {"BACK":>14} {"Δ %":>10}')
    print('  ' + '-' * 75)
    print(f'  {"N overshoot frames":<32} {o["n_overshoot"]:>14} {b["n_overshoot"]:>14}')
    print(f'  {"Mean OS magnitude (1/m)":<32} {o["mean_os"]:>14.5f} {b["mean_os"]:>14.5f} {pct(o["mean_os"], b["mean_os"]):>9.1f}%')
    print(f'  {"P95 OS magnitude (1/m)":<32} {o["p95_os"]:>14.5f} {b["p95_os"]:>14.5f} {pct(o["p95_os"], b["p95_os"]):>9.1f}%')
    print(f'  {"P99 OS magnitude (1/m)":<32} {o["p99_os"]:>14.5f} {b["p99_os"]:>14.5f} {pct(o["p99_os"], b["p99_os"]):>9.1f}%')
    print(f'  {"Detector coincidence on OS":<32} {100*o["coincidence_with_p4rel"]:>13.1f}% {100*b["coincidence_with_p4rel"]:>13.1f}%')

    print(f'\n## Per-direction (left/right curves)\n')
    print(f'  {"Direction":<10} {"Drive":<6} {"N curve":>9} {"N OS":>8} {"Mean OS":>10} {"P95 OS":>10} {"Trip %":>8} {"Coinc %":>9}')
    print('  ' + '-' * 80)
    for direction in ['left', 'right']:
        for drive_lbl, drive_s in [('OUT', out_s), ('BACK', back_s)]:
            d = drive_s[direction]
            if d is None:
                continue
            print(f'  {direction:<10} {drive_lbl:<6} {d["n_curve_frames"]:>9} {d["n_overshoot"]:>8} '
                  f'{d["mean_os"]:>10.5f} {d["p95_os"]:>10.5f} {100*d["trip_rate"]:>7.1f}% {100*d["coincidence"]:>8.1f}%')

    # Mirror comparisons (same physical curves)
    print(f'\n## Mirror-paired comparisons (same physical curves)\n')
    print(f'  OUT left curves vs BACK right curves should both be the same physical road curves')
    print(f'  Path 4 (symmetric design) should improve both pairs equally')
    print()
    pairs = [
        ('OUT left', out_s.get('left'), 'BACK right', back_s.get('right')),
        ('OUT right', out_s.get('right'), 'BACK left', back_s.get('left')),
    ]
    for label_a, a, label_b, b in pairs:
        if a is None or b is None:
            continue
        delta = pct(a['mean_os'], b['mean_os'])
        print(f'  {label_a} vs {label_b}: mean_os {a["mean_os"]:.5f} → {b["mean_os"]:.5f}  ({delta:+.1f}%)')

    print(f'\n## By curve magnitude bin\n')
    print(f'  {"Bin":<10} {"Drive":<6} {"N":>6} {"N OS":>6} {"Mean OS":>10} {"P95 OS":>10} {"Trip %":>8}')
    print('  ' + '-' * 65)
    for bin_lbl in ['gentle', 'moderate', 'sharp']:
        for drive_lbl, drive_s in [('OUT', out_s), ('BACK', back_s)]:
            d = drive_s[bin_lbl]
            if d is None:
                continue
            print(f'  {bin_lbl:<10} {drive_lbl:<6} {d["n"]:>6} {d["n_os"]:>6} '
                  f'{d["mean_os"]:>10.5f} {d["p95_os"]:>10.5f} {100*d["trip_rate"]:>7.1f}%')
        # Per-bin delta
        d_o = out_s[bin_lbl]; d_b = back_s[bin_lbl]
        if d_o and d_b:
            delta = pct(d_o['mean_os'], d_b['mean_os'])
            print(f'  {bin_lbl:<10} {"DELTA":<6} {"":>6} {"":>6} {"":>10} {delta:>+9.1f}%')

    # Per-trip reduction estimate
    print(f'\n## Per-trip reduction estimate\n')
    print(f'  Hypothesis: drive_wide_change ≈ coincidence × per_trip_reduction')
    print(f'  BACK coincidence (p4Rel during OS): {100*b["coincidence_with_p4rel"]:.1f}%')
    drive_wide_change = pct(o['mean_os'], b['mean_os'])
    if b['coincidence_with_p4rel'] > 0:
        per_trip_implied = drive_wide_change / b['coincidence_with_p4rel']
        print(f'  Drive-wide OS change (mean): {drive_wide_change:+.1f}%')
        print(f'  Implied per-trip reduction: {per_trip_implied:+.1f}%')

    # Verdict
    print(f'\n' + '=' * 96)
    print(f'VERDICT')
    print('=' * 96)
    drive_wide = pct(o['mean_os'], b['mean_os'])
    p95_change = pct(o['p95_os'], b['p95_os'])

    print(f'  Mean OS change (drive-wide): {drive_wide:+.1f}%  (target: ≤-15% for gate)')
    print(f'  P95 OS change: {p95_change:+.1f}%  (informational)')

    if drive_wide <= -15.0:
        print(f'  ✓ PASS — Path 4 reduces drive-wide mean overshoot ≥15%')
    elif drive_wide <= -5.0:
        print(f'  ◐ MARGINAL — Path 4 helps but below 15% gate. Consider tuning detector/tau.')
    elif drive_wide >= 5.0:
        print(f'  ✗ FAIL — Path 4 made overshoot WORSE. Disable and investigate.')
    else:
        print(f'  ◯ INCONCLUSIVE — change within noise floor (±5%). Need more samples.')

    # Direction symmetry check
    out_l = out_s.get('left'); out_r = out_s.get('right')
    back_l = back_s.get('left'); back_r = back_s.get('right')
    if all([out_l, out_r, back_l, back_r]):
        left_delta = pct(out_l['mean_os'], back_l['mean_os'])
        right_delta = pct(out_r['mean_os'], back_r['mean_os'])
        sym_gap = abs(left_delta - right_delta)
        print(f'\n  Direction symmetry: LEFT changed {left_delta:+.1f}%, RIGHT changed {right_delta:+.1f}% (gap {sym_gap:.1f}pp)')
        if sym_gap > 20:
            print(f'  WARN: large asymmetry — Path 4 effect direction-dependent. Investigate EPAS bias interaction.')


if __name__ == '__main__':
    main()
