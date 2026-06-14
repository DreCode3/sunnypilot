#!/usr/bin/env python3
"""Analyze Explorer ST drive logs for curve oversteer and low-speed hunting."""

import sys
import os
import glob
import numpy as np
from collections import defaultdict

# Add sunnypilot root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('PYTHONPATH', '.')

from openpilot.tools.lib.logreader import LogReader


def load_route(route_dir):
  """Load all segments from a route directory, return sorted log messages."""
  files = sorted(glob.glob(os.path.join(route_dir, 'rlog_seg*.zst')),
                 key=lambda f: int(f.split('seg')[1].split('.')[0]))
  print(f"Loading {len(files)} segments from {route_dir}...")

  all_msgs = []
  for i, f in enumerate(files):
    try:
      lr = LogReader(f)
      msgs = list(lr)
      all_msgs.extend(msgs)
      if (i + 1) % 10 == 0:
        print(f"  loaded {i+1}/{len(files)} segments...")
    except Exception as e:
      print(f"  skip seg {i}: {e}")

  print(f"  Total messages: {len(all_msgs)}")
  return all_msgs


def extract_data(msgs):
  """Extract relevant time series from log messages."""
  data = defaultdict(list)

  for msg in msgs:
    t = msg.logMonoTime / 1e9  # seconds

    if msg.which() == 'carState':
      cs = msg.carState
      data['cs_t'].append(t)
      data['v_ego'].append(cs.vEgo)
      data['steering_angle'].append(cs.steeringAngleDeg)
      data['steering_torque'].append(cs.steeringTorque)
      data['steering_pressed'].append(cs.steeringPressed)
      data['yaw_rate'].append(cs.yawRate)

    elif msg.which() == 'carControl':
      cc = msg.carControl
      data['cc_t'].append(t)
      data['lat_active'].append(cc.latActive)
      data['actuator_curvature'].append(cc.actuators.curvature)

    elif msg.which() == 'controlsState':
      cs = msg.controlsState
      data['ctrl_t'].append(t)
      data['desired_curvature'].append(cs.desiredCurvature)
      data['curvature'].append(cs.curvature)

    elif msg.which() == 'selfdriveState':
      ss = msg.selfdriveState
      data['ss_t'].append(t)
      data['enabled'].append(ss.enabled)
      data['active'].append(ss.active)

    elif msg.which() == 'carOutput':
      co = msg.carOutput
      data['co_t'].append(t)
      data['co_actuator_curvature'].append(co.actuatorsOutput.curvature)

    elif msg.which() == 'lateralPlan' or msg.which() == 'lateralPlanDEPRECATED':
      try:
        lp = getattr(msg, msg.which())
        data['lp_t'].append(t)
        data['lp_curvature'].append(lp.curvature)
        data['lp_curvature_rate'].append(lp.curvatureRate)
      except Exception:
        pass

  # Convert to numpy
  for k in data:
    data[k] = np.array(data[k])

  return data


def analyze_curves(data):
  """Analyze sharp curve behavior — look for oversteer."""
  print("\n" + "="*80)
  print("CURVE ANALYSIS — Sharp Curve Oversteer Detection")
  print("="*80)

  if len(data['cs_t']) == 0 or len(data['co_t']) == 0:
    print("No data available")
    return

  # Use carState timestamps as primary
  t = data['cs_t']
  v = data['v_ego']
  yaw = data['yaw_rate']
  angle = data['steering_angle']

  # Compute measured curvature from yaw rate
  measured_curv = np.where(v > 1.0, -yaw / np.maximum(v, 0.1), 0.0)

  # Interpolate actuator curvature to carState timestamps
  if len(data['co_t']) > 1:
    applied_curv = np.interp(t, data['co_t'], data['co_actuator_curvature'])
  elif len(data['cc_t']) > 1:
    applied_curv = np.interp(t, data['cc_t'], data['actuator_curvature'])
  else:
    print("No actuator data")
    return

  # Interpolate desired curvature
  if len(data['ctrl_t']) > 1:
    desired_curv = np.interp(t, data['ctrl_t'], data['desired_curvature'])
  else:
    desired_curv = np.zeros_like(t)

  # Find curve segments: |curvature| > 0.002 (meaningful curves) at speed > 10 m/s
  in_curve = (np.abs(measured_curv) > 0.002) & (v > 10.0)

  # Find curve entry/exit transitions
  curve_diff = np.diff(in_curve.astype(int))
  curve_starts = np.where(curve_diff == 1)[0]
  curve_ends = np.where(curve_diff == -1)[0]

  if len(curve_starts) == 0:
    print("No significant curves detected at speed")
    return

  # Pair starts with ends
  if len(curve_ends) > 0 and curve_ends[0] < curve_starts[0]:
    curve_ends = curve_ends[1:]
  if len(curve_starts) > len(curve_ends):
    curve_starts = curve_starts[:len(curve_ends)]

  print(f"\nFound {len(curve_starts)} curve segments at speed > 10 m/s (22 mph)")

  # Classify curves by sharpness
  sharp_curves = []  # |curvature| > 0.004
  moderate_curves = []  # 0.002 - 0.004

  for start, end in zip(curve_starts, curve_ends):
    if end - start < 10:  # skip very short segments
      continue

    seg_t = t[start:end]
    seg_v = v[start:end]
    seg_meas = measured_curv[start:end]
    seg_app = applied_curv[start:end]
    seg_des = desired_curv[start:end]

    peak_curv = np.max(np.abs(seg_meas))
    mean_v = np.mean(seg_v)
    duration = seg_t[-1] - seg_t[0]

    # Oversteer: measured curvature significantly exceeds applied curvature (same sign)
    # Check if the car turned MORE than we asked
    sign = np.sign(np.mean(seg_meas))
    signed_meas = seg_meas * sign
    signed_app = seg_app * sign

    overshoot = signed_meas - signed_app
    max_overshoot = np.max(overshoot)
    mean_overshoot = np.mean(overshoot[overshoot > 0]) if np.any(overshoot > 0) else 0

    # Lateral acceleration
    lat_accel = seg_meas * (seg_v ** 2)
    peak_lat_accel = np.max(np.abs(lat_accel))

    curve_info = {
      'start_t': seg_t[0],
      'duration': duration,
      'peak_curv': peak_curv,
      'mean_speed_mph': mean_v * 2.237,
      'peak_lat_accel': peak_lat_accel,
      'max_overshoot': max_overshoot,
      'mean_overshoot': mean_overshoot,
      'peak_applied': np.max(np.abs(seg_app)),
      'peak_desired': np.max(np.abs(seg_des)),
    }

    if peak_curv > 0.004:
      sharp_curves.append(curve_info)
    else:
      moderate_curves.append(curve_info)

  # Print sharp curve analysis
  print(f"\n--- SHARP CURVES (|curvature| > 0.004, ~R < 250m) ---")
  print(f"Count: {len(sharp_curves)}")
  if sharp_curves:
    print(f"{'Time':>8s} {'Dur':>5s} {'Speed':>7s} {'PkCurv':>8s} {'PkApp':>8s} {'PkDes':>8s} {'Overshoot':>10s} {'LatAccel':>9s}")
    for c in sorted(sharp_curves, key=lambda x: x['max_overshoot'], reverse=True):
      rel_t = c['start_t'] - t[0]
      print(f"{rel_t:>7.0f}s {c['duration']:>4.1f}s {c['mean_speed_mph']:>5.1f}mph "
            f"{c['peak_curv']:>8.5f} {c['peak_applied']:>8.5f} {c['peak_desired']:>8.5f} "
            f"{c['max_overshoot']:>+9.5f} {c['peak_lat_accel']:>8.2f}m/s²")

    overshooting = [c for c in sharp_curves if c['max_overshoot'] > 0.001]
    print(f"\n  Curves with significant overshoot (>0.001): {len(overshooting)}/{len(sharp_curves)}")
    if overshooting:
      avg_overshoot = np.mean([c['max_overshoot'] for c in overshooting])
      print(f"  Average peak overshoot: {avg_overshoot:.5f} (1/m)")
      print(f"  Worst overshoot: {max(c['max_overshoot'] for c in overshooting):.5f}")

  print(f"\n--- MODERATE CURVES (0.002 < |curvature| < 0.004) ---")
  print(f"Count: {len(moderate_curves)}")
  if moderate_curves:
    overshooting = [c for c in moderate_curves if c['max_overshoot'] > 0.0005]
    print(f"  Curves with overshoot (>0.0005): {len(overshooting)}/{len(moderate_curves)}")
    if overshooting:
      avg_overshoot = np.mean([c['max_overshoot'] for c in overshooting])
      print(f"  Average peak overshoot: {avg_overshoot:.5f}")

  return sharp_curves, moderate_curves


def analyze_hunting(data):
  """Analyze low-speed highway hunting (oscillation)."""
  print("\n" + "="*80)
  print("HUNTING ANALYSIS — Low-Speed Highway Oscillation")
  print("="*80)

  if len(data['cs_t']) == 0:
    print("No data available")
    return

  t = data['cs_t']
  v = data['v_ego']
  angle = data['steering_angle']
  yaw = data['yaw_rate']
  measured_curv = np.where(v > 1.0, -yaw / np.maximum(v, 0.1), 0.0)

  if len(data['co_t']) > 1:
    applied_curv = np.interp(t, data['co_t'], data['co_actuator_curvature'])
  elif len(data['cc_t']) > 1:
    applied_curv = np.interp(t, data['cc_t'], data['actuator_curvature'])
  else:
    print("No actuator data")
    return

  if len(data['ctrl_t']) > 1:
    desired_curv = np.interp(t, data['ctrl_t'], data['desired_curvature'])
  else:
    desired_curv = np.zeros_like(t)

  # Define speed bins
  speed_bins = [
    ("Low speed (15-30 mph / 7-13 m/s)", 7, 13),
    ("Medium speed (30-45 mph / 13-20 m/s)", 13, 20),
    ("Highway (45-65 mph / 20-29 m/s)", 20, 29),
    ("High highway (65-80 mph / 29-36 m/s)", 29, 36),
  ]

  print(f"\n--- STEERING OSCILLATION BY SPEED BIN ---")
  print(f"(Higher values = more hunting/oscillation)")
  print(f"{'Speed bin':>42s} {'N pts':>7s} {'StdAngle':>9s} {'StdCurv':>9s} {'StdApplied':>11s} {'DesVsApp':>9s} {'MeasVsApp':>10s}")

  for name, v_min, v_max in speed_bins:
    # Only straight-ish sections (low curvature = highway driving)
    mask = (v >= v_min) & (v < v_max) & (np.abs(measured_curv) < 0.003)
    n = np.sum(mask)
    if n < 100:
      print(f"{name:>42s} {n:>7d}    (insufficient data)")
      continue

    # Compute oscillation metrics over 2-second windows
    seg_angle = angle[mask]
    seg_curv = measured_curv[mask]
    seg_app = applied_curv[mask]
    seg_des = desired_curv[mask]
    seg_t = t[mask]

    # Standard deviation of steering angle (detrended)
    # Use rolling windows of ~2s (100 samples at 50Hz or 40 at 20Hz)
    window = min(100, n // 3)
    if window < 10:
      window = 10

    # Compute rolling std
    def rolling_std(arr, w):
      stds = []
      for i in range(0, len(arr) - w, w // 2):
        stds.append(np.std(arr[i:i+w]))
      return np.array(stds) if stds else np.array([0.0])

    angle_std = np.median(rolling_std(seg_angle, window))
    curv_std = np.median(rolling_std(seg_curv, window))
    app_std = np.median(rolling_std(seg_app, window))

    # Tracking error: desired vs applied
    des_vs_app = np.sqrt(np.mean((seg_des - seg_app) ** 2))
    # Measured vs applied error
    meas_vs_app = np.sqrt(np.mean((seg_curv - seg_app) ** 2))

    print(f"{name:>42s} {n:>7d} {angle_std:>8.3f}° {curv_std:>8.6f} {app_std:>10.6f} {des_vs_app:>8.6f} {meas_vs_app:>9.6f}")

  # Detailed low-speed hunting analysis
  print(f"\n--- LOW-SPEED HUNTING DEEP DIVE (7-13 m/s, straight sections) ---")
  mask = (v >= 7) & (v < 13) & (np.abs(measured_curv) < 0.003)
  if np.sum(mask) < 200:
    print("Insufficient low-speed straight driving data")
    return

  seg_t = t[mask]
  seg_angle = angle[mask]
  seg_curv = measured_curv[mask]
  seg_app = applied_curv[mask]
  seg_v = v[mask]

  # Find zero-crossings of applied curvature (sign changes = oscillation)
  sign_changes = np.sum(np.diff(np.sign(seg_app)) != 0)
  duration = seg_t[-1] - seg_t[0]

  # Frequency: zero crossings per second / 2 = oscillation frequency
  if duration > 0:
    osc_freq = sign_changes / duration / 2
  else:
    osc_freq = 0

  print(f"  Total time in low-speed straight: {duration:.1f}s")
  print(f"  Applied curvature sign changes: {sign_changes}")
  print(f"  Oscillation frequency: {osc_freq:.3f} Hz ({1/osc_freq:.1f}s period)" if osc_freq > 0 else "  No oscillation detected")
  print(f"  Steering angle range: {np.min(seg_angle):.2f}° to {np.max(seg_angle):.2f}°")
  print(f"  Steering angle std: {np.std(seg_angle):.3f}°")
  print(f"  Applied curvature range: {np.min(seg_app):.6f} to {np.max(seg_app):.6f}")
  print(f"  Applied curvature std: {np.std(seg_app):.6f}")

  # Check if curvature error clamping is happening at low speed
  # (v < 9 m/s bypasses CURVATURE_ERROR clamp, v > 9 applies it)
  mask_below9 = (seg_v < 9)
  mask_above9 = (seg_v >= 9)
  if np.sum(mask_below9) > 50 and np.sum(mask_above9) > 50:
    print(f"\n  Speed < 9 m/s (no curv error clamp):")
    print(f"    Steering std: {np.std(seg_angle[mask_below9]):.3f}°, Applied curv std: {np.std(seg_app[mask_below9]):.6f}")
    print(f"  Speed >= 9 m/s (curv error clamp active):")
    print(f"    Steering std: {np.std(seg_angle[mask_above9]):.3f}°, Applied curv std: {np.std(seg_app[mask_above9]):.6f}")


def analyze_overall_stats(data):
  """Print overall drive statistics."""
  print("\n" + "="*80)
  print("OVERALL DRIVE STATISTICS")
  print("="*80)

  if len(data['cs_t']) == 0:
    print("No data")
    return

  t = data['cs_t']
  v = data['v_ego']

  duration = t[-1] - t[0]
  print(f"  Duration: {duration/60:.1f} minutes")
  print(f"  Speed range: {np.min(v)*2.237:.1f} - {np.max(v)*2.237:.1f} mph")
  print(f"  Mean speed: {np.mean(v)*2.237:.1f} mph")

  if len(data.get('ss_t', [])) > 0:
    enabled_pct = np.mean(data['enabled']) * 100
    active_pct = np.mean(data['active']) * 100
    print(f"  OP enabled: {enabled_pct:.1f}%")
    print(f"  OP active: {active_pct:.1f}%")

  # Speed histogram
  print(f"\n  Speed distribution:")
  bins = [(0, 7, "0-15 mph"), (7, 13, "15-30 mph"), (13, 20, "30-45 mph"),
          (20, 29, "45-65 mph"), (29, 36, "65-80 mph"), (36, 50, "80+ mph")]
  for v_min, v_max, label in bins:
    pct = np.mean((v >= v_min) & (v < v_max)) * 100
    bar = "█" * int(pct / 2)
    print(f"    {label:>10s}: {pct:5.1f}% {bar}")


def main():
  if len(sys.argv) < 2:
    print("Usage: analyze_drive.py <route_dir> [route_dir2 ...]")
    sys.exit(1)

  for route_dir in sys.argv[1:]:
    print(f"\n{'#'*80}")
    print(f"# ANALYZING: {route_dir}")
    print(f"{'#'*80}")

    msgs = load_route(route_dir)
    if not msgs:
      print("No messages loaded!")
      continue

    data = extract_data(msgs)
    analyze_overall_stats(data)
    analyze_curves(data)
    analyze_hunting(data)


if __name__ == '__main__':
  main()
