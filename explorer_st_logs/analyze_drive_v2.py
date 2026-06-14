#!/usr/bin/env python3
"""Analyze Explorer ST drive logs — v2: only engaged periods, proper filtering."""

import sys
import os
import glob
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openpilot.tools.lib.logreader import LogReader


def load_route(route_dir):
  """Load all segments, return list of messages."""
  files = sorted(glob.glob(os.path.join(route_dir, 'rlog_seg*.zst')),
                 key=lambda f: int(f.split('seg')[1].split('.')[0]))
  print(f"Loading {len(files)} segments from {route_dir}...")

  all_msgs = []
  for i, f in enumerate(files):
    try:
      lr = LogReader(f)
      all_msgs.extend(list(lr))
      if (i + 1) % 10 == 0:
        print(f"  loaded {i+1}/{len(files)} segments...")
    except Exception as e:
      print(f"  skip seg {i}: {e}")

  print(f"  Total messages: {len(all_msgs)}")
  return all_msgs


def extract_data(msgs):
  """Extract time-aligned data arrays."""
  data = defaultdict(list)

  for msg in msgs:
    t = msg.logMonoTime / 1e9
    w = msg.which()

    if w == 'carState':
      cs = msg.carState
      data['cs_t'].append(t)
      data['v_ego'].append(cs.vEgo)
      data['steering_angle'].append(cs.steeringAngleDeg)
      data['steering_torque'].append(cs.steeringTorque)
      data['steering_pressed'].append(cs.steeringPressed)
      data['yaw_rate'].append(cs.yawRate)

    elif w == 'carControl':
      cc = msg.carControl
      data['cc_t'].append(t)
      data['lat_active'].append(cc.latActive)
      data['cc_curvature'].append(cc.actuators.curvature)

    elif w == 'controlsState':
      cs = msg.controlsState
      data['ctrl_t'].append(t)
      data['desired_curvature'].append(cs.desiredCurvature)
      data['measured_curvature'].append(cs.curvature)

    elif w == 'carOutput':
      co = msg.carOutput
      data['co_t'].append(t)
      data['applied_curvature'].append(co.actuatorsOutput.curvature)

    elif w == 'selfdriveState':
      ss = msg.selfdriveState
      data['ss_t'].append(t)
      data['enabled'].append(ss.enabled)
      data['active'].append(ss.active)

  for k in data:
    data[k] = np.array(data[k])
  return data


def build_unified_timeline(data):
  """Interpolate all signals to carState timestamps."""
  t = data['cs_t']
  v = data['v_ego']
  yaw = data['yaw_rate']
  angle = data['steering_angle']
  torque = data['steering_torque']

  measured_curv = np.where(v > 0.5, -yaw / np.maximum(v, 0.1), 0.0)

  # Interpolate lat_active (boolean -> nearest)
  if len(data['cc_t']) > 1:
    lat_active = np.interp(t, data['cc_t'], data['lat_active'].astype(float)) > 0.5
    cc_curv = np.interp(t, data['cc_t'], data['cc_curvature'])
  else:
    lat_active = np.zeros(len(t), dtype=bool)
    cc_curv = np.zeros(len(t))

  # Applied curvature (what actually went to the CAN bus)
  if len(data['co_t']) > 1:
    applied_curv = np.interp(t, data['co_t'], data['applied_curvature'])
  else:
    applied_curv = cc_curv

  # Desired curvature from planner
  if len(data['ctrl_t']) > 1:
    desired_curv = np.interp(t, data['ctrl_t'], data['desired_curvature'])
    ctrl_measured_curv = np.interp(t, data['ctrl_t'], data['measured_curvature'])
  else:
    desired_curv = np.zeros(len(t))
    ctrl_measured_curv = measured_curv

  return {
    't': t, 'v': v, 'angle': angle, 'torque': torque,
    'yaw': yaw, 'measured_curv': measured_curv,
    'lat_active': lat_active, 'applied_curv': applied_curv,
    'desired_curv': desired_curv, 'ctrl_measured_curv': ctrl_measured_curv,
  }


def analyze_overall(u):
  """Overall drive stats."""
  print("\n" + "="*80)
  print("OVERALL DRIVE STATISTICS")
  print("="*80)

  t, v = u['t'], u['v']
  duration = t[-1] - t[0]
  lat = u['lat_active']

  print(f"  Duration: {duration/60:.1f} minutes")
  print(f"  Speed range: {np.min(v)*2.237:.1f} - {np.max(v)*2.237:.1f} mph")
  print(f"  Mean speed: {np.mean(v)*2.237:.1f} mph")

  engaged_pct = np.mean(lat) * 100
  engaged_time = np.sum(lat) * np.median(np.diff(t))
  print(f"  Lateral engaged: {engaged_pct:.1f}% ({engaged_time/60:.1f} min)")

  # Speed distribution while engaged
  if np.sum(lat) > 100:
    print(f"\n  Speed distribution (ENGAGED ONLY):")
    bins = [(0, 7, "0-15 mph"), (7, 13, "15-30 mph"), (13, 20, "30-45 mph"),
            (20, 29, "45-65 mph"), (29, 36, "65-80 mph"), (36, 50, "80+ mph")]
    eng_v = v[lat]
    for v_min, v_max, label in bins:
      pct = np.mean((eng_v >= v_min) & (eng_v < v_max)) * 100
      bar = "█" * int(pct / 2)
      print(f"    {label:>10s}: {pct:5.1f}% {bar}")


def find_engaged_curve_events(u):
  """Find curve events ONLY while lat control is active."""
  t, v, lat = u['t'], u['v'], u['lat_active']
  meas = u['measured_curv']
  app = u['applied_curv']
  des = u['desired_curv']
  angle = u['angle']

  # Only engaged periods at meaningful speed
  engaged_and_moving = lat & (v > 5.0)

  # Find continuous engaged segments
  diffs = np.diff(engaged_and_moving.astype(int))
  starts = np.where(diffs == 1)[0] + 1
  ends = np.where(diffs == -1)[0] + 1
  if engaged_and_moving[0]:
    starts = np.insert(starts, 0, 0)
  if engaged_and_moving[-1]:
    ends = np.append(ends, len(engaged_and_moving))
  if len(starts) > len(ends):
    starts = starts[:len(ends)]

  curves = []
  for seg_start, seg_end in zip(starts, ends):
    if seg_end - seg_start < 20:  # skip < 1s
      continue

    seg_t = t[seg_start:seg_end]
    seg_v = v[seg_start:seg_end]
    seg_meas = meas[seg_start:seg_end]
    seg_app = app[seg_start:seg_end]
    seg_des = des[seg_start:seg_end]
    seg_angle = angle[seg_start:seg_end]

    # Find sub-segments with significant curvature
    in_curve = np.abs(seg_meas) > 0.001
    if not np.any(in_curve):
      continue

    # Find peak curvature in this engaged segment
    peak_idx = np.argmax(np.abs(seg_meas))
    peak_meas = seg_meas[peak_idx]
    peak_app = seg_app[peak_idx]
    peak_des = seg_des[peak_idx]

    # Tracking error: how well does applied follow desired?
    # And overshoot: does measured exceed applied?
    sign = np.sign(peak_meas) if abs(peak_meas) > 0.0005 else 1.0
    signed_meas = seg_meas * sign
    signed_app = seg_app * sign
    signed_des = seg_des * sign

    # Overshoot: measured > applied (car turned more than requested)
    overshoot = signed_meas - signed_app
    max_overshoot = np.max(overshoot)

    # Tracking lag: applied vs desired
    tracking_err = signed_des - signed_app
    max_tracking_lag = np.max(tracking_err)  # positive = applied is behind desired

    peak_curv = np.max(np.abs(seg_meas))
    mean_speed = np.mean(seg_v)
    lat_accel = seg_meas * (seg_v ** 2)

    curves.append({
      'start_t': seg_t[0] - t[0],
      'duration': seg_t[-1] - seg_t[0],
      'mean_speed_mph': mean_speed * 2.237,
      'peak_curv': peak_curv,
      'peak_applied': np.max(np.abs(seg_app)),
      'peak_desired': np.max(np.abs(seg_des)),
      'peak_lat_accel': np.max(np.abs(lat_accel)),
      'max_overshoot': max_overshoot,
      'max_tracking_lag': max_tracking_lag,
      'peak_angle': seg_angle[peak_idx],
    })

  return curves


def analyze_curves(u):
  """Analyze sharp curve behavior while engaged."""
  print("\n" + "="*80)
  print("CURVE ANALYSIS — Engaged Periods Only")
  print("="*80)

  curves = find_engaged_curve_events(u)
  if not curves:
    print("No engaged curve events found")
    return

  # Filter to actual curves (peak curvature > 0.002)
  real_curves = [c for c in curves if c['peak_curv'] > 0.002]
  sharp = [c for c in real_curves if c['peak_curv'] > 0.004]
  moderate = [c for c in real_curves if 0.002 < c['peak_curv'] <= 0.004]

  print(f"\nTotal engaged segments with curves: {len(real_curves)}")

  for label, subset in [("SHARP CURVES (|curv| > 0.004)", sharp),
                         ("MODERATE CURVES (0.002 < |curv| < 0.004)", moderate)]:
    print(f"\n--- {label} ---")
    print(f"Count: {len(subset)}")
    if not subset:
      continue

    print(f"{'Time':>8s} {'Dur':>5s} {'Speed':>7s} {'PkMeas':>8s} {'PkAppl':>8s} {'PkDesir':>8s} "
          f"{'Overshoot':>10s} {'TrkLag':>8s} {'LatAcc':>7s}")
    for c in sorted(subset, key=lambda x: x['max_overshoot'], reverse=True):
      print(f"{c['start_t']:>7.0f}s {c['duration']:>4.1f}s {c['mean_speed_mph']:>5.1f}mph "
            f"{c['peak_curv']:>8.5f} {c['peak_applied']:>8.5f} {c['peak_desired']:>8.5f} "
            f"{c['max_overshoot']:>+9.5f} {c['max_tracking_lag']:>+7.5f} {c['peak_lat_accel']:>6.2f}")

    overshooters = [c for c in subset if c['max_overshoot'] > 0.0005]
    laggers = [c for c in subset if c['max_tracking_lag'] > 0.001]
    print(f"\n  Overshoot (meas > applied by >0.0005): {len(overshooters)}/{len(subset)}")
    if overshooters:
      print(f"    Mean: {np.mean([c['max_overshoot'] for c in overshooters]):.5f}")
      print(f"    Worst: {max(c['max_overshoot'] for c in overshooters):.5f}")
    print(f"  Tracking lag (desired > applied by >0.001): {len(laggers)}/{len(subset)}")
    if laggers:
      print(f"    Mean: {np.mean([c['max_tracking_lag'] for c in laggers]):.5f}")
      print(f"    Worst: {max(c['max_tracking_lag'] for c in laggers):.5f}")


def analyze_hunting(u):
  """Analyze low-speed hunting during engaged periods."""
  print("\n" + "="*80)
  print("HUNTING ANALYSIS — Engaged Low-Speed Oscillation")
  print("="*80)

  t, v, lat = u['t'], u['v'], u['lat_active']
  angle = u['angle']
  meas = u['measured_curv']
  app = u['applied_curv']
  des = u['desired_curv']

  speed_bins = [
    ("Low speed (15-30 mph / 7-13 m/s)", 7, 13),
    ("Medium speed (30-45 mph / 13-20 m/s)", 13, 20),
    ("Highway (45-65 mph / 20-29 m/s)", 20, 29),
    ("High highway (65-80 mph / 29-36 m/s)", 29, 36),
  ]

  print(f"\n--- STEERING OSCILLATION BY SPEED (ENGAGED, ~straight sections) ---")
  print(f"{'Speed bin':>42s} {'Secs':>6s} {'AngStd':>7s} {'AppStd':>9s} {'DesVsApp':>9s} {'MeasVsApp':>10s}")

  for name, v_min, v_max in speed_bins:
    # Engaged, in speed range, roughly straight
    mask = lat & (v >= v_min) & (v < v_max) & (np.abs(meas) < 0.003)
    n = np.sum(mask)
    if n < 50:
      print(f"{name:>42s}   (insufficient data)")
      continue

    dt = np.median(np.diff(t))
    secs = n * dt

    seg_angle = angle[mask]
    seg_app = app[mask]
    seg_des = des[mask]
    seg_meas = meas[mask]

    def rolling_std(arr, w):
      stds = []
      for i in range(0, len(arr) - w, w // 2):
        stds.append(np.std(arr[i:i+w]))
      return np.array(stds) if stds else np.array([0.0])

    window = min(100, n // 3)
    if window < 10:
      window = 10

    angle_std = np.median(rolling_std(seg_angle, window))
    app_std = np.median(rolling_std(seg_app, window))
    des_vs_app = np.sqrt(np.mean((seg_des - seg_app) ** 2))
    meas_vs_app = np.sqrt(np.mean((seg_meas - seg_app) ** 2))

    print(f"{name:>42s} {secs:>5.0f}s {angle_std:>6.3f}° {app_std:>8.6f} {des_vs_app:>8.6f} {meas_vs_app:>9.6f}")

  # Deep dive on low speed
  print(f"\n--- LOW-SPEED ENGAGED HUNTING DEEP DIVE ---")
  for v_min, v_max, label in [(7, 10, "7-10 m/s (15-22 mph)"), (10, 13, "10-13 m/s (22-30 mph)")]:
    mask = lat & (v >= v_min) & (v < v_max) & (np.abs(meas) < 0.003)
    n = np.sum(mask)
    if n < 50:
      print(f"\n  {label}: insufficient data ({n} points)")
      continue

    dt = np.median(np.diff(t))
    secs = n * dt
    seg_app = app[mask]
    seg_angle = angle[mask]
    seg_des = des[mask]
    seg_meas = meas[mask]

    # Zero crossings of applied curvature
    sign_changes = np.sum(np.diff(np.sign(seg_app)) != 0)
    osc_freq = sign_changes / secs / 2 if secs > 0 else 0

    print(f"\n  {label}: {secs:.1f}s of engaged data")
    print(f"    Steering angle std: {np.std(seg_angle):.3f}°")
    print(f"    Applied curv std: {np.std(seg_app):.6f}")
    print(f"    Applied curv range: [{np.min(seg_app):.6f}, {np.max(seg_app):.6f}]")
    print(f"    Desired curv std: {np.std(seg_des):.6f}")
    print(f"    Desired vs applied RMSE: {np.sqrt(np.mean((seg_des - seg_app)**2)):.6f}")
    print(f"    Measured vs applied RMSE: {np.sqrt(np.mean((seg_meas - seg_app)**2)):.6f}")
    print(f"    Sign changes: {sign_changes}, freq: {osc_freq:.3f} Hz" +
          (f" ({1/osc_freq:.1f}s period)" if osc_freq > 0.001 else ""))

    # Check if desired curvature itself is oscillating
    des_sign_changes = np.sum(np.diff(np.sign(seg_des)) != 0)
    des_osc_freq = des_sign_changes / secs / 2 if secs > 0 else 0
    print(f"    DESIRED curv sign changes: {des_sign_changes}, freq: {des_osc_freq:.3f} Hz")
    print(f"    -> Hunting source: {'PLANNER (desired oscillates)' if des_osc_freq > 0.05 else 'CONTROLLER (desired stable, applied oscillates)' if osc_freq > des_osc_freq * 2 else 'MIXED'}")


def main():
  if len(sys.argv) < 2:
    print("Usage: analyze_drive_v2.py <route_dir> [route_dir2 ...]")
    sys.exit(1)

  for route_dir in sys.argv[1:]:
    print(f"\n{'#'*80}")
    print(f"# ANALYZING: {route_dir}")
    print(f"{'#'*80}")

    msgs = load_route(route_dir)
    if not msgs:
      continue

    data = extract_data(msgs)
    u = build_unified_timeline(data)
    analyze_overall(u)
    analyze_curves(u)
    analyze_hunting(u)


if __name__ == '__main__':
  main()
