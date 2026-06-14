#!/usr/bin/env python3
"""Analyze Explorer ST drive logs — v3: comparison across drives, improved hunting metrics."""

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
  files = sorted(glob.glob(os.path.join(route_dir, 'rlog_*.zst')),
                 key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))
  print(f"Loading {len(files)} segments from {os.path.basename(route_dir)}...")
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
  for k in data:
    data[k] = np.array(data[k])
  return data


def build_unified_timeline(data):
  t = data['cs_t']
  v = data['v_ego']
  yaw = data['yaw_rate']
  angle = data['steering_angle']
  measured_curv = np.where(v > 0.5, -yaw / np.maximum(v, 0.1), 0.0)

  if len(data['cc_t']) > 1:
    lat_active = np.interp(t, data['cc_t'], data['lat_active'].astype(float)) > 0.5
    cc_curv = np.interp(t, data['cc_t'], data['cc_curvature'])
  else:
    lat_active = np.zeros(len(t), dtype=bool)
    cc_curv = np.zeros(len(t))

  if len(data['co_t']) > 1:
    applied_curv = np.interp(t, data['co_t'], data['applied_curvature'])
  else:
    applied_curv = cc_curv

  if len(data['ctrl_t']) > 1:
    desired_curv = np.interp(t, data['ctrl_t'], data['desired_curvature'])
    ctrl_measured_curv = np.interp(t, data['ctrl_t'], data['measured_curvature'])
  else:
    desired_curv = np.zeros(len(t))
    ctrl_measured_curv = measured_curv

  return {
    't': t, 'v': v, 'angle': angle,
    'yaw': yaw, 'measured_curv': measured_curv,
    'lat_active': lat_active, 'applied_curv': applied_curv,
    'desired_curv': desired_curv, 'ctrl_measured_curv': ctrl_measured_curv,
  }


def analyze_overall(u, label=""):
  print(f"\n{'='*80}")
  print(f"OVERALL DRIVE STATISTICS — {label}")
  print(f"{'='*80}")

  t, v = u['t'], u['v']
  duration = t[-1] - t[0]
  lat = u['lat_active']

  print(f"  Duration: {duration/60:.1f} minutes")
  print(f"  Speed range: {np.min(v)*2.237:.1f} - {np.max(v)*2.237:.1f} mph")
  print(f"  Mean speed: {np.mean(v)*2.237:.1f} mph")

  engaged_pct = np.mean(lat) * 100
  engaged_time = np.sum(lat) * np.median(np.diff(t))
  print(f"  Lateral engaged: {engaged_pct:.1f}% ({engaged_time/60:.1f} min)")

  if np.sum(lat) > 100:
    print(f"\n  Speed distribution (ENGAGED ONLY):")
    bins = [(0, 7, "0-15 mph"), (7, 13, "15-30 mph"), (13, 20, "30-45 mph"),
            (20, 29, "45-65 mph"), (29, 36, "65-80 mph"), (36, 50, "80+ mph")]
    eng_v = v[lat]
    for v_min, v_max, lbl in bins:
      pct = np.mean((eng_v >= v_min) & (eng_v < v_max)) * 100
      bar = "█" * int(pct / 2)
      print(f"    {lbl:>10s}: {pct:5.1f}% {bar}")

  return {'duration': duration, 'engaged_pct': engaged_pct, 'engaged_time': engaged_time,
          'mean_speed': np.mean(v)}


def analyze_curves(u, label=""):
  print(f"\n{'='*80}")
  print(f"CURVE ANALYSIS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  meas = u['measured_curv']
  app = u['applied_curv']
  des = u['desired_curv']

  engaged_and_moving = lat & (v > 5.0)
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
    if seg_end - seg_start < 20:
      continue
    seg_meas = meas[seg_start:seg_end]
    seg_app = app[seg_start:seg_end]
    seg_des = des[seg_start:seg_end]
    seg_v = v[seg_start:seg_end]

    if not np.any(np.abs(seg_meas) > 0.001):
      continue

    peak_idx = np.argmax(np.abs(seg_meas))
    peak_meas = seg_meas[peak_idx]
    sign = np.sign(peak_meas) if abs(peak_meas) > 0.0005 else 1.0
    signed_meas = seg_meas * sign
    signed_app = seg_app * sign

    overshoot = signed_meas - signed_app
    max_overshoot = np.max(overshoot)
    peak_curv = np.max(np.abs(seg_meas))
    lat_accel = seg_meas * (seg_v ** 2)

    curves.append({
      'peak_curv': peak_curv,
      'peak_applied': np.max(np.abs(seg_app)),
      'peak_desired': np.max(np.abs(seg_des)),
      'peak_lat_accel': np.max(np.abs(lat_accel)),
      'max_overshoot': max_overshoot,
      'mean_speed_mph': np.mean(seg_v) * 2.237,
      'tracking_rmse': np.sqrt(np.mean((seg_des - seg_app)**2)),
    })

  sharp = [c for c in curves if c['peak_curv'] > 0.004]
  moderate = [c for c in curves if 0.002 < c['peak_curv'] <= 0.004]
  gentle = [c for c in curves if 0.001 < c['peak_curv'] <= 0.002]

  metrics = {}
  for cat_label, subset in [("Sharp (>0.004)", sharp), ("Moderate (0.002-0.004)", moderate), ("Gentle (0.001-0.002)", gentle)]:
    print(f"\n  {cat_label}: {len(subset)} events")
    if subset:
      overshoots = [c['max_overshoot'] for c in subset]
      overshooters = [o for o in overshoots if o > 0.0005]
      tracking = [c['tracking_rmse'] for c in subset]
      print(f"    Mean overshoot: {np.mean(overshoots):+.5f}")
      print(f"    Max overshoot:  {np.max(overshoots):+.5f}")
      print(f"    Overshoot events (>0.0005): {len(overshooters)}/{len(subset)} ({100*len(overshooters)/len(subset):.0f}%)")
      print(f"    Mean tracking RMSE: {np.mean(tracking):.6f}")
      metrics[cat_label] = {
        'count': len(subset),
        'mean_overshoot': np.mean(overshoots),
        'max_overshoot': np.max(overshoots),
        'overshoot_rate': len(overshooters)/len(subset) if subset else 0,
        'mean_tracking_rmse': np.mean(tracking),
      }

  return metrics


def analyze_hunting(u, label=""):
  print(f"\n{'='*80}")
  print(f"HUNTING ANALYSIS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  angle = u['angle']
  meas = u['measured_curv']
  app = u['applied_curv']
  des = u['desired_curv']

  speed_bins = [
    ("7-10 m/s (15-22 mph)", 7, 10),
    ("10-13 m/s (22-30 mph)", 10, 13),
    ("13-20 m/s (30-45 mph)", 13, 20),
    ("20-29 m/s (45-65 mph)", 20, 29),
    ("29-36 m/s (65-80 mph)", 29, 36),
  ]

  print(f"\n{'Speed bin':>30s} {'Secs':>6s} {'AngStd':>7s} {'AppStd':>9s} {'DesStd':>9s} {'OscFreq':>8s} {'DesOscF':>8s} {'Source':>10s}")

  metrics = {}
  for name, v_min, v_max in speed_bins:
    mask = lat & (v >= v_min) & (v < v_max) & (np.abs(meas) < 0.003)
    n = np.sum(mask)
    if n < 50:
      print(f"{name:>30s}   (insufficient data, {n} pts)")
      continue

    dt = np.median(np.diff(t))
    secs = n * dt

    seg_angle = angle[mask]
    seg_app = app[mask]
    seg_des = des[mask]

    angle_std = np.std(seg_angle)
    app_std = np.std(seg_app)
    des_std = np.std(seg_des)

    app_sign_changes = np.sum(np.diff(np.sign(seg_app)) != 0)
    des_sign_changes = np.sum(np.diff(np.sign(seg_des)) != 0)
    osc_freq = app_sign_changes / secs / 2 if secs > 0 else 0
    des_osc_freq = des_sign_changes / secs / 2 if secs > 0 else 0

    if des_osc_freq > 0.05:
      source = "PLANNER"
    elif osc_freq > des_osc_freq * 2:
      source = "CONTROLLER"
    else:
      source = "MIXED"

    print(f"{name:>30s} {secs:>5.0f}s {angle_std:>6.3f}° {app_std:>8.6f} {des_std:>8.6f} {osc_freq:>7.3f}Hz {des_osc_freq:>7.3f}Hz {source:>10s}")

    metrics[name] = {
      'secs': secs,
      'angle_std': angle_std,
      'app_std': app_std,
      'des_std': des_std,
      'osc_freq': osc_freq,
      'des_osc_freq': des_osc_freq,
      'source': source,
    }

  # Detailed low-speed analysis
  print(f"\n--- LOW-SPEED HUNTING DEEP DIVE ---")
  for v_min, v_max, lbl in [(7, 10, "7-10 m/s (15-22 mph)"), (10, 13, "10-13 m/s (22-30 mph)")]:
    mask = lat & (v >= v_min) & (v < v_max) & (np.abs(meas) < 0.003)
    n = np.sum(mask)
    if n < 50:
      print(f"\n  {lbl}: insufficient data ({n} points)")
      continue

    dt = np.median(np.diff(t))
    secs = n * dt
    seg_app = app[mask]
    seg_des = des[mask]
    seg_meas = meas[mask]
    seg_angle = angle[mask]

    sign_changes = np.sum(np.diff(np.sign(seg_app)) != 0)
    osc_freq = sign_changes / secs / 2 if secs > 0 else 0
    des_sign_changes = np.sum(np.diff(np.sign(seg_des)) != 0)
    des_osc_freq = des_sign_changes / secs / 2 if secs > 0 else 0

    print(f"\n  {lbl}: {secs:.1f}s of engaged data")
    print(f"    Steering angle std: {np.std(seg_angle):.3f}°")
    print(f"    Applied curv std:   {np.std(seg_app):.6f}")
    print(f"    Applied curv range: [{np.min(seg_app):.6f}, {np.max(seg_app):.6f}]")
    print(f"    Desired curv std:   {np.std(seg_des):.6f}")
    print(f"    Des vs app RMSE:    {np.sqrt(np.mean((seg_des - seg_app)**2)):.6f}")
    print(f"    Meas vs app RMSE:   {np.sqrt(np.mean((seg_meas - seg_app)**2)):.6f}")
    print(f"    Applied osc freq:   {osc_freq:.3f} Hz" + (f" ({1/osc_freq:.1f}s period)" if osc_freq > 0.001 else ""))
    print(f"    Desired osc freq:   {des_osc_freq:.3f} Hz" + (f" ({1/des_osc_freq:.1f}s period)" if des_osc_freq > 0.001 else ""))

  return metrics


def print_comparison(label_a, label_b, hunt_a, hunt_b, curve_a, curve_b):
  print(f"\n{'='*80}")
  print(f"COMPARISON: {label_a}  vs  {label_b}")
  print(f"{'='*80}")

  print(f"\n--- HUNTING COMPARISON (negative = improvement) ---")
  print(f"{'Speed bin':>30s} │ {'AngStd_A':>8s} {'AngStd_B':>8s} {'Δ%':>7s} │ {'AppStd_A':>9s} {'AppStd_B':>9s} {'Δ%':>7s} │ {'OscHz_A':>7s} {'OscHz_B':>7s} {'Δ%':>7s}")

  common_bins = sorted(set(hunt_a.keys()) & set(hunt_b.keys()))
  for name in common_bins:
    a = hunt_a[name]
    b = hunt_b[name]
    ang_pct = ((b['angle_std'] - a['angle_std']) / a['angle_std'] * 100) if a['angle_std'] > 0 else 0
    app_pct = ((b['app_std'] - a['app_std']) / a['app_std'] * 100) if a['app_std'] > 0 else 0
    osc_pct = ((b['osc_freq'] - a['osc_freq']) / a['osc_freq'] * 100) if a['osc_freq'] > 0 else 0
    print(f"{name:>30s} │ {a['angle_std']:>7.3f}° {b['angle_std']:>7.3f}° {ang_pct:>+6.1f}% │ "
          f"{a['app_std']:>8.6f} {b['app_std']:>8.6f} {app_pct:>+6.1f}% │ "
          f"{a['osc_freq']:>6.3f} {b['osc_freq']:>6.3f} {osc_pct:>+6.1f}%")

  print(f"\n--- CURVE OVERSHOOT COMPARISON (negative = improvement) ---")
  print(f"{'Category':>25s} │ {'MnOS_A':>9s} {'MnOS_B':>9s} {'Δ%':>8s} │ {'MxOS_A':>9s} {'MxOS_B':>9s} │ {'RMSE_A':>8s} {'RMSE_B':>8s} {'Δ%':>7s}")
  common_cats = sorted(set(curve_a.keys()) & set(curve_b.keys()))
  for cat in common_cats:
    a = curve_a[cat]
    b = curve_b[cat]
    mn_pct = ((b['mean_overshoot'] - a['mean_overshoot']) / abs(a['mean_overshoot']) * 100) if abs(a['mean_overshoot']) > 1e-6 else 0
    rmse_pct = ((b['mean_tracking_rmse'] - a['mean_tracking_rmse']) / a['mean_tracking_rmse'] * 100) if a['mean_tracking_rmse'] > 1e-6 else 0
    print(f"{cat:>25s} │ {a['mean_overshoot']:>+8.5f} {b['mean_overshoot']:>+8.5f} {mn_pct:>+7.1f}% │ "
          f"{a['max_overshoot']:>+8.5f} {b['max_overshoot']:>+8.5f} │ "
          f"{a['mean_tracking_rmse']:>7.6f} {b['mean_tracking_rmse']:>7.6f} {rmse_pct:>+6.1f}%")


def main():
  if len(sys.argv) < 2:
    print("Usage: analyze_drive_v3.py <route_dir> [route_dir2]")
    print("  With 2 dirs: first is baseline (before), second is test (after)")
    sys.exit(1)

  results = []
  for route_dir in sys.argv[1:]:
    name = os.path.basename(route_dir)
    print(f"\n{'#'*80}")
    print(f"# ANALYZING: {name}")
    print(f"{'#'*80}")

    msgs = load_route(route_dir)
    if not msgs:
      continue

    data = extract_data(msgs)
    u = build_unified_timeline(data)
    overall = analyze_overall(u, name)
    curve_metrics = analyze_curves(u, name)
    hunt_metrics = analyze_hunting(u, name)
    results.append((name, overall, curve_metrics, hunt_metrics))

  if len(results) == 2:
    (name_a, _, curve_a, hunt_a) = results[0]
    (name_b, _, curve_b, hunt_b) = results[1]
    print_comparison(name_a, name_b, hunt_a, hunt_b, curve_a, curve_b)


if __name__ == '__main__':
  main()
