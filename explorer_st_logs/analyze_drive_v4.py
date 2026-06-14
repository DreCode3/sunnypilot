#!/usr/bin/env python3
"""
Explorer ST Drive Analysis — v4
Deep analysis: filter efficacy, FFT power spectrum, lateral path error,
steering interventions, EPAS tracking quality, sub-7 m/s hunting, full comparison.
"""

import sys
import os
import glob
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openpilot.tools.lib.logreader import LogReader


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_route(route_dir):
  # Format 1 (old): single dir with rlog_0.zst, rlog_1.zst, ...
  files = sorted(glob.glob(os.path.join(route_dir, 'rlog_*.zst')),
                 key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

  # Format 2 (new): per-segment dirs {route}--{n}/rlog.zst next to route_dir
  if not files:
    route_name = os.path.basename(route_dir)
    parent = os.path.dirname(route_dir)
    seg_dirs = sorted(glob.glob(os.path.join(parent, route_name + '--*')),
                      key=lambda d: int(d.rsplit('--', 1)[-1]))
    # prefer rlog.zst over qlog.zst
    for sd in seg_dirs:
      rlog = os.path.join(sd, 'rlog.zst')
      qlog = os.path.join(sd, 'qlog.zst')
      if os.path.exists(rlog):
        files.append(rlog)
      elif os.path.exists(qlog):
        files.append(qlog)

  route_name = os.path.basename(route_dir)
  print(f"  Loading {len(files)} segments from {route_name}...")
  all_msgs = []
  for i, f in enumerate(files):
    try:
      lr = LogReader(f)
      all_msgs.extend(list(lr))
      if (i + 1) % 10 == 0:
        print(f"    {i+1}/{len(files)} segments...")
    except Exception as e:
      print(f"    skip seg {i}: {e}")
  print(f"    {len(all_msgs):,} messages loaded")
  return all_msgs


def extract_data(msgs):
  """Extract all relevant signals from log messages."""
  data = defaultdict(list)
  for msg in msgs:
    t = msg.logMonoTime / 1e9
    w = msg.which()

    if w == 'carState':
      cs = msg.carState
      data['cs_t'].append(t)
      data['v_ego'].append(cs.vEgoRaw)
      data['steering_angle'].append(cs.steeringAngleDeg)
      data['steering_torque'].append(cs.steeringTorque)
      data['steering_pressed'].append(cs.steeringPressed)
      data['yaw_rate'].append(cs.yawRate)
      data['a_ego'].append(cs.aEgo)

    elif w == 'carControl':
      cc = msg.carControl
      data['cc_t'].append(t)
      data['lat_active'].append(cc.latActive)
      data['long_active'].append(cc.longActive)
      # post-filter applied curvature (our carcontroller output)
      data['cc_curvature'].append(cc.actuators.curvature)

    elif w == 'carOutput':
      co = msg.carOutput
      data['co_t'].append(t)
      data['co_curvature'].append(co.actuatorsOutput.curvature)

    elif w == 'controlsState':
      cs2 = msg.controlsState
      data['ctrl_t'].append(t)
      data['ctrl_desired_curv'].append(getattr(cs2, 'desiredCurvature', 0.0))
      data['ctrl_measured_curv'].append(getattr(cs2, 'curvature', 0.0))

    elif w == 'modelV2':
      mv = msg.modelV2
      data['mv_t'].append(t)
      # Pre-filter planner curvature (model output before our carcontroller)
      action = mv.action
      data['mv_desired_curv'].append(action.desiredCurvature)
      # Lane centering error: midpoint of left+right lane lines at current position
      # laneLines[1]=left, laneLines[2]=right; y convention: negative=left, positive=right
      # midpoint > 0 → car is left of lane center; midpoint < 0 → car is right of lane center
      if len(mv.laneLines) > 2 and len(mv.laneLines[1].y) > 0 and len(mv.laneLines[2].y) > 0:
        ll_y = mv.laneLines[1].y[0]
        rl_y = mv.laneLines[2].y[0]
        data['mv_path_y'].append((ll_y + rl_y) / 2.0)  # lane centering offset (m)
        data['mv_lane_width'].append(rl_y - ll_y)
      else:
        data['mv_path_y'].append(0.0)
        data['mv_lane_width'].append(3.7)
      # Future predicted path: position.y at ~1s lookahead (index 10 at 0.1s steps) for path preview
      if len(mv.position.y) > 10:
        data['mv_pos_y1s'].append(list(mv.position.y)[10])
      else:
        data['mv_pos_y1s'].append(0.0)
      # Lane line probabilities (left=1, right=2)
      if len(mv.laneLineProbs) > 2:
        data['mv_ll_prob'].append(mv.laneLineProbs[1])
        data['mv_rl_prob'].append(mv.laneLineProbs[2])
      else:
        data['mv_ll_prob'].append(0.0)
        data['mv_rl_prob'].append(0.0)

    elif w == 'selfdriveState':
      sd = msg.selfdriveState
      data['sd_t'].append(t)
      data['sd_active'].append(sd.active)

  for k in data:
    data[k] = np.array(data[k], dtype=float if k not in ('steering_pressed',) else bool)
  # Compute delta-curvature (curvature rate of change) from modelV2 desired curvature
  if len(data['mv_desired_curv']) > 1:
    dt_mv = np.diff(data['mv_t'])
    dt_mv = np.where(dt_mv > 0, dt_mv, 0.05)
    dcurv = np.diff(data['mv_desired_curv']) / dt_mv
    data['mv_dcurv'] = np.concatenate([[0.0], dcurv])
  else:
    data['mv_dcurv'] = np.zeros_like(data['mv_desired_curv'])
  return data


def build_timeline(data):
  """Interpolate all signals onto the carState timebase."""
  t = data['cs_t']
  if len(t) < 2:
    return None

  def interp(src_t, src_v, default=0.0):
    if len(src_t) > 1:
      return np.interp(t, src_t, src_v)
    return np.full(len(t), default)

  v = data['v_ego']
  yaw = data['yaw_rate']
  measured_curv = np.where(v > 0.5, -yaw / np.maximum(v, 0.1), 0.0)

  lat_active = interp(data['cc_t'], data['lat_active'].astype(float)) > 0.5
  cc_curv     = interp(data['cc_t'], data['cc_curvature'])   # post-filter (our output)
  co_curv     = interp(data['co_t'], data['co_curvature']) if len(data['co_t']) > 1 else cc_curv
  mv_curv     = interp(data['mv_t'], data['mv_desired_curv'])  # pre-filter (model output)
  ctrl_curv   = interp(data['ctrl_t'], data['ctrl_desired_curv'])
  path_y      = interp(data['mv_t'], data['mv_path_y'])
  lane_width  = interp(data['mv_t'], data['mv_lane_width'])
  pos_y1s     = interp(data['mv_t'], data['mv_pos_y1s'])
  ll_prob     = interp(data['mv_t'], data['mv_ll_prob'])
  rl_prob     = interp(data['mv_t'], data['mv_rl_prob'])
  mv_dcurv    = interp(data['mv_t'], data['mv_dcurv'])
  steer_press = interp(data['cs_t'], data['steering_pressed'].astype(float)) > 0.5

  # Use carOutput curvature if available, else carControl
  applied_curv = co_curv if len(data['co_t']) > 1 else cc_curv

  return {
    't': t,
    'v': v,
    'angle': data['steering_angle'],
    'torque': data['steering_torque'],
    'steer_pressed': steer_press,
    'a_ego': data['a_ego'],
    'yaw': yaw,
    'lat_active': lat_active,
    'measured_curv': measured_curv,
    'applied_curv': applied_curv,    # post-filter carcontroller output
    'cc_curv': cc_curv,              # same as applied in most cases
    'mv_curv': mv_curv,              # pre-filter model desired curvature
    'ctrl_curv': ctrl_curv,          # controlsState desired curvature
    'path_y': path_y,                # lane centering offset (m): + = left of lane center
    'lane_width': lane_width,        # estimated lane width (m)
    'pos_y1s': pos_y1s,              # predicted lateral deviation at 1s lookahead (m)
    'll_prob': ll_prob,
    'rl_prob': rl_prob,
    'mv_dcurv': mv_dcurv,            # model curvature rate of change (1/m/s)
  }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_overview(u, label):
  print(f"\n{'='*80}")
  print(f"OVERVIEW — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  dt = np.median(np.diff(t))
  duration = t[-1] - t[0]
  eng_time = np.sum(lat) * dt

  print(f"  Duration:          {duration/60:.1f} min")
  print(f"  Speed range:       {np.min(v)*2.237:.1f} – {np.max(v)*2.237:.1f} mph")
  print(f"  Mean speed:        {np.mean(v)*2.237:.1f} mph  (engaged: {np.mean(v[lat])*2.237:.1f} mph)")
  print(f"  Lateral engaged:   {100*np.mean(lat):.1f}%  ({eng_time/60:.1f} min)")

  # Engagement events
  changes = np.diff(lat.astype(int))
  engages = np.sum(changes == 1)
  disengages = np.sum(changes == -1)
  print(f"  Engage events:     {engages}")
  print(f"  Disengage events:  {disengages}")

  # Steering interventions while engaged
  sp = u['steer_pressed']
  interventions = np.diff((lat & sp).astype(int))
  n_interventions = np.sum(interventions == 1)
  intervention_secs = np.sum(lat & sp) * dt
  print(f"  Steer overrides:   {n_interventions} events  ({intervention_secs:.1f}s total, {100*intervention_secs/eng_time:.1f}% of engaged time)")

  # Speed distribution engaged only
  bins = [(0,7,"0-15 mph"),(7,13,"15-30 mph"),(13,20,"30-45 mph"),
          (20,29,"45-65 mph"),(29,36,"65-80 mph"),(36,50,"80+ mph")]
  print(f"\n  Speed distribution (engaged):")
  eng_v = v[lat]
  for vlo, vhi, lbl in bins:
    pct = np.mean((eng_v >= vlo) & (eng_v < vhi)) * 100
    print(f"    {lbl:>10s}: {pct:5.1f}%  {'█'*int(pct/2)}")

  # Acceleration profile (traffic characterization)
  a = u['a_ego'][lat]
  hard_brake = np.sum(a < -2.0) * dt
  hard_accel = np.sum(a > 1.5) * dt
  print(f"\n  Traffic characterization (engaged):")
  print(f"    Hard braking (<-2.0 m/s²): {hard_brake:.1f}s")
  print(f"    Hard acceleration (>1.5):  {hard_accel:.1f}s")
  print(f"    Mean |accel|:              {np.mean(np.abs(a)):.3f} m/s²")

  return {'duration': duration, 'eng_time': eng_time, 'engages': engages,
          'n_interventions': n_interventions, 'intervention_secs': intervention_secs}


def analyze_filter_efficacy(u, label):
  """Compare modelV2 pre-filter curvature vs carcontroller post-filter output."""
  print(f"\n{'='*80}")
  print(f"FILTER EFFICACY — {label}")
  print(f"{'='*80}")
  print("  (modelV2 = pre-filter planner output; applied = post-filter carcontroller output)")

  t, v, lat = u['t'], u['v'], u['lat_active']
  mv    = u['mv_curv']      # pre-filter
  app   = u['applied_curv'] # post-filter
  meas  = u['measured_curv']

  speed_bins = [
    ("0.5-2 m/s  (1-4 mph)",   0.5,  2.0),
    ("2-4 m/s   (4-9 mph)",    2.0,  4.0),
    ("4-7 m/s   (9-16 mph)",   4.0,  7.0),
    ("7-10 m/s  (16-22 mph)",  7.0, 10.0),
    ("10-13 m/s (22-30 mph)", 10.0, 13.0),
    ("13-20 m/s (30-45 mph)", 13.0, 20.0),
    ("20-29 m/s (45-65 mph)", 20.0, 29.0),
  ]

  print(f"\n{'Bin':<28} {'Secs':>5} {'MV_Std':>9} {'App_Std':>9} {'Reduction':>10} {'RMSE(mv-app)':>13} {'Filter_Active':>14}")
  print('-'*95)

  metrics = {}
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 20:
      print(f"  {name:<26} <20pts")
      continue
    dt = np.median(np.diff(t))
    secs = n * dt
    mv_std  = np.std(mv[mask])
    app_std = np.std(app[mask])
    rmse    = np.sqrt(np.mean((mv[mask] - app[mask])**2))
    reduction = (mv_std - app_std) / mv_std * 100 if mv_std > 0 else 0
    # filter_active: fraction of samples where applied != modelV2 desired
    diverged = np.sum(np.abs(mv[mask] - app[mask]) > 0.0002) / n * 100
    print(f"  {name:<26} {secs:>5.0f}s {mv_std:>9.6f} {app_std:>9.6f} {reduction:>+9.1f}% {rmse:>13.6f} {diverged:>13.1f}%")
    metrics[name] = {'mv_std': mv_std, 'app_std': app_std, 'rmse': rmse, 'reduction': reduction}

  return metrics


def analyze_fft(u, label):
  """FFT power spectrum of applied curvature by speed bin to identify dominant oscillation frequencies."""
  print(f"\n{'='*80}")
  print(f"OSCILLATION POWER SPECTRUM (FFT) — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  app = u['applied_curv']
  mv  = u['mv_curv']
  dt  = np.median(np.diff(t))

  speed_bins = [
    ("2-4 m/s   (4-9 mph)",    2.0,  4.0),
    ("7-10 m/s  (16-22 mph)",  7.0, 10.0),
    ("13-20 m/s (30-45 mph)", 13.0, 20.0),
    ("20-29 m/s (45-65 mph)", 20.0, 29.0),
  ]

  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 100:
      continue

    sig = app[mask] - np.mean(app[mask])
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(np.fft.rfft(sig))**2

    # Top 3 dominant frequencies above 0.05 Hz
    valid = freqs > 0.05
    top_idx = np.argsort(power[valid])[-3:][::-1]
    top_freqs = freqs[valid][top_idx]
    top_powers = power[valid][top_idx]
    total_power = np.sum(power[valid])

    mv_sig = mv[mask] - np.mean(mv[mask])
    mv_power = np.abs(np.fft.rfft(mv_sig))**2

    print(f"\n  {name}  ({n*dt:.0f}s)")
    print(f"  {'Freq':>8s}  {'Period':>8s}  {'App Power%':>11s}  {'MV Power%':>11s}")
    for f, p in zip(top_freqs, top_powers):
      mv_p = np.interp(f, freqs[valid], mv_power[valid]) if len(mv_power[valid]) > 0 else 0
      period = 1/f if f > 0 else 999
      print(f"    {f:>6.3f} Hz  {period:>6.1f}s    {100*p/total_power:>9.1f}%   {100*mv_p/np.sum(mv_power[valid]) if np.sum(mv_power[valid])>0 else 0:>9.1f}%")


def analyze_hunting(u, label):
  """Standard hunting analysis by speed bin with oscillation source identification."""
  print(f"\n{'='*80}")
  print(f"HUNTING ANALYSIS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  angle = u['angle']
  meas  = u['measured_curv']
  app   = u['applied_curv']
  mv    = u['mv_curv']
  dt    = np.median(np.diff(t))

  speed_bins = [
    ("7-10 m/s (15-22 mph)",   7,  10),
    ("10-13 m/s (22-30 mph)", 10,  13),
    ("13-20 m/s (30-45 mph)", 13,  20),
    ("20-29 m/s (45-65 mph)", 20,  29),
    ("29-36 m/s (65-80 mph)", 29,  36),
  ]

  print(f"\n{'Bin':>30s} {'Secs':>6s} {'AngStd':>7s} {'AppStd':>9s} {'MV_Std':>9s} {'OscHz':>7s} {'MV_Hz':>7s} {'Source':>10s}")

  metrics = {}
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi) & (np.abs(meas) < 0.003)
    n = np.sum(mask)
    if n < 50:
      print(f"{name:>30s}  (insufficient, {n} pts)")
      continue
    secs = n * dt

    ang_std = np.std(angle[mask])
    app_std = np.std(app[mask])
    mv_std  = np.std(mv[mask])

    app_sc = np.sum(np.diff(np.sign(app[mask] - np.mean(app[mask]))) != 0)
    mv_sc  = np.sum(np.diff(np.sign(mv[mask] - np.mean(mv[mask]))) != 0)
    app_hz = app_sc / secs / 2
    mv_hz  = mv_sc / secs / 2

    source = "PLANNER" if mv_hz > 0.05 else ("CONTROLLER" if app_hz > mv_hz * 2 else "MIXED")

    print(f"{name:>30s} {secs:>5.0f}s {ang_std:>6.3f}° {app_std:>8.6f} {mv_std:>8.6f} {app_hz:>6.3f}Hz {mv_hz:>6.3f}Hz {source:>10s}")
    metrics[name] = {'secs': secs, 'angle_std': ang_std, 'app_std': app_std,
                     'mv_std': mv_std, 'osc_freq': app_hz, 'mv_osc_freq': mv_hz, 'source': source}

  return metrics


def analyze_low_speed(u, label):
  """Deep dive into sub-7 m/s stop-and-go range."""
  print(f"\n{'='*80}")
  print(f"STOP-AND-GO DEEP DIVE (<7 m/s) — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  app  = u['applied_curv']
  mv   = u['mv_curv']
  angle = u['angle']
  dt   = np.median(np.diff(t))

  bins = [
    ("0.5-1.5 m/s (1-3 mph)",    0.5,  1.5),
    ("1.5-3 m/s  (3-7 mph)",     1.5,  3.0),
    ("3-5 m/s    (7-11 mph)",    3.0,  5.0),
    ("5-7 m/s    (11-16 mph)",   5.0,  7.0),
    ("7-10 m/s   (16-22 mph)",   7.0, 10.0),  # for context
  ]

  print(f"\n{'Bin':<28} {'Secs':>5} {'AngStd':>7} {'AppStd':>10} {'MV_Std':>10} {'AppAmp':>10} {'RMSE(mv-app)':>13} {'OscHz':>7}")
  print('-'*100)

  metrics = {}
  for name, vlo, vhi in bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 10:
      print(f"  {name:<26}  <10pts")
      continue
    secs = n * dt
    app_s  = app[mask]
    mv_s   = mv[mask]
    ang_s  = angle[mask]
    rmse   = np.sqrt(np.mean((mv_s - app_s)**2))
    amp    = np.max(app_s) - np.min(app_s)
    sc     = np.sum(np.diff(np.sign(app_s - np.mean(app_s))) != 0)
    hz     = sc / secs / 2 if secs > 0 else 0
    print(f"  {name:<26} {secs:>5.0f}s {np.std(ang_s):>6.3f}° {np.std(app_s):>10.6f} {np.std(mv_s):>10.6f} {amp:>10.6f} {rmse:>13.6f} {hz:>7.3f}Hz")
    metrics[name] = {'secs': secs, 'app_std': np.std(app_s), 'mv_std': np.std(mv_s),
                     'amp': amp, 'rmse': rmse, 'hz': hz}

  return metrics


def analyze_curves(u, label):
  """Curve overshoot and EPAS tracking quality."""
  print(f"\n{'='*80}")
  print(f"CURVE ANALYSIS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  meas = u['measured_curv']
  app  = u['applied_curv']
  mv   = u['mv_curv']
  dt   = np.median(np.diff(t))

  # Find engaged+moving segments
  mask = lat & (v > 5.0)
  diffs = np.diff(mask.astype(int))
  starts = np.where(diffs == 1)[0] + 1
  ends   = np.where(diffs == -1)[0] + 1
  if mask[0]: starts = np.insert(starts, 0, 0)
  if mask[-1]: ends = np.append(ends, len(mask))
  if len(starts) > len(ends): starts = starts[:len(ends)]

  curves = []
  for s, e in zip(starts, ends):
    if e - s < 20: continue
    seg_meas = meas[s:e]
    seg_app  = app[s:e]
    seg_mv   = mv[s:e]
    seg_v    = v[s:e]
    if not np.any(np.abs(seg_meas) > 0.001): continue
    peak_idx = np.argmax(np.abs(seg_meas))
    sign = np.sign(seg_meas[peak_idx]) if abs(seg_meas[peak_idx]) > 0.0005 else 1.0
    os_val = np.max((seg_meas - seg_app) * sign)
    curves.append({
      'peak_curv': np.max(np.abs(seg_meas)),
      'max_overshoot': os_val,
      'tracking_rmse': np.sqrt(np.mean((seg_mv - seg_app)**2)),
      'epas_rmse': np.sqrt(np.mean((seg_meas - seg_app)**2)),
      'speed_mph': np.mean(seg_v) * 2.237,
    })

  sharp    = [c for c in curves if c['peak_curv'] > 0.004]
  moderate = [c for c in curves if 0.002 < c['peak_curv'] <= 0.004]
  gentle   = [c for c in curves if 0.001 < c['peak_curv'] <= 0.002]

  metrics = {}
  for cat_lbl, subset in [("Sharp (>0.004)", sharp), ("Moderate (0.002-0.004)", moderate), ("Gentle (0.001-0.002)", gentle)]:
    print(f"\n  {cat_lbl}: {len(subset)} events")
    if not subset: continue
    os_vals = [c['max_overshoot'] for c in subset]
    trk     = [c['tracking_rmse'] for c in subset]
    epas    = [c['epas_rmse'] for c in subset]
    overshooters = sum(1 for o in os_vals if o > 0.0005)
    print(f"    Mean overshoot:    {np.mean(os_vals):+.5f} 1/m")
    print(f"    Max overshoot:     {np.max(os_vals):+.5f} 1/m")
    print(f"    Overshoot events:  {overshooters}/{len(subset)} ({100*overshooters/len(subset):.0f}%)")
    print(f"    MV→App RMSE:       {np.mean(trk):.6f}  (filter lag on curves)")
    print(f"    EPAS tracking:     {np.mean(epas):.6f}  (EPAS vs commanded)")
    metrics[cat_lbl] = {'count': len(subset), 'mean_overshoot': np.mean(os_vals),
                        'max_overshoot': np.max(os_vals), 'overshoot_rate': overshooters/len(subset),
                        'tracking_rmse': np.mean(trk), 'epas_rmse': np.mean(epas)}

  # EPAS tracking quality by speed
  print(f"\n  EPAS tracking quality (measured vs applied curvature) by speed:")
  print(f"  {'Bin':<22} {'Secs':>5} {'RMSE':>9} {'Bias':>9} {'AngStd':>8}")
  for name, vlo, vhi in [("7-13 m/s (15-30 mph)", 7, 13), ("13-20 m/s (30-45 mph)", 13, 20),
                           ("20-29 m/s (45-65 mph)", 20, 29), ("29-36 m/s (65-80 mph)", 29, 36)]:
    m = lat & (v >= vlo) & (v < vhi)
    if np.sum(m) < 50: continue
    secs = np.sum(m) * dt
    err  = meas[m] - app[m]
    print(f"    {name:<20} {secs:>5.0f}s {np.sqrt(np.mean(err**2)):>9.6f} {np.mean(err):>+9.6f} {np.std(u['angle'][m]):>7.3f}°")

  return metrics


def analyze_lateral_path_error(u, label):
  """Lane centering quality from modelV2 laneLines — how well we're centered in lane.

  path_y = (laneLines[1].y[0] + laneLines[2].y[0]) / 2
  Convention: +y = right of car, so:
    path_y > 0: lane center is to the right → car is left of center
    path_y < 0: lane center is to the left  → car is right of center
  """
  print(f"\n{'='*80}")
  print(f"LANE CENTERING — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  path_y     = u['path_y']
  lane_width = u['lane_width']
  pos_y1s    = u['pos_y1s']
  ll_prob    = u['ll_prob']
  rl_prob    = u['rl_prob']
  dt = np.median(np.diff(t))

  # Engaged with good lane line confidence (any speed)
  good_ll = lat & (ll_prob > 0.5) & (rl_prob > 0.5) & (lane_width > 2.5) & (lane_width < 5.5)
  n = np.sum(good_ll)
  if n < 100:
    print(f"  Insufficient data with good lane lines ({n} pts)")
    return {}

  py = path_y[good_ll]
  lw = lane_width[good_ll]
  py1s = pos_y1s[good_ll]
  secs = n * dt

  print(f"\n  Good lane confidence: {secs:.0f}s ({100*n/len(t):.1f}% of drive)")
  print(f"  Lane width: mean={np.mean(lw):.2f}m  std={np.std(lw):.2f}m")
  print(f"\n  Lane centering offset (+ = car left of center, - = car right of center):")
  print(f"    Mean:         {np.mean(py):+.3f} m  (systematic bias direction)")
  print(f"    Std:          {np.std(py):.3f} m  (centering consistency)")
  print(f"    P95 abs:      {np.percentile(np.abs(py), 95):.3f} m")
  print(f"    P99 abs:      {np.percentile(np.abs(py), 99):.3f} m")
  print(f"    >0.15m:       {100*np.mean(np.abs(py) > 0.15):.1f}% of time")
  print(f"    >0.30m:       {100*np.mean(np.abs(py) > 0.30):.1f}% of time")
  print(f"    >0.50m:       {100*np.mean(np.abs(py) > 0.50):.1f}% of time")

  # 1s lookahead predicted deviation
  print(f"\n  Predicted path at 1s lookahead (how curved the path ahead is):")
  print(f"    Mean |pos_y1s|: {np.mean(np.abs(py1s)):.3f} m")
  print(f"    Std:            {np.std(py1s):.3f} m")

  # By speed bin
  print(f"\n  Lane centering by speed:")
  for name, vlo, vhi in [("0-15 mph (0-7 m/s)", 0, 7), ("15-30 mph (7-13 m/s)", 7, 13),
                           ("30-45 mph (13-20 m/s)", 13, 20), ("45-65 mph (20-29 m/s)", 20, 29),
                           ("65-80 mph (29-36 m/s)", 29, 36)]:
    m = good_ll & (v >= vlo) & (v < vhi)
    if np.sum(m) < 50: continue
    py_s = path_y[m]
    print(f"    {name:<24}: mean={np.mean(py_s):+.3f}m  std={np.std(py_s):.3f}m  P95={np.percentile(np.abs(py_s),95):.3f}m")

  return {'mean_offset': np.mean(py), 'std_offset': np.std(py), 'p95': np.percentile(np.abs(py), 95)}


def analyze_steering_interventions(u, label):
  """When and why the driver is overriding."""
  print(f"\n{'='*80}")
  print(f"STEERING INTERVENTIONS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  sp  = u['steer_pressed']
  app = u['applied_curv']
  dt  = np.median(np.diff(t))

  engaged_override = lat & sp
  if np.sum(engaged_override) < 5:
    print("  No significant steering interventions while engaged.")
    return {}

  # Find intervention events
  changes = np.diff(engaged_override.astype(int))
  starts = np.where(changes == 1)[0] + 1
  ends   = np.where(changes == -1)[0] + 1
  if engaged_override[0]: starts = np.insert(starts, 0, 0)
  if engaged_override[-1]: ends = np.append(ends, len(engaged_override))
  if len(starts) > len(ends): starts = starts[:len(ends)]

  durations = [(ends[i] - starts[i]) * dt for i in range(len(starts))]
  speeds    = [np.mean(v[starts[i]:ends[i]]) * 2.237 for i in range(len(starts))]
  curv_at   = [np.mean(app[starts[i]:ends[i]]) for i in range(len(starts))]

  print(f"  Total interventions: {len(starts)}")
  print(f"  Mean duration:       {np.mean(durations):.1f}s")
  print(f"  Total time:          {np.sum(durations):.1f}s ({100*np.sum(engaged_override)*dt / (np.sum(lat)*dt):.1f}% of engaged)")
  print(f"\n  Speed distribution of interventions:")
  for vlo, vhi, lbl in [(0,15,"0-15 mph"),(15,30,"15-30 mph"),(30,45,"30-45 mph"),(45,65,"45-65 mph"),(65,100,"65+ mph")]:
    cnt = sum(1 for s in speeds if vlo <= s < vhi)
    print(f"    {lbl}: {cnt} ({100*cnt/len(speeds):.0f}%)")

  return {'n_interventions': len(starts), 'mean_duration': np.mean(durations), 'total_secs': np.sum(durations)}


def analyze_epas_bias(u, label):
  """Detailed EPAS response bias: measured curvature vs applied curvature.

  Positive bias = EPAS over-delivers (turns more than commanded).
  Negative bias = EPAS under-delivers (understeer).
  """
  print(f"\n{'='*80}")
  print(f"EPAS TRACKING & BIAS — {label}")
  print(f"{'='*80}")
  print("  (measured = yawRate/vEgo; applied = carcontroller curvature output)")
  print("  Bias = measured - applied  (+= EPAS over-delivers, -= EPAS under-steers)")

  t, v, lat = u['t'], u['v'], u['lat_active']
  meas = u['measured_curv']
  app  = u['applied_curv']
  dt   = np.median(np.diff(t))

  speed_bins = [
    ("0.5-2 m/s  (1-4 mph)",    0.5,  2.0),
    ("2-4 m/s   (4-9 mph)",     2.0,  4.0),
    ("4-7 m/s   (9-16 mph)",    4.0,  7.0),
    ("7-10 m/s  (16-22 mph)",   7.0, 10.0),
    ("10-13 m/s (22-30 mph)",  10.0, 13.0),
    ("13-20 m/s (30-45 mph)",  13.0, 20.0),
    ("20-29 m/s (45-65 mph)",  20.0, 29.0),
    ("29-36 m/s (65-80 mph)",  29.0, 36.0),
  ]

  print(f"\n{'Bin':<28} {'Secs':>5} {'Bias(mean)':>11} {'Bias(std)':>10} {'RMSE':>10} {'|Bias|>0.001':>13}")
  print('-'*82)

  metrics = {}
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 20:
      continue
    secs = n * dt
    bias = meas[mask] - app[mask]
    rmse = np.sqrt(np.mean(bias**2))
    large_bias_pct = 100 * np.mean(np.abs(bias) > 0.001)
    print(f"  {name:<26} {secs:>5.0f}s {np.mean(bias):>+11.6f} {np.std(bias):>10.6f} {rmse:>10.6f} {large_bias_pct:>12.1f}%")
    metrics[name] = {'bias': np.mean(bias), 'bias_std': np.std(bias), 'rmse': rmse}

  # Curvature-dependent bias: does bias scale with commanded curvature magnitude?
  print(f"\n  Bias by commanded curvature magnitude (engaged, all speeds >4 m/s):")
  mask_h = lat & (v > 4.0)
  if np.sum(mask_h) > 200:
    curv_bins = [(0, 0.001, "straight (<0.001)"), (0.001, 0.003, "gentle (0.001-0.003)"),
                 (0.003, 0.006, "moderate (0.003-0.006)"), (0.006, 0.02, "sharp (>0.006)")]
    print(f"  {'Category':<28} {'Secs':>5} {'Bias':>10} {'EPAS_RMSE':>10}")
    for clo, chi, clbl in curv_bins:
      cm = mask_h & (np.abs(app) >= clo) & (np.abs(app) < chi)
      if np.sum(cm) < 20: continue
      bias = meas[cm] - app[cm]
      secs = np.sum(cm) * dt
      rmse = np.sqrt(np.mean(bias**2))
      print(f"    {clbl:<26} {secs:>5.0f}s {np.mean(bias):>+10.6f} {rmse:>10.6f}")

  return metrics


def analyze_response_latency(u, label):
  """Estimate steering response latency: lag from model curvature change to measured yaw response.

  Uses cross-correlation of delta(mv_curv) with delta(measured_curv) to find peak lag.
  This measures end-to-end: model → filter → controller → EPAS → yaw measurement.
  """
  print(f"\n{'='*80}")
  print(f"STEERING RESPONSE LATENCY — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  mv   = u['mv_curv']
  meas = u['measured_curv']
  app  = u['applied_curv']
  dt   = np.median(np.diff(t))

  speed_bins = [
    ("7-13 m/s (15-30 mph)",   7.0, 13.0),
    ("13-20 m/s (30-45 mph)", 13.0, 20.0),
    ("20-29 m/s (45-65 mph)", 20.0, 29.0),
  ]

  max_lag_steps = int(2.0 / dt)  # search up to 2s lag

  print(f"\n  Cross-correlation peak lag (model→measured yaw, lower=faster response)")
  print(f"  {'Bin':<28} {'Secs':>5} {'Mod→Meas lag':>13} {'Mod→App lag':>13} {'EPAS lag':>10}")
  print('-'*75)

  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 100:
      continue
    secs = n * dt
    mv_s   = mv[mask] - np.mean(mv[mask])
    meas_s = meas[mask] - np.mean(meas[mask])
    app_s  = app[mask] - np.mean(app[mask])

    # Cross-correlate mv_curv with measured_curv
    if len(mv_s) > max_lag_steps * 2:
      xcorr_meas = np.correlate(meas_s, mv_s, mode='full')
      lags = np.arange(len(xcorr_meas)) - (len(mv_s) - 1)
      valid = (lags >= 0) & (lags <= max_lag_steps)
      lag_meas_steps = lags[valid][np.argmax(xcorr_meas[valid])]
      lag_meas_s = lag_meas_steps * dt

      xcorr_app = np.correlate(app_s, mv_s, mode='full')
      lag_app_steps = lags[valid][np.argmax(xcorr_app[valid])]
      lag_app_s = lag_app_steps * dt

      # EPAS lag = measured lag - app lag
      epas_lag = lag_meas_s - lag_app_s
      print(f"  {name:<28} {secs:>5.0f}s {lag_meas_s:>12.2f}s {lag_app_s:>12.2f}s {epas_lag:>9.2f}s")
    else:
      print(f"  {name:<28} (insufficient data)")

  print(f"\n  Note: Mod→App lag = filter/controller delay; EPAS lag = EPAS mechanical response")
  print(f"  Total Mod→Meas = filter + controller + EPAS response chain")


def analyze_curvature_rate(u, label):
  """Analyze curvature rate of change — how fast the model is requesting steering changes."""
  print(f"\n{'='*80}")
  print(f"CURVATURE RATE DEMAND — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  mv_dcurv = u['mv_dcurv']
  app      = u['applied_curv']
  dt       = np.median(np.diff(t))

  speed_bins = [
    ("0-7 m/s   (0-15 mph)",    0.0,  7.0),
    ("7-13 m/s  (15-30 mph)",   7.0, 13.0),
    ("13-20 m/s (30-45 mph)",  13.0, 20.0),
    ("20-29 m/s (45-65 mph)",  20.0, 29.0),
  ]

  print(f"\n  Rate limits (software): ~0.0025 at 5m/s, 0.0012 at 16m/s, 0.00015 at 25m/s (per step/0.05s)")
  print(f"  So max rate (per second): 0.050 at 5m/s, 0.024 at 16m/s, 0.003 at 25m/s")
  print(f"\n  {'Bin':<28} {'Secs':>5} {'Mean|dC|':>10} {'P95|dC|':>10} {'P99|dC|':>10} {'HitLimit%':>10}")
  print('-'*78)

  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 20:
      continue
    secs = n * dt
    dc = np.abs(mv_dcurv[mask])
    # Rate limit at this speed bin midpoint (per second = per step * 20Hz)
    vmid = (vlo + vhi) / 2
    rate_limit = float(np.interp(vmid, [5, 16, 25], [0.0025, 0.0012, 0.00015])) * 20  # /s
    hit_limit_pct = 100 * np.mean(dc > rate_limit * 0.8)  # within 80% of limit
    print(f"  {name:<28} {secs:>5.0f}s {np.mean(dc):>10.4f} {np.percentile(dc,95):>10.4f} {np.percentile(dc,99):>10.4f} {hit_limit_pct:>9.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison(results):
  if len(results) < 2:
    return

  baseline = results[0]
  print(f"\n{'#'*80}")
  print(f"# COMPARISON SUMMARY vs BASELINE ({baseline['label']})")
  print(f"{'#'*80}")

  # Hunting comparison
  print(f"\n{'─'*80}")
  print(f"HUNTING: Applied Curvature Std (lower = less oscillation)")
  print(f"{'─'*80}")
  bins = sorted(set.intersection(*[set(r['hunt'].keys()) for r in results]))
  hdr = f"{'Bin':<30s}"
  for r in results:
    hdr += f"  {r['label'][:12]:>12s}"
  hdr += f"  {'vs Base':>8s}"
  print(hdr)
  for b in bins:
    row = f"{b:<30s}"
    base_std = results[0]['hunt'][b]['app_std']
    for r in results:
      std = r['hunt'][b]['app_std']
      row += f"  {std:>12.6f}"
    last_std = results[-1]['hunt'][b]['app_std']
    pct = (last_std - base_std) / base_std * 100 if base_std > 0 else 0
    row += f"  {pct:>+7.1f}%"
    print(row)

  # Low-speed comparison
  print(f"\n{'─'*80}")
  print(f"STOP-AND-GO: Applied Curvature Std (lower = less hunting)")
  print(f"{'─'*80}")
  ls_bins = sorted(set.intersection(*[set(r['low_speed'].keys()) for r in results]))
  hdr = f"{'Bin':<28s}"
  for r in results:
    hdr += f"  {r['label'][:12]:>12s}"
  hdr += f"  {'vs Base':>8s}"
  print(hdr)
  for b in ls_bins:
    row = f"{b:<28s}"
    base_std = results[0]['low_speed'][b]['app_std']
    for r in results:
      std = r['low_speed'][b].get('app_std', 0)
      row += f"  {std:>12.6f}"
    last_std = results[-1]['low_speed'][b]['app_std']
    pct = (last_std - base_std) / base_std * 100 if base_std > 0 else 0
    row += f"  {pct:>+7.1f}%"
    print(row)

  # Curve overshoot comparison
  print(f"\n{'─'*80}")
  print(f"CURVE OVERSHOOT: Mean overshoot (lower = less EPAS overshoot)")
  print(f"{'─'*80}")
  cat_bins = sorted(set.intersection(*[set(r['curves'].keys()) for r in results]))
  hdr = f"{'Category':<26s}"
  for r in results:
    hdr += f"  {r['label'][:12]:>12s}"
  hdr += f"  {'vs Base':>8s}"
  print(hdr)
  for b in cat_bins:
    row = f"{b:<26s}"
    base_os = results[0]['curves'][b]['mean_overshoot']
    for r in results:
      os = r['curves'][b]['mean_overshoot']
      row += f"  {os:>+12.5f}"
    last_os = results[-1]['curves'][b]['mean_overshoot']
    pct = (last_os - base_os) / abs(base_os) * 100 if abs(base_os) > 1e-6 else 0
    row += f"  {pct:>+7.1f}%"
    print(row)

  # Filter efficacy
  print(f"\n{'─'*80}")
  print(f"FILTER EFFICACY: MV→App reduction % (higher = filter doing more work)")
  print(f"{'─'*80}")
  eff_bins = sorted(set.intersection(*[set(r['filter'].keys()) for r in results if r['filter']]))
  if eff_bins:
    hdr = f"{'Bin':<28s}"
    for r in results:
      hdr += f"  {r['label'][:12]:>12s}"
    print(hdr)
    for b in eff_bins:
      row = f"{b:<28s}"
      for r in results:
        red = r['filter'].get(b, {}).get('reduction', 0)
        row += f"  {red:>+11.1f}%"
      print(row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
  if len(sys.argv) < 2:
    print("Usage: analyze_drive_v4.py <route_dir> [route_dir2 ...]")
    sys.exit(1)

  results = []
  for route_dir in sys.argv[1:]:
    label = os.path.basename(route_dir)
    print(f"\n{'#'*80}")
    print(f"# {label}")
    print(f"{'#'*80}")

    msgs = load_route(route_dir)
    if not msgs:
      continue

    data = extract_data(msgs)
    u    = build_timeline(data)
    if u is None:
      continue

    overview    = analyze_overview(u, label)
    filter_eff  = analyze_filter_efficacy(u, label)
    analyze_fft(u, label)
    hunt        = analyze_hunting(u, label)
    low_speed   = analyze_low_speed(u, label)
    curves      = analyze_curves(u, label)
    lane_center = analyze_lateral_path_error(u, label)
    analyze_steering_interventions(u, label)
    analyze_epas_bias(u, label)
    analyze_response_latency(u, label)
    analyze_curvature_rate(u, label)

    results.append({'label': label, 'overview': overview, 'filter': filter_eff,
                    'hunt': hunt, 'low_speed': low_speed, 'curves': curves,
                    'lane_center': lane_center})

  print_comparison(results)


if __name__ == '__main__':
  main()
