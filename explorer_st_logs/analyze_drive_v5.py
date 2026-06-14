#!/usr/bin/env python3
"""
Explorer ST Drive Analysis — v5
Extends v4 with 5 new deep-analysis sections:
  1. Curve dynamics: entry/exit timing, EPAS bias vs curvature rate (dC/dt)
  2. Override behavior: trigger delta, post-override snap, re-engage quality
  3. Lane centering detail: offset rate of change, centering active%, convergence %
  4. SteerRatio validation + predicted vs desired curvature quality
  5. Comfort metrics: lateral jerk, ISO 2631 lateral acceleration distribution
"""

import sys
import os
import glob
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
  from scipy.ndimage import uniform_filter1d
  from scipy.signal import welch
  HAS_SCIPY = True
except ImportError:
  HAS_SCIPY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openpilot.tools.lib.logreader import LogReader

try:
  from openpilot.selfdrive.modeld.constants import ModelConstants
  T_IDXS = list(ModelConstants.T_IDXS)
except Exception:
  T_IDXS = [i * 0.1 for i in range(33)]  # fallback

# Ford Explorer ST constants
WHEELBASE  = 3.025   # m (119.1 in, 6th-gen 2020+ Explorer)
STEER_RATIO_ASSUMED = 15.0


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


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_data(msgs):
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
      action = mv.action
      data['mv_desired_curv'].append(action.desiredCurvature)

      if len(mv.laneLines) > 2 and len(mv.laneLines[1].y) > 0 and len(mv.laneLines[2].y) > 0:
        ll_y = mv.laneLines[1].y[0]
        rl_y = mv.laneLines[2].y[0]
        data['mv_path_y'].append((ll_y + rl_y) / 2.0)
        data['mv_lane_width'].append(rl_y - ll_y)
      else:
        data['mv_path_y'].append(0.0)
        data['mv_lane_width'].append(3.7)

      if len(mv.position.y) > 10:
        data['mv_pos_y1s'].append(list(mv.position.y)[10])
      else:
        data['mv_pos_y1s'].append(0.0)

      if len(mv.laneLineProbs) > 2:
        data['mv_ll_prob'].append(mv.laneLineProbs[1])
        data['mv_rl_prob'].append(mv.laneLineProbs[2])
      else:
        data['mv_ll_prob'].append(0.0)
        data['mv_rl_prob'].append(0.0)

      # v5: orientationRate.z for predicted curvature quality
      oz = list(mv.orientationRate.z)
      if len(oz) >= 2:
        # Sample at ~0.5s lookahead (index 5 in T_IDXS ≈ 0.5s) and at current (index 0)
        data['mv_orient_z0'].append(oz[0])
        # Interpolate at curvature_lookup_time=0.5s
        idxs = T_IDXS[:len(oz)]
        data['mv_orient_z05'].append(float(np.interp(0.5, idxs, oz)))
      else:
        data['mv_orient_z0'].append(0.0)
        data['mv_orient_z05'].append(0.0)

    elif w == 'selfdriveState':
      sd = msg.selfdriveState
      data['sd_t'].append(t)
      data['sd_active'].append(sd.active)

  for k in data:
    data[k] = np.array(data[k], dtype=float if k not in ('steering_pressed',) else bool)

  if len(data['mv_desired_curv']) > 1:
    dt_mv = np.diff(data['mv_t'])
    dt_mv = np.where(dt_mv > 0, dt_mv, 0.05)
    dcurv = np.diff(data['mv_desired_curv']) / dt_mv
    data['mv_dcurv'] = np.concatenate([[0.0], dcurv])
  else:
    data['mv_dcurv'] = np.zeros_like(data['mv_desired_curv'])
  return data


# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline(data):
  t = data['cs_t']
  if len(t) < 2:
    return None

  def interp(src_t, src_v, default=0.0):
    if len(src_t) > 1:
      return np.interp(t, src_t, src_v)
    return np.full(len(t), default)

  v   = data['v_ego']
  yaw = data['yaw_rate']
  measured_curv = np.where(v > 0.5, -yaw / np.maximum(v, 0.1), 0.0)

  lat_active  = interp(data['cc_t'], data['lat_active'].astype(float)) > 0.5
  cc_curv     = interp(data['cc_t'], data['cc_curvature'])
  co_curv     = interp(data['co_t'], data['co_curvature']) if len(data['co_t']) > 1 else cc_curv
  mv_curv     = interp(data['mv_t'], data['mv_desired_curv'])
  ctrl_curv   = interp(data['ctrl_t'], data['ctrl_desired_curv'])
  path_y      = interp(data['mv_t'], data['mv_path_y'])
  lane_width  = interp(data['mv_t'], data['mv_lane_width'])
  pos_y1s     = interp(data['mv_t'], data['mv_pos_y1s'])
  ll_prob     = interp(data['mv_t'], data['mv_ll_prob'])
  rl_prob     = interp(data['mv_t'], data['mv_rl_prob'])
  mv_dcurv    = interp(data['mv_t'], data['mv_dcurv'])
  steer_press = interp(data['cs_t'], data['steering_pressed'].astype(float)) > 0.5

  # v5: predicted curvature from orientationRate (what the PC blend uses)
  orient_z0   = interp(data['mv_t'], data['mv_orient_z0'])
  orient_z05  = interp(data['mv_t'], data['mv_orient_z05'])
  predicted_curv = np.where(v > 0.5, orient_z05 / np.maximum(v, 0.1), 0.0)

  applied_curv = co_curv if len(data['co_t']) > 1 else cc_curv

  # v5: derived signals
  dt      = np.median(np.diff(t))
  app_dcurv = np.gradient(applied_curv, dt)           # d(applied)/dt (1/m/s)
  lat_accel_cmd  = applied_curv * v**2                # commanded lateral accel (m/s²)
  lat_accel_meas = yaw * v                            # measured lateral accel (m/s²)
  lat_jerk  = np.gradient(lat_accel_meas, dt)         # lateral jerk (m/s³)

  return {
    't': t, 'v': v, 'angle': data['steering_angle'],
    'torque': data['steering_torque'], 'steer_pressed': steer_press,
    'a_ego': data['a_ego'], 'yaw': yaw,
    'lat_active': lat_active, 'measured_curv': measured_curv,
    'applied_curv': applied_curv, 'cc_curv': cc_curv,
    'mv_curv': mv_curv, 'ctrl_curv': ctrl_curv,
    'path_y': path_y, 'lane_width': lane_width, 'pos_y1s': pos_y1s,
    'll_prob': ll_prob, 'rl_prob': rl_prob,
    'mv_dcurv': mv_dcurv, 'predicted_curv': predicted_curv,
    'app_dcurv': app_dcurv,
    'lat_accel_cmd': lat_accel_cmd, 'lat_accel_meas': lat_accel_meas,
    'lat_jerk': lat_jerk,
  }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(sig, window):
  """Simple moving average."""
  if HAS_SCIPY:
    return uniform_filter1d(sig.astype(float), size=max(1, window))
  kernel = np.ones(max(1, window)) / max(1, window)
  return np.convolve(sig.astype(float), kernel, mode='same')


def _find_curve_events(t, v, lat, applied_curv, min_peak=0.003, min_dur=0.4, min_sep=3.0):
  """Return list of dicts {start, end, peak, peak_val} for curve events."""
  dt = np.median(np.diff(t))
  smooth_win = max(1, int(0.2 / dt))
  smoothed = _smooth(np.abs(applied_curv), smooth_win)
  engaged  = lat & (v > 5.0)

  events, in_curve, start_idx = [], False, 0
  for i in range(len(t)):
    if engaged[i] and smoothed[i] > min_peak * 0.4:
      if not in_curve:
        in_curve, start_idx = True, i
    else:
      if in_curve:
        end_idx = i
        dur = (end_idx - start_idx) * dt
        if dur >= min_dur:
          seg  = applied_curv[start_idx:end_idx]
          pidx = np.argmax(np.abs(seg))
          pval = seg[pidx]
          if abs(pval) >= min_peak:
            events.append({'start': start_idx, 'end': end_idx,
                           'peak': start_idx + pidx, 'peak_val': pval})
        in_curve = False
  if in_curve:
    end_idx = len(t)
    seg  = applied_curv[start_idx:end_idx]
    pidx = np.argmax(np.abs(seg))
    pval = seg[pidx]
    if abs(pval) >= min_peak and (end_idx - start_idx) * dt >= min_dur:
      events.append({'start': start_idx, 'end': end_idx,
                     'peak': start_idx + pidx, 'peak_val': pval})

  # Merge nearby events
  merged = []
  for ev in events:
    if merged and (ev['start'] - merged[-1]['end']) * dt < min_sep:
      merged[-1]['end'] = ev['end']
      if abs(ev['peak_val']) > abs(merged[-1]['peak_val']):
        merged[-1]['peak'] = ev['peak']
        merged[-1]['peak_val'] = ev['peak_val']
    else:
      merged.append(dict(ev))
  return merged


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING V4 ANALYSIS SECTIONS (unchanged)
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
  changes = np.diff(lat.astype(int))
  engages = np.sum(changes == 1)
  disengages = np.sum(changes == -1)
  print(f"  Engage events:     {engages}")
  print(f"  Disengage events:  {disengages}")
  sp = u['steer_pressed']
  interventions = np.diff((lat & sp).astype(int))
  n_int = np.sum(interventions == 1)
  int_secs = np.sum(lat & sp) * dt
  print(f"  Steer overrides:   {n_int} events  ({int_secs:.1f}s total, {100*int_secs/eng_time:.1f}% of engaged time)")
  bins = [(0,7,"0-15 mph"),(7,13,"15-30 mph"),(13,20,"30-45 mph"),
          (20,29,"45-65 mph"),(29,36,"65-80 mph"),(36,50,"80+ mph")]
  print(f"\n  Speed distribution (engaged):")
  eng_v = v[lat]
  for vlo, vhi, lbl in bins:
    pct = np.mean((eng_v >= vlo) & (eng_v < vhi)) * 100
    print(f"    {lbl:>10s}: {pct:5.1f}%  {'█'*int(pct/2)}")
  a = u['a_ego'][lat]
  print(f"\n  Traffic characterization (engaged):")
  print(f"    Hard braking (<-2.0 m/s²): {np.sum(a < -2.0)*dt:.1f}s")
  print(f"    Hard acceleration (>1.5):  {np.sum(a > 1.5)*dt:.1f}s")
  print(f"    Mean |accel|:              {np.mean(np.abs(a)):.3f} m/s²")
  return {'duration': duration, 'eng_time': eng_time, 'engages': engages,
          'n_interventions': n_int, 'intervention_secs': int_secs}


def analyze_filter_efficacy(u, label):
  print(f"\n{'='*80}")
  print(f"FILTER EFFICACY — {label}")
  print(f"{'='*80}")
  print("  (modelV2 = pre-filter planner output; applied = post-filter carcontroller output)")
  t, v, lat = u['t'], u['v'], u['lat_active']
  mv, app = u['mv_curv'], u['applied_curv']
  speed_bins = [
    ("0.5-2 m/s  (1-4 mph)",   0.5,  2.0), ("2-4 m/s   (4-9 mph)",    2.0,  4.0),
    ("4-7 m/s   (9-16 mph)",   4.0,  7.0), ("7-10 m/s  (16-22 mph)",  7.0, 10.0),
    ("10-13 m/s (22-30 mph)", 10.0, 13.0), ("13-20 m/s (30-45 mph)", 13.0, 20.0),
    ("20-29 m/s (45-65 mph)", 20.0, 29.0),
  ]
  print(f"\n{'Bin':<28} {'Secs':>5} {'MV_Std':>9} {'App_Std':>9} {'Reduction':>10} {'RMSE(mv-app)':>13} {'Filter_Active':>14}")
  print('-'*95)
  metrics = {}
  dt = np.median(np.diff(t))
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 20:
      print(f"  {name:<26} <20pts"); continue
    secs = n * dt
    mv_std, app_std = np.std(mv[mask]), np.std(app[mask])
    rmse = np.sqrt(np.mean((mv[mask] - app[mask])**2))
    reduction = (mv_std - app_std) / mv_std * 100 if mv_std > 0 else 0
    diverged = np.sum(np.abs(mv[mask] - app[mask]) > 0.0002) / n * 100
    print(f"  {name:<26} {secs:>5.0f}s {mv_std:>9.6f} {app_std:>9.6f} {reduction:>+9.1f}% {rmse:>13.6f} {diverged:>13.1f}%")
    metrics[name] = {'mv_std': mv_std, 'app_std': app_std, 'rmse': rmse, 'reduction': reduction}
  return metrics


def analyze_fft(u, label):
  print(f"\n{'='*80}")
  print(f"OSCILLATION POWER SPECTRUM (FFT) — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  app, mv = u['applied_curv'], u['mv_curv']
  dt = np.median(np.diff(t))
  speed_bins = [("2-4 m/s   (4-9 mph)", 2.0, 4.0), ("7-10 m/s  (16-22 mph)", 7.0, 10.0),
                ("13-20 m/s (30-45 mph)", 13.0, 20.0), ("20-29 m/s (45-65 mph)", 20.0, 29.0)]
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 100: continue
    if HAS_SCIPY:
      freqs, psd_app = welch(app[mask], fs=1/dt, nperseg=min(256, n//2))
      _, psd_mv  = welch(mv[mask],  fs=1/dt, nperseg=min(256, n//2))
    else:
      sig = app[mask] - np.mean(app[mask])
      freqs = np.fft.rfftfreq(n, d=dt)
      psd_app = np.abs(np.fft.rfft(sig))**2
      psd_mv  = np.abs(np.fft.rfft(mv[mask] - np.mean(mv[mask])))**2
    valid = freqs > 0.05
    top_idx = np.argsort(psd_app[valid])[-3:][::-1]
    top_freqs = freqs[valid][top_idx]
    total_app = np.sum(psd_app[valid])
    total_mv  = np.sum(psd_mv[valid])
    print(f"\n  {name}  ({n*dt:.0f}s)")
    print(f"  {'Freq':>8s}  {'Period':>8s}  {'App Power%':>11s}  {'MV Power%':>11s}")
    for f in top_freqs:
      p_app = float(np.interp(f, freqs[valid], psd_app[valid]))
      p_mv  = float(np.interp(f, freqs[valid], psd_mv[valid]))
      period = 1/f if f > 0 else 999
      print(f"    {f:>6.3f} Hz  {period:>6.1f}s    {100*p_app/total_app:>9.1f}%   {100*p_mv/total_mv if total_mv>0 else 0:>9.1f}%")


def analyze_hunting(u, label):
  print(f"\n{'='*80}")
  print(f"HUNTING ANALYSIS — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  angle, meas, app, mv = u['angle'], u['measured_curv'], u['applied_curv'], u['mv_curv']
  dt = np.median(np.diff(t))
  speed_bins = [("7-10 m/s (15-22 mph)", 7, 10), ("10-13 m/s (22-30 mph)", 10, 13),
                ("13-20 m/s (30-45 mph)", 13, 20), ("20-29 m/s (45-65 mph)", 20, 29),
                ("29-36 m/s (65-80 mph)", 29, 36)]
  print(f"\n{'Bin':>30s} {'Secs':>6s} {'AngStd':>7s} {'AppStd':>9s} {'MV_Std':>9s} {'OscHz':>7s} {'MV_Hz':>7s} {'Source':>10s}")
  metrics = {}
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi) & (np.abs(meas) < 0.003)
    n = np.sum(mask)
    if n < 50:
      print(f"{name:>30s}  (insufficient, {n} pts)"); continue
    secs = n * dt
    ang_std, app_std, mv_std = np.std(angle[mask]), np.std(app[mask]), np.std(mv[mask])
    app_sc = np.sum(np.diff(np.sign(app[mask] - np.mean(app[mask]))) != 0)
    mv_sc  = np.sum(np.diff(np.sign(mv[mask]  - np.mean(mv[mask])))  != 0)
    app_hz, mv_hz = app_sc / secs / 2, mv_sc / secs / 2
    source = "PLANNER" if mv_hz > 0.05 else ("CONTROLLER" if app_hz > mv_hz * 2 else "MIXED")
    print(f"{name:>30s} {secs:>5.0f}s {ang_std:>6.3f}° {app_std:>8.6f} {mv_std:>8.6f} {app_hz:>6.3f}Hz {mv_hz:>6.3f}Hz {source:>10s}")
    metrics[name] = {'secs': secs, 'angle_std': ang_std, 'app_std': app_std,
                     'mv_std': mv_std, 'osc_freq': app_hz, 'mv_osc_freq': mv_hz, 'source': source}
  return metrics


def analyze_low_speed(u, label):
  print(f"\n{'='*80}")
  print(f"STOP-AND-GO DEEP DIVE (<7 m/s) — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  app, mv, angle = u['applied_curv'], u['mv_curv'], u['angle']
  dt = np.median(np.diff(t))
  bins = [("0.5-1.5 m/s (1-3 mph)", 0.5, 1.5), ("1.5-3 m/s  (3-7 mph)", 1.5, 3.0),
          ("3-5 m/s    (7-11 mph)", 3.0, 5.0), ("5-7 m/s    (11-16 mph)", 5.0, 7.0),
          ("7-10 m/s   (16-22 mph)", 7.0, 10.0)]
  print(f"\n{'Bin':<28} {'Secs':>5} {'AngStd':>7} {'AppStd':>10} {'MV_Std':>10} {'AppAmp':>10} {'RMSE(mv-app)':>13} {'OscHz':>7}")
  print('-'*100)
  metrics = {}
  for name, vlo, vhi in bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 10:
      print(f"  {name:<26}  <10pts"); continue
    secs = n * dt
    app_s, mv_s, ang_s = app[mask], mv[mask], angle[mask]
    rmse = np.sqrt(np.mean((mv_s - app_s)**2))
    amp  = np.max(app_s) - np.min(app_s)
    sc   = np.sum(np.diff(np.sign(app_s - np.mean(app_s))) != 0)
    hz   = sc / secs / 2 if secs > 0 else 0
    print(f"  {name:<26} {secs:>5.0f}s {np.std(ang_s):>6.3f}° {np.std(app_s):>10.6f} {np.std(mv_s):>10.6f} {amp:>10.6f} {rmse:>13.6f} {hz:>7.3f}Hz")
    metrics[name] = {'secs': secs, 'app_std': np.std(app_s), 'mv_std': np.std(mv_s),
                     'amp': amp, 'rmse': rmse, 'hz': hz}
  return metrics


def analyze_curves(u, label):
  print(f"\n{'='*80}")
  print(f"CURVE ANALYSIS — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  meas, app, mv = u['measured_curv'], u['applied_curv'], u['mv_curv']
  dt = np.median(np.diff(t))
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
    seg_meas, seg_app, seg_mv, seg_v = meas[s:e], app[s:e], mv[s:e], v[s:e]
    if not np.any(np.abs(seg_meas) > 0.001): continue
    peak_idx = np.argmax(np.abs(seg_meas))
    sign = np.sign(seg_meas[peak_idx]) if abs(seg_meas[peak_idx]) > 0.0005 else 1.0
    os_val = np.max((seg_meas - seg_app) * sign)
    curves.append({'peak_curv': np.max(np.abs(seg_meas)), 'max_overshoot': os_val,
                   'tracking_rmse': np.sqrt(np.mean((seg_mv - seg_app)**2)),
                   'epas_rmse': np.sqrt(np.mean((seg_meas - seg_app)**2)),
                   'speed_mph': np.mean(seg_v) * 2.237})
  sharp    = [c for c in curves if c['peak_curv'] > 0.004]
  moderate = [c for c in curves if 0.002 < c['peak_curv'] <= 0.004]
  gentle   = [c for c in curves if 0.001 < c['peak_curv'] <= 0.002]
  metrics = {}
  for cat_lbl, subset in [("Sharp (>0.004)", sharp), ("Moderate (0.002-0.004)", moderate),
                           ("Gentle (0.001-0.002)", gentle)]:
    print(f"\n  {cat_lbl}: {len(subset)} events")
    if not subset: continue
    os_vals = [c['max_overshoot'] for c in subset]
    trk, epas = [c['tracking_rmse'] for c in subset], [c['epas_rmse'] for c in subset]
    oshooters = sum(1 for o in os_vals if o > 0.0005)
    print(f"    Mean overshoot:    {np.mean(os_vals):+.5f} 1/m")
    print(f"    Max overshoot:     {np.max(os_vals):+.5f} 1/m")
    print(f"    Overshoot events:  {oshooters}/{len(subset)} ({100*oshooters/len(subset):.0f}%)")
    print(f"    MV→App RMSE:       {np.mean(trk):.6f}  (filter lag on curves)")
    print(f"    EPAS tracking:     {np.mean(epas):.6f}  (EPAS vs commanded)")
    metrics[cat_lbl] = {'count': len(subset), 'mean_overshoot': np.mean(os_vals),
                        'max_overshoot': np.max(os_vals), 'overshoot_rate': oshooters/len(subset),
                        'tracking_rmse': np.mean(trk), 'epas_rmse': np.mean(epas)}
  print(f"\n  EPAS tracking quality (measured vs applied curvature) by speed:")
  print(f"  {'Bin':<22} {'Secs':>5} {'RMSE':>9} {'Bias':>9} {'AngStd':>8}")
  for name, vlo, vhi in [("7-13 m/s (15-30 mph)", 7, 13), ("13-20 m/s (30-45 mph)", 13, 20),
                           ("20-29 m/s (45-65 mph)", 20, 29), ("29-36 m/s (65-80 mph)", 29, 36)]:
    m = lat & (v >= vlo) & (v < vhi)
    if np.sum(m) < 50: continue
    secs = np.sum(m) * dt
    err = meas[m] - app[m]
    print(f"    {name:<20} {secs:>5.0f}s {np.sqrt(np.mean(err**2)):>9.6f} {np.mean(err):>+9.6f} {np.std(u['angle'][m]):>7.3f}°")
  return metrics


def analyze_lateral_path_error(u, label):
  print(f"\n{'='*80}")
  print(f"LANE CENTERING — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  path_y, lane_width, pos_y1s = u['path_y'], u['lane_width'], u['pos_y1s']
  ll_prob, rl_prob = u['ll_prob'], u['rl_prob']
  dt = np.median(np.diff(t))
  good_ll = lat & (ll_prob > 0.5) & (rl_prob > 0.5) & (lane_width > 2.5) & (lane_width < 5.5)
  n = np.sum(good_ll)
  if n < 100:
    print(f"  Insufficient data with good lane lines ({n} pts)"); return {}
  py = path_y[good_ll]
  lw = lane_width[good_ll]
  py1s = pos_y1s[good_ll]
  secs = n * dt
  print(f"\n  Good lane confidence: {secs:.0f}s ({100*n/len(t):.1f}% of drive)")
  print(f"  Lane width: mean={np.mean(lw):.2f}m  std={np.std(lw):.2f}m")
  print(f"\n  Lane centering offset (+ = car left of center, - = car right of center):")
  print(f"    Mean:         {np.mean(py):+.3f} m")
  print(f"    Std:          {np.std(py):.3f} m")
  print(f"    P95 abs:      {np.percentile(np.abs(py), 95):.3f} m")
  print(f"    P99 abs:      {np.percentile(np.abs(py), 99):.3f} m")
  print(f"    >0.15m:       {100*np.mean(np.abs(py) > 0.15):.1f}% of time")
  print(f"    >0.30m:       {100*np.mean(np.abs(py) > 0.30):.1f}% of time")
  print(f"    >0.50m:       {100*np.mean(np.abs(py) > 0.50):.1f}% of time")
  print(f"\n  Predicted path at 1s lookahead:")
  print(f"    Mean |pos_y1s|: {np.mean(np.abs(py1s)):.3f} m  Std: {np.std(py1s):.3f} m")
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
  print(f"\n{'='*80}")
  print(f"STEERING INTERVENTIONS — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  sp, app = u['steer_pressed'], u['applied_curv']
  dt = np.median(np.diff(t))
  engaged_override = lat & sp
  if np.sum(engaged_override) < 5:
    print("  No significant steering interventions while engaged."); return {}
  changes = np.diff(engaged_override.astype(int))
  starts = np.where(changes == 1)[0] + 1
  ends   = np.where(changes == -1)[0] + 1
  if engaged_override[0]: starts = np.insert(starts, 0, 0)
  if engaged_override[-1]: ends = np.append(ends, len(engaged_override))
  if len(starts) > len(ends): starts = starts[:len(ends)]
  durations = [(ends[i] - starts[i]) * dt for i in range(len(starts))]
  speeds    = [np.mean(v[starts[i]:ends[i]]) * 2.237 for i in range(len(starts))]
  print(f"  Total interventions: {len(starts)}")
  print(f"  Mean duration:       {np.mean(durations):.1f}s")
  print(f"  Total time:          {np.sum(durations):.1f}s ({100*np.sum(engaged_override)*dt/(np.sum(lat)*dt):.1f}% of engaged)")
  print(f"\n  Speed distribution of interventions:")
  for vlo, vhi, lbl in [(0,15,"0-15 mph"),(15,30,"15-30 mph"),(30,45,"30-45 mph"),(45,65,"45-65 mph"),(65,100,"65+ mph")]:
    cnt = sum(1 for s in speeds if vlo <= s < vhi)
    print(f"    {lbl}: {cnt} ({100*cnt/len(speeds):.0f}%)")
  return {'n_interventions': len(starts), 'mean_duration': np.mean(durations), 'total_secs': np.sum(durations)}


def analyze_epas_bias(u, label):
  print(f"\n{'='*80}")
  print(f"EPAS TRACKING & BIAS — {label}")
  print(f"{'='*80}")
  print("  (measured = yawRate/vEgo; applied = carcontroller curvature output)")
  print("  Bias = measured - applied  (+= EPAS over-delivers, -= EPAS under-steers)")
  t, v, lat = u['t'], u['v'], u['lat_active']
  meas, app = u['measured_curv'], u['applied_curv']
  dt = np.median(np.diff(t))
  speed_bins = [("0.5-2 m/s  (1-4 mph)", 0.5, 2.0), ("2-4 m/s   (4-9 mph)", 2.0, 4.0),
                ("4-7 m/s   (9-16 mph)", 4.0, 7.0), ("7-10 m/s  (16-22 mph)", 7.0, 10.0),
                ("10-13 m/s (22-30 mph)", 10.0, 13.0), ("13-20 m/s (30-45 mph)", 13.0, 20.0),
                ("20-29 m/s (45-65 mph)", 20.0, 29.0), ("29-36 m/s (65-80 mph)", 29.0, 36.0)]
  print(f"\n{'Bin':<28} {'Secs':>5} {'Bias(mean)':>11} {'Bias(std)':>10} {'RMSE':>10} {'|Bias|>0.001':>13}")
  print('-'*82)
  metrics = {}
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 20: continue
    secs = n * dt
    bias = meas[mask] - app[mask]
    rmse = np.sqrt(np.mean(bias**2))
    large_pct = 100 * np.mean(np.abs(bias) > 0.001)
    print(f"  {name:<26} {secs:>5.0f}s {np.mean(bias):>+11.6f} {np.std(bias):>10.6f} {rmse:>10.6f} {large_pct:>12.1f}%")
    metrics[name] = {'bias': np.mean(bias), 'bias_std': np.std(bias), 'rmse': rmse}
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
      print(f"    {clbl:<26} {secs:>5.0f}s {np.mean(bias):>+10.6f} {np.sqrt(np.mean(bias**2)):>10.6f}")
  return metrics


def analyze_response_latency(u, label):
  print(f"\n{'='*80}")
  print(f"STEERING RESPONSE LATENCY — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  mv, meas, app = u['mv_curv'], u['measured_curv'], u['applied_curv']
  dt = np.median(np.diff(t))
  speed_bins = [("7-13 m/s (15-30 mph)", 7.0, 13.0), ("13-20 m/s (30-45 mph)", 13.0, 20.0),
                ("20-29 m/s (45-65 mph)", 20.0, 29.0)]
  max_lag_steps = int(2.0 / dt)
  print(f"\n  Cross-correlation peak lag (model→measured yaw, lower=faster response)")
  print(f"  {'Bin':<28} {'Secs':>5} {'Mod→Meas lag':>13} {'Mod→App lag':>13} {'EPAS lag':>10}")
  print('-'*75)
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 100: continue
    secs = n * dt
    mv_s   = mv[mask] - np.mean(mv[mask])
    meas_s = meas[mask] - np.mean(meas[mask])
    app_s  = app[mask] - np.mean(app[mask])
    if len(mv_s) > max_lag_steps * 2:
      xcorr = np.correlate(meas_s, mv_s, mode='full')
      lags  = np.arange(len(xcorr)) - (len(mv_s) - 1)
      valid = (lags >= 0) & (lags <= max_lag_steps)
      lag_meas_s = lags[valid][np.argmax(xcorr[valid])] * dt
      xcorr_app  = np.correlate(app_s, mv_s, mode='full')
      lag_app_s  = lags[valid][np.argmax(xcorr_app[valid])] * dt
      print(f"  {name:<28} {secs:>5.0f}s {lag_meas_s:>12.2f}s {lag_app_s:>12.2f}s {lag_meas_s-lag_app_s:>9.2f}s")
  print(f"\n  Note: Mod→App lag = filter/controller delay; EPAS lag = EPAS mechanical response")


def analyze_curvature_rate(u, label):
  print(f"\n{'='*80}")
  print(f"CURVATURE RATE DEMAND — {label}")
  print(f"{'='*80}")
  t, v, lat = u['t'], u['v'], u['lat_active']
  mv_dcurv, app = u['mv_dcurv'], u['applied_curv']
  dt = np.median(np.diff(t))
  speed_bins = [("0-7 m/s   (0-15 mph)", 0.0, 7.0), ("7-13 m/s  (15-30 mph)", 7.0, 13.0),
                ("13-20 m/s (30-45 mph)", 13.0, 20.0), ("20-29 m/s (45-65 mph)", 20.0, 29.0)]
  print(f"\n  Rate limits: ~0.0025 at 5m/s, 0.0012 at 16m/s, 0.00015 at 25m/s (per step/0.05s)")
  print(f"\n  {'Bin':<28} {'Secs':>5} {'Mean|dC|':>10} {'P95|dC|':>10} {'P99|dC|':>10} {'HitLimit%':>10}")
  print('-'*78)
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 20: continue
    secs = n * dt
    dc = np.abs(mv_dcurv[mask])
    vmid = (vlo + vhi) / 2
    rate_limit = float(np.interp(vmid, [5, 16, 25], [0.0025, 0.0012, 0.00015])) * 20
    hit_pct = 100 * np.mean(dc > rate_limit * 0.8)
    print(f"  {name:<28} {secs:>5.0f}s {np.mean(dc):>10.4f} {np.percentile(dc,95):>10.4f} {np.percentile(dc,99):>10.4f} {hit_pct:>9.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# NEW V5 ANALYSIS SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_curve_dynamics(u, label):
  """
  V5 Section 1: Curve dynamics
  A) EPAS bias vs curvature rate (dC/dt) — distinguishes static gain error from dynamic lag
  B) Curve event phase analysis — where in the turn does overshoot happen?
  """
  print(f"\n{'='*80}")
  print(f"CURVE DYNAMICS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  meas      = u['measured_curv']
  app       = u['applied_curv']
  app_dcurv = u['app_dcurv']    # d(applied)/dt
  dt        = np.median(np.diff(t))

  # ── A: EPAS bias binned by curvature rate (dC/dt) ──────────────────────────
  print(f"\n  A) EPAS Bias vs Curvature Rate (dC/dt)")
  print(f"  If bias flat across all rates → static gain error → feed-forward correction will work")
  print(f"  If bias rises with rate → dynamic lag/overshoot → harder to fix with feed-forward")
  print()

  rate_bins = [
    ("zero/static (<0.002/s)",     0.000, 0.002),
    ("slow       (0.002-0.01/s)",  0.002, 0.010),
    ("medium     (0.01-0.05/s)",   0.010, 0.050),
    ("fast       (0.05-0.20/s)",   0.050, 0.200),
    ("very fast  (>0.20/s)",       0.200, 99.0 ),
  ]

  mask_base = lat & (v > 7.0)
  print(f"  {'Rate bin':<28} {'Secs':>5} {'Bias(mean)':>11} {'Bias(std)':>10} {'RMSE':>10}  {'Interpretation'}")
  print('  ' + '-'*82)
  bias_by_rate = {}
  for rname, rlo, rhi in rate_bins:
    m = mask_base & (np.abs(app_dcurv) >= rlo) & (np.abs(app_dcurv) < rhi)
    n = np.sum(m)
    if n < 20: continue
    secs = n * dt
    bias = meas[m] - app[m]
    bmean, bstd = np.mean(bias), np.std(bias)
    rmse = np.sqrt(np.mean(bias**2))
    bias_by_rate[rname] = bmean
    print(f"  {rname:<28} {secs:>5.0f}s {bmean:>+11.6f} {bstd:>10.6f} {rmse:>10.6f}")

  # Summarize pattern
  if len(bias_by_rate) >= 3:
    vals = list(bias_by_rate.values())
    bias_range = max(vals) - min(vals)
    if bias_range < 0.001:
      print(f"\n  → Bias variation across rates: {bias_range:.6f}  (STATIC GAIN — feed-forward correction should work well)")
    elif bias_range < 0.003:
      print(f"\n  → Bias variation across rates: {bias_range:.6f}  (MIXED — some dynamic component but feed-forward will help)")
    else:
      print(f"\n  → Bias variation across rates: {bias_range:.6f}  (DYNAMIC LAG — significant rate-dependent overshoot)")

  # ── B: Curve event phase analysis ──────────────────────────────────────────
  print(f"\n  B) Curve Event Phase Analysis")
  print(f"  Entry=30% rise  Peak=apex  Exit=30% fall  (positive = EPAS over-delivers)")
  print()

  events = _find_curve_events(t, v, lat, app, min_peak=0.002, min_dur=0.4)
  if not events:
    print(f"  No suitable curve events found.")
    return {}

  sharp_ev    = [e for e in events if abs(e['peak_val']) > 0.004]
  moderate_ev = [e for e in events if 0.002 < abs(e['peak_val']) <= 0.004]

  phase_summary = {}
  for cat_lbl, ev_list in [("Sharp (>0.004 1/m)", sharp_ev), ("Moderate (0.002-0.004 1/m)", moderate_ev)]:
    if len(ev_list) < 1:
      print(f"  {cat_lbl}: no events"); continue

    entry_bias_vals, peak_bias_vals, exit_bias_vals = [], [], []
    entry_lag_ms, peak_lag_ms = [], []

    for ev in ev_list:
      peak = ev['peak']
      sign = np.sign(ev['peak_val'])
      peak_app_val = app[peak]
      if abs(peak_app_val) < 0.001: continue

      # Entry: find where applied first crosses 30% of peak on the rise before peak
      threshold = 0.30 * abs(peak_app_val)
      pre_range = range(max(0, peak - int(3.0/dt)), peak)
      app_pre  = app[list(pre_range)]  * sign
      meas_pre = meas[list(pre_range)] * sign

      app_entry_idx  = next((i for i, x in enumerate(app_pre)  if x >= threshold), None)
      meas_entry_idx = next((i for i, x in enumerate(meas_pre) if x >= threshold), None)

      if app_entry_idx is not None and meas_entry_idx is not None:
        lag_ms = (meas_entry_idx - app_entry_idx) * dt * 1000
        entry_lag_ms.append(lag_ms)
        # Bias at entry point
        bi = list(pre_range)[app_entry_idx]
        entry_bias_vals.append((meas[bi] - app[bi]) * sign)

      # Peak bias
      peak_bias_vals.append((meas[peak] - app[peak]) * sign)

      # Peak lag: find where measured peaks after commanded peak
      post_range = range(peak, min(len(t), peak + int(2.0/dt)))
      meas_post  = meas[list(post_range)] * sign
      meas_peak_offset = np.argmax(meas_post)
      peak_lag_ms.append(meas_peak_offset * dt * 1000)

      # Exit: 30% of peak on fall after peak
      exit_range = range(peak, min(len(t), peak + int(3.0/dt)))
      app_post  = app[list(exit_range)]  * sign
      meas_post2 = meas[list(exit_range)] * sign
      app_exit_idx = next((i for i, x in enumerate(app_post) if x <= threshold), None)
      if app_exit_idx is not None:
        ei = list(exit_range)[app_exit_idx]
        exit_bias_vals.append((meas[ei] - app[ei]) * sign)

    n_ev = len(ev_list)
    print(f"  {cat_lbl} ({n_ev} events):")
    if entry_bias_vals:
      print(f"    Entry bias (at 30% rise):  {np.mean(entry_bias_vals):+.6f} 1/m  "
            f"({'+' if np.mean(entry_bias_vals)>0 else ''}{'EPAS ahead' if np.mean(entry_bias_vals)>0 else 'EPAS behind'})")
    if entry_lag_ms:
      print(f"    Entry lag (app→meas):      {np.mean(entry_lag_ms):+.0f} ms  "
            f"({'meas lags' if np.mean(entry_lag_ms)>0 else 'meas leads'})")
    if peak_bias_vals:
      print(f"    Peak bias (at apex):       {np.mean(peak_bias_vals):+.6f} 1/m  ← primary overshoot")
    if peak_lag_ms:
      print(f"    Measured peak lag:         {np.mean(peak_lag_ms):+.0f} ms after commanded peak")
    if exit_bias_vals:
      print(f"    Exit bias (at 30% fall):   {np.mean(exit_bias_vals):+.6f} 1/m  "
            f"({'still overshooting on exit' if np.mean(exit_bias_vals)>0 else 'settled by exit'})")

    phase_summary[cat_lbl] = {
      'n': n_ev,
      'entry_bias': np.mean(entry_bias_vals) if entry_bias_vals else None,
      'peak_bias': np.mean(peak_bias_vals) if peak_bias_vals else None,
      'exit_bias': np.mean(exit_bias_vals) if exit_bias_vals else None,
      'entry_lag_ms': np.mean(entry_lag_ms) if entry_lag_ms else None,
      'peak_lag_ms': np.mean(peak_lag_ms) if peak_lag_ms else None,
    }
    print()

  # Root cause summary
  if phase_summary:
    print(f"  Root cause interpretation:")
    for cat, ps in phase_summary.items():
      eb = ps.get('entry_bias') or 0
      pb = ps.get('peak_bias') or 0
      xb = ps.get('exit_bias') or 0
      el = ps.get('entry_lag_ms') or 0
      if pb > 0 and eb < pb * 0.3:
        print(f"    {cat}: Overshoot mainly at PEAK, small at entry → EPAS static gain overshoot")
        print(f"      Feed-forward correction (subtract ~{pb:.4f} from sharp commands) should fix this")
      elif pb > 0 and eb > pb * 0.5:
        print(f"    {cat}: Overshoot starts at ENTRY, large at peak → EPAS dynamic lag + anticipation")
        print(f"      Reduce EMA tau or curvature_lookup_time to reduce")
      elif xb > pb * 0.5:
        print(f"    {cat}: Exit overshoot large → EPAS slow to release, consider slew-rate ramp")

  return phase_summary


def analyze_override_behavior(u, label):
  """
  V5 Section 2: Driver override trigger analysis and post-override transition quality.
  Measures how hard the car was fighting at each override, and how smoothly it re-engages.
  """
  print(f"\n{'='*80}")
  print(f"OVERRIDE BEHAVIOR — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  sp        = u['steer_pressed']
  app       = u['applied_curv']
  meas      = u['measured_curv']
  angle     = u['angle']
  app_dcurv = u['app_dcurv']
  dt        = np.median(np.diff(t))

  engaged_sp = lat & sp
  if np.sum(engaged_sp) < 3:
    print("  No engaged steering override events found.")
    return {}

  changes = np.diff(engaged_sp.astype(int))
  starts  = np.where(changes == 1)[0] + 1
  ends    = np.where(changes == -1)[0] + 1
  if engaged_sp[0]: starts = np.insert(starts, 0, 0)
  if engaged_sp[-1]: ends = np.append(ends, len(engaged_sp))
  if len(starts) > len(ends): starts = starts[:len(ends)]

  n_events    = len(starts)
  snap_window = int(1.0 / dt)   # 1s window after release

  # Per-event metrics
  trigger_deltas = []    # |applied - measured| at override start (fight magnitude)
  trigger_angles = []    # steering angle at override start
  trigger_speeds = []    # speed at override start (mph)
  override_deltas = []   # mean |applied - measured| during override
  post_snaps     = []    # max |d(applied)/dt| in 1s after release (jerk from snap-back)
  post_re_rmse   = []    # RMSE(applied - measured) in 1s after release (re-engage quality)

  for i in range(n_events):
    s, e = starts[i], ends[i]
    if s >= len(t) or e > len(t): continue

    # Trigger stats
    trigger_deltas.append(abs(app[s] - meas[s]))
    trigger_angles.append(abs(angle[s]))
    trigger_speeds.append(v[s] * 2.237)

    # During override
    override_deltas.append(np.mean(np.abs(app[s:e] - meas[s:e])))

    # Post-override
    post_end = min(len(t), e + snap_window)
    if post_end > e and lat[e] if e < len(lat) else False:
      # Only measure post if still engaged after release
      still_engaged = lat[e:post_end]
      if np.sum(still_engaged) > 5:
        post_app  = app[e:post_end][still_engaged]
        post_meas = meas[e:post_end][still_engaged]
        post_dc   = np.abs(app_dcurv[e:post_end][still_engaged])
        post_snaps.append(np.percentile(post_dc, 95))
        post_re_rmse.append(np.sqrt(np.mean((post_app - post_meas)**2)))

  print(f"\n  Events analyzed: {n_events}")
  print()
  print(f"  AT OVERRIDE TRIGGER (how hard was the fight?):")
  print(f"    Mean |app - meas| at trigger:  {np.mean(trigger_deltas):.6f} 1/m")
  print(f"    P50 |app - meas|:              {np.median(trigger_deltas):.6f} 1/m")
  print(f"    P90 |app - meas|:              {np.percentile(trigger_deltas, 90):.6f} 1/m")
  fight_pct = 100 * np.mean(np.array(trigger_deltas) > 0.001)
  print(f"    > 0.001 1/m (non-trivial fight): {fight_pct:.0f}% of overrides")
  print(f"    Mean angle at trigger:         {np.mean(trigger_angles):.1f}°")
  print()
  print(f"  DURING OVERRIDE (mean curvature fight magnitude):")
  print(f"    Mean |app - meas|:             {np.mean(override_deltas):.6f} 1/m")
  print()

  # Speed distribution of trigger deltas
  print(f"  Trigger delta by speed (fight magnitude at each speed range):")
  for vlo, vhi, lbl in [(0,15,"0-15 mph"),(15,30,"15-30 mph"),(30,45,"30-45 mph"),(45,65,"45-65 mph")]:
    idx = [i for i, s in enumerate(trigger_speeds) if vlo <= s < vhi]
    if not idx: continue
    d = [trigger_deltas[i] for i in idx]
    print(f"    {lbl:>10s}: mean={np.mean(d):.6f}  P90={np.percentile(d,90):.6f}  n={len(d)}")

  print()
  if post_snaps:
    print(f"  POST-OVERRIDE RE-ENGAGE QUALITY (1s after driver releases):")
    print(f"    P95 curvature rate (snap-back jerk):  {np.mean(post_snaps):.6f} 1/m/s")
    print(f"    Mean RMSE(applied-measured):          {np.mean(post_re_rmse):.6f} 1/m")
    snap_ok = np.mean(np.array(post_snaps) < 0.005)
    print(f"    Smooth re-engage (<0.005/s rate):     {100*snap_ok:.0f}% of events")
    if np.mean(post_snaps) < 0.005:
      print(f"    → Re-engage is smooth (good)")
    elif np.mean(post_snaps) < 0.02:
      print(f"    → Minor snap on re-engage")
    else:
      print(f"    → Significant snap on re-engage — driver can feel the system grabbing back")
  else:
    print(f"  POST-OVERRIDE: insufficient data (overrides at end of engagement segments)")

  return {'n_events': n_events, 'mean_trigger_delta': np.mean(trigger_deltas),
          'fight_pct': fight_pct,
          'mean_post_snap': np.mean(post_snaps) if post_snaps else None}


def analyze_lane_centering_detail(u, label):
  """
  V5 Section 3: Lane centering deep dive.
  Rate of change of offset, centering feature active%, convergence vs divergence.
  """
  print(f"\n{'='*80}")
  print(f"LANE CENTERING DETAIL — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  path_y    = u['path_y']
  lane_width = u['lane_width']
  ll_prob   = u['ll_prob']
  rl_prob   = u['rl_prob']
  dt        = np.median(np.diff(t))

  good_ll = lat & (ll_prob > 0.5) & (rl_prob > 0.5) & (lane_width > 2.5) & (lane_width < 5.5)
  if np.sum(good_ll) < 200:
    print(f"  Insufficient lane line data ({np.sum(good_ll)} pts)"); return {}

  # ── Centering feature active conditions ──────────────────────────────────
  # Feature fires when: v > 7 m/s, ll_prob > 0.6, rl_prob > 0.6, lane_width 2.5-5.5
  # (gate lowered from 15.0 to 7.0 m/s in our last commit)
  centering_active = good_ll & (v > 7.0) & (ll_prob > 0.6) & (rl_prob > 0.6)

  print(f"\n  Centering feature active% by speed:")
  print(f"  {'Speed bin':<24} {'Secs w/good LL':>15} {'Centering active':>17} {'Active%':>8}")
  print('  ' + '-'*68)
  for name, vlo, vhi in [("15-30 mph (7-13 m/s)", 7, 13), ("30-45 mph (13-20 m/s)", 13, 20),
                           ("45-65 mph (20-29 m/s)", 20, 29), ("65-80 mph (29-36 m/s)", 29, 36)]:
    good_bin   = good_ll & (v >= vlo) & (v < vhi)
    active_bin = centering_active & (v >= vlo) & (v < vhi)
    n_good, n_act = np.sum(good_bin), np.sum(active_bin)
    if n_good < 50: continue
    print(f"  {name:<24} {n_good*dt:>14.0f}s {n_act*dt:>16.0f}s {100*n_act/n_good:>7.1f}%")

  # ── Lane offset rate of change ──────────────────────────────────────────
  # Smooth path_y then differentiate to get drift rate
  smooth_win = max(1, int(0.5 / dt))  # 0.5s smoothing
  py_smooth  = _smooth(path_y, smooth_win)
  py_rate    = np.gradient(py_smooth, dt)  # m/s  (+ = drifting further left)

  print(f"\n  Lane offset drift rate (d(offset)/dt):")
  print(f"  Positive = car drifting left (bias increasing), Negative = converging to center")
  print()
  for name, vlo, vhi in [("15-30 mph (7-13 m/s)", 7, 13), ("30-45 mph (13-20 m/s)", 13, 20),
                           ("45-65 mph (20-29 m/s)", 20, 29)]:
    m = good_ll & (v >= vlo) & (v < vhi)
    if np.sum(m) < 100: continue
    r = py_rate[m]
    print(f"  {name:<24}: mean={np.mean(r):+.4f} m/s  std={np.std(r):.4f}  P10={np.percentile(r,10):+.4f}  P90={np.percentile(r,90):+.4f}")

  # ── Convergence analysis when left of center ───────────────────────────
  print(f"\n  Convergence analysis (when car is >0.10m left of center):")
  print(f"  Expected: centering active → negative rate (converging); inactive → positive (drifting)")
  print()

  left_mask = good_ll & (path_y > 0.10)  # car is left of center
  if np.sum(left_mask) > 100:
    # Active vs inactive centering while left of center
    act_left   = left_mask & centering_active
    inact_left = left_mask & ~centering_active & (v > 7.0)

    n_act = np.sum(act_left)
    n_inact = np.sum(inact_left)

    if n_act > 50:
      rate_act = py_rate[act_left]
      converging_pct = 100 * np.mean(rate_act < 0)
      print(f"  While centering ACTIVE & left of center ({n_act*dt:.0f}s):")
      print(f"    Mean drift rate:    {np.mean(rate_act):+.4f} m/s")
      print(f"    Converging (%<0):   {converging_pct:.0f}%")
      print(f"    → {'Centering is correcting' if converging_pct > 50 else 'Centering is NOT correcting (gain may be too small)'}")
    else:
      print(f"  Centering active & left of center: insufficient data ({n_act*dt:.0f}s)")

    if n_inact > 50:
      rate_inact = py_rate[inact_left]
      print(f"\n  While centering INACTIVE & left of center ({n_inact*dt:.0f}s):")
      print(f"    Mean drift rate:    {np.mean(rate_inact):+.4f} m/s")
      print(f"    → {'Natural drift toward center (model corrects without centering feature)' if np.mean(rate_inact) < 0 else 'Unmitigated leftward drift'}")
  else:
    print(f"  Insufficient samples with car left of center + good lane lines.")

  # ── Summary: is centering working? ─────────────────────────────────────
  print(f"\n  Summary:")
  n_total_good = np.sum(good_ll & (v > 7.0))
  n_active     = np.sum(centering_active)
  if n_total_good > 0:
    active_pct = 100 * n_active / n_total_good
    print(f"  Centering active {active_pct:.0f}% of engaged highway time with good lane lines")
    if active_pct < 40:
      print(f"  → Low activation rate. Lane line confidence often below 0.6 threshold.")
      print(f"    Consider lowering confidence threshold or using a longer-tail smoothing.")
    else:
      print(f"  → Activation rate looks adequate.")

  return {'centering_active_pct': 100*n_active/n_total_good if n_total_good > 0 else 0}


def analyze_steer_ratio(u, label):
  """
  V5 Section 4: SteerRatio validation and predicted curvature quality.
  A) Infers effective steerRatio from steeringAngleDeg / (measured_curv * WHEELBASE)
  B) Compares predicted curvature (orientationRate blend) vs desired curvature (MPC output)
  """
  print(f"\n{'='*80}")
  print(f"STEER RATIO VALIDATION & PREDICTED CURVATURE — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  meas     = u['measured_curv']
  app      = u['applied_curv']
  angle    = u['angle']
  mv_curv  = u['mv_curv']
  pred     = u['predicted_curv']
  dt       = np.median(np.diff(t))

  # ── A: Effective steerRatio ───────────────────────────────────────────────
  print(f"\n  A) Effective SteerRatio from telemetry")
  print(f"  Formula: eff_SR = |steeringAngleDeg * π/180| / (|measured_curv| * WHEELBASE)")
  print(f"  Assumed SR = {STEER_RATIO_ASSUMED:.1f}  |  WHEELBASE = {WHEELBASE:.3f} m")
  print()

  # Filter for reliable conditions: moderate speed, non-trivial curvature
  sr_mask = lat & (v > 15.0) & (np.abs(meas) > 0.002) & (np.abs(angle) > 3.0) & (np.abs(meas) < 0.015)
  n_sr = np.sum(sr_mask)

  if n_sr > 100:
    angle_rad = angle[sr_mask] * np.pi / 180.0
    meas_c    = meas[sr_mask]
    # Sign alignment: in openpilot, negative yaw = right, positive angle = right (Ford)
    # So: effective_sr = angle / (meas_curv * WB), both should have same sign
    valid_sign = np.sign(angle_rad) == np.sign(-meas_c)  # Ford: right turn = positive angle, negative curv
    if np.mean(valid_sign) > 0.5:
      eff_sr = np.abs(angle_rad[valid_sign]) / (np.abs(meas_c[valid_sign]) * WHEELBASE)
    else:
      eff_sr = np.abs(angle_rad) / (np.abs(meas_c) * WHEELBASE)

    # Filter outliers
    eff_sr = eff_sr[(eff_sr > 5) & (eff_sr < 30)]

    if len(eff_sr) > 50:
      print(f"  Samples used: {len(eff_sr)}")
      print(f"  Effective SR:  mean={np.mean(eff_sr):.2f}  median={np.median(eff_sr):.2f}  std={np.std(eff_sr):.2f}")
      print(f"                 P10={np.percentile(eff_sr,10):.2f}  P90={np.percentile(eff_sr,90):.2f}")
      error_pct = (np.median(eff_sr) - STEER_RATIO_ASSUMED) / STEER_RATIO_ASSUMED * 100
      print(f"  Assumed {STEER_RATIO_ASSUMED:.1f} vs measured {np.median(eff_sr):.2f}: {error_pct:+.1f}% error")
      if abs(error_pct) < 5:
        print(f"  → steerRatio assumption is accurate (within 5%)")
      elif abs(error_pct) < 15:
        print(f"  → Minor steerRatio mismatch — curvature calculations off by ~{abs(error_pct):.0f}%")
        print(f"     Consider tuning steerRatio to {np.median(eff_sr):.1f}")
      else:
        print(f"  → Significant steerRatio mismatch — curvature estimates unreliable")

      # By speed
      print(f"\n  Effective SR by speed:")
      for name, vlo, vhi in [("15-25 mph (7-11 m/s)", 7, 11), ("25-45 mph (11-20 m/s)", 11, 20),
                               ("45-65 mph (20-29 m/s)", 20, 29)]:
        sm = sr_mask & (v >= vlo) & (v < vhi)
        a_r = angle[sm] * np.pi / 180.0
        m_c = meas[sm]
        if np.sum(sm) < 50: continue
        sr_bin = np.abs(a_r) / (np.abs(m_c) * WHEELBASE)
        sr_bin = sr_bin[(sr_bin > 5) & (sr_bin < 30)]
        if len(sr_bin) < 20: continue
        print(f"    {name:<24}: median={np.median(sr_bin):.2f}  std={np.std(sr_bin):.2f}")
    else:
      print(f"  Insufficient samples after filtering ({n_sr} raw, {len(eff_sr)} after outlier removal)")
  else:
    print(f"  Insufficient data for steerRatio validation ({n_sr} samples with suitable conditions)")

  # ── B: Predicted curvature quality ────────────────────────────────────────
  print(f"\n  B) Predicted Curvature Quality (orientationRate blend input)")
  print(f"  pred = orientationRate.z@0.5s / vEgo  vs  mv_curv = MPC desiredCurvature")
  print(f"  If pred is noisier → PC blend may be adding oscillation at some speeds")
  print()

  speed_bins = [("7-13 m/s (15-30 mph)", 7, 13), ("13-20 m/s (30-45 mph)", 13, 20),
                ("20-29 m/s (45-65 mph)", 20, 29)]
  print(f"  {'Speed bin':<24} {'Secs':>5} {'MV_Std':>9} {'Pred_Std':>10} {'Ratio':>7} {'Corr':>7}  {'Interpretation'}")
  print('  ' + '-'*80)

  for name, vlo, vhi in speed_bins:
    m = lat & (v >= vlo) & (v < vhi)
    n = np.sum(m)
    if n < 100: continue
    secs     = n * dt
    mv_s     = mv_curv[m]
    pred_s   = pred[m]
    mv_std   = np.std(mv_s)
    pred_std = np.std(pred_s)
    ratio    = pred_std / mv_std if mv_std > 1e-6 else 0
    corr     = float(np.corrcoef(mv_s, pred_s)[0, 1]) if len(mv_s) > 10 else 0
    if ratio < 0.8:
      interp_str = "pred smoother (good blend input)"
    elif ratio < 1.2:
      interp_str = "similar noise (neutral)"
    else:
      interp_str = f"pred {ratio:.1f}x noisier (blend adding noise!)"
    print(f"  {name:<24} {secs:>5.0f}s {mv_std:>9.6f} {pred_std:>10.6f} {ratio:>6.2f}x {corr:>6.3f}  {interp_str}")

  return {}


def analyze_comfort(u, label):
  """
  V5 Section 5: Lateral comfort metrics — jerk, ISO 2631 lateral acceleration.
  Lateral accel commanded vs measured shows EPAS over-delivery in acceleration units.
  """
  print(f"\n{'='*80}")
  print(f"LATERAL COMFORT METRICS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  lat_accel_cmd  = u['lat_accel_cmd']   # applied_curv * v^2  (m/s²)
  lat_accel_meas = u['lat_accel_meas']  # yawRate * v          (m/s²)
  lat_jerk       = u['lat_jerk']        # d(lat_accel_meas)/dt (m/s³)
  dt             = np.median(np.diff(t))

  mask_h = lat & (v > 7.0)  # highway-ish only for meaningful lat accel
  if np.sum(mask_h) < 200:
    print(f"  Insufficient engaged highway data"); return {}

  la_cmd  = lat_accel_cmd[mask_h]
  la_meas = lat_accel_meas[mask_h]
  jerk    = lat_jerk[mask_h]

  # Clip extreme outliers (very low speed yawRate/vEgo noise)
  la_meas_clip = np.clip(la_meas, -5.0, 5.0)
  jerk_clip    = np.clip(jerk, -20.0, 20.0)

  print(f"\n  ISO 2631-1 lateral comfort scale (RMS m/s²):")
  print(f"  < 0.315: Not uncomfortable  0.315-0.63: A little uncomfortable")
  print(f"  0.5-1.0: Fairly  0.8-1.6: Uncomfortable  > 2.0: Extremely uncomfortable")

  rms_meas = np.sqrt(np.mean(la_meas_clip**2))
  rms_cmd  = np.sqrt(np.mean(la_cmd**2))
  print(f"\n  Lateral acceleration (engaged, v > 7 m/s):")
  print(f"    RMS commanded:  {rms_cmd:.3f} m/s²")
  print(f"    RMS measured:   {rms_meas:.3f} m/s²")
  print(f"    EPAS over-delivery: {rms_meas - rms_cmd:+.3f} m/s² RMS")
  print(f"    P95 measured:   {np.percentile(np.abs(la_meas_clip), 95):.3f} m/s²")
  print(f"    P99 measured:   {np.percentile(np.abs(la_meas_clip), 99):.3f} m/s²")

  # ISO 2631 comfort rating
  if rms_meas < 0.315:
    comfort = "Not uncomfortable"
  elif rms_meas < 0.63:
    comfort = "A little uncomfortable"
  elif rms_meas < 1.0:
    comfort = "Fairly uncomfortable"
  elif rms_meas < 1.6:
    comfort = "Uncomfortable"
  else:
    comfort = "Very/extremely uncomfortable"
  print(f"    ISO 2631 rating: {comfort}")

  # Thresholds
  print(f"\n  Time above comfort thresholds:")
  for thresh, lbl in [(0.5, "0.5 m/s² (onset of discomfort)"),
                      (1.0, "1.0 m/s² (uncomfortable)"),
                      (2.0, "2.0 m/s² (very uncomfortable)")]:
    pct = 100 * np.mean(np.abs(la_meas_clip) > thresh)
    print(f"    >{thresh} m/s²: {pct:.1f}%  ({pct/100*np.sum(mask_h)*dt:.0f}s)")

  # Lateral jerk
  rms_jerk = np.sqrt(np.mean(jerk_clip**2))
  print(f"\n  Lateral jerk (d(lat_accel)/dt):")
  print(f"    RMS:   {rms_jerk:.3f} m/s³")
  print(f"    P95:   {np.percentile(np.abs(jerk_clip), 95):.3f} m/s³")
  print(f"    P99:   {np.percentile(np.abs(jerk_clip), 99):.3f} m/s³")
  jerk_pct = 100 * np.mean(np.abs(jerk_clip) > 2.0)
  print(f"    >2.0 m/s³: {jerk_pct:.1f}%  (ISO 2631 jerk discomfort threshold)")

  # Lateral acceleration distribution histogram
  print(f"\n  Lateral acceleration distribution (|measured|, engaged highway):")
  bins = [(0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]
  for lo, hi in bins:
    pct = 100 * np.mean((np.abs(la_meas_clip) >= lo) & (np.abs(la_meas_clip) < hi))
    bar = '█' * int(pct / 2)
    print(f"    {lo:.1f}-{hi:.1f} m/s²: {pct:5.1f}%  {bar}")

  # By speed bin
  print(f"\n  Lateral acceleration by speed:")
  print(f"  {'Speed bin':<24} {'RMS cmd':>9} {'RMS meas':>9} {'Over-del':>9} {'P95 meas':>9} {'Comfort'}")
  print('  ' + '-'*72)
  for name, vlo, vhi in [("15-30 mph (7-13 m/s)", 7, 13), ("30-45 mph (13-20 m/s)", 13, 20),
                           ("45-65 mph (20-29 m/s)", 20, 29)]:
    m = lat & (v >= vlo) & (v < vhi)
    if np.sum(m) < 100: continue
    cmd_s  = lat_accel_cmd[m]
    meas_s = np.clip(lat_accel_meas[m], -5, 5)
    rms_c  = np.sqrt(np.mean(cmd_s**2))
    rms_m  = np.sqrt(np.mean(meas_s**2))
    p95    = np.percentile(np.abs(meas_s), 95)
    oc     = rms_m - rms_c
    if rms_m < 0.315: cf = "OK"
    elif rms_m < 0.63: cf = "mild"
    elif rms_m < 1.0:  cf = "noticeable"
    else:               cf = "uncomfortable"
    print(f"  {name:<24} {rms_c:>9.3f} {rms_m:>9.3f} {oc:>+9.3f} {p95:>9.3f} {cf}")

  # Commanded vs measured comparison (EPAS over-delivery in accel units)
  print(f"\n  Commanded vs measured lateral accel — EPAS over-delivery breakdown:")
  mask_curv = lat & (v > 7.0)
  curv_bins = [(0, 0.5, "straight"), (0.5, 1.0, "gentle"), (1.0, 2.0, "moderate"), (2.0, 5.0, "sharp")]
  print(f"  {'|cmd| bin (m/s²)':<22} {'Secs':>5} {'cmd':>8} {'meas':>8} {'over-del':>10}  {'pct'}")
  print('  ' + '-'*65)
  for lo, hi, lbl in curv_bins:
    m = mask_curv & (np.abs(lat_accel_cmd) >= lo) & (np.abs(lat_accel_cmd) < hi)
    if np.sum(m) < 20: continue
    secs  = np.sum(m) * dt
    cmd_v = np.mean(np.abs(lat_accel_cmd[m]))
    meas_v= np.mean(np.abs(np.clip(lat_accel_meas[m], -5, 5)))
    od    = meas_v - cmd_v
    od_pct= od / cmd_v * 100 if cmd_v > 0 else 0
    print(f"  {lbl:<22} {secs:>5.0f}s {cmd_v:>8.3f} {meas_v:>8.3f} {od:>+10.3f}  {od_pct:>+5.1f}%")

  return {'rms_lat_accel': rms_meas, 'rms_jerk': rms_jerk, 'comfort_rating': comfort}


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

  def _compare_table(title, metric_key, sub_key, fmt='.6f', lower_better=True):
    all_bins = [set(r[metric_key].keys()) for r in results if r.get(metric_key)]
    if not all_bins: return
    common = sorted(set.intersection(*all_bins))
    if not common: return
    print(f"\n{'─'*80}")
    print(f"{title}")
    print(f"{'─'*80}")
    hdr = f"{'Bin':<30s}"
    for r in results: hdr += f"  {r['label'][:12]:>12s}"
    hdr += f"  {'vs Base':>8s}"
    print(hdr)
    for b in common:
      row = f"{b:<30s}"
      base_val = results[0][metric_key].get(b, {}).get(sub_key, None)
      last_val  = results[-1][metric_key].get(b, {}).get(sub_key, None)
      for r in results:
        val = r[metric_key].get(b, {}).get(sub_key, None)
        row += f"  {val:{fmt}}" if val is not None else f"  {'—':>12s}"
      if base_val and last_val and base_val != 0:
        pct = (last_val - base_val) / abs(base_val) * 100
        direction = '↓' if (pct < 0) == lower_better else '↑'
        row += f"  {pct:>+6.1f}% {direction}"
      print(row)

  _compare_table("HUNTING: Applied Curvature Std (lower = better)",
                 'hunt', 'app_std', fmt='>12.6f')
  _compare_table("STOP-AND-GO: Applied Curvature Std (lower = better)",
                 'low_speed', 'app_std', fmt='>12.6f')
  _compare_table("CURVE OVERSHOOT: Mean overshoot (lower = better)",
                 'curves', 'mean_overshoot', fmt='>+12.5f')
  _compare_table("FILTER EFFICACY: Reduction % (higher = filter doing more)",
                 'filter', 'reduction', fmt='>+11.1f', lower_better=False)

  # Override behavior comparison
  if any(r.get('override') for r in results):
    print(f"\n{'─'*80}")
    print(f"OVERRIDE FIGHT: Mean trigger delta (lower = less fighting)")
    print(f"{'─'*80}")
    hdr = f"{'Drive':<30s}  {'Trigger Delta':>14s}  {'Fight %':>8s}  {'Post Snap':>10s}"
    print(hdr)
    for r in results:
      ov = r.get('override', {})
      if not ov: continue
      td = ov.get('mean_trigger_delta')
      fp = ov.get('fight_pct')
      ps = ov.get('mean_post_snap')
      print(f"  {r['label']:<28s}  {td:>14.6f}  {fp:>7.0f}%  {'—' if ps is None else f'{ps:.6f}'}")

  # Comfort comparison
  if any(r.get('comfort') for r in results):
    print(f"\n{'─'*80}")
    print(f"COMFORT: RMS lateral accel (lower = more comfortable)")
    print(f"{'─'*80}")
    for r in results:
      cm = r.get('comfort', {})
      if not cm: continue
      print(f"  {r['label']}: RMS={cm.get('rms_lat_accel', 0):.3f} m/s²  "
            f"Jerk={cm.get('rms_jerk', 0):.3f} m/s³  Rating: {cm.get('comfort_rating', '—')}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
  if len(sys.argv) < 2:
    print("Usage: analyze_drive_v5.py <route_dir> [route_dir2 ...]")
    sys.exit(1)

  print(f"scipy available: {HAS_SCIPY}")

  results = []
  for route_dir in sys.argv[1:]:
    label = os.path.basename(route_dir)
    print(f"\n{'#'*80}")
    print(f"# {label}")
    print(f"{'#'*80}")

    msgs = load_route(route_dir)
    if not msgs: continue

    data = extract_data(msgs)
    u    = build_timeline(data)
    if u is None: continue

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

    # v5 new sections
    curve_dyn  = analyze_curve_dynamics(u, label)
    override   = analyze_override_behavior(u, label)
    analyze_lane_centering_detail(u, label)
    analyze_steer_ratio(u, label)
    comfort    = analyze_comfort(u, label)

    results.append({
      'label': label, 'overview': overview, 'filter': filter_eff,
      'hunt': hunt, 'low_speed': low_speed, 'curves': curves,
      'lane_center': lane_center, 'curve_dyn': curve_dyn,
      'override': override, 'comfort': comfort,
    })

  print_comparison(results)


if __name__ == '__main__':
  main()
