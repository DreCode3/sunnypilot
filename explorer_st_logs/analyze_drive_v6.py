#!/usr/bin/env python3
"""
Explorer ST Drive Analysis — v6
Extends v5 with 2 new sections:
  6. Steering micro-oscillation: reversals/mile, curvature jitter, smoothness ratio
  7. PI controller diagnostics: parsed LC:/CP: log messages, PI state, pipeline stats
All v5 sections retained:
  1. Curve dynamics: entry/exit timing, EPAS bias vs curvature rate (dC/dt)
  2. Override behavior: trigger delta, post-override snap, re-engage quality
  3. Lane centering detail: offset rate of change, centering active%, convergence %
  4. SteerRatio validation + predicted vs desired curvature quality
  5. Comfort metrics: lateral jerk, ISO 2631 lateral acceleration distribution
"""

import sys
import os
import glob
import re
import json
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
  from scipy.ndimage import uniform_filter1d
  from scipy.signal import welch, coherence as sp_coherence, csd as sp_csd
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

    elif w == 'logMessage':
      try:
        txt = msg.logMessage
        parsed = json.loads(txt)
        msg_str = parsed.get('msg', '')
        if msg_str.startswith('LC: '):
          data['lc_logs'].append((t, msg_str[4:]))
        elif msg_str.startswith('CP: '):
          data['cp_logs'].append((t, msg_str[4:]))
        elif msg_str.startswith('CX1: '):
          data.setdefault('cx1_logs', []).append((t, msg_str[5:]))
      except Exception:
        pass

  # Preserve raw log lists — don't convert to numpy
  lc_logs = data.pop('lc_logs', [])
  cp_logs = data.pop('cp_logs', [])
  cx1_logs = data.pop('cx1_logs', [])

  for k in data:
    data[k] = np.array(data[k], dtype=float if k not in ('steering_pressed',) else bool)

  data['lc_logs'] = lc_logs
  data['cp_logs'] = cp_logs
  data['cx1_logs'] = cx1_logs

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
# NEW V6 ANALYSIS SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _count_reversals(signal, threshold):
  """Count sign changes in diff(signal) where |change| > threshold."""
  d = np.diff(signal)
  # Zero out changes below threshold
  d_filt = np.where(np.abs(d) >= threshold, d, 0.0)
  # Remove zeros for sign-change detection
  nonzero = d_filt[d_filt != 0]
  if len(nonzero) < 2:
    return 0
  signs = np.sign(nonzero)
  return int(np.sum(np.diff(signs) != 0))


def analyze_micro_oscillation(u, label):
  """
  V6 Section 6: Steering micro-oscillation metrics for engaged highway driving (>25 m/s).
  Reversals/mile, curvature jitter, steering angle jitter, lateral accel jerk,
  and smoothness ratio vs planner desired curvature.
  """
  print(f"\n{'='*80}")
  print(f"STEERING MICRO-OSCILLATION — {label}")
  print(f"{'='*80}")
  print(f"  Highway driving: engaged & v > 25 m/s (56 mph)")

  t, v, lat = u['t'], u['v'], u['lat_active']
  angle     = u['angle']
  app       = u['applied_curv']
  mv_curv   = u['mv_curv']
  yaw       = u['yaw']
  dt        = np.median(np.diff(t))

  speed_bins = [
    ("25-29 m/s (56-65 mph)", 25.0, 29.0),
    ("29-33 m/s (65-74 mph)", 29.0, 33.0),
    ("33-38 m/s (74-85 mph)", 33.0, 38.0),
    ("ALL >25 m/s (>56 mph)", 25.0, 999.0),
  ]

  # ── 1. Steering angle reversals per mile ──────────────────────────────────
  print(f"\n  1) Steering angle reversals per mile")
  print(f"     A reversal = steering angle derivative changes sign (above threshold)")
  print()
  angle_thresholds = [0.05, 0.10, 0.20]
  hdr = f"  {'Speed bin':<28} {'Secs':>5} {'Miles':>6}"
  for th in angle_thresholds:
    hdr += f" {f'>{th:.2f} deg':>10}"
  print(hdr)
  print('  ' + '-'*(28+5+6+10*len(angle_thresholds)+len(angle_thresholds)+4))

  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 50:
      print(f"  {name:<28} <50pts"); continue
    secs = n * dt
    # Distance in miles: sum(v*dt) for engaged samples
    miles = np.sum(v[mask]) * dt / 1609.34
    if miles < 0.01:
      print(f"  {name:<28} <0.01 mi"); continue
    ang = angle[mask]
    row = f"  {name:<28} {secs:>5.0f}s {miles:>5.2f}mi"
    for th in angle_thresholds:
      rev = _count_reversals(ang, th)
      rpm = rev / miles
      row += f" {rpm:>10.1f}"
    print(row)

  # ── 2. Curvature command reversals per mile ───────────────────────────────
  print(f"\n  2) Curvature command reversals per mile (threshold >1e-6 1/m)")
  print()
  print(f"  {'Speed bin':<28} {'Miles':>6} {'Rev/mi':>10}")
  print('  ' + '-'*48)
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 50: continue
    miles = np.sum(v[mask]) * dt / 1609.34
    if miles < 0.01: continue
    rev = _count_reversals(app[mask], 1e-6)
    print(f"  {name:<28} {miles:>5.2f}mi {rev/miles:>10.1f}")

  # ── 3. Curvature jitter (std of d(curvature)/dt at 100Hz) ────────────────
  print(f"\n  3) Curvature jitter  (std of d(curvature)/dt)")
  print(f"     Effective sample rate: {1/dt:.0f} Hz")
  print()
  print(f"  {'Speed bin':<28} {'Secs':>5} {'Curv jitter':>12} {'Unit':>10}")
  print('  ' + '-'*60)
  curv_jitter_by_bin = {}
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 50: continue
    secs = n * dt
    dc_dt = np.diff(app[mask]) / dt
    jitter = np.std(dc_dt)
    curv_jitter_by_bin[name] = jitter
    print(f"  {name:<28} {secs:>5.0f}s {jitter:>12.6f} {'1/m/s':>10}")

  # ── 4. Steering angle jitter (std of d(angle)/dt) ────────────────────────
  print(f"\n  4) Steering angle jitter  (std of d(steeringAngle)/dt)")
  print()
  print(f"  {'Speed bin':<28} {'Secs':>5} {'Ang jitter':>12} {'Unit':>10}")
  print('  ' + '-'*60)
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 50: continue
    secs = n * dt
    da_dt = np.diff(angle[mask]) / dt
    jitter = np.std(da_dt)
    print(f"  {name:<28} {secs:>5.0f}s {jitter:>12.4f} {'deg/s':>10}")

  # ── 5. Lateral acceleration jerk (std of d(lat_accel)/dt) ─────────────────
  print(f"\n  5) Lateral acceleration jerk  (std of d(yawRate*speed)/dt)")
  print()
  print(f"  {'Speed bin':<28} {'Secs':>5} {'Lat jerk std':>13} {'Unit':>10}")
  print('  ' + '-'*62)
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 50: continue
    secs = n * dt
    lat_accel = yaw[mask] * v[mask]
    dlat_dt = np.diff(lat_accel) / dt
    jitter = np.std(dlat_dt)
    print(f"  {name:<28} {secs:>5.0f}s {jitter:>13.4f} {'m/s^3':>10}")

  # ── 6. Smoothness ratio vs planner ───────────────────────────────────────
  print(f"\n  6) Smoothness ratio (controller curvature jitter / planner desired curvature jitter)")
  print(f"     Ratio < 1.0 = controller smoother than planner (good)")
  print(f"     Ratio > 1.0 = controller adding oscillation (bad)")
  print()
  print(f"  {'Speed bin':<28} {'Ctrl jitter':>12} {'Plan jitter':>12} {'Ratio':>7}")
  print('  ' + '-'*64)
  for name, vlo, vhi in speed_bins:
    mask = lat & (v >= vlo) & (v < vhi)
    n = np.sum(mask)
    if n < 50: continue
    dc_app = np.diff(app[mask]) / dt
    dc_mv  = np.diff(mv_curv[mask]) / dt
    ctrl_j = np.std(dc_app)
    plan_j = np.std(dc_mv)
    ratio = ctrl_j / plan_j if plan_j > 1e-9 else float('inf')
    marker = "  <-- adding oscillation" if ratio > 1.2 else ("  <-- smoothing" if ratio < 0.8 else "")
    print(f"  {name:<28} {ctrl_j:>12.6f} {plan_j:>12.6f} {ratio:>6.2f}x{marker}")

  return curv_jitter_by_bin


def _parse_kv_line(line):
  """Parse key=value pairs from an LC: or CP: log line, handling | separators."""
  result = {}
  # Remove pipe separators
  line = line.replace('|', ' ')
  # Match key=value pairs (value can be a number, possibly with sign)
  for m in re.finditer(r'(\w+)=([-+]?\d*\.?\d+(?:e[-+]?\d+)?)', line):
    key = m.group(1)
    val_str = m.group(2)
    try:
      result[key] = float(val_str)
    except ValueError:
      pass
  return result


def analyze_pi_diagnostics(data, u, label):
  """
  V6 Section 7: PI Controller diagnostics from LC: and CP: log messages.
  Parses JSON logMessage events, extracts PI state, and reports statistics.
  """
  print(f"\n{'='*80}")
  print(f"PI CONTROLLER DIAGNOSTICS — {label}")
  print(f"{'='*80}")

  lc_logs = data.get('lc_logs', [])
  cp_logs = data.get('cp_logs', [])

  t, v, lat = u['t'], u['v'], u['lat_active']
  dt = np.median(np.diff(t))

  print(f"\n  Log messages found: {len(lc_logs)} LC,  {len(cp_logs)} CP")

  if not lc_logs and not cp_logs:
    print("  No LC: or CP: log messages found in this route.")
    print("  (PI controller logging may not be enabled in this build)")
    return {}

  # ── Parse LC logs ─────────────────────────────────────────────────────────
  lc_parsed = []
  for ts, line in lc_logs:
    kv = _parse_kv_line(line)
    if kv:
      kv['_t'] = ts
      lc_parsed.append(kv)

  # ── Parse CP logs ─────────────────────────────────────────────────────────
  cp_parsed = []
  for ts, line in cp_logs:
    kv = _parse_kv_line(line)
    if kv:
      kv['_t'] = ts
      cp_parsed.append(kv)

  # ── LC stats (lane centering PI state) ────────────────────────────────────
  if lc_parsed:
    print(f"\n  ── LC (Lane Centering) PI State ──")
    print(f"  Parsed entries: {len(lc_parsed)}")

    # Extract arrays for each field
    lc_fields = {}
    for key in ['off', 'll', 'pos', 'scl', 'conf', 'wid', 'int', 'P', 'I', 'curv', 'spd']:
      vals = [e[key] for e in lc_parsed if key in e]
      if vals:
        lc_fields[key] = np.array(vals)

    print(f"\n  {'Field':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'P5':>10} {'P95':>10}")
    print('  ' + '-'*72)
    field_labels = {
      'off': 'offset', 'int': 'integral', 'P': 'P-term', 'I': 'I-term',
      'curv': 'curvature', 'spd': 'speed', 'll': 'laneline', 'pos': 'position',
      'scl': 'scale', 'conf': 'confidence', 'wid': 'width'
    }
    for key in ['off', 'int', 'P', 'I', 'curv', 'spd', 'll', 'pos', 'scl', 'conf', 'wid']:
      if key not in lc_fields: continue
      arr = lc_fields[key]
      lbl = field_labels.get(key, key)
      print(f"  {lbl:<10} {np.mean(arr):>10.4f} {np.std(arr):>10.4f} "
            f"{np.min(arr):>10.4f} {np.max(arr):>10.4f} "
            f"{np.percentile(arr,5):>10.4f} {np.percentile(arr,95):>10.4f}")

    # PI state summary
    if 'off' in lc_fields:
      print(f"\n  PI state summary:")
      print(f"    Offset  mean={np.mean(lc_fields['off']):+.4f}  std={np.std(lc_fields['off']):.4f}")
    if 'int' in lc_fields:
      print(f"    Integral mean={np.mean(lc_fields['int']):+.6f}  range=[{np.min(lc_fields['int']):.6f}, {np.max(lc_fields['int']):.6f}]")
    if 'P' in lc_fields:
      print(f"    P-term   mean={np.mean(lc_fields['P']):+.6f}  std={np.std(lc_fields['P']):.6f}  P95={np.percentile(np.abs(lc_fields['P']),95):.6f}")
    if 'I' in lc_fields:
      print(f"    I-term   mean={np.mean(lc_fields['I']):+.6f}  std={np.std(lc_fields['I']):.6f}  P95={np.percentile(np.abs(lc_fields['I']),95):.6f}")

    # Speed-binned PI state
    if 'spd' in lc_fields and 'off' in lc_fields:
      print(f"\n  Speed-binned PI state (LC):")
      print(f"  {'Speed bin':<28} {'N':>5} {'off':>10} {'int':>10} {'P':>10} {'I':>10}")
      print('  ' + '-'*78)
      spd = lc_fields['spd']
      for name, vlo, vhi in [("15-30 mph (7-13 m/s)", 7, 13), ("30-45 mph (13-20 m/s)", 13, 20),
                               ("45-65 mph (20-29 m/s)", 20, 29), ("65-80 mph (29-36 m/s)", 29, 36)]:
        sm = (spd >= vlo) & (spd < vhi)
        n = np.sum(sm)
        if n < 10: continue
        row = f"  {name:<28} {n:>5}"
        for key in ['off', 'int', 'P', 'I']:
          if key in lc_fields and len(lc_fields[key]) == len(spd):
            row += f" {np.mean(lc_fields[key][sm]):>+10.6f}"
          else:
            row += f" {'--':>10}"
        print(row)

  # ── CP stats (curvature pipeline) ─────────────────────────────────────────
  if cp_parsed:
    print(f"\n  ── CP (Curvature Pipeline) State ──")
    print(f"  Parsed entries: {len(cp_parsed)}")

    cp_fields = {}
    for key in ['des', 'pred', 'ema', 'preRL', 'RL', 'send', 'meas', 'ovr', 'rst', 'ramp', 'rlClip', 'aw', 'ang', 'tq']:
      vals = [e[key] for e in cp_parsed if key in e]
      if vals:
        cp_fields[key] = np.array(vals)

    # Pipeline curvature stages
    curv_keys = ['des', 'pred', 'ema', 'preRL', 'RL', 'send', 'meas']
    avail = [k for k in curv_keys if k in cp_fields]
    if avail:
      print(f"\n  Pipeline curvature stages:")
      print(f"  {'Stage':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
      print('  ' + '-'*54)
      for key in avail:
        arr = cp_fields[key]
        print(f"  {key:<10} {np.mean(arr):>10.6f} {np.std(arr):>10.6f} {np.min(arr):>10.6f} {np.max(arr):>10.6f}")

    # Boolean flags: override, reset, ramp, rateLimit clip, anti-windup
    n_total = len(cp_parsed)
    flag_summary = {}
    print(f"\n  Pipeline flags (% of {n_total} samples):")
    for key, lbl in [('ovr', 'Override active'), ('rst', 'Reset'), ('ramp', 'Ramp'),
                      ('rlClip', 'Rate-limit clip'), ('aw', 'Anti-windup')]:
      if key not in cp_fields: continue
      pct = 100 * np.mean(cp_fields[key] > 0.5)
      flag_summary[lbl] = pct
      print(f"    {lbl:<22}: {pct:>6.1f}%  ({int(np.sum(cp_fields[key] > 0.5)):>6} / {n_total})")

    # Speed-binned CP state (use 'ang' to infer speed bin via steering angle,
    # or match timestamps to timeline speeds)
    if 'send' in cp_fields and 'meas' in cp_fields:
      print(f"\n  Curvature tracking (send vs meas):")
      send = cp_fields['send']
      meas_cp = cp_fields['meas']
      err = meas_cp - send
      rmse = np.sqrt(np.mean(err**2))
      print(f"    RMSE(meas - send):  {rmse:.6f} 1/m")
      print(f"    Bias(meas - send):  {np.mean(err):+.6f} 1/m")

    # Speed-binned CP state using timeline interpolation
    cp_times = np.array([e['_t'] for e in cp_parsed])
    if len(cp_times) > 10 and len(t) > 10:
      cp_speeds = np.interp(cp_times, t, u['v'])
      print(f"\n  Speed-binned CP pipeline:")
      print(f"  {'Speed bin':<28} {'N':>5} {'ovr%':>6} {'rlClip%':>8} {'aw%':>6} {'RMSE(s-m)':>10}")
      print('  ' + '-'*68)
      for name, vlo, vhi in [("15-30 mph (7-13 m/s)", 7, 13), ("30-45 mph (13-20 m/s)", 13, 20),
                               ("45-65 mph (20-29 m/s)", 20, 29), ("65-80 mph (29-36 m/s)", 29, 36)]:
        sm = (cp_speeds >= vlo) & (cp_speeds < vhi)
        n = np.sum(sm)
        if n < 10: continue
        row = f"  {name:<28} {n:>5}"
        ovr_pct = 100 * np.mean(cp_fields['ovr'][sm] > 0.5) if 'ovr' in cp_fields else 0
        rl_pct  = 100 * np.mean(cp_fields['rlClip'][sm] > 0.5) if 'rlClip' in cp_fields else 0
        aw_pct  = 100 * np.mean(cp_fields['aw'][sm] > 0.5) if 'aw' in cp_fields else 0
        row += f" {ovr_pct:>5.1f}% {rl_pct:>7.1f}% {aw_pct:>5.1f}%"
        if 'send' in cp_fields and 'meas' in cp_fields:
          err_bin = cp_fields['meas'][sm] - cp_fields['send'][sm]
          row += f" {np.sqrt(np.mean(err_bin**2)):>10.6f}"
        print(row)

  return {'n_lc': len(lc_parsed), 'n_cp': len(cp_parsed)}


# ─────────────────────────────────────────────────────────────────────────────
# PATH 4 A/B ANALYSIS (asymmetric release-side EMA)
# ─────────────────────────────────────────────────────────────────────────────

# CX1 schema v2: 29 positional fields. Index map for clarity below.
CX1_FIELDS = ['frame', 'v', 'yr', 'aLat', 'cmd', 'rate', 'meas', 'des', 'pred', 'ema',
              'preRL', 'rl', 'cmdInt', 'rateInt', 'ang', 'dAng', 'tq', 'ovr', 'lc',
              'lookT', 'blend', 'cFac', 'lOff', 'lInt', 'pmd', 'burst',
              'p4Rel', 'p4Tau', 'p4On']


def _parse_cx1_logs(cx1_logs):
  """Parse positional CX1 log lines into a dict of numpy arrays.
     Returns ({field: np.array}, t_log) — t_log is the carlog timestamp per row."""
  rows = []
  t_logs = []
  for ts, line in cx1_logs:
    if line.startswith('SCHEMA='):
      continue
    parts = line.split()
    if len(parts) < len(CX1_FIELDS):
      continue
    try:
      rows.append([float(p) for p in parts[:len(CX1_FIELDS)]])
      t_logs.append(ts)
    except ValueError:
      continue
  if not rows:
    return {}, np.array([])
  arr = np.array(rows)
  out = {f: arr[:, i] for i, f in enumerate(CX1_FIELDS)}
  return out, np.array(t_logs)


def _path4_metrics(cx1, mask, label):
  """Compute the six acceptance-gate metrics on a subset of CX1 rows.
     Returns dict of metrics. All inputs already filtered to engaged + speed-relevant."""
  if not mask.any() or mask.sum() < 50:
    return None
  sub = {k: v[mask] for k, v in cx1.items()}
  m = {}
  m['n_samples'] = int(mask.sum())
  m['mean_speed'] = float(np.mean(sub['v']))

  # 1. Overshoot magnitude per sample where on a curve (|cmd|>0.001)
  curve_mask = np.abs(sub['cmd']) > 0.001
  if curve_mask.sum() > 10:
    os_signed = sub['meas'][curve_mask] - sub['cmd'][curve_mask]
    # Only count overshoots where measured passes BEYOND the command in same direction
    cmd_sign = np.sign(sub['cmd'][curve_mask])
    overshoot_only = os_signed * cmd_sign  # positive when meas overshoots past cmd
    overshoot_only = overshoot_only[overshoot_only > 0]
    m['mean_overshoot'] = float(np.mean(overshoot_only)) if len(overshoot_only) else 0.0
    m['p95_overshoot'] = float(np.percentile(overshoot_only, 95)) if len(overshoot_only) else 0.0
    m['n_overshoots'] = int(len(overshoot_only))
  else:
    m['mean_overshoot'] = 0.0
    m['p95_overshoot'] = 0.0
    m['n_overshoots'] = 0

  # 2. Lane offset std (lOff field)
  m['lane_off_std'] = float(np.std(sub['lOff']))
  m['lane_off_p95_abs'] = float(np.percentile(np.abs(sub['lOff']), 95))

  # 3. PI integral excursion peaks (lInt field)
  m['lInt_p95_abs'] = float(np.percentile(np.abs(sub['lInt']), 95))
  m['lInt_max_abs'] = float(np.max(np.abs(sub['lInt'])))

  # 4. Lateral comfort RMS (aLat field)
  m['rms_aLat'] = float(np.sqrt(np.mean(sub['aLat'] ** 2)))

  # 5. Detector flutter rate — fraction of consecutive-frame transitions in p4Rel
  # Only valid where samples are consecutive in frame number AND in 20-30 m/s band
  hwy_mask = (sub['v'] >= 20) & (sub['v'] < 30)
  if hwy_mask.sum() > 50:
    f_hwy = sub['frame'][hwy_mask]
    p4r_hwy = sub['p4Rel'][hwy_mask]
    # Pairs where frame N+1 = frame N + step (5-10 frames at typical CX1 rate)
    df = np.diff(f_hwy)
    consecutive = df <= 12  # allow 10Hz curve cadence + slack
    if consecutive.any():
      flips = np.diff(p4r_hwy) != 0
      m['flutter_rate_hwy_pct'] = 100.0 * float(np.sum(flips & consecutive) / max(1, np.sum(consecutive)))
    else:
      m['flutter_rate_hwy_pct'] = 0.0
  else:
    m['flutter_rate_hwy_pct'] = 0.0

  # 6. Curvature reversal rate at >56 mph (25 m/s) — sign reversals in cmd per "mile"
  # Rate is per sample, scaled to per-mile via mean speed
  fast_mask = sub['v'] >= 25.0
  if fast_mask.sum() > 50:
    cmd_fast = sub['cmd'][fast_mask]
    f_fast = sub['frame'][fast_mask]
    df_fast = np.diff(f_fast)
    consecutive_fast = df_fast <= 12
    sign_changes = np.diff(np.sign(cmd_fast)) != 0
    n_reversals = int(np.sum(sign_changes & consecutive_fast))
    # Time over consecutive samples — assume ~10Hz when consecutive
    time_consec_sec = float(np.sum(consecutive_fast) * 0.1)
    miles = (np.mean(sub['v'][fast_mask]) * time_consec_sec) / 1609.34
    m['reversals_per_mile_fast'] = float(n_reversals / max(0.001, miles))
    m['n_reversals_fast'] = n_reversals
    m['miles_fast'] = float(miles)
  else:
    m['reversals_per_mile_fast'] = 0.0
    m['n_reversals_fast'] = 0
    m['miles_fast'] = 0.0

  # Helpful side metrics: detector trip rate and tau distribution
  m['detector_trip_pct'] = 100.0 * float(np.mean(sub['p4Rel'] > 0.5))
  m['mean_smooth_tau'] = float(np.mean(sub['p4Tau']))
  return m


def analyze_path4_ab(data, u, label):
  """V6 Path 4 A/B comparison. Splits CX1 logs by p4On and compares acceptance gates."""
  print(f"\n{'='*80}")
  print(f"PATH 4 A/B ANALYSIS — {label}")
  print(f"{'='*80}")

  cx1_logs = data.get('cx1_logs', [])
  if not cx1_logs:
    print("  No CX1: log messages found. (Path 4 telemetry requires CX1-instrumented build.)")
    return {}

  cx1, t_log = _parse_cx1_logs(cx1_logs)
  if not cx1:
    print(f"  CX1 logs found ({len(cx1_logs)}) but parsing produced 0 rows. Schema mismatch?")
    return {}

  print(f"  CX1 rows parsed: {len(cx1['frame']):,}")
  n_on = int(np.sum(cx1['p4On'] > 0.5))
  n_off = int(np.sum(cx1['p4On'] < 0.5))
  print(f"  Path 4 ON  samples: {n_on:,}")
  print(f"  Path 4 OFF samples: {n_off:,}")

  if n_on < 50 or n_off < 50:
    print("\n  Insufficient samples in one or both conditions (<50). Need a paired drive with toggle flip.")
    return {'n_on': n_on, 'n_off': n_off}

  # Engaged + relevant-speed mask: above 4 m/s where EMA actually runs
  engaged = (cx1['v'] >= 4.0) & (cx1['ovr'] < 0.5)
  off_mask = engaged & (cx1['p4On'] < 0.5)
  on_mask  = engaged & (cx1['p4On'] > 0.5)

  off_m = _path4_metrics(cx1, off_mask, 'OFF')
  on_m  = _path4_metrics(cx1, on_mask, 'ON')

  if off_m is None or on_m is None:
    print("\n  Insufficient engaged samples for one of the conditions.")
    return {'n_on': n_on, 'n_off': n_off}

  # Acceptance gates
  print(f"\n  ── Sample summary ──")
  print(f"  {'Condition':<10} {'N':>8} {'Mean spd m/s':>14}")
  print(f"  {'OFF':<10} {off_m['n_samples']:>8} {off_m['mean_speed']:>14.2f}")
  print(f"  {'ON':<10} {on_m['n_samples']:>8} {on_m['mean_speed']:>14.2f}")

  def _pct_change(off, on):
    if off == 0:
      return float('nan')
    return (on - off) / abs(off) * 100.0

  rows = [
    # (label, off_value, on_value, lower_better, format, gate_text)
    ('Mean overshoot (1/m)',       off_m['mean_overshoot'],   on_m['mean_overshoot'],   True,  '.6f', '≥15% reduction'),
    ('P95 overshoot (1/m)',        off_m['p95_overshoot'],    on_m['p95_overshoot'],    True,  '.6f', 'no regression'),
    ('Overshoot count',            off_m['n_overshoots'],     on_m['n_overshoots'],     True,  '.0f', 'informational'),
    ('Lane offset std (m)',        off_m['lane_off_std'],     on_m['lane_off_std'],     True,  '.4f', 'no regression'),
    ('Lane offset P95|abs| (m)',   off_m['lane_off_p95_abs'], on_m['lane_off_p95_abs'], True,  '.4f', 'no regression'),
    ('PI integral P95|abs|',       off_m['lInt_p95_abs'],     on_m['lInt_p95_abs'],     True,  '.4f', 'no regression'),
    ('PI integral max|abs|',       off_m['lInt_max_abs'],     on_m['lInt_max_abs'],     True,  '.4f', 'no regression'),
    ('Lateral RMS aLat (m/s²)',    off_m['rms_aLat'],         on_m['rms_aLat'],         True,  '.4f', 'no regression'),
    ('Reversals/mile @>56mph',     off_m['reversals_per_mile_fast'], on_m['reversals_per_mile_fast'], True, '.1f', 'should drop'),
    ('Detector trip %',            off_m['detector_trip_pct'], on_m['detector_trip_pct'], False, '.1f', 'should match (detector runs both)'),
    ('Flutter rate @20-30m/s %',   off_m['flutter_rate_hwy_pct'], on_m['flutter_rate_hwy_pct'], True, '.1f', '<10% acceptable'),
    ('Mean smooth_tau (s)',        off_m['mean_smooth_tau'],  on_m['mean_smooth_tau'],  False, '.4f', 'ON should be higher'),
  ]

  print(f"\n  ── Acceptance gates (OFF vs ON) ──")
  print(f"  {'Metric':<28} {'OFF':>12} {'ON':>12} {'Δ %':>9} {'Gate':<32}")
  print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*9} {'-'*32}")
  for name, ov, nv, lower_better, fmt, gate in rows:
    pct = _pct_change(ov, nv)
    direction = '↓' if (pct < 0) == lower_better else '↑'
    pct_str = f"{pct:+.1f}{direction}" if not np.isnan(pct) else '   --'
    print(f"  {name:<28} {ov:>12{fmt}} {nv:>12{fmt}} {pct_str:>9} {gate:<32}")

  # Verdict line: did we meet the 15% mean OS reduction with no regressions?
  print(f"\n  ── Verdict against QA gates ──")
  os_pct = _pct_change(off_m['mean_overshoot'], on_m['mean_overshoot'])
  off_std_pct = _pct_change(off_m['lane_off_std'], on_m['lane_off_std'])
  pi_p95_pct = _pct_change(off_m['lInt_p95_abs'], on_m['lInt_p95_abs'])
  rms_pct = _pct_change(off_m['rms_aLat'], on_m['rms_aLat'])
  flutter_ok = on_m['flutter_rate_hwy_pct'] < 10.0
  rev_pct = _pct_change(off_m['reversals_per_mile_fast'], on_m['reversals_per_mile_fast'])

  def _check(name, condition, detail):
    sym = 'PASS' if condition else 'FAIL'
    print(f"    [{sym}] {name}: {detail}")
    return condition

  passes = []
  passes.append(_check('Mean OS reduction ≥15%',  os_pct <= -15.0, f"got {os_pct:+.1f}%"))
  passes.append(_check('Lane offset std',         off_std_pct <= 5.0, f"changed {off_std_pct:+.1f}% (allow ≤+5%)"))
  passes.append(_check('PI integral P95',         pi_p95_pct <= 10.0, f"changed {pi_p95_pct:+.1f}% (allow ≤+10%)"))
  passes.append(_check('Lateral RMS',             rms_pct <= 5.0, f"changed {rms_pct:+.1f}% (allow ≤+5%)"))
  passes.append(_check('Flutter <10% at hwy',     flutter_ok, f"ON flutter={on_m['flutter_rate_hwy_pct']:.1f}%"))
  passes.append(_check('Reversal rate @>56mph',   rev_pct <= 5.0, f"changed {rev_pct:+.1f}% (target ↓)"))

  n_pass = sum(passes)
  print(f"\n    SUMMARY: {n_pass}/{len(passes)} gates passed.")
  if n_pass == len(passes):
    print("    → Path 4 meets all QA gates. Ready to default-enable for Mode 0.")
  elif passes[0] and n_pass >= 4:
    print("    → Mixed result: OS reduction met, some side effects. Review individual gates.")
  else:
    print("    → Path 4 does NOT meet primary OS gate. Tune smooth_tau_release or revert.")

  return {'off': off_m, 'on': on_m, 'gates_passed': n_pass, 'gates_total': len(passes)}


# ─────────────────────────────────────────────────────────────────────────────
# DISTURBANCE vs HUNTING ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_disturbance_vs_hunting(u, label):
  """
  V6 Section 8: Disturbance vs Hunting Analysis.
  Determines whether lateral oscillations are caused by the controller (hunting)
  or external disturbances (wind, road crown, bumps).
  Requires scipy.
  """
  if not HAS_SCIPY:
    print(f"\n{'='*80}")
    print(f"DISTURBANCE vs HUNTING ANALYSIS — {label}")
    print(f"{'='*80}")
    print("  Skipped: scipy not available")
    return {}

  print(f"\n{'='*80}")
  print(f"DISTURBANCE vs HUNTING ANALYSIS — {label}")
  print(f"{'='*80}")

  t, v, lat = u['t'], u['v'], u['lat_active']
  app   = u['applied_curv']   # commanded curvature (carOutput)
  meas  = u['measured_curv']  # measured curvature (-yawRate/vEgo)
  path_y = u['path_y']        # lane offset from modelV2
  lat_accel_meas = u['lat_accel_meas']
  dt    = np.median(np.diff(t))
  fs    = 1.0 / dt
  results = {}

  # Highway mask: engaged & v > 25 m/s
  hwy_mask = lat & (v > 25.0)
  n_hwy = np.sum(hwy_mask)
  hwy_secs = n_hwy * dt

  if n_hwy < 500:
    print(f"  Insufficient highway data ({n_hwy} samples, {hwy_secs:.0f}s). Need >500 samples.")
    return results

  print(f"  Highway data: {n_hwy} samples ({hwy_secs:.0f}s) at {fs:.0f} Hz")

  # Extract contiguous highway segments (for spectral analysis, need contiguous data)
  hwy_indices = np.where(hwy_mask)[0]
  breaks = np.where(np.diff(hwy_indices) > 1)[0]
  segments = []
  start = 0
  for b in breaks:
    seg = hwy_indices[start:b+1]
    if len(seg) >= 200:  # at least 2s of contiguous data
      segments.append(seg)
    start = b + 1
  last_seg = hwy_indices[start:]
  if len(last_seg) >= 200:
    segments.append(last_seg)

  if not segments:
    print("  No contiguous highway segments long enough for spectral analysis.")
    return results

  # Use the longest contiguous segment for spectral analysis
  longest = max(segments, key=len)
  app_seg  = app[longest]
  meas_seg = meas[longest]
  path_seg = path_y[longest]
  v_seg    = v[longest]

  print(f"  Longest contiguous segment: {len(longest)} samples ({len(longest)*dt:.0f}s)")

  # ── 1. Lead/Lag Analysis (Cross-correlation) ──────────────────────────────
  print(f"\n  1) Lead/Lag Analysis (Cross-correlation of curvature derivatives)")
  print(f"     Positive lag = command leads (hunting), Negative = measured leads (disturbance)")

  speed_bins_xcorr = [
    ("25-29 m/s (56-65 mph)", 25.0, 29.0),
    ("29-33 m/s (65-74 mph)", 29.0, 33.0),
    ("ALL >25 m/s",           25.0, 999.0),
  ]

  print(f"\n  {'Speed bin':<28} {'Secs':>5} {'Peak Lag':>10} {'Peak r':>8} {'Interpretation':>18}")
  print('  ' + '-'*75)

  xcorr_results = {}
  for name, vlo, vhi in speed_bins_xcorr:
    # Gather contiguous data within speed range for this bin
    bin_mask = hwy_mask & (v >= vlo) & (v < vhi)
    n_bin = np.sum(bin_mask)
    if n_bin < 200:
      print(f"  {name:<28} {'<200pts':>5}")
      continue
    secs = n_bin * dt

    # For cross-correlation, use all highway data in bin (interpolated so already uniform)
    bin_app  = app[bin_mask]
    bin_meas = meas[bin_mask]

    # Compute derivatives
    d_app  = np.gradient(bin_app, dt)
    d_meas = np.gradient(bin_meas, dt)

    # Normalize
    d_app_n  = (d_app - np.mean(d_app))
    d_meas_n = (d_meas - np.mean(d_meas))
    norm = np.sqrt(np.sum(d_app_n**2) * np.sum(d_meas_n**2))

    if norm < 1e-20:
      print(f"  {name:<28} {secs:>5.0f}s {'—':>10} {'—':>8} {'no signal':>18}")
      continue

    # Cross-correlate within +/- 500ms
    max_lag_samples = int(0.5 * fs)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    xcorr = np.zeros(len(lags))
    for i, lag in enumerate(lags):
      if lag >= 0:
        xcorr[i] = np.sum(d_app_n[:len(d_app_n)-lag] * d_meas_n[lag:]) / norm
      else:
        xcorr[i] = np.sum(d_app_n[-lag:] * d_meas_n[:len(d_meas_n)+lag]) / norm

    peak_idx = np.argmax(xcorr)
    peak_lag_ms = lags[peak_idx] * dt * 1000
    peak_r = xcorr[peak_idx]

    if peak_lag_ms > 10:
      interp_str = "HUNTING (cmd leads)"
    elif peak_lag_ms < -10:
      interp_str = "DISTURB (meas leads)"
    else:
      interp_str = "SIMULTANEOUS"

    print(f"  {name:<28} {secs:>5.0f}s {peak_lag_ms:>+8.1f}ms {peak_r:>8.3f} {interp_str:>18}")
    xcorr_results[name] = {'lag_ms': peak_lag_ms, 'peak_r': peak_r, 'interp': interp_str}

  results['xcorr'] = xcorr_results

  # ── 2. Frequency Content Analysis ─────────────────────────────────────────
  print(f"\n  2) Frequency Content Analysis (Welch PSD)")
  print(f"     Narrowband (peaked) = hunting, Broadband (flat) = disturbance")

  nperseg = min(512, len(longest) // 2)
  if nperseg < 64:
    print("     Segment too short for reliable spectral analysis.")
  else:
    freqs_w, psd_cmd = welch(app_seg - np.mean(app_seg), fs=fs, nperseg=nperseg)
    _, psd_meas      = welch(meas_seg - np.mean(meas_seg), fs=fs, nperseg=nperseg)

    # Only consider 0.05-5 Hz range
    valid = (freqs_w >= 0.05) & (freqs_w <= 5.0)
    freqs_v = freqs_w[valid]
    psd_cmd_v = psd_cmd[valid]
    psd_meas_v = psd_meas[valid]

    total_cmd  = np.sum(psd_cmd_v)
    total_meas = np.sum(psd_meas_v)

    if total_cmd > 0 and total_meas > 0:
      # Top 3 peaks for each
      for sig_name, psd_v, total_p in [("Commanded", psd_cmd_v, total_cmd),
                                         ("Measured",  psd_meas_v, total_meas)]:
        top3_idx = np.argsort(psd_v)[-3:][::-1]
        print(f"\n     {sig_name} curvature — top 3 frequency peaks:")
        print(f"     {'Freq':>8s}  {'Period':>8s}  {'Power%':>8s}  {'NB Ratio':>9s}  {'Type':>12s}")
        dom_freq = freqs_v[top3_idx[0]]
        dom_pct  = 100.0 * psd_v[top3_idx[0]] / total_p

        for idx in top3_idx:
          f = freqs_v[idx]
          pct = 100.0 * psd_v[idx] / total_p
          period = 1.0 / f if f > 0 else 999.0
          # Narrowband ratio: power within +/-0.02 Hz of peak / total
          band_mask = (freqs_v >= f - 0.02) & (freqs_v <= f + 0.02)
          nb_power = np.sum(psd_v[band_mask])
          nb_ratio = nb_power / total_p if total_p > 0 else 0
          nb_type = "narrowband" if nb_ratio > 0.15 else "broadband"
          print(f"     {f:>7.3f} Hz  {period:>6.1f}s    {pct:>6.1f}%  {nb_ratio:>8.3f}   {nb_type:>12s}")

        if sig_name == "Commanded":
          results['dom_freq_cmd'] = float(dom_freq)
          results['dom_pct_cmd'] = float(dom_pct)
          # Narrowband ratio for dominant peak
          band_mask = (freqs_v >= dom_freq - 0.02) & (freqs_v <= dom_freq + 0.02)
          results['nb_ratio_cmd'] = float(np.sum(psd_cmd_v[band_mask]) / total_cmd)
        else:
          results['dom_freq_meas'] = float(dom_freq)
          results['dom_pct_meas'] = float(dom_pct)
          band_mask = (freqs_v >= dom_freq - 0.02) & (freqs_v <= dom_freq + 0.02)
          results['nb_ratio_meas'] = float(np.sum(psd_meas_v[band_mask]) / total_meas)

    else:
      print("     Insufficient spectral power for analysis.")

  # ── 3. Spectral Coherence ─────────────────────────────────────────────────
  print(f"\n  3) Spectral Coherence (commanded curvature vs lane offset)")
  print(f"     High coherence (>0.5) at a frequency = systematic oscillation (hunting)")
  print(f"     Low coherence = random disturbances")

  if nperseg >= 64 and len(longest) >= 128:
    freqs_c, coh = sp_coherence(app_seg - np.mean(app_seg),
                                 path_seg - np.mean(path_seg),
                                 fs=fs, nperseg=min(256, nperseg))
    valid_c = (freqs_c >= 0.05) & (freqs_c <= 5.0)
    freqs_cv = freqs_c[valid_c]
    coh_v = coh[valid_c]

    if len(coh_v) > 0:
      # Report coherence at dominant oscillation frequency
      dom_freq = results.get('dom_freq_cmd', 0.0)
      if dom_freq > 0:
        coh_at_dom = float(np.interp(dom_freq, freqs_cv, coh_v))
        print(f"     Coherence at dominant freq ({dom_freq:.3f} Hz): {coh_at_dom:.3f}")
        if coh_at_dom > 0.5:
          print(f"     --> HIGH coherence: systematic oscillation (hunting likely)")
        elif coh_at_dom > 0.3:
          print(f"     --> MODERATE coherence: partial systematic component")
        else:
          print(f"     --> LOW coherence: random/disturbance-driven")
        results['coherence_at_dom'] = coh_at_dom

      # Also find max coherence frequency
      max_coh_idx = np.argmax(coh_v)
      max_coh_freq = freqs_cv[max_coh_idx]
      max_coh_val = coh_v[max_coh_idx]
      print(f"     Peak coherence: {max_coh_val:.3f} at {max_coh_freq:.3f} Hz ({1/max_coh_freq:.1f}s period)")
      results['max_coherence'] = float(max_coh_val)
      results['max_coherence_freq'] = float(max_coh_freq)
    else:
      print("     Insufficient coherence data.")
  else:
    print("     Segment too short for coherence analysis.")

  # ── 4. Phase Analysis ─────────────────────────────────────────────────────
  print(f"\n  4) Phase Analysis at dominant oscillation frequency")
  print(f"     0 deg = in-phase (hunting), 180 deg = anti-phase (active damping)")

  dom_freq = results.get('dom_freq_cmd', 0.0)
  if dom_freq > 0 and nperseg >= 64:
    freqs_p, pxy = sp_csd(app_seg - np.mean(app_seg),
                           meas_seg - np.mean(meas_seg),
                           fs=fs, nperseg=min(256, nperseg))
    valid_p = (freqs_p >= 0.05) & (freqs_p <= 5.0)
    freqs_pv = freqs_p[valid_p]
    pxy_v = pxy[valid_p]

    if len(pxy_v) > 0:
      # Interpolate cross-spectral density at dominant frequency
      csd_at_dom = np.interp(dom_freq, freqs_pv, pxy_v)
      phase_deg = float(np.degrees(np.angle(csd_at_dom)))
      print(f"     Phase at {dom_freq:.3f} Hz: {phase_deg:+.1f} deg")

      if abs(phase_deg) < 45:
        phase_interp = "IN-PHASE (hunting — cmd and car move together)"
      elif abs(phase_deg) > 135:
        phase_interp = "ANTI-PHASE (active damping or disturbance rejection)"
      elif phase_deg > 0:
        phase_interp = f"COMMAND LEADS by {phase_deg:.0f} deg (hunting with delay)"
      else:
        phase_interp = f"MEASURED LEADS by {abs(phase_deg):.0f} deg (disturbance response)"
      print(f"     --> {phase_interp}")
      results['phase_deg'] = phase_deg
      results['phase_interp'] = phase_interp
    else:
      print("     Insufficient data for phase analysis.")
  else:
    print("     No dominant frequency identified; skipping phase analysis.")

  # ── 5. Lateral Disturbance Detection ──────────────────────────────────────
  print(f"\n  5) Lateral Disturbance Detection")
  print(f"     External disturbance = large measured lateral accel change with small command change")

  # Use all highway data
  hwy_app  = app[hwy_mask]
  hwy_meas_la = lat_accel_meas[hwy_mask]
  hwy_v = v[hwy_mask]

  # Compute changes over ~100ms window
  win = max(1, int(0.1 / dt))
  d_meas_la = np.abs(np.concatenate([[0]*win, hwy_meas_la[win:] - hwy_meas_la[:-win]]))
  d_cmd_curv = np.abs(np.concatenate([[0]*win, hwy_app[win:] - hwy_app[:-win]]))

  # Thresholds
  lat_accel_thresh = 0.3   # m/s^2 change = significant lateral accel event
  cmd_thresh = 0.0003      # 1/m change = significant command change

  disturbance_events = (d_meas_la > lat_accel_thresh) & (d_cmd_curv < cmd_thresh)
  command_events = (d_cmd_curv > cmd_thresh)
  total_osc_events = disturbance_events | command_events

  n_disturbance = np.sum(disturbance_events)
  n_command = np.sum(command_events)
  n_total_osc = np.sum(total_osc_events)

  # Distance in miles
  distance_mi = np.sum(hwy_v) * dt / 1609.34
  if distance_mi > 0:
    dist_per_mi = n_disturbance / distance_mi
    cmd_per_mi = n_command / distance_mi
  else:
    dist_per_mi = 0
    cmd_per_mi = 0

  dist_ratio = n_disturbance / n_total_osc * 100 if n_total_osc > 0 else 0

  print(f"     Highway distance: {distance_mi:.1f} miles")
  print(f"     Disturbance events (lat accel >{lat_accel_thresh} m/s^2, cmd <{cmd_thresh} 1/m): "
        f"{n_disturbance} ({dist_per_mi:.0f}/mile)")
  print(f"     Command-driven events (cmd >{cmd_thresh} 1/m): "
        f"{n_command} ({cmd_per_mi:.0f}/mile)")
  print(f"     Disturbance ratio: {dist_ratio:.1f}% of all oscillation events")

  if dist_ratio > 60:
    print(f"     --> Oscillations are DISTURBANCE-DOMINATED")
  elif dist_ratio < 30:
    print(f"     --> Oscillations are COMMAND-DOMINATED (hunting)")
  else:
    print(f"     --> MIXED: both disturbance and hunting contribute")

  results['n_disturbance'] = int(n_disturbance)
  results['n_command'] = int(n_command)
  results['disturbance_ratio'] = float(dist_ratio)
  results['dist_per_mi'] = float(dist_per_mi)
  results['cmd_per_mi'] = float(cmd_per_mi)

  # ── 6. Summary Verdict ────────────────────────────────────────────────────
  print(f"\n  6) Summary Verdict")
  print(f"  {'─'*70}")

  evidence_hunting = 0
  evidence_disturb = 0
  evidence_lines = []

  # Cross-correlation evidence
  all_xcorr = xcorr_results.get("ALL >25 m/s", {})
  if all_xcorr:
    lag = all_xcorr.get('lag_ms', 0)
    if lag > 10:
      evidence_hunting += 1
      evidence_lines.append(f"  Cross-corr: command leads by {lag:+.1f}ms (hunting)")
    elif lag < -10:
      evidence_disturb += 1
      evidence_lines.append(f"  Cross-corr: measured leads by {lag:+.1f}ms (disturbance)")
    else:
      evidence_lines.append(f"  Cross-corr: simultaneous ({lag:+.1f}ms, inconclusive)")

  # Narrowband ratio evidence
  nb = results.get('nb_ratio_cmd', 0)
  if nb > 0.15:
    evidence_hunting += 1
    evidence_lines.append(f"  Spectral: narrowband ratio {nb:.3f} (>0.15 = hunting)")
  else:
    evidence_disturb += 1
    evidence_lines.append(f"  Spectral: narrowband ratio {nb:.3f} (<0.15 = broadband/disturbance)")

  # Coherence evidence
  coh_dom = results.get('coherence_at_dom', 0)
  if coh_dom > 0.5:
    evidence_hunting += 1
    evidence_lines.append(f"  Coherence: {coh_dom:.3f} at dominant freq (>0.5 = systematic)")
  elif coh_dom > 0:
    evidence_disturb += 1
    evidence_lines.append(f"  Coherence: {coh_dom:.3f} at dominant freq (<0.5 = random)")

  # Phase evidence
  phase = results.get('phase_deg', None)
  if phase is not None:
    if abs(phase) < 45:
      evidence_hunting += 1
      evidence_lines.append(f"  Phase: {phase:+.1f} deg (in-phase = hunting)")
    elif abs(phase) > 135:
      evidence_disturb += 1
      evidence_lines.append(f"  Phase: {phase:+.1f} deg (anti-phase = damping/disturbance)")
    else:
      evidence_lines.append(f"  Phase: {phase:+.1f} deg (intermediate, inconclusive)")

  # Disturbance ratio evidence
  if dist_ratio > 60:
    evidence_disturb += 1
    evidence_lines.append(f"  Event ratio: {dist_ratio:.0f}% disturbance-driven (>60% = disturbance)")
  elif dist_ratio < 30:
    evidence_hunting += 1
    evidence_lines.append(f"  Event ratio: {dist_ratio:.0f}% disturbance-driven (<30% = hunting)")
  else:
    evidence_lines.append(f"  Event ratio: {dist_ratio:.0f}% disturbance-driven (30-60% = mixed)")

  if evidence_hunting > evidence_disturb + 1:
    verdict = "HUNTING-DOMINANT"
  elif evidence_disturb > evidence_hunting + 1:
    verdict = "DISTURBANCE-DOMINANT"
  else:
    verdict = "MIXED"

  print(f"\n  Evidence (hunting={evidence_hunting}, disturbance={evidence_disturb}):")
  for line in evidence_lines:
    print(f"    {line}")
  print(f"\n  >>> VERDICT: {verdict} <<<")
  results['verdict'] = verdict
  results['evidence_hunting'] = evidence_hunting
  results['evidence_disturb'] = evidence_disturb

  return results


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
    print("Usage: analyze_drive_v6.py <route_dir> [route_dir2 ...]")
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

    # v6 new sections
    micro_osc  = analyze_micro_oscillation(u, label)
    pi_diag    = analyze_pi_diagnostics(data, u, label)
    dist_hunt  = analyze_disturbance_vs_hunting(u, label)
    path4_ab   = analyze_path4_ab(data, u, label)

    results.append({
      'label': label, 'overview': overview, 'filter': filter_eff,
      'hunt': hunt, 'low_speed': low_speed, 'curves': curves,
      'lane_center': lane_center, 'curve_dyn': curve_dyn,
      'override': override, 'comfort': comfort,
      'micro_osc': micro_osc, 'pi_diag': pi_diag,
      'dist_hunt': dist_hunt, 'path4_ab': path4_ab,
    })

  print_comparison(results)


if __name__ == '__main__':
  main()
