#!/usr/bin/env python3
"""
Explorer ST Longitudinal Control Analysis — v1
Analyzes driving logs for acceleration, braking, following, and comfort metrics.

Sections:
  1. Overview: duration, distance, speed stats, engage/disengage, overrides
  2. Acceleration Response: commanded vs actual, gain by accel bin and speed bin
  3. Response Lag: cross-correlation delay between commanded and actual accel
  4. Gas/Brake Transitions: zero-crossing counts, coasting opportunity analysis
  5. Jerk Analysis: longitudinal jerk stats, speed-binned, ISO comfort, asymmetry
  6. Following Distance: lead distance, time gap distribution, T_FOLLOW comparison
  7. Braking Events: hard brake catalog, brake onset rate
  8. Acceleration Events: unnecessary accel, speed overshoot
  9. Stopping Behavior: stop events, creep, resume profile
 10. Speed Maintenance: cruise speed holding smoothness
 11. PID State: accel output, integral windup detection
"""

import sys
import os
import glob
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
  from scipy.signal import correlate
  HAS_SCIPY = True
except ImportError:
  HAS_SCIPY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openpilot.tools.lib.logreader import LogReader

# Ford Explorer ST constants
ACCEL_MAX = 2.0       # m/s^2
ACCEL_MIN = -3.5      # m/s^2
MIN_GAS = -0.5
INACTIVE_GAS = -5.0
MPS_TO_MPH = 2.23694

# T_FOLLOW targets from planner (comma stock profiles)
T_FOLLOW_TARGETS = {
  'close': 1.25,
  'medium': 1.45,
  'far': 1.75,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_route(route_dir, sample_every=1):
  """Load rlog files from route directory. sample_every=N loads every Nth segment."""
  files = sorted(glob.glob(os.path.join(route_dir, 'rlog_*.zst')),
                 key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

  # Format 2: per-segment dirs {route}--{n}/rlog.zst
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

  if sample_every > 1:
    files = files[::sample_every]

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
      data['a_ego'].append(cs.aEgo)
      data['gas_pressed'].append(cs.gasPressed)
      data['brake_pressed'].append(cs.brakePressed)
      data['steering_pressed'].append(cs.steeringPressed)
      data['v_cruise'].append(cs.cruiseState.speed)  # m/s
      data['cruise_enabled'].append(cs.cruiseState.enabled)
      data['standstill'].append(cs.standstill)

    elif w == 'carControl':
      cc = msg.carControl
      data['cc_t'].append(t)
      data['cc_accel'].append(cc.actuators.accel)
      data['long_active'].append(cc.longActive)
      data['cc_enabled'].append(cc.enabled)

    elif w == 'carOutput':
      co = msg.carOutput
      data['co_t'].append(t)
      data['co_accel'].append(co.actuatorsOutput.accel)

    elif w == 'radarState':
      rs = msg.radarState
      data['rs_t'].append(t)
      lead = rs.leadOne
      data['lead_status'].append(lead.status)
      data['lead_dRel'].append(lead.dRel)
      data['lead_vRel'].append(lead.vRel)
      data['lead_aRel'].append(lead.aRel)

    elif w == 'longitudinalPlan':
      lp = msg.longitudinalPlan
      data['lp_t'].append(t)
      speeds = list(lp.speeds)
      accels = list(lp.accels)
      jerks = list(lp.jerks)
      data['lp_speed0'].append(speeds[0] if speeds else 0.0)
      data['lp_accel0'].append(accels[0] if accels else 0.0)
      data['lp_jerk0'].append(jerks[0] if jerks else 0.0)

  # Convert to numpy
  for k in data:
    data[k] = np.array(data[k], dtype=float if k not in ('gas_pressed', 'brake_pressed',
                        'steering_pressed', 'cruise_enabled', 'standstill', 'lead_status') else bool)

  return data


# ─────────────────────────────────────────────────────────────────────────────
# TIMELINE BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_timeline(data):
  t = data.get('cs_t', np.array([]))
  if len(t) < 2:
    return None

  def interp(src_t, src_v, default=0.0):
    if len(src_t) > 1:
      return np.interp(t, src_t, src_v)
    return np.full(len(t), default)

  def interp_bool(src_t, src_v):
    if len(src_t) > 1:
      return np.interp(t, src_t, src_v.astype(float)) > 0.5
    return np.full(len(t), False)

  u = {}
  u['t'] = t
  u['dt'] = np.median(np.diff(t))
  u['v'] = data['v_ego']
  u['a_ego'] = data['a_ego']
  u['gas_pressed'] = data['gas_pressed']
  u['brake_pressed'] = data['brake_pressed']
  u['steering_pressed'] = data['steering_pressed']
  u['v_cruise'] = data['v_cruise']
  u['cruise_enabled'] = data['cruise_enabled']
  u['standstill'] = data['standstill']

  # Interpolate carControl onto carState timeline
  cc_t = data.get('cc_t', np.array([]))
  u['cc_accel'] = interp(cc_t, data.get('cc_accel', np.array([])))
  u['long_active'] = interp_bool(cc_t, data.get('long_active', np.array([])))
  u['cc_enabled'] = interp_bool(cc_t, data.get('cc_enabled', np.array([])))

  # carOutput
  co_t = data.get('co_t', np.array([]))
  u['co_accel'] = interp(co_t, data.get('co_accel', np.array([])))

  # radarState
  rs_t = data.get('rs_t', np.array([]))
  u['lead_status'] = interp_bool(rs_t, data.get('lead_status', np.array([])))
  u['lead_dRel'] = interp(rs_t, data.get('lead_dRel', np.array([])))
  u['lead_vRel'] = interp(rs_t, data.get('lead_vRel', np.array([])))
  u['lead_aRel'] = interp(rs_t, data.get('lead_aRel', np.array([])))

  # longitudinalPlan
  lp_t = data.get('lp_t', np.array([]))
  u['lp_speed0'] = interp(lp_t, data.get('lp_speed0', np.array([])))
  u['lp_accel0'] = interp(lp_t, data.get('lp_accel0', np.array([])))
  u['lp_jerk0'] = interp(lp_t, data.get('lp_jerk0', np.array([])))

  # Derived: longitudinal jerk (with light smoothing to suppress sensor noise)
  dt_arr = np.diff(t)
  dt_arr = np.where(dt_arr > 0, dt_arr, u['dt'])
  raw_jerk = np.concatenate([[0.0], np.diff(u['a_ego']) / dt_arr])
  # 5-sample moving average (~0.05s at 100Hz) to smooth sensor noise
  kernel = 5
  if len(raw_jerk) > kernel:
    cumsum = np.cumsum(np.insert(raw_jerk, 0, 0))
    smoothed = (cumsum[kernel:] - cumsum[:-kernel]) / kernel
    # Pad edges
    u['jerk'] = np.concatenate([raw_jerk[:kernel//2], smoothed, raw_jerk[-(kernel - kernel//2 - 1):]])
    # Ensure same length
    if len(u['jerk']) != len(t):
      u['jerk'] = raw_jerk  # fallback
  else:
    u['jerk'] = raw_jerk

  return u


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def analyze_overview(u, label):
  print(f"\n{'='*80}")
  print(f"1. OVERVIEW — {label}")
  print(f"{'='*80}")

  t, v, dt = u['t'], u['v'], u['dt']
  duration = t[-1] - t[0]
  distance = np.sum(v[:-1] * np.diff(t))

  print(f"\n  Duration:      {duration/60:.1f} min ({duration:.0f} s)")
  print(f"  Distance:      {distance/1609.34:.1f} mi ({distance/1000:.1f} km)")
  print(f"  Speed mean:    {np.mean(v)*MPS_TO_MPH:.1f} mph ({np.mean(v):.1f} m/s)")
  print(f"  Speed max:     {np.max(v)*MPS_TO_MPH:.1f} mph")
  print(f"  Speed std:     {np.std(v)*MPS_TO_MPH:.1f} mph")

  # Longitudinal engaged
  long_active = u['long_active']
  active_pct = 100 * np.mean(long_active)

  # Count engage/disengage transitions
  changes = np.diff(long_active.astype(int))
  engages = np.sum(changes == 1)
  disengages = np.sum(changes == -1)

  print(f"\n  Long engaged:  {active_pct:.1f}%")
  print(f"  Engages:       {engages}")
  print(f"  Disengages:    {disengages}")

  # Gas/brake overrides while engaged
  gas_override = long_active & u['gas_pressed']
  brake_override = long_active & u['brake_pressed']

  # Count distinct events (transitions from not-pressed to pressed)
  gas_events = np.sum(np.diff(gas_override.astype(int)) == 1)
  brake_events = np.sum(np.diff(brake_override.astype(int)) == 1)
  gas_pct = 100 * np.mean(gas_override) if np.any(long_active) else 0
  brake_pct = 100 * np.mean(brake_override) if np.any(long_active) else 0

  print(f"\n  Gas overrides:   {gas_events} events ({gas_pct:.1f}% of engaged time)")
  print(f"  Brake overrides: {brake_events} events ({brake_pct:.1f}% of engaged time)")

  # Accel stats while engaged
  mask = long_active & (v > 0.5)
  if np.any(mask):
    a = u['a_ego'][mask]
    print(f"\n  Accel (engaged, moving):")
    print(f"    Mean:  {np.mean(a):+.3f} m/s^2")
    print(f"    Std:   {np.std(a):.3f} m/s^2")
    print(f"    Min:   {np.min(a):+.3f} m/s^2")
    print(f"    Max:   {np.max(a):+.3f} m/s^2")

  return {'duration': duration, 'distance': distance, 'active_pct': active_pct}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: ACCELERATION RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_accel_response(u, label):
  print(f"\n{'='*80}")
  print(f"2. ACCELERATION RESPONSE — {label}")
  print(f"{'='*80}")

  mask = u['long_active'] & (u['v'] > 0.5)
  cmd = u['cc_accel'][mask]
  act = u['a_ego'][mask]
  v = u['v'][mask]

  if len(cmd) < 10:
    print("  Insufficient data")
    return {}

  # A) Binned by command level
  bins = [
    ("Hard brake     (<-2.0)",     -99.0, -2.0),
    ("Moderate brake (-2 to -1)",  -2.0,  -1.0),
    ("Light brake    (-1 to -0.3)",-1.0,  -0.3),
    ("Coast          (-0.3 to 0)", -0.3,   0.0),
    ("Light accel    (0 to 0.5)",   0.0,   0.5),
    ("Moderate accel (0.5 to 1)",   0.5,   1.0),
    ("Hard accel     (>1.0)",       1.0,  99.0),
  ]

  print(f"\n  A) Response by Command Level")
  print(f"  {'Bin':<32} {'N':>6} {'Cmd(mean)':>10} {'Act(mean)':>10} {'Gain':>7} {'Overshoot':>10}")
  print('  ' + '-' * 82)

  for bname, blo, bhi in bins:
    m = (cmd >= blo) & (cmd < bhi)
    n = np.sum(m)
    if n < 10:
      print(f"  {bname:<32} {n:>6}  (too few)")
      continue
    c_mean = np.mean(cmd[m])
    a_mean = np.mean(act[m])
    gain = a_mean / c_mean if abs(c_mean) > 0.01 else float('nan')
    # Overshoot: how much actual exceeds commanded (in magnitude)
    if c_mean < 0:
      overshoot = a_mean - c_mean  # negative means overbrake
    else:
      overshoot = a_mean - c_mean  # positive means over-accel
    print(f"  {bname:<32} {n:>6} {c_mean:>+10.3f} {a_mean:>+10.3f} {gain:>7.2f} {overshoot:>+10.3f}")

  # B) Speed-binned gain
  speed_bins = [
    ("0-15 mph",    0,    6.7),
    ("15-30 mph",   6.7, 13.4),
    ("30-45 mph",  13.4, 20.1),
    ("45-60 mph",  20.1, 26.8),
    ("60-75 mph",  26.8, 33.5),
    ("75+ mph",    33.5, 99.0),
  ]

  print(f"\n  B) Response Gain by Speed (engaged, |cmd| > 0.1)")
  print(f"  {'Speed Range':<16} {'N':>6} {'Brake Gain':>11} {'Accel Gain':>11}")
  print('  ' + '-' * 50)

  for sname, slo, shi in speed_bins:
    sm = (v >= slo) & (v < shi) & (np.abs(cmd) > 0.1)
    if np.sum(sm) < 20:
      continue
    brake_m = sm & (cmd < -0.1)
    accel_m = sm & (cmd > 0.1)
    bg = np.mean(act[brake_m]) / np.mean(cmd[brake_m]) if np.sum(brake_m) > 10 else float('nan')
    ag = np.mean(act[accel_m]) / np.mean(cmd[accel_m]) if np.sum(accel_m) > 10 else float('nan')
    print(f"  {sname:<16} {np.sum(sm):>6} {bg:>11.3f} {ag:>11.3f}")

  return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: RESPONSE LAG
# ─────────────────────────────────────────────────────────────────────────────

def analyze_response_lag(u, label):
  print(f"\n{'='*80}")
  print(f"3. RESPONSE LAG — {label}")
  print(f"{'='*80}")

  if not HAS_SCIPY:
    print("  scipy not available — skipping cross-correlation")
    return {}

  mask = u['long_active'] & (u['v'] > 2.0)
  cmd = u['cc_accel'][mask]
  act = u['a_ego'][mask]
  dt = u['dt']

  if len(cmd) < 200:
    print("  Insufficient data")
    return {}

  # Split into segments of ~60s for multiple measurements
  seg_len = int(60.0 / dt)
  n_segs = len(cmd) // seg_len
  if n_segs < 1:
    n_segs = 1
    seg_len = len(cmd)

  delays = []
  max_lag_samples = int(2.0 / dt)  # search up to 2 seconds

  for i in range(n_segs):
    s = i * seg_len
    e = s + seg_len
    c = cmd[s:e] - np.mean(cmd[s:e])
    a = act[s:e] - np.mean(act[s:e])

    if np.std(c) < 0.05 or np.std(a) < 0.05:
      continue

    corr = correlate(a, c, mode='full')
    mid = len(c) - 1
    # Only look at positive lags (actual lags behind commanded)
    search_start = mid
    search_end = min(mid + max_lag_samples, len(corr))
    if search_end <= search_start:
      continue
    peak = np.argmax(corr[search_start:search_end])
    delay_ms = peak * dt * 1000
    delays.append(delay_ms)

  if delays:
    print(f"\n  Cross-correlation delay (commanded -> actual):")
    print(f"    Segments analyzed: {len(delays)}")
    print(f"    Mean delay:    {np.mean(delays):.0f} ms")
    print(f"    Median delay:  {np.median(delays):.0f} ms")
    print(f"    Std:           {np.std(delays):.0f} ms")
    print(f"    Min:           {np.min(delays):.0f} ms")
    print(f"    Max:           {np.max(delays):.0f} ms")
    return {'delay_mean_ms': np.mean(delays), 'delay_std_ms': np.std(delays)}
  else:
    print("  Could not compute delay (insufficient variation)")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: GAS/BRAKE TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_transitions(u, label):
  print(f"\n{'='*80}")
  print(f"4. GAS/BRAKE TRANSITIONS — {label}")
  print(f"{'='*80}")

  mask = u['long_active'] & (u['v'] > 0.5)
  a = u['a_ego'][mask]
  t = u['t'][mask]
  cmd = u['cc_accel'][mask]
  dt = u['dt']

  if len(a) < 100:
    print("  Insufficient data")
    return {}

  duration_min = (t[-1] - t[0]) / 60.0 if len(t) > 1 else 1.0

  # Count sign transitions at different thresholds
  thresholds = [0.0, 0.1, 0.3]
  print(f"\n  A) Acceleration Sign Transitions (aEgo)")
  print(f"  {'Threshold':>12} {'Pos->Neg':>10} {'Neg->Pos':>10} {'Total':>8} {'Per min':>10}")
  print('  ' + '-' * 55)

  for thresh in thresholds:
    pos = a > thresh
    neg = a < -thresh
    p2n = np.sum(pos[:-1] & neg[1:])
    n2p = np.sum(neg[:-1] & pos[1:])
    total = p2n + n2p
    per_min = total / duration_min if duration_min > 0 else 0
    print(f"  {thresh:>+12.1f} {p2n:>10} {n2p:>10} {total:>8} {per_min:>10.1f}")

  # B) Coasting opportunity
  print(f"\n  B) Coasting Opportunity Analysis")
  print(f"  Examines when light braking (-0.5 to 0 m/s^2) was commanded")
  print(f"  vs what engine braking / coasting would provide (~-0.3 m/s^2)")

  light_brake = (cmd >= -0.5) & (cmd < 0.0)
  n_light = np.sum(light_brake)
  if n_light > 10:
    cmd_light = cmd[light_brake]
    act_light = a[light_brake]
    # Engine braking typically provides about -0.2 to -0.3 m/s^2
    engine_brake_approx = -0.25
    could_coast = np.sum(cmd_light > engine_brake_approx)
    coast_pct = 100.0 * could_coast / n_light
    print(f"    Light brake commands:  {n_light} samples ({n_light*dt:.1f}s)")
    print(f"    Mean commanded:       {np.mean(cmd_light):+.3f} m/s^2")
    print(f"    Mean actual:          {np.mean(act_light):+.3f} m/s^2")
    print(f"    Could coast instead:  {could_coast} ({coast_pct:.1f}%) — commanded > {engine_brake_approx} m/s^2")
  else:
    print(f"    Too few light brake commands ({n_light} samples)")

  return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: JERK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_jerk(u, label):
  print(f"\n{'='*80}")
  print(f"5. JERK ANALYSIS — {label}")
  print(f"{'='*80}")

  mask = u['long_active'] & (u['v'] > 0.5)
  jerk = u['jerk'][mask]
  v = u['v'][mask]
  a = u['a_ego'][mask]
  dt = u['dt']

  if len(jerk) < 100:
    print("  Insufficient data")
    return {}

  # Filter out spurious jerk from data gaps (|jerk| > 50 is clearly noise)
  valid = np.abs(jerk) < 50
  jerk = jerk[valid]
  v = v[valid]
  a = a[valid]

  # A) Overall stats
  print(f"\n  A) Longitudinal Jerk Statistics (engaged, moving)")
  print(f"    Mean:     {np.mean(jerk):+.3f} m/s^3")
  print(f"    Std:      {np.std(jerk):.3f} m/s^3")
  print(f"    P5:       {np.percentile(jerk, 5):+.3f} m/s^3")
  print(f"    P95:      {np.percentile(jerk, 95):+.3f} m/s^3")
  print(f"    P99:      {np.percentile(jerk, 99):+.3f} m/s^3")
  print(f"    P1:       {np.percentile(jerk, 1):+.3f} m/s^3")
  abs_jerk = np.abs(jerk)
  print(f"    |Jerk| mean:  {np.mean(abs_jerk):.3f} m/s^3")
  print(f"    |Jerk| P95:   {np.percentile(abs_jerk, 95):.3f} m/s^3")
  print(f"    |Jerk| P99:   {np.percentile(abs_jerk, 99):.3f} m/s^3")

  # B) ISO comfort thresholds
  # ISO 2631-1: longitudinal jerk comfort thresholds
  # "Not uncomfortable" < 0.5, "A little" 0.5-1.0, "Fairly" 1.0-2.0, "Uncomfortable" > 2.0
  print(f"\n  B) ISO 2631 Longitudinal Jerk Distribution")
  comfort_bins = [
    ("Not uncomfortable", 0.0, 0.5),
    ("A little uncomfortable", 0.5, 1.0),
    ("Fairly uncomfortable", 1.0, 2.0),
    ("Uncomfortable", 2.0, 5.0),
    ("Very uncomfortable", 5.0, 999),
  ]
  for cname, clo, chi in comfort_bins:
    pct = 100.0 * np.mean((abs_jerk >= clo) & (abs_jerk < chi))
    print(f"    {cname:<28} {pct:>6.1f}%")

  # C) Speed-binned jerk
  speed_bins = [
    ("0-15 mph",    0,    6.7),
    ("15-30 mph",   6.7, 13.4),
    ("30-45 mph",  13.4, 20.1),
    ("45-60 mph",  20.1, 26.8),
    ("60+ mph",    26.8, 99.0),
  ]

  print(f"\n  C) Jerk by Speed")
  print(f"  {'Speed':<14} {'N':>7} {'|Jerk| mean':>12} {'|Jerk| P95':>12} {'|Jerk| P99':>12}")
  print('  ' + '-' * 62)

  for sname, slo, shi in speed_bins:
    sm = (v >= slo) & (v < shi)
    n = np.sum(sm)
    if n < 20:
      continue
    j = np.abs(jerk[sm])
    print(f"  {sname:<14} {n:>7} {np.mean(j):>12.3f} {np.percentile(j,95):>12.3f} {np.percentile(j,99):>12.3f}")

  # D) Asymmetry: braking jerk vs acceleration jerk
  print(f"\n  D) Jerk Asymmetry (braking vs acceleration)")
  braking = a < -0.1
  accel = a > 0.1
  if np.sum(braking) > 20 and np.sum(accel) > 20:
    j_brake = np.abs(jerk[braking])
    j_accel = np.abs(jerk[accel])
    print(f"    During braking:  |jerk| mean={np.mean(j_brake):.3f}, P95={np.percentile(j_brake,95):.3f} m/s^3")
    print(f"    During accel:    |jerk| mean={np.mean(j_accel):.3f}, P95={np.percentile(j_accel,95):.3f} m/s^3")
    ratio = np.mean(j_brake) / np.mean(j_accel) if np.mean(j_accel) > 0 else float('nan')
    print(f"    Brake/accel ratio: {ratio:.2f}x")
  else:
    print("    Insufficient data for asymmetry analysis")

  return {'jerk_mean': np.mean(abs_jerk), 'jerk_p95': np.percentile(abs_jerk, 95)}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FOLLOWING DISTANCE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_following(u, label):
  print(f"\n{'='*80}")
  print(f"6. FOLLOWING DISTANCE — {label}")
  print(f"{'='*80}")

  mask = u['long_active'] & u['lead_status'] & (u['v'] > 2.0)
  if np.sum(mask) < 100:
    print("  No lead vehicle data or insufficient engaged following")
    return {}

  d = u['lead_dRel'][mask]
  v = u['v'][mask]
  vrel = u['lead_vRel'][mask]
  dt = u['dt']

  # Time gap = distance / ego speed
  tgap = d / np.maximum(v, 0.5)

  # Following time stats
  print(f"\n  A) Following with Lead Vehicle")
  following_time = np.sum(mask) * dt
  total_active = np.sum(u['long_active']) * dt
  print(f"    Following time:    {following_time:.0f}s ({100*following_time/total_active:.1f}% of engaged time)")
  print(f"    Lead distance:     mean={np.mean(d):.1f}m, std={np.std(d):.1f}m")
  print(f"    Lead rel speed:    mean={np.mean(vrel)*MPS_TO_MPH:+.1f} mph")

  print(f"\n  B) Time Gap Distribution")
  print(f"    Mean:    {np.mean(tgap):.2f}s")
  print(f"    Std:     {np.std(tgap):.2f}s")
  print(f"    Median:  {np.median(tgap):.2f}s")
  print(f"    P5:      {np.percentile(tgap, 5):.2f}s (closest)")
  print(f"    P10:     {np.percentile(tgap, 10):.2f}s")
  print(f"    P25:     {np.percentile(tgap, 25):.2f}s")
  print(f"    P75:     {np.percentile(tgap, 75):.2f}s")
  print(f"    Max:     {np.max(tgap):.2f}s")

  # Histogram
  hist_bins = [0, 0.5, 1.0, 1.25, 1.45, 1.75, 2.0, 2.5, 3.0, 5.0, 999]
  hist_labels = ["<0.5s", "0.5-1.0s", "1.0-1.25s", "1.25-1.45s", "1.45-1.75s",
                 "1.75-2.0s", "2.0-2.5s", "2.5-3.0s", "3.0-5.0s", ">5.0s"]
  print(f"\n  C) Time Gap Histogram")
  for i, hlabel in enumerate(hist_labels):
    pct = 100 * np.mean((tgap >= hist_bins[i]) & (tgap < hist_bins[i+1]))
    bar = '#' * int(pct / 2)
    print(f"    {hlabel:<14} {pct:>5.1f}% {bar}")

  # Compare to T_FOLLOW targets
  print(f"\n  D) Time vs T_FOLLOW Targets")
  for tname, target in T_FOLLOW_TARGETS.items():
    below = 100 * np.mean(tgap < target)
    print(f"    Below {tname} ({target:.2f}s): {below:.1f}%")

  # Minimum safe following (2-second rule)
  below_2s = 100 * np.mean(tgap < 2.0)
  below_1s = 100 * np.mean(tgap < 1.0)
  print(f"\n    Below 2.0s (2-second rule): {below_2s:.1f}%")
  print(f"    Below 1.0s (dangerous):     {below_1s:.1f}%")

  return {'tgap_mean': np.mean(tgap), 'tgap_p5': np.percentile(tgap, 5)}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: BRAKING EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_braking(u, label):
  print(f"\n{'='*80}")
  print(f"7. BRAKING EVENTS — {label}")
  print(f"{'='*80}")

  mask = u['long_active']
  a = u['a_ego']
  t = u['t']
  v = u['v']
  dt = u['dt']
  jerk = u['jerk']
  lead_d = u['lead_dRel']
  lead_s = u['lead_status']

  if np.sum(mask) < 100:
    print("  Insufficient data")
    return {}

  duration_min = np.sum(mask) * dt / 60.0

  # Count events at thresholds
  thresholds = [1.0, 2.0, 3.0]
  print(f"\n  A) Braking Event Counts (engaged)")
  for thresh in thresholds:
    # Find events: decel exceeds threshold (not just individual samples)
    hard = mask & (a < -thresh)
    # Count distinct events (transitions into hard braking)
    events = np.sum(np.diff(hard.astype(int)) == 1)
    per_hr = events / (duration_min / 60) if duration_min > 0 else 0
    print(f"    Decel > {thresh:.1f} m/s^2: {events:>4} events ({per_hr:.1f}/hr)")

  # B) Catalog of hard brake events (> 2.0 m/s^2)
  hard_mask = mask & (a < -1.5)
  event_starts = np.where(np.diff(hard_mask.astype(int)) == 1)[0] + 1
  print(f"\n  B) Hard Brake Events (peak decel > 1.5 m/s^2)")

  if len(event_starts) > 0:
    print(f"  {'#':>3} {'Time':>8} {'Init Spd':>10} {'Peak Dec':>10} {'Duration':>10} {'Lead Dist':>10}")
    print('  ' + '-' * 58)

    events_shown = 0
    for idx in event_starts:
      # Find end of event
      end = idx
      while end < len(a) - 1 and a[end] < -0.5:
        end += 1
      if end - idx < 2:
        continue

      init_speed = v[idx] * MPS_TO_MPH
      peak_decel = np.min(a[idx:end])
      event_duration = (t[end] - t[idx])
      lead_dist = lead_d[idx] if lead_s[idx] else float('nan')

      events_shown += 1
      elapsed = t[idx] - t[0]
      time_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"
      ld_str = f"{lead_dist:.1f}m" if not np.isnan(lead_dist) else "no lead"
      print(f"  {events_shown:>3} {time_str:>8} {init_speed:>9.1f}mph {peak_decel:>+10.2f} {event_duration:>9.1f}s {ld_str:>10}")

      if events_shown >= 20:
        remaining = len(event_starts) - events_shown
        if remaining > 0:
          print(f"  ... and {remaining} more events")
        break

  # C) Brake onset rate (jerk at brake initiation)
  # Find moments where braking begins: a crosses below -0.3
  brake_onset_mask = mask[:-1] & (a[:-1] > -0.3) & (a[1:] <= -0.3)
  onset_indices = np.where(brake_onset_mask)[0]
  if len(onset_indices) > 5:
    onset_jerks = []
    for oi in onset_indices:
      # Average jerk over the first 0.5s of braking
      end_oi = min(oi + int(0.5 / dt), len(jerk) - 1)
      if end_oi > oi:
        onset_jerks.append(np.mean(jerk[oi:end_oi]))
    if onset_jerks:
      onset_jerks = np.array(onset_jerks)
      print(f"\n  C) Brake Onset Rate (jerk at brake initiation, first 0.5s)")
      print(f"    Events:  {len(onset_jerks)}")
      print(f"    Mean:    {np.mean(onset_jerks):+.3f} m/s^3")
      print(f"    P5:      {np.percentile(onset_jerks, 5):+.3f} m/s^3")
      print(f"    P95:     {np.percentile(onset_jerks, 95):+.3f} m/s^3")

  return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: ACCELERATION EVENTS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_accel_events(u, label):
  print(f"\n{'='*80}")
  print(f"8. ACCELERATION EVENTS — {label}")
  print(f"{'='*80}")

  mask = u['long_active'] & (u['v'] > 0.5)
  a = u['a_ego']
  t = u['t']
  v = u['v']
  v_cruise = u['v_cruise']
  dt = u['dt']

  if np.sum(mask) < 100:
    print("  Insufficient data")
    return {}

  duration_min = np.sum(mask) * dt / 60.0

  # A) Accel event counts
  thresholds = [0.5, 1.0, 1.5]
  print(f"\n  A) Acceleration Event Counts (engaged)")
  for thresh in thresholds:
    accel_mask = mask & (a > thresh)
    events = np.sum(np.diff(accel_mask.astype(int)) == 1)
    per_hr = events / (duration_min / 60) if duration_min > 0 else 0
    print(f"    Accel > {thresh:.1f} m/s^2: {events:>4} events ({per_hr:.1f}/hr)")

  # B) Unnecessary acceleration: accel > 0.5 followed by brake within 5s
  print(f"\n  B) Unnecessary Acceleration (accel > 0.5 then brake within 5s)")
  accel_starts = np.where(np.diff((mask & (a > 0.5)).astype(int)) == 1)[0] + 1
  unnecessary = 0
  for ai in accel_starts:
    # Look ahead 5 seconds
    lookahead = int(5.0 / dt)
    end = min(ai + lookahead, len(a))
    window = a[ai:end]
    if np.any(window < -0.3):
      unnecessary += 1

  if len(accel_starts) > 0:
    pct = 100 * unnecessary / len(accel_starts)
    print(f"    Accel events:       {len(accel_starts)}")
    print(f"    Followed by brake:  {unnecessary} ({pct:.1f}%)")
  else:
    print(f"    No acceleration events detected")

  # C) Speed overshoot past cruise speed
  print(f"\n  C) Speed Overshoot Past Cruise Speed")
  # Look at moments where v > v_cruise and system is engaged
  cruise_valid = mask & (v_cruise > 5.0)  # cruise speed must be meaningful
  if np.sum(cruise_valid) > 100:
    overshoot = v[cruise_valid] - v_cruise[cruise_valid]
    overshoot_mph = overshoot * MPS_TO_MPH
    over = overshoot_mph > 1.0  # more than 1 mph over
    if np.any(over):
      print(f"    Time > 1 mph over cruise:  {np.sum(over)*dt:.1f}s ({100*np.mean(over):.1f}%)")
      print(f"    Time > 2 mph over cruise:  {np.sum(overshoot_mph > 2.0)*dt:.1f}s")
      print(f"    Time > 5 mph over cruise:  {np.sum(overshoot_mph > 5.0)*dt:.1f}s")
      print(f"    Max overshoot:             {np.max(overshoot_mph):.1f} mph")
      print(f"    Mean overshoot (when >0):  {np.mean(overshoot_mph[overshoot_mph > 0]):+.1f} mph")
    else:
      print(f"    No significant speed overshoot detected")
  else:
    print(f"    Insufficient cruise data")

  return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: STOPPING BEHAVIOR
# ─────────────────────────────────────────────────────────────────────────────

def analyze_stopping(u, label):
  print(f"\n{'='*80}")
  print(f"9. STOPPING BEHAVIOR — {label}")
  print(f"{'='*80}")

  t = u['t']
  v = u['v']
  a = u['a_ego']
  dt = u['dt']
  long_active = u['long_active']

  # Find stop events: speed drops below 0.5 m/s while engaged
  moving = long_active & (v > 2.0)
  stopped = long_active & (v < 0.5)
  stop_transitions = np.where(np.diff(stopped.astype(int)) == 1)[0] + 1

  # Filter: must have been moving recently (within 10s before)
  stop_events = []
  lookback = int(10.0 / dt)
  for si in stop_transitions:
    start = max(0, si - lookback)
    if np.any(moving[start:si]):
      stop_events.append(si)

  print(f"\n  Stop events (engaged): {len(stop_events)}")

  if len(stop_events) == 0:
    return {}

  # Analyze each stop
  final_decels = []
  creep_distances = []
  resume_profiles = []

  print(f"\n  A) Stop Event Details")
  print(f"  {'#':>3} {'Time':>8} {'Approach Spd':>13} {'Final Decel':>12} {'Creep':>8}")
  print('  ' + '-' * 50)

  for idx, si in enumerate(stop_events[:15]):
    elapsed = t[si] - t[0]
    time_str = f"{int(elapsed//60)}:{int(elapsed%60):02d}"

    # Approach speed: max speed in the 5s before stop
    lookback_5s = int(5.0 / dt)
    start = max(0, si - lookback_5s)
    approach_speed = np.max(v[start:si]) * MPS_TO_MPH

    # Final decel: deceleration in last 1s before stop
    lookback_1s = int(1.0 / dt)
    start_1s = max(0, si - lookback_1s)
    final_decel = np.mean(a[start_1s:si]) if si > start_1s else 0
    final_decels.append(final_decel)

    # Creep after stop: any forward movement in the 3s after stopping
    lookahead_3s = int(3.0 / dt)
    end_3s = min(si + lookahead_3s, len(v))
    creep_speed = np.max(v[si:end_3s]) if end_3s > si else 0
    creep_distances.append(creep_speed)

    creep_str = f"{creep_speed*MPS_TO_MPH:.1f}mph" if creep_speed > 0.1 else "none"
    print(f"  {idx+1:>3} {time_str:>8} {approach_speed:>12.1f}mph {final_decel:>+12.2f} {creep_str:>8}")

  if len(stop_events) > 15:
    print(f"  ... and {len(stop_events)-15} more")

  if final_decels:
    print(f"\n  B) Final Decel Statistics")
    final_decels = np.array(final_decels)
    print(f"    Mean final decel:  {np.mean(final_decels):+.3f} m/s^2")
    print(f"    Std:               {np.std(final_decels):.3f} m/s^2")
    print(f"    Gentlest:          {np.max(final_decels):+.3f} m/s^2")
    print(f"    Hardest:           {np.min(final_decels):+.3f} m/s^2")

  # C) Resume acceleration profile (0 to 5 m/s = 0 to 11 mph)
  print(f"\n  C) Resume Acceleration Profile (0 -> 11 mph)")
  resume_times = []
  resume_accels = []
  for si in stop_events:
    # Find when speed exceeds 5 m/s after stop
    lookahead = int(15.0 / dt)  # up to 15s
    end = min(si + lookahead, len(v))
    window_v = v[si:end]
    crosses = np.where(window_v > 5.0)[0]
    if len(crosses) > 0:
      resume_idx = crosses[0]
      resume_time = resume_idx * dt
      resume_times.append(resume_time)
      # Mean accel during resume
      mean_accel = np.mean(a[si:si+resume_idx]) if resume_idx > 0 else 0
      resume_accels.append(mean_accel)

  if resume_times:
    print(f"    Events analyzed: {len(resume_times)}")
    print(f"    Mean time 0->11 mph:  {np.mean(resume_times):.1f}s")
    print(f"    Mean accel:           {np.mean(resume_accels):+.3f} m/s^2")
    print(f"    Fastest:              {np.min(resume_times):.1f}s")
    print(f"    Slowest:              {np.max(resume_times):.1f}s")

  return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: SPEED MAINTENANCE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_speed_maintenance(u, label):
  print(f"\n{'='*80}")
  print(f"10. SPEED MAINTENANCE — {label}")
  print(f"{'='*80}")

  v = u['v']
  a = u['a_ego']
  v_cruise = u['v_cruise']
  long_active = u['long_active']
  lead_status = u['lead_status']
  dt = u['dt']

  # At cruise: within 2 mph of vCruise, no lead, engaged
  cruise_tol = 2.0 / MPS_TO_MPH  # 2 mph in m/s
  at_cruise = long_active & (v_cruise > 5.0) & (np.abs(v - v_cruise) < cruise_tol) & ~lead_status

  n_cruise = np.sum(at_cruise)
  if n_cruise < 100:
    print("  Insufficient cruise-holding data (no lead, within 2 mph of set speed)")
    # Try with lead vehicle too
    at_cruise_with_lead = long_active & (v_cruise > 5.0) & (np.abs(v - v_cruise) < cruise_tol)
    n2 = np.sum(at_cruise_with_lead)
    if n2 > 100:
      print(f"  (Including with lead vehicle: {n2*dt:.0f}s available)")
      at_cruise = at_cruise_with_lead
      n_cruise = n2
    else:
      return {}

  cruise_time = n_cruise * dt
  v_at_cruise = v[at_cruise]
  a_at_cruise = a[at_cruise]
  jerk_at_cruise = u['jerk'][at_cruise]

  print(f"\n  Cruise-holding time: {cruise_time:.0f}s ({cruise_time/60:.1f} min)")
  print(f"  Set speed range:    {np.min(v_cruise[at_cruise])*MPS_TO_MPH:.0f}-{np.max(v_cruise[at_cruise])*MPS_TO_MPH:.0f} mph")

  print(f"\n  Speed Stability:")
  speed_err = (v_at_cruise - v_cruise[at_cruise]) * MPS_TO_MPH
  print(f"    Speed error mean:  {np.mean(speed_err):+.2f} mph")
  print(f"    Speed error std:   {np.std(speed_err):.2f} mph")
  print(f"    Speed error P5:    {np.percentile(speed_err, 5):+.2f} mph")
  print(f"    Speed error P95:   {np.percentile(speed_err, 95):+.2f} mph")

  print(f"\n  Acceleration Smoothness at Cruise:")
  print(f"    Accel mean:  {np.mean(a_at_cruise):+.4f} m/s^2")
  print(f"    Accel std:   {np.std(a_at_cruise):.4f} m/s^2")
  print(f"    Accel P95:   {np.percentile(np.abs(a_at_cruise), 95):.4f} m/s^2")

  valid_jerk = np.abs(jerk_at_cruise) < 50
  if np.sum(valid_jerk) > 10:
    j = jerk_at_cruise[valid_jerk]
    print(f"    Jerk std:    {np.std(j):.4f} m/s^3")
    print(f"    |Jerk| P95:  {np.percentile(np.abs(j), 95):.4f} m/s^3")

  # Speed-binned cruise quality
  speed_bins = [
    ("30-45 mph",  13.4, 20.1),
    ("45-60 mph",  20.1, 26.8),
    ("60-75 mph",  26.8, 33.5),
    ("75+ mph",    33.5, 99.0),
  ]

  print(f"\n  Speed-Binned Cruise Quality:")
  print(f"  {'Speed':<14} {'Time':>6} {'Spd Std':>10} {'Acc Std':>10} {'|Acc| P95':>10}")
  print('  ' + '-' * 55)

  for sname, slo, shi in speed_bins:
    sm = at_cruise & (v >= slo) & (v < shi)
    n = np.sum(sm)
    if n < 20:
      continue
    secs = n * dt
    v_std = np.std(v[sm]) * MPS_TO_MPH
    a_std = np.std(a[sm])
    a_p95 = np.percentile(np.abs(a[sm]), 95)
    print(f"  {sname:<14} {secs:>5.0f}s {v_std:>10.3f} {a_std:>10.4f} {a_p95:>10.4f}")

  return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: PID STATE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pid_state(u, label):
  print(f"\n{'='*80}")
  print(f"11. PID STATE — {label}")
  print(f"{'='*80}")

  co_accel = u['co_accel']
  cc_accel = u['cc_accel']
  a_ego = u['a_ego']
  v = u['v']
  long_active = u['long_active']
  dt = u['dt']

  mask = long_active & (v > 0.5)
  if np.sum(mask) < 100:
    print("  Insufficient data")
    return {}

  # carOutput accel vs carControl accel shows what the PID modified
  cmd = cc_accel[mask]
  out = co_accel[mask]
  act = a_ego[mask]

  print(f"\n  A) Accel Pipeline")
  print(f"    carControl.accel (planner cmd):  mean={np.mean(cmd):+.3f}, std={np.std(cmd):.3f}")
  print(f"    carOutput.accel  (PID output):   mean={np.mean(out):+.3f}, std={np.std(out):.3f}")
  print(f"    carState.aEgo    (actual):        mean={np.mean(act):+.3f}, std={np.std(act):.3f}")

  pid_delta = out - cmd
  print(f"\n    PID adjustment (output - cmd):   mean={np.mean(pid_delta):+.4f}, std={np.std(pid_delta):.4f}")

  tracking_err = act - out
  print(f"    Tracking error (actual - output): mean={np.mean(tracking_err):+.4f}, std={np.std(tracking_err):.4f}")

  # B) Integral windup detection
  # Look for sustained error in one direction (> 3s continuous)
  print(f"\n  B) Integral Windup Detection")
  error = tracking_err  # already masked
  sustained_threshold = 0.3  # m/s^2
  sustained_frames = int(3.0 / dt)

  # Positive sustained error
  pos_err = error > sustained_threshold
  neg_err = error < -sustained_threshold

  def count_sustained(mask_arr, min_len):
    """Count events where mask is continuously True for min_len frames."""
    events = 0
    total_frames = 0
    run = 0
    for val in mask_arr:
      if val:
        run += 1
      else:
        if run >= min_len:
          events += 1
          total_frames += run
        run = 0
    if run >= min_len:
      events += 1
      total_frames += run
    return events, total_frames

  pos_events, pos_frames = count_sustained(pos_err, sustained_frames)
  neg_events, neg_frames = count_sustained(neg_err, sustained_frames)

  print(f"    Sustained positive error (>{sustained_threshold} m/s^2, >3s): {pos_events} events, {pos_frames*dt:.1f}s total")
  print(f"    Sustained negative error (<-{sustained_threshold} m/s^2, >3s): {neg_events} events, {neg_frames*dt:.1f}s total")

  if pos_events + neg_events > 0:
    print(f"    ** Possible windup detected — check if integral accumulates during these periods")
  else:
    print(f"    No sustained error detected — PID tracking appears healthy")

  # C) Error by speed range
  print(f"\n  C) Tracking Error by Speed")
  speed_bins = [
    ("0-15 mph",    0,    6.7),
    ("15-30 mph",   6.7, 13.4),
    ("30-45 mph",  13.4, 20.1),
    ("45-60 mph",  20.1, 26.8),
    ("60+ mph",    26.8, 99.0),
  ]
  print(f"  {'Speed':<14} {'N':>7} {'Err Mean':>10} {'Err Std':>10} {'|Err| P95':>10}")
  print('  ' + '-' * 55)

  v_masked = v[mask]
  for sname, slo, shi in speed_bins:
    sm = (v_masked >= slo) & (v_masked < shi)
    n = np.sum(sm)
    if n < 20:
      continue
    e = tracking_err[sm]
    print(f"  {sname:<14} {n:>7} {np.mean(e):>+10.4f} {np.std(e):>10.4f} {np.percentile(np.abs(e),95):>10.4f}")

  return {}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
  if len(sys.argv) < 2:
    print("Usage: python analyze_long_v1.py <route_dir> [--sample N]")
    print("  --sample N  load every Nth segment (default: 1 = all)")
    sys.exit(1)

  # Parse args
  route_dirs = []
  sample_every = 1
  i = 1
  while i < len(sys.argv):
    if sys.argv[i] == '--sample':
      sample_every = int(sys.argv[i+1])
      i += 2
    else:
      route_dirs.append(sys.argv[i])
      i += 1

  for route_dir in route_dirs:
    label = os.path.basename(route_dir)
    print(f"\n{'#'*80}")
    print(f"# LONGITUDINAL ANALYSIS: {label}")
    print(f"{'#'*80}")

    msgs = load_route(route_dir, sample_every=sample_every)
    if not msgs:
      continue

    data = extract_data(msgs)
    u = build_timeline(data)
    if u is None:
      print("  No carState data found")
      continue

    analyze_overview(u, label)
    analyze_accel_response(u, label)
    analyze_response_lag(u, label)
    analyze_transitions(u, label)
    analyze_jerk(u, label)
    analyze_following(u, label)
    analyze_braking(u, label)
    analyze_accel_events(u, label)
    analyze_stopping(u, label)
    analyze_speed_maintenance(u, label)
    analyze_pid_state(u, label)

    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE — {label}")
    print(f"{'='*80}")


if __name__ == '__main__':
  main()
