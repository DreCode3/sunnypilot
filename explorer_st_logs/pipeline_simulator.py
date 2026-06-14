#!/usr/bin/env python3
"""Replay carcontroller pipeline at production 20 Hz steer rate.

QA round 5 rewrite — applies all 6 fixes flagged by adversarial QA:
  1. STEER_STEP = 5 (20 Hz steer rate, not 50)
  2. Per-tick speed-interpolated SMOOTH_TAU (engage 0.12/0.04 mode-0)
  3. Per-tick SMOOTH_ALPHA = 1 - exp(-0.05/tau_i)
  4. Reconstruct lane_offset from modelV2.laneLines + position; drive full PI
  5. Rate-up direction: sign-match + magnitude growth (lateral.py:81)
  6. ZOH on model_des (latest pre-tick value), not linear interp

Also matches production deadband (0.001 → 0 at 7 m/s), bias offset (+0.000035),
override logic, post-reset ramp (simplified — flagged events excluded), and
anti-windup integral rollback. Path 4 release-tau branch off by default.

Usage:
  ./pipeline_simulator.py <route_prefix> [--mag {sharp,moderate,gentle}]
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

try:
    from scipy.signal import correlate as sp_correlate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION PARAMETERS — sourced from carcontroller.py + values.py + lateral.py
# ─────────────────────────────────────────────────────────────────────────────

DT_CTRL = 0.01                  # 100 Hz tick (controlsd)
STEER_STEP = 5                  # carcontroller steers every 5 ticks → 20 Hz
SMOOTH_DT = DT_CTRL * STEER_STEP  # 0.05s — per-EMA-tick interval

# Mode-0 active config (memory baseline)
SMOOTH_TAU_ENGAGE = (0.12, 0.04)    # [low-speed, highway] — interp on speed via [4,7,25]
SMOOTH_TAU_RELEASE = (0.18, 0.15)   # Path 4 release branch (only when path4_enabled)
PATH4_ENABLED = False               # match memory baseline

# Blend (pred + des)
PC_BLEND_RATIO_V = [7.,  20., 27., 35.]
PC_BLEND_RATIO_Y = [0.10, 0.30, 0.20, 0.10]
LOOKUP_TIME_V = [15., 25.]
LOOKUP_TIME_Y = [0.50, 0.35]

# Deadband — speed-interp 0.001 → 0
DEADBAND_V = [0., 4., 7.]
DEADBAND_Y = [0.001, 0.001, 0.0]

# PI controller — memory baseline
KP = 0.0001
KI = 0.0002
INTEGRAL_CAP = 0.3
INTEGRAL_GATE_THRESHOLD = 0.005
INTEGRAL_DECAY_OFF_GATE = 0.98
INTEGRAL_DECAY_ZERO_CROSS = 0.97
LANE_OFFSET_EMA_TAU = 1.5
ENABLE_LANE_POSITIONING = True   # per memory
PI_BIAS = 0.000035               # apply_curvature += 0.000035 (carcontroller.py:383)

# Rate limit (Mode 0)
RATE_UP_V = [5., 16., 25.]
RATE_UP_Y = [0.0025, 0.0015, 0.00018]
RATE_DOWN_V = [5., 16., 25.]
RATE_DOWN_Y = [0.0025, 0.0018, 0.00028]

CURVATURE_ERROR = 0.004
MAX_CURVATURE = 0.02
LANELINE_CONFIDENCE_GATE = 0.6
LANE_POSITION_LP_TIME = 0.2      # interp position.y at 0.2s lookahead


# ─────────────────────────────────────────────────────────────────────────────
# RLOG INGESTION (now pulling lane data too)
# ─────────────────────────────────────────────────────────────────────────────

def find_rlogs(route_prefix):
    files = sorted(glob.glob(os.path.join(route_prefix, 'rlog_*.zst')))
    if files:
        return files
    seg_dirs = sorted(glob.glob(route_prefix + '--*'),
                      key=lambda d: int(d.rsplit('--', 1)[-1]))
    out = []
    for sd in seg_dirs:
        rlog = os.path.join(sd, 'rlog.zst')
        if os.path.exists(rlog):
            out.append(rlog)
    return out


def _T_IDXS():
    """ModelConstants.T_IDXS (verified vs production by QA)."""
    return np.array([0.0, 0.00976562, 0.0390625, 0.087890625, 0.15625, 0.244140625,
                     0.3515625, 0.4785156, 0.625, 0.7910156, 0.9765625, 1.181640625,
                     1.40625, 1.650390625, 1.9140625, 2.197265625, 2.5,
                     2.8222656, 3.1640625, 3.5253906, 3.90625, 4.306640625,
                     4.7265625, 5.166015625, 5.625, 6.1035156, 6.6015625,
                     7.119140625, 7.65625, 8.212890625, 8.7890625, 9.384765625, 10.0])


def pull_signals(route_prefix):
    from openpilot.tools.lib.logreader import LogReader

    files = find_rlogs(route_prefix)
    if not files:
        return None

    out = {
        # modelV2 (20 Hz)
        'model_t': [], 'model_des_curv': [], 'model_orient_rate': [],
        'model_lane_left_y': [], 'model_lane_right_y': [],
        'model_lane_left_prob': [], 'model_lane_right_prob': [],
        'model_position_y': [],
        # carControl (100 Hz)
        'cc_t': [], 'cc_curv': [], 'cc_lat_active': [],
        # carState (100 Hz)
        'cs_t': [], 'cs_v': [], 'cs_yaw': [], 'cs_steer_press': [], 'cs_steer_angle': [],
        # controlsState (100 Hz)
        'csstate_t': [], 'csstate_curv': [],
    }

    n_msgs_dropped = 0
    for fp in files:
        try:
            for msg in LogReader(fp):
                w = msg.which()
                t = msg.logMonoTime / 1e9
                if w == 'modelV2':
                    mv2 = msg.modelV2
                    if not hasattr(mv2, 'action'):
                        continue
                    out['model_t'].append(t)
                    out['model_des_curv'].append(float(mv2.action.desiredCurvature))
                    try:
                        orz = np.array(mv2.orientationRate.z, dtype=np.float32)
                    except Exception:
                        orz = np.zeros(33, dtype=np.float32)
                    out['model_orient_rate'].append(orz)
                    # Lane lines [1] = left, [2] = right; .y is array per T_IDXS
                    try:
                        ll = mv2.laneLines
                        lp = mv2.laneLineProbs
                        if len(ll) > 2 and len(lp) > 2:
                            out['model_lane_left_y'].append(float(ll[1].y[0]))
                            out['model_lane_right_y'].append(float(ll[2].y[0]))
                            out['model_lane_left_prob'].append(float(lp[1]))
                            out['model_lane_right_prob'].append(float(lp[2]))
                        else:
                            out['model_lane_left_y'].append(float('nan'))
                            out['model_lane_right_y'].append(float('nan'))
                            out['model_lane_left_prob'].append(0.0)
                            out['model_lane_right_prob'].append(0.0)
                    except Exception:
                        out['model_lane_left_y'].append(float('nan'))
                        out['model_lane_right_y'].append(float('nan'))
                        out['model_lane_left_prob'].append(0.0)
                        out['model_lane_right_prob'].append(0.0)
                    # position.y array
                    try:
                        py = np.array(mv2.position.y, dtype=np.float32)
                    except Exception:
                        py = np.zeros(33, dtype=np.float32)
                    out['model_position_y'].append(py)
                elif w == 'carControl':
                    cc = msg.carControl
                    out['cc_t'].append(t)
                    out['cc_curv'].append(float(cc.actuators.curvature))
                    out['cc_lat_active'].append(bool(cc.latActive))
                elif w == 'carState':
                    cs = msg.carState
                    out['cs_t'].append(t)
                    out['cs_v'].append(float(cs.vEgoRaw))
                    out['cs_yaw'].append(float(cs.yawRate))
                    out['cs_steer_press'].append(bool(cs.steeringPressed))
                    out['cs_steer_angle'].append(float(cs.steeringAngleDeg))
                elif w == 'controlsState':
                    cs2 = msg.controlsState
                    out['csstate_t'].append(t)
                    out['csstate_curv'].append(float(cs2.curvature))
        except Exception:
            n_msgs_dropped += 1

    # to arrays
    for k in out:
        if k in ('model_orient_rate', 'model_position_y'):
            out[k] = np.array(out[k]) if out[k] else np.zeros((0, 33))
        elif k in ('cc_lat_active', 'cs_steer_press'):
            out[k] = np.array(out[k], dtype=bool)
        else:
            out[k] = np.array(out[k])
    if n_msgs_dropped:
        print(f'  (note: {n_msgs_dropped} log msgs dropped during parse)')
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION-MATCHING PIPELINE RECONSTRUCTION (20 Hz steer cadence)
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_cx1_lint(arr, t_cx1, t_query):
    """Get CX1's lInt (lane_centering_integral) at the nearest CX1 sample to t_query.

    Used for QA round 7 Bug C: warm-start the simulator's integral so PI behavior
    matches production's persistent-across-drives integral.
    """
    if arr is None or len(arr) == 0:
        return 0.0
    idx = int(np.argmin(np.abs(t_cx1 - t_query)))
    return float(arr[idx, IDX['lInt']])


def reconstruct_pipeline(grid, initial_integral=0.0, smooth_tau_override=None):
    """Reconstruct ema, preRL, cmd at 20Hz steer cadence.

    Matches carcontroller.py:227-470 (Mode 0 active, Path 4 off).

    QA round 6 fixes:
    - Bug 3: do NOT reset cmd_last on inactive ticks (production keeps the state)
    - Bug 4: defer integral decay until first valid lane data is seen (state
      starts at 0 but production restored from LaneBiasIntegral param; this
      avoids decaying-toward-zero before there's anything to decay)

    QA round 7 fixes:
    - Bug A: inactive branch sets cmd_out=0, cmd_last=0 (not held value)
    - Bug C: integral can be warm-started via initial_integral parameter
    """
    n = len(grid['_t'])
    v = grid['_v']
    yaw = grid['_yaw']
    des = grid['_des_zoh']
    pred = grid['_pred']
    meas = grid['_meas']
    active = grid['_active']
    pressed = grid['_steer_pressed']
    steer_deg = grid['_steer_angle_deg']
    ll_left = grid['_lane_left_y']
    ll_right = grid['_lane_right_y']
    ll_lprob = grid['_lane_left_prob']
    ll_rprob = grid['_lane_right_prob']
    pos_y_02 = grid['_position_y_at_02']

    blend_out = np.zeros(n)
    ema_out = np.zeros(n)
    preRL_out = np.zeros(n)
    cmd_out = np.zeros(n)
    lInt_out = np.zeros(n)
    PI_out = np.zeros(n)         # combined Kp*off + Ki*int per tick
    pi_clipped_count = 0

    # State
    smooth_last = 0.0
    cmd_last = 0.0
    integral = float(initial_integral)  # Bug C: warm-start from CX1 lInt
    lane_offset_ema = 0.0
    reset_steering_last = False
    lane_data_seen = bool(initial_integral != 0.0)  # if warm-started, consider lane "seen"

    for i in range(n):
        if not active[i]:
            # Inactive: production's apply_std_steer_angle_limits at lateral.py:88-89
            # sets new_apply_angle = steering_angle (passed as 0.0 from carcontroller.py:565
            # via the inactive branch), then clips. Output is 0.0, NOT held value.
            # QA round 7 Bug A fix: write 0.0, not cmd_last.
            smooth_last = 0.0
            integral = 0.0
            lane_offset_ema = 0.0
            cmd_last = 0.0  # production: apply_curvature_last reassigned to 0 (carcontroller.py:564)
            blend_out[i] = ema_out[i] = preRL_out[i] = cmd_out[i] = 0.0
            lInt_out[i] = integral
            PI_out[i] = 0.0
            continue

        # ─── BLEND (pred + des) ────────────────────────────────────────────
        b = float(np.interp(v[i], PC_BLEND_RATIO_V, PC_BLEND_RATIO_Y))
        apply_curv = pred[i] * b + des[i] * (1 - b)
        blend_out[i] = apply_curv

        # ─── DEADBAND (only when active AND >= 4 m/s with EMA below; below 4 deadband holds) ──
        deadband = float(np.interp(v[i], DEADBAND_V, DEADBAND_Y))
        if abs(apply_curv - smooth_last) <= deadband:
            apply_curv = smooth_last

        # ─── EMA (only above 4 m/s) ─────────────────────────────────────────
        if v[i] >= 4.0:
            # Allow tau override for what-if sweeps; falls back to memory baseline.
            tau_low = smooth_tau_override[0] if smooth_tau_override else SMOOTH_TAU_ENGAGE[0]
            tau_high = smooth_tau_override[1] if smooth_tau_override else SMOOTH_TAU_ENGAGE[1]
            engage_tau = float(np.interp(v[i], [4., 7., 25.],
                                         [tau_low, tau_low, tau_high]))
            # Path 4 release branch (off by default per memory)
            if PATH4_ENABLED:
                release_tau = float(np.interp(v[i], [4., 7., 25.],
                                              [SMOOTH_TAU_RELEASE[0], SMOOTH_TAU_RELEASE[0],
                                               SMOOTH_TAU_RELEASE[1]]))
                is_release = (abs(smooth_last) > 0.0005
                              and abs(apply_curv) < abs(smooth_last)
                              and apply_curv * smooth_last > 0)
                smooth_tau = release_tau if is_release else engage_tau
            else:
                smooth_tau = engage_tau
            smooth_alpha = 1.0 - np.exp(-SMOOTH_DT / smooth_tau)
            apply_curv = float(smooth_alpha * apply_curv + (1.0 - smooth_alpha) * smooth_last)
        smooth_last = apply_curv
        ema_out[i] = apply_curv

        # ─── PI (lane centering) ───────────────────────────────────────────
        lc_step = 0.0
        lane_offset = 0.0
        pi_p = 0.0
        pi_i = 0.0
        if (ENABLE_LANE_POSITIONING and v[i] > 7.0
                and not np.isnan(ll_left[i]) and not np.isnan(ll_right[i])):
            lp = ll_lprob[i]
            rp = ll_rprob[i]
            lane_width = ll_right[i] + (-ll_left[i])
            width_tol = float(np.interp(lane_width, [3.75, 4.25], [0.81, 0.59]))
            conf = min(lp, rp, width_tol)
            if conf > LANELINE_CONFIDENCE_GATE:
                lane_data_seen = True  # Bug 4: track first valid lane sighting
                laneline_scale = float(np.interp(conf, [0.6, 0.8], [0.0, 1.0]))
                path_offset_lanelines = (ll_left[i] + ll_right[i]) / 2
                path_offset_position = pos_y_02[i]
                lane_offset_raw = (path_offset_position * (1 - laneline_scale)
                                   + path_offset_lanelines * laneline_scale)
                lc_ema_alpha = 1.0 - np.exp(-SMOOTH_DT / LANE_OFFSET_EMA_TAU)
                lane_offset_ema = float(lc_ema_alpha * lane_offset_raw
                                        + (1.0 - lc_ema_alpha) * lane_offset_ema)
                lane_offset = lane_offset_ema

                # Integral accumulation/decay
                if abs(apply_curv) < INTEGRAL_GATE_THRESHOLD and not pressed[i]:
                    lc_step = lane_offset * SMOOTH_DT
                    integral += lc_step
                    integral = float(np.clip(integral, -INTEGRAL_CAP, INTEGRAL_CAP))
                    if lane_offset * integral < 0:
                        integral *= INTEGRAL_DECAY_ZERO_CROSS
                else:
                    integral *= INTEGRAL_DECAY_OFF_GATE

                pi_p = KP * lane_offset
                pi_i = KI * integral
                apply_curv += pi_p + pi_i
            elif lane_data_seen:
                integral *= INTEGRAL_DECAY_OFF_GATE
            # else: never seen lane data — leave integral as-is (Bug 4 fix)
        elif lane_data_seen:
            integral *= INTEGRAL_DECAY_OFF_GATE
        PI_out[i] = pi_p + pi_i

        # Bias offset
        apply_curv += PI_BIAS

        # Measured curvature (used for override)
        current_curv = -yaw[i] / max(v[i], 0.1)

        # ─── OVERRIDE LOGIC ─────────────────────────────────────────────────
        human_turn = pressed[i] and abs(steer_deg[i]) > 45.0
        reset_steering = human_turn
        if reset_steering:
            apply_curv = current_curv
            smooth_last = current_curv
            integral = 0.0
            lane_offset_ema = 0.0
        elif pressed[i]:
            override_alpha = 0.6
            apply_curv = override_alpha * current_curv + (1 - override_alpha) * smooth_last
            smooth_last = apply_curv
        else:
            if reset_steering_last and not reset_steering:
                cmd_last = current_curv  # ramp from actual
        reset_steering_last = reset_steering

        preRL_out[i] = apply_curv

        # ─── apply_ford_curvature_limits — production order ─────────────────
        # 1. Curvature-error clip vs measured — ONLY above 9 m/s (carcontroller.py:50)
        # 2. apply_std_steer_angle_limits: sign-aware rate limit, then abs clip
        if v[i] > 9.0:
            apply_curv = float(np.clip(apply_curv,
                                       current_curv - CURVATURE_ERROR,
                                       current_curv + CURVATURE_ERROR))

        # Sign-aware rate limit (lateral.py:81)
        steer_up = (cmd_last * apply_curv >= 0.0) and (abs(apply_curv) > abs(cmd_last))
        if steer_up:
            rate_cap = float(np.interp(v[i], RATE_UP_V, RATE_UP_Y))
        else:
            rate_cap = float(np.interp(v[i], RATE_DOWN_V, RATE_DOWN_Y))
        delta = apply_curv - cmd_last
        delta = float(np.clip(delta, -rate_cap, rate_cap))
        apply_curv_rl = cmd_last + delta
        apply_curv_rl = float(np.clip(apply_curv_rl, -MAX_CURVATURE, MAX_CURVATURE))

        # Anti-windup: if RL clipped in same direction as integral step, undo it
        if lc_step != 0.0:
            rl_clip = preRL_out[i] - apply_curv_rl
            if rl_clip * lc_step > 0:
                integral -= lc_step
                integral = float(np.clip(integral, -1.0, 1.0))
                pi_clipped_count += 1

        cmd_out[i] = apply_curv_rl
        cmd_last = apply_curv_rl
        lInt_out[i] = integral

    return {
        'blend': blend_out,
        'ema': ema_out,
        'preRL': preRL_out,
        'cmd': cmd_out,
        'lInt': lInt_out,
        'PI': PI_out,
        'pi_clipped_count': pi_clipped_count,
    }


def crossings_on_signal(t_grid, sig_grid):
    """Crossings/sec via sign-change of d(sig)/dt above log-precision threshold.

    Bug 2 fix: removed dead/incorrect xor computation that was overwritten anyway.
    """
    if len(t_grid) < 3:
        return None
    dt = float(t_grid[1] - t_grid[0])
    rate = np.diff(sig_grid) / dt
    threshold = RATE_THRESHOLD_K * LOG_PRECISION / dt
    signs = np.where(np.abs(rate) > threshold, np.sign(rate), 0.0)
    # Count sign changes among non-zero (above-threshold) samples
    last = 0.0
    crossings = 0
    for s in signs:
        if s != 0:
            if last != 0 and s != last:
                crossings += 1
            last = s
    duration = float(t_grid[-1] - t_grid[0])
    cps = float(crossings / duration) if duration > 0 else 0.0
    return {
        'crossings_per_sec': cps,
        'zero_crossings': crossings,
        'std_signal': float(np.std(sig_grid)),
        'std_rate': float(np.std(rate)),
        'peak_rate': float(np.max(np.abs(rate))),
        'n_samples': len(t_grid),
    }


WARMUP_SEC = 8.0  # Bug 1 fix: warm up state for lane_offset_ema (tau=1.5s)
                  # and lane_centering_integral before scoring crossings.


def build_event_grid(sig, ev, window_sec=3.0, warmup_sec=WARMUP_SEC):
    """Build a uniform 20 Hz grid for this event and interpolate inputs.

    Bug 1 fix: pre-window warmup (default 8s) is prepended to the grid so
    lane_offset_ema (tau=1.5s, needs ~4-5s) and lane_centering_integral can
    converge before the scoring window. Scoring is done only on the latter
    `window_sec*2` second window.

    Uses ZOH (zero-order hold) for model_des and modelV2 lane/position signals
    (production reads them via 'latest value at tick time'). Uses linear interp
    only for carControl/carState which natively sample at 100 Hz (so a 50ms
    grid is densely covered).
    """
    t_lo = ev['t_apex'] - window_sec - warmup_sec
    t_hi = ev['t_apex'] + window_sec
    n = int(np.ceil((t_hi - t_lo) / SMOOTH_DT)) + 1
    t_grid = t_lo + np.arange(n) * SMOOTH_DT  # 20 Hz
    # Mark which samples are inside the scoring window (last 2*window_sec seconds)
    score_start_t = ev['t_apex'] - window_sec
    score_mask = t_grid >= score_start_t

    # ── carControl + carState (100 Hz native) — linear interp is fine ──
    cc_mask = (sig['cc_t'] >= t_lo - 0.2) & (sig['cc_t'] <= t_hi + 0.2)
    cs_mask = (sig['cs_t'] >= t_lo - 0.2) & (sig['cs_t'] <= t_hi + 0.2)
    if cc_mask.sum() < 3 or cs_mask.sum() < 3:
        return None

    des_grid = np.interp(t_grid, sig['cc_t'][cc_mask], sig['cc_curv'][cc_mask])
    active = np.interp(t_grid, sig['cc_t'][cc_mask],
                       sig['cc_lat_active'][cc_mask].astype(float)) > 0.5
    v_grid = np.interp(t_grid, sig['cs_t'][cs_mask], sig['cs_v'][cs_mask])
    yaw_grid = np.interp(t_grid, sig['cs_t'][cs_mask], sig['cs_yaw'][cs_mask])
    pressed = np.interp(t_grid, sig['cs_t'][cs_mask],
                        sig['cs_steer_press'][cs_mask].astype(float)) > 0.5
    steer_deg = np.interp(t_grid, sig['cs_t'][cs_mask], sig['cs_steer_angle'][cs_mask])

    csst_mask = (sig['csstate_t'] >= t_lo - 0.2) & (sig['csstate_t'] <= t_hi + 0.2)
    if csst_mask.sum() >= 3:
        meas_grid = np.interp(t_grid, sig['csstate_t'][csst_mask], sig['csstate_curv'][csst_mask])
    else:
        meas_grid = np.zeros_like(t_grid)

    # ── modelV2 (20 Hz native) — ZOH ──
    m_mask = (sig['model_t'] >= t_lo - 0.5) & (sig['model_t'] <= t_hi + 0.5)
    if m_mask.sum() < 3:
        return None
    tm = sig['model_t'][m_mask]
    md = sig['model_des_curv'][m_mask]
    mor = sig['model_orient_rate'][m_mask]   # (N, 33)
    mll = sig['model_lane_left_y'][m_mask]
    mlr = sig['model_lane_right_y'][m_mask]
    mlp = sig['model_lane_left_prob'][m_mask]
    mrp = sig['model_lane_right_prob'][m_mask]
    mpy = sig['model_position_y'][m_mask]    # (N, 33)

    # ZOH: latest model message <= t_grid_i
    nearest = np.searchsorted(tm, t_grid, side='right') - 1
    nearest = np.clip(nearest, 0, len(tm) - 1)

    model_des_grid = md[nearest]
    lane_left_y = mll[nearest]
    lane_right_y = mlr[nearest]
    lane_left_prob = mlp[nearest]
    lane_right_prob = mrp[nearest]

    # pred = orientationRate.z[at lookup_time] / vEgo (fallback to des at vEgo <= 1)
    T_IDXS = _T_IDXS()
    lookup_grid = np.interp(v_grid, LOOKUP_TIME_V, LOOKUP_TIME_Y)
    pred_grid = np.zeros_like(t_grid)
    for i, mi in enumerate(nearest):
        if v_grid[i] > 1.0:
            pred_grid[i] = float(np.interp(lookup_grid[i], T_IDXS, mor[mi])) / v_grid[i]
        else:
            pred_grid[i] = model_des_grid[i]

    # position.y at LANE_POSITION_LP_TIME (0.2s) — ZOH on message, interp on T_IDXS
    pos_y_02_grid = np.zeros_like(t_grid)
    for i, mi in enumerate(nearest):
        pos_y_02_grid[i] = float(np.interp(LANE_POSITION_LP_TIME, T_IDXS, mpy[mi]))

    return {
        '_t': t_grid,
        '_v': v_grid,
        '_yaw': yaw_grid,
        '_des_zoh': model_des_grid,     # ZOH'd modelV2.action.desiredCurvature
        '_des_actuator': des_grid,      # what carControl.actuators.curvature shows (post-clip)
        '_pred': pred_grid,
        '_meas': meas_grid,
        '_active': active,
        '_steer_pressed': pressed,
        '_steer_angle_deg': steer_deg,
        '_lane_left_y': lane_left_y,
        '_lane_right_y': lane_right_y,
        '_lane_left_prob': lane_left_prob,
        '_lane_right_prob': lane_right_prob,
        '_position_y_at_02': pos_y_02_grid,
        '_score_mask': score_mask,      # Bug 1: mask of grid points inside scoring window
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

SWEEP_TAU_SETTINGS = [
    ('baseline (0.12, 0.04)', (0.12, 0.04)),
    ('softer    (0.15, 0.06)', (0.15, 0.06)),
    ('soft      (0.18, 0.08)', (0.18, 0.08)),
    ('softest   (0.25, 0.12)', (0.25, 0.12)),
    ('extreme   (0.35, 0.18)', (0.35, 0.18)),
]


def _xcorr_lag_ms(des, cmd, dt):
    """True cross-correlation lag with sub-sample parabolic interp.

    Positive lag → cmd lags des. Returns lag in ms.
    QA round 9 fix: argmax-based lag was sample-quantized to 50ms steps and
    sensitive to multi-peak/flat-top events. Cross-correlation is the proper
    phase-shift measure.
    """
    if not HAS_SCIPY or len(des) < 10 or len(cmd) != len(des):
        return float('nan')
    des_d = des - des.mean()
    cmd_d = cmd - cmd.mean()
    if des_d.std() < 1e-9 or cmd_d.std() < 1e-9:
        return float('nan')
    corr = sp_correlate(cmd_d, des_d, mode='full')
    norm = np.sqrt((des_d ** 2).sum() * (cmd_d ** 2).sum())
    if norm < 1e-12:
        return float('nan')
    corr = corr / norm
    peak = int(np.argmax(corr))
    lag_int = peak - (len(des) - 1)
    # Parabolic sub-sample fit
    sub = 0.0
    if 0 < peak < len(corr) - 1:
        y_lo, y_pk, y_hi = corr[peak - 1], corr[peak], corr[peak + 1]
        denom = y_lo - 2 * y_pk + y_hi
        if abs(denom) > 1e-12:
            sub = 0.5 * (y_lo - y_hi) / denom
    return float((lag_int + sub) * dt * 1000.0)


def _event_metrics(grid, rec, event_duration=None):
    """Compute the rich per-event metric set.

    QA round 11 additions:
    - Tier 1.3: integral envelope (min, max, range, drift) — addresses pi_cps
      insensitivity to slow integral drift
    - Tier 1.4: lag normalized by event duration — addresses xcorr-lag SNR
      bias on big-amplitude sharp curves
    - Frequency-domain power in 0.5-3 Hz band — the actual subjectively-felt
      wobble band (Tier 3 prep)
    """
    sm = grid['_score_mask']
    t = grid['_t'][sm]
    des = grid['_des_zoh'][sm]
    cmd = rec['cmd'][sm]
    pi = rec['PI'][sm]
    lint = rec['lInt'][sm]

    cps = crossings_on_signal(t, cmd)
    pi_cps_res = crossings_on_signal(t, pi)
    if cps is None:
        return None

    des_max = float(np.max(np.abs(des))) if len(des) else 0.0
    cmd_max = float(np.max(np.abs(cmd))) if len(cmd) else 0.0
    peak_ratio = cmd_max / des_max if des_max > 1e-6 else float('nan')
    overshoot = (cmd_max - des_max) / des_max if des_max > 1e-6 else float('nan')

    lag_ms = _xcorr_lag_ms(des, cmd, SMOOTH_DT)
    # Tier 1.4: normalize lag by event duration (sharp events shorter → 8ms lag
    # is bigger fraction than on long moderate events)
    dur_for_norm = event_duration if event_duration else (t[-1] - t[0])
    lag_pct_of_dur = (lag_ms / 1000.0 / dur_for_norm * 100.0) if dur_for_norm > 0 else float('nan')

    # Tier 3: 0.5-3 Hz band power on cmd (predicted wheel-felt oscillation)
    cmd_band_power = _band_power(cmd, 1.0 / SMOOTH_DT, 0.5, 3.0)
    des_band_power = _band_power(des, 1.0 / SMOOTH_DT, 0.5, 3.0)

    return {
        'cmd_cps': cps['crossings_per_sec'],
        'pi_cps': pi_cps_res['crossings_per_sec'] if pi_cps_res else 0.0,
        'lag_ms': lag_ms,
        'lag_pct_of_event': lag_pct_of_dur,
        'peak_mag_ratio': peak_ratio,
        'overshoot_pct': overshoot,
        'rms_cmd': float(np.sqrt(np.mean(cmd ** 2))),
        'rms_tracking_err': float(np.sqrt(np.mean((cmd - des) ** 2))),
        'rms_pi': float(np.sqrt(np.mean(pi ** 2))),
        # Tier 1.3: integral envelope
        'integral_min': float(np.min(lint)),
        'integral_max': float(np.max(lint)),
        'integral_range': float(np.max(lint) - np.min(lint)),
        'integral_mean_abs': float(np.mean(np.abs(lint))),
        # Tier 3 prep: 0.5-3 Hz band power
        'cmd_band_power_0_5_to_3': cmd_band_power,
        'des_band_power_0_5_to_3': des_band_power,
    }


def _band_power(sig, fs_hz, f_lo, f_hi):
    """Power in [f_lo, f_hi] Hz band via Hann-windowed rFFT."""
    if not HAS_SCIPY or len(sig) < 10:
        return float('nan')
    try:
        from scipy.fft import rfft, rfftfreq
    except ImportError:
        return float('nan')
    s = sig - np.mean(sig)
    w = np.hanning(len(s))
    yf = rfft(s * w)
    freqs = rfftfreq(len(s), d=1.0 / fs_hz)
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not mask.any():
        return 0.0
    return float(np.sum(np.abs(yf[mask]) ** 2))


def _verify_mode_0(arr, t_cx1, events):
    """Tier 1.1: Verify each event was driven under Mode 0 (current tuning).

    CX1's `p4Tau` field actually carries `cx1_last_smooth_tau` (schema label is
    legacy — see carcontroller.py:547). For Mode 0, expected tau is the speed-
    interp from (0.12 at v<=4 m/s) → (0.04 at v>=25 m/s).

    Returns set of event indices to EXCLUDE (mode != 0 detected).
    """
    tau_field = IDX['p4Tau']
    v_field = IDX['v']
    excluded = set()
    mode_anomalies = 0
    for i, ev in enumerate(events):
        # Get CX1 samples within event time
        mask = (t_cx1 >= ev['t_start']) & (t_cx1 <= ev['t_end'])
        if mask.sum() < 2:
            continue
        v_in = arr[mask, v_field]
        tau_in = arr[mask, tau_field]
        # Expected Mode 0 tau via speed interp [4, 7, 25] → [0.12, 0.12, 0.04]
        expected = np.interp(v_in, [4., 7., 25.], [0.12, 0.12, 0.04])
        # Allow ±10% slack for floating-point / release-branch transients
        deviation = np.abs(tau_in - expected) / np.maximum(expected, 1e-6)
        # If >25% of event has deviation > 10%, flag as non-Mode-0
        if (deviation > 0.10).mean() > 0.25:
            excluded.add(i)
            mode_anomalies += 1
    return excluded, mode_anomalies


def run_tau_sweep(route_prefix, magnitude='moderate'):
    """QA round-11 rewrite: Mode 0 verification, event-weighted+P90, integral envelope,
    duration-normalized lag.
    """
    print(f'[1/3] Loading CX1 (events only)...', flush=True)
    arr, t_cx1 = load_cx1(route_prefix)
    events = detect_curve_events(arr, t_cx1)
    events = [e for e in events if e['magnitude_class'] == magnitude]
    print(f'  {len(events)} {magnitude} events')

    print(f'[2/3] Pulling rlog signals...', flush=True)
    sig = pull_signals(route_prefix)
    if sig is None:
        print('  No rlogs.')
        return
    print(f'  modelV2={len(sig["model_t"])}, carControl={len(sig["cc_t"])}')

    # Tier 1.1: Mode 0 verification via cx1_last_smooth_tau (logged as p4Tau)
    excluded_mode, n_mode_anom = _verify_mode_0(arr, t_cx1, events)
    if n_mode_anom > 0:
        print(f'  WARNING: {n_mode_anom} events excluded (smooth_tau deviates from Mode 0)')

    print(f'[3/3] Building event grids + tau sweep...\n', flush=True)
    grids = []
    initial_integrals = []
    event_durations = []
    n_filtered_mode = 0
    for i, ev in enumerate(events):
        if i in excluded_mode:
            n_filtered_mode += 1
            continue
        g = build_event_grid(sig, ev)
        if g is None:
            continue
        score_mask = g['_score_mask']
        pressed = g['_steer_pressed']
        if pressed[score_mask].any() or pressed[~score_mask].any():
            continue
        grids.append(g)
        initial_integrals.append(_nearest_cx1_lint(arr, t_cx1, g['_t'][0]))
        event_durations.append(float(ev['duration_sec']))
    print(f'  {len(grids)} events kept ({n_filtered_mode} mode-excluded)\n')
    if not grids:
        return

    # Run all settings and collect per-event metrics
    per_setting = {label: [] for label, _ in SWEEP_TAU_SETTINGS}
    for label, tau in SWEEP_TAU_SETTINGS:
        for g, ii, dur in zip(grids, initial_integrals, event_durations):
            rec = reconstruct_pipeline(g, initial_integral=ii, smooth_tau_override=tau)
            m = _event_metrics(g, rec, event_duration=dur)
            if m is not None:
                per_setting[label].append(m)

    # Headline table — medians with per-route detail and all the missed metrics
    route_id = os.path.basename(route_prefix.rstrip('/')).split('--')[0]
    print(f'{"="*150}')
    print(f'TAU SWEEP — {len(grids)} {magnitude} events, route {route_id} (Mode 0 assumed)')
    print(f'{"="*150}')
    # PRIMARY METRIC after Phase 1: cmd_band 0.5-3 Hz (better correlated with felt aLat
    # oscillation than cmd_cps; R²=0.272 vs R²=0.034 per phase1_validation.py).
    print(f'  {"Setting":<26} {"cmd_band":>11} {"Δband%":>7} '
          f'{"cmd_cps":>9} {"Δcps%":>7} {"lag_ms":>8} {"lag_P90":>8} '
          f'{"peak_med":>9} {"track_err":>10} {"int_rng":>8} {"score":>7}')
    print('  ' + '-' * 150)
    baseline = None
    for label, _ in SWEEP_TAU_SETTINGS:
        rows = per_setting[label]
        if not rows:
            print(f'  {label:<26} (no events)')
            continue
        # Primary metric: cmd_band 0.5-3 Hz (Phase 1: R²=0.272 with aLat, 8x better than cps)
        band_vals = [r['cmd_band_power_0_5_to_3'] for r in rows]
        cps_vals = [r['cmd_cps'] for r in rows]
        lag_vals = [r['lag_ms'] for r in rows if not np.isnan(r['lag_ms'])]
        peak_vals = [r['peak_mag_ratio'] for r in rows if not np.isnan(r['peak_mag_ratio'])]
        band_med = float(np.median(band_vals))
        cps_med = float(np.median(cps_vals))
        lag_med = float(np.median(lag_vals)) if lag_vals else float('nan')
        lag_p90 = float(np.percentile(lag_vals, 90)) if lag_vals else float('nan')
        peak_med = float(np.median(peak_vals)) if peak_vals else float('nan')
        track_rms_med = float(np.median([r['rms_tracking_err'] for r in rows]))
        int_rng_med = float(np.median([r['integral_range'] for r in rows]))

        if baseline is None:
            assert label == SWEEP_TAU_SETTINGS[0][0], 'Baseline row must be first SWEEP_TAU_SETTINGS entry'
            baseline = {'band': band_med, 'cps': cps_med, 'lag': lag_med, 'peak': peak_med}
            d_band = '0%'
            d_cps = '0%'
            score = 0.0
        else:
            d_band_pct = 100 * (band_med - baseline['band']) / max(baseline['band'], 1e-12)
            d_cps_pct = 100 * (cps_med - baseline['cps']) / max(baseline['cps'], 1e-3)
            d_lag_ms = lag_med - baseline['lag']
            d_peak_pct = 100 * (peak_med - baseline['peak']) / baseline['peak']
            d_band = f'{d_band_pct:+.0f}%'
            d_cps = f'{d_cps_pct:+.0f}%'
            # Composite score (re-anchored to band): cmd_band reduction is the target;
            # lag and peak loss are costs.
            #   Weights: 1 unit per 10% band reduction, 0.01 per ms lag, 0.5 per |peak%|
            score = (-d_band_pct / 10
                     - 0.01 * d_lag_ms
                     - 0.5 * abs(d_peak_pct))
        print(f'  {label:<26} {band_med:>11.3e} {d_band:>7} '
              f'{cps_med:>9.2f} {d_cps:>7} {lag_med:>8.0f} {lag_p90:>8.0f} '
              f'{peak_med:>9.3f} {track_rms_med:>10.5f} {int_rng_med:>8.4f} '
              f'{score:>+7.2f}')

    # Per-event spread at each setting (key for showing route-level variance)
    print(f'\n  Per-event cmd_cps at each setting:')
    print(f'  {"Setting":<26}', end='')
    for i in range(len(per_setting[SWEEP_TAU_SETTINGS[0][0]])):
        print(f'{f"ev{i}":>6}', end='')
    print()
    for label, _ in SWEEP_TAU_SETTINGS:
        rows = per_setting[label]
        print(f'  {label:<26}', end='')
        for r in rows:
            print(f'{r["cmd_cps"]:>6.2f}', end='')
        print()

    # Per-event PI cps spread (addresses round-10 concern #6 — non-monotonic ev1 bounce)
    print(f'\n  Per-event pi_cps at each setting (look for PI-driven non-monotonic cps):')
    print(f'  {"Setting":<26}', end='')
    for i in range(len(per_setting[SWEEP_TAU_SETTINGS[0][0]])):
        print(f'{f"ev{i}":>6}', end='')
    print()
    for label, _ in SWEEP_TAU_SETTINGS:
        rows = per_setting[label]
        print(f'  {label:<26}', end='')
        for r in rows:
            print(f'{r["pi_cps"]:>6.2f}', end='')
        print()

    print(f'\n  COLUMN GUIDE:')
    print(f'    cmd_band        FFT power 0.5-3 Hz on cmd — PRIMARY metric (R²=0.272 with aLat)')
    print(f'    cmd_cps         crossings/sec of cmd (R²=0.034 with aLat — weak, kept for context)')
    print(f'    lag_ms          cross-correlation phase lag of cmd vs des (positive = cmd lags)')
    print(f'    peak_ratio      max|cmd| / max|des| (1.0 = full delivery; <1 = EMA blunted apex)')
    print(f'    track_err_rms   sqrt(mean((cmd - des)^2)) (rising = EMA losing real planner content)')
    print(f'    int_rng         integral envelope range — drift indicator (Tier 1.3)')
    print(f'    score           composite: -Δband%/10 - 0.01*Δlag_ms - 0.5*|Δpeak%|. Higher = better.')
    print(f'                    Weights are explicit; reader can re-anchor for own priorities.')
    print(f'\n  CAVEATS:')
    print(f'    - Sweep is predictive — only baseline (current production) was validated against CX1.')
    print(f'    - N={len(grids)} events on a single route. Run on multiple routes to bracket variance.')
    print(f'    - Mode 0 assumed. If FordCurveMode changed mid-route, results are wrong for those events.')
    print(f'    - LANE_OFFSET_EMA_TAU (=1.5s) is NOT swept — only the curvature EMA changes.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route_prefix')
    ap.add_argument('--mag', default='moderate', choices=['sharp', 'moderate', 'gentle'])
    ap.add_argument('--sweep-tau', action='store_true',
                    help='Sweep smooth_tau values and report cmd_cps + apex-lag per setting')
    args = ap.parse_args()

    if args.sweep_tau:
        run_tau_sweep(args.route_prefix, magnitude=args.mag)
        return

    print('[1/3] Loading CX1 (for event detection only)...', flush=True)
    arr, t_cx1 = load_cx1(args.route_prefix)
    print(f'  {len(arr)} CX1 rows')
    events = detect_curve_events(arr, t_cx1)
    events = [e for e in events if e['magnitude_class'] == args.mag]
    print(f'  {len(events)} {args.mag} events')

    print('[2/3] Pulling rlog signals at native rates...', flush=True)
    sig = pull_signals(args.route_prefix)
    if sig is None:
        print('  No rlogs.')
        return
    print(f'  modelV2: {len(sig["model_t"])} msgs, '
          f'carControl: {len(sig["cc_t"])}, carState: {len(sig["cs_t"])}')

    print(f'[3/3] Reconstructing pipeline at production 20 Hz steer cadence...\n')

    print('=' * 130)
    print('PRODUCTION-MATCHED 20 Hz PIPELINE')
    print('  STEER_STEP=5, tau speed-interp (0.12/0.04), ZOH model_des, full PI,'
          ' sign-aware rate limit')
    print('=' * 130)
    print(f'  {"Idx":>4} {"v_mph":>6} {"dur":>5} '
          f'{"model":>7} {"act_des":>8} {"pred":>7} {"ema":>7} {"preRL":>7} {"cmd":>7} '
          f'{"m→ema":>7} {"ema→cmd":>8} {"sim-act":>9} {"flags":<10}')
    print('  ' + '-' * 130)

    rows = {k: [] for k in ['model', 'act_des', 'pred', 'ema', 'preRL', 'cmd', 'PI']}
    m_ema_deltas = []
    ema_cmd_deltas = []
    sim_vs_act_med = []   # Bug 5: end-to-end validation (vs carcontrol input)
    sim_vs_act_std = []
    sim_vs_cx1_med = []   # Bug 5 strengthen: vs CX1 cmd field (production output)
    sim_vs_cx1_std = []
    n_filtered_press = 0
    n_skipped = 0

    for i, ev in enumerate(events):
        grid = build_event_grid(sig, ev)
        if grid is None:
            n_skipped += 1
            continue

        # Bug 6 fix: skip events spanning press transitions in the scoring window
        score_mask = grid['_score_mask']
        pressed_score = grid['_steer_pressed'][score_mask]
        if pressed_score.any() and not pressed_score.all():
            n_filtered_press += 1
            continue
        # Skip events that were fully overridden
        if pressed_score.all():
            n_filtered_press += 1
            continue
        # QA round 7 Bug B fix: also skip events with press in WARMUP window
        # (warmup-press triggers override branch that resets integral/EMA,
        # poisoning state before scoring starts)
        pressed_warmup = grid['_steer_pressed'][~score_mask]
        if pressed_warmup.any():
            n_filtered_press += 1
            continue

        # QA round 7 Bug C: warm-start integral from CX1 lInt at warmup start.
        # CX1 logs lInt; pull the nearest CX1 sample to event_start - WARMUP_SEC
        # and use that as the initial integral state (mirrors production's
        # persistent integral across drives).
        warmup_start_t = grid['_t'][0]
        initial_integral = _nearest_cx1_lint(arr, t_cx1, warmup_start_t)

        rec = reconstruct_pipeline(grid, initial_integral=initial_integral)

        # Bug 1 fix: score crossings only on the post-warmup window
        t = grid['_t'][score_mask]
        des_zoh_s = grid['_des_zoh'][score_mask]
        des_act_s = grid['_des_actuator'][score_mask]
        pred_s = grid['_pred'][score_mask]
        ema_s = rec['ema'][score_mask]
        preRL_s = rec['preRL'][score_mask]
        cmd_s = rec['cmd'][score_mask]
        PI_s = rec['PI'][score_mask]

        m_cps = crossings_on_signal(t, des_zoh_s)
        ad_cps = crossings_on_signal(t, des_act_s)
        pr_cps = crossings_on_signal(t, pred_s)
        em_cps = crossings_on_signal(t, ema_s)
        pl_cps = crossings_on_signal(t, preRL_s)
        cd_cps = crossings_on_signal(t, cmd_s)
        pi_cps = crossings_on_signal(t, PI_s)

        if not all([m_cps, ad_cps, pr_cps, em_cps, pl_cps, cd_cps]):
            continue

        m_e = em_cps['crossings_per_sec'] - m_cps['crossings_per_sec']
        e_c = cd_cps['crossings_per_sec'] - em_cps['crossings_per_sec']
        m_ema_deltas.append(m_e)
        ema_cmd_deltas.append(e_c)
        rows['model'].append(m_cps['crossings_per_sec'])
        rows['act_des'].append(ad_cps['crossings_per_sec'])
        rows['pred'].append(pr_cps['crossings_per_sec'])
        rows['ema'].append(em_cps['crossings_per_sec'])
        rows['preRL'].append(pl_cps['crossings_per_sec'])
        rows['cmd'].append(cd_cps['crossings_per_sec'])
        rows['PI'].append(pi_cps['crossings_per_sec'])

        # Bug 5: end-to-end validation — sim cmd vs actual carControl curvature.
        # Compare with one-tick lag (production cmd at tick i → act_des at tick i+1 next cycle).
        diff = cmd_s - des_act_s
        sim_vs_act_med.append(float(np.median(diff)))
        sim_vs_act_std.append(float(np.std(diff)))

        # QA round 7 Bug 5 strengthen: also compare to CX1 `cmd` field (TRUE
        # production output to CAN, post-pipeline). CX1 is sparse so only
        # compare at CX1 sample times within the scoring window.
        score_t_lo = ev['t_apex'] - 3.0
        score_t_hi = ev['t_apex'] + 3.0
        cx1_mask = (t_cx1 >= score_t_lo) & (t_cx1 <= score_t_hi)
        if cx1_mask.sum() >= 3:
            t_cx1_in = t_cx1[cx1_mask]
            cmd_cx1_in = arr[cx1_mask, IDX['cmd']]
            sim_cmd_at_cx1 = np.interp(t_cx1_in, grid['_t'], rec['cmd'])
            cx1_diff = sim_cmd_at_cx1 - cmd_cx1_in
            sim_vs_cx1_med.append(float(np.median(cx1_diff)))
            sim_vs_cx1_std.append(float(np.std(cx1_diff)))
        else:
            sim_vs_cx1_med.append(float('nan'))
            sim_vs_cx1_std.append(float('nan'))

        flags = []
        if grid['_steer_pressed'].any():
            flags.append('press(warmup)')
        flag_str = ','.join(flags) if flags else ''

        print(f'  {i:>4} {ev["mean_v_mph"]:>6.1f} {ev["duration_sec"]:>5.1f} '
              f'{m_cps["crossings_per_sec"]:>7.3f} {ad_cps["crossings_per_sec"]:>8.3f} '
              f'{pr_cps["crossings_per_sec"]:>7.3f} {em_cps["crossings_per_sec"]:>7.3f} '
              f'{pl_cps["crossings_per_sec"]:>7.3f} {cd_cps["crossings_per_sec"]:>7.3f} '
              f'{m_e:>+7.3f} {e_c:>+8.3f} {sim_vs_act_std[-1]:>9.5f} {flag_str:<10}')

    if not rows['model']:
        print(f'  No events scored (filtered: {n_filtered_press} press, {n_skipped} skip)')
        return

    def s(arr):
        return f'{np.median(arr):.3f}  IQR [{np.percentile(arr, 25):.3f}, {np.percentile(arr, 75):.3f}]'
    print(f'\n  SUMMARY across {len(rows["model"])} events '
          f'({n_filtered_press} excluded for press, {n_skipped} for missing data):')
    for k in ['model', 'act_des', 'pred', 'ema', 'preRL', 'cmd', 'PI']:
        print(f'    {k:<10} {s(rows[k])}')
    print(f'    model→ema Δ {np.median(m_ema_deltas):+.3f}  IQR '
          f'[{np.percentile(m_ema_deltas, 25):+.3f}, {np.percentile(m_ema_deltas, 75):+.3f}]')
    print(f'    ema→cmd Δ   {np.median(ema_cmd_deltas):+.3f}  IQR '
          f'[{np.percentile(ema_cmd_deltas, 25):+.3f}, {np.percentile(ema_cmd_deltas, 75):+.3f}]')

    # Bug 5 end-to-end validation summary
    print(f'\n  END-TO-END VALIDATION:')
    print(f'    [vs carControl.actuators.curvature (carcontroller INPUT)]')
    print(f'    median |sim - act|: {np.median(np.abs(sim_vs_act_med)):.5f} 1/m')
    print(f'    median sample std:  {np.median(sim_vs_act_std):.5f} 1/m')
    cx1_valid = [v for v in sim_vs_cx1_std if not np.isnan(v)]
    cx1_med_valid = [v for v in sim_vs_cx1_med if not np.isnan(v)]
    if cx1_valid:
        print(f'    [vs CX1 cmd field (carcontroller OUTPUT — the true target)]')
        print(f'    median |sim - cx1|: {np.median(np.abs(cx1_med_valid)):.5f} 1/m')
        print(f'    median sample std:  {np.median(cx1_valid):.5f} 1/m '
              f'(if >5e-4, sim not tracking production cmd)')

    # Interpretation
    print(f'\n  INTERPRETATION:')
    m_med = np.median(rows['model'])
    cmd_med = np.median(rows['cmd'])
    if m_med > 0.01:
        pct = 100 * (1 - cmd_med / m_med)
        print(f'    Total model→cmd reduction: {pct:.1f}% (from {m_med:.2f} to {cmd_med:.2f} cps)')
    print(f'    Blend/EMA stage: model→ema removes {-np.median(m_ema_deltas):.2f} cps median')
    print(f'    PI+RL stages: ema→cmd removes {-np.median(ema_cmd_deltas):.2f} cps median')


if __name__ == '__main__':
    main()
