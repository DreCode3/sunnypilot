# 2021 Ford Explorer ST — SunnyPilot Development Guide

Complete implementation guide for every change beyond stock sunnypilot on the `2021_explorer_st-mici` branch.
A developer unfamiliar with this vehicle should be able to recreate the entire implementation from this document.

Last updated: 2026-04-08 (evening — final config for extended testing)

---

## Quick Reference: Files Modified

| File | Changes |
|------|---------|
| `opendbc_repo/opendbc/car/ford/carcontroller.py` | Curvature pipeline, PI centering, BP Long, override handling, gas EMA |
| `opendbc_repo/opendbc/car/ford/fordcan.py` | AccPrpl_A_Pred, ramp_type/precision_type params, brake/precharge split |
| `opendbc_repo/opendbc/car/ford/values.py` | ACCEL_MAX, rate limits, CURVATURE_ERROR, curve modes, steerRatio |
| `opendbc_repo/opendbc/car/ford/interface.py` | steerActuatorDelay, Ki/Kp tuning, hardcoded long control |
| `opendbc_repo/opendbc/safety/modes/ford.h` | max_angle_error, rate limits, reset bypass latch, 4-signal validation |
| `selfdrive/monitoring/helpers.py` | Phone threshold, yaw offset bounds, steering-aware yaw tolerance |
| `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py` | Tightened T_FOLLOW gaps |
| `launch_chffrplus.sh` | Scons lock cleanup on boot |
| `selfdrive/controls/lib/longitudinal_planner.py` | Lane change gas gating fix |
| `system/loggerd/deleter.py` | MIN_PERCENT raised |
| `selfdrive/locationd/locationd.py` | Cherry-pick: cam odo delay |
| `selfdrive/locationd/paramsd.py` | Cherry-pick: timestamp fix |
| `selfdrive/locationd/torqued.py` | Cherry-pick: timestamp fix |

---

## CRITICAL: Ford CAN Sign Convention

**The code NEGATES curvature before sending to CAN** (carcontroller.py):
```python
can_sends.append(fordcan.create_lat_ctl_msg(packer, CAN, CC.latActive,
                 0., 0., -apply_curv_send, -apply_curvature_rate, ...))
#                        ^^^^^^^^^^^^^^^^^
#                        NEGATED!
```

- In code: positive curvature = LEFT turn
- On CAN wire: positive curvature = RIGHT turn (negated)
- `current_curvature = -yawRate / vEgo` → positive = LEFT (Ford yawRate positive = RIGHT)

See `explorer_st_logs/ford_can_reference.md` for full CAN documentation.

---

## 1. Imports & State Variables

### carcontroller.py — Imports
```python
import math
import numpy as np
from collections import deque
from cereal import log, messaging
from opendbc.can import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, apply_hysteresis, structs
from opendbc.car.carlog import carlog
from opendbc.car.lateral import ISO_LATERAL_ACCEL, apply_std_steer_angle_limits
from opendbc.car.ford import fordcan
from opendbc.car.ford.values import CarControllerParams, FordFlags, CAR
from opendbc.car.interfaces import CarControllerBase, V_CRUISE_MAX
from selfdrive.modeld.constants import ModelConstants
from openpilot.common.params import Params

LongCtrlState = structs.CarControl.Actuators.LongControlState
VisualAlert = structs.CarControl.HUDControl.VisualAlert
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
```

### carcontroller.py — __init__ State Variables
```python
def __init__(self, dbc_names, CP, CP_SP):
    super().__init__(dbc_names, CP, CP_SP)
    self.packer = CANPacker(dbc_names[Bus.pt])
    self.CAN = fordcan.CanBus(CP)
    self.params = Params()

    self.apply_curvature_last = 0
    self.anti_overshoot_curvature_last = 0
    self.accel = 0.0
    self.gas = 0.0
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.lead_distance_bars_last = None
    self.distance_bar_frame = 0

    # === BluePilot Longitudinal State ===
    self._bp_long_active_last = False
    self.bp_gas_last = 0.0
    self.bp_accel_last = 0.0
    self.gas_ema = 0.0
    self.bpSpeedAllow = False
    self.op_brake_actuate_last = False
    self.MAX_URBAN_SPEED_MPH = 45.0
    self.following_accel_ROC = 0.004
    self.brake_actuate_target = -0.19
    self.brake_actuate_release = -0.08
    self.precharge_actuate_target = -0.15
    self.precharge_actuate_release = -0.08
    self.disable_BP_long_UI = False
    self.disable_downhill_comp_UI = True

    # === SubMaster for model + radar ===
    self.sm = messaging.SubMaster(['modelV2', 'radarState'])
    self.model = None

    # === Predicted Curvature Blending ===
    self.pc_blend_ratio = 0.30

    # === Low-Speed Curvature Stabilizer ===
    self.smooth_curvature_last = 0.0

    # === Curvature Rate Computation ===
    self.curvature_rate_delta_t = 0.3
    self.curvature_rate_deque = deque(maxlen=7)  # 0.3s at 20Hz

    # === PI Lane Centering ===
    self.enable_lane_positioning = True
    self.lane_offset_ema = 0.0
    self.lc_kp = 0.0005  # was 0.0001 — PI Phase 1: 5x for meaningful curve correction
    self.lc_ki = 0.0002
    self.lane_centering_integral_save_counter = 0

    # Persistent integral: restore from previous drive
    try:
      saved = self.params.get("LaneBiasIntegral")
      self.lane_centering_integral = float(saved) if saved else 0.0
      carlog.info("LC: restored integral=%.4f" % self.lane_centering_integral)
    except Exception:
      self.lane_centering_integral = 0.0

    # === Driver Override ===
    self.reset_steering_last = False
    self.post_reset_ramp_active = False

    # === FordCurveMode ===
    self.curve_mode = 0
    self._apply_curve_mode(0)
```

---

## 2. Lateral Control Pipeline

### 2.1 Predicted Curvature Blending
**Purpose**: Anticipate curves by blending model's predicted curvature with planner's desired.
```python
if CS.out.vEgoRaw > 1.0 and self.model is not None and len(self.model.orientationRate.z) >= len(ModelConstants.T_IDXS):
  curvatures = np.array(self.model.orientationRate.z) / CS.out.vEgoRaw
  lookup_time = self._get_curvature_lookup_time(CS.out.vEgoRaw)
  predicted_curvature = float(np.interp(lookup_time, ModelConstants.T_IDXS, curvatures))
else:
  predicted_curvature = desired_curvature

# Speed-dependent blend: high for surface streets, low for highway (reduces hunting)
blend = float(np.interp(CS.out.vEgoRaw, [7., 20., 27., 35.], [0.10, 0.30, 0.20, 0.10]))
apply_curvature = (predicted_curvature * blend) + (desired_curvature * (1 - blend))
```

### 2.2 Low-Speed Curvature Stabilizer
**Purpose**: Eliminate stop-and-go hunting without highway lag.
```python
smooth_dt = DT_CTRL * CarControllerParams.STEER_STEP  # 0.05s

# Deadband at low speed, EMA at higher speed
deadband = float(np.interp(CS.out.vEgoRaw, [0., 4., 7.], [0.001, 0.001, 0.0]))
if abs(apply_curvature - self.smooth_curvature_last) <= deadband:
  apply_curvature = self.smooth_curvature_last

if CS.out.vEgoRaw >= 4.0:
  smooth_tau = float(np.interp(CS.out.vEgoRaw, [4., 7., 25.],
                     [self._smooth_tau[0], self._smooth_tau[0], self._smooth_tau[1]]))
  smooth_alpha = 1.0 - np.exp(-smooth_dt / smooth_tau)
  apply_curvature = float(smooth_alpha * apply_curvature + (1.0 - smooth_alpha) * self.smooth_curvature_last)
self.smooth_curvature_last = apply_curvature
```

### 2.3 PI Lane Centering Controller
**Purpose**: Correct persistent lane offset (road crown, camera mount, alignment).
```python
lc_integral_step = 0.0
if (self.enable_lane_positioning and self.model is not None
    and len(self.model.laneLines) > 2 and len(self.model.laneLineProbs) > 2
    and CS.out.vEgoRaw > 7.0):

  left_y = self.model.laneLines[1].y[0]
  right_y = self.model.laneLines[2].y[0]
  left_prob = self.model.laneLineProbs[1]
  right_prob = self.model.laneLineProbs[2]
  lane_width = right_y + (-left_y)
  width_tolerance = float(np.interp(lane_width, [3.75, 4.25], [0.81, 0.59]))
  laneline_confidence = min(left_prob, right_prob, width_tolerance)

  if laneline_confidence > 0.6:
    laneline_scale = float(np.interp(laneline_confidence, [0.6, 0.8], [0.0, 1.0]))
    path_offset_lanelines = (left_y + right_y) / 2
    path_offset_position = float(np.interp(0.2, ModelConstants.T_IDXS, self.model.position.y))
    lane_offset_raw = path_offset_position * (1 - laneline_scale) + path_offset_lanelines * laneline_scale

    # EMA smooth to filter lane line noise (tau=1.5s)
    lc_ema_alpha = 1.0 - np.exp(-smooth_dt / 1.5)
    self.lane_offset_ema = float(lc_ema_alpha * lane_offset_raw + (1.0 - lc_ema_alpha) * self.lane_offset_ema)
    lane_offset = self.lane_offset_ema

    # Integral gate: straights only, no driver override
    if abs(apply_curvature) < 0.005 and not CS.out.steeringPressed:
      lc_integral_step = lane_offset * smooth_dt
      self.lane_centering_integral += lc_integral_step
      # Speed-dependent cap
      int_cap = float(np.interp(CS.out.vEgoRaw, [20., 30.], [0.3, 1.0]))
      self.lane_centering_integral = float(np.clip(self.lane_centering_integral, -int_cap, int_cap))
      # Zero-crossing decay: 0.97x when offset and integral disagree in sign
      if lane_offset * self.lane_centering_integral < 0:
        self.lane_centering_integral *= 0.97
    else:
      self.lane_centering_integral *= 0.995  # was 0.98 — retain 61% through 5s curve vs 13%

    # Persist integral every 10s
    self.lane_centering_integral_save_counter += 1
    if self.lane_centering_integral_save_counter >= 200:
      self.params.put_nonblocking("LaneBiasIntegral", str(round(self.lane_centering_integral, 4)))
      self.lane_centering_integral_save_counter = 0

    pi_p = self.lc_kp * lane_offset
    pi_i = self.lc_ki * self.lane_centering_integral
    apply_curvature += pi_p + pi_i

    if self.frame % 100 == 0:
      carlog.info("LC: off=%.3f ll=%.3f pos=%.4f scl=%.2f conf=%.2f wid=%.2f int=%.4f P=%.6f I=%.6f curv=%.6f spd=%.1f" % (
        lane_offset, path_offset_lanelines, path_offset_position, laneline_scale, laneline_confidence,
        lane_width, self.lane_centering_integral, pi_p, pi_i, apply_curvature, CS.out.vEgoRaw))
  else:
    self.lane_centering_integral *= 0.98
else:
  self.lane_centering_integral *= 0.98
```

### 2.4 Minimal EPAS Bias Compensation
**Purpose**: Compensate for constant component of +0.000245 1/m leftward EPAS over-delivery. PI handles the rest.
```python
# Data-driven interpolation from 10+ test routes:
# offset 0.000000 → +0.20m left, offset 0.000050 → -0.14m right → zero at ~0.000035.
# No curve fade needed at this magnitude (10% of EPAS bias).
apply_curvature += 0.000035
```
**Key**: Adding to `apply_curvature` shifts RIGHTWARD on the CAN wire because curvature is NEGATED before send.
**History**: Extensive tuning Apr 6-8. Values tried: 0.000245 (wrong sign initially, then overcorrected), 0.000080 (overcorrected), 0.000050 (overcorrected), 0.000025 (under-corrected +0.116m), 0.000035 (centered at +0.005m over 28mi).

### 2.5 Driver Override Handling
**Purpose**: Make EPAS cooperative during driver steering instead of fighting.
```python
current_curvature = -CS.out.yawRate / max(CS.out.vEgoRaw, 0.1)

human_turn = CS.out.steeringPressed and abs(CS.out.steeringAngleDeg) > 45.0
reset_steering = human_turn

if reset_steering:
  apply_curvature = current_curvature
  self.smooth_curvature_last = current_curvature
  ramp_type = 3  # Immediate
  self.curvature_rate_deque.clear()
  self.post_reset_ramp_active = False
  self.lane_centering_integral = 0.0
  self.lane_offset_ema = 0.0
elif CS.out.steeringPressed:
  override_alpha = 0.6
  apply_curvature = (override_alpha * current_curvature +
                     (1.0 - override_alpha) * self.smooth_curvature_last)
  self.smooth_curvature_last = apply_curvature
  self.curvature_rate_deque.clear()
else:
  if self.reset_steering_last and not reset_steering:
    self.post_reset_ramp_active = True
    self.apply_curvature_last = current_curvature

self.reset_steering_last = reset_steering
```

### 2.6 Rate Limiting with Anti-Windup
```python
apply_curvature_pre_rl = apply_curvature
self.apply_curvature_last = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last,
    current_curvature, CS.out.vEgoRaw, 0., CC.latActive, self.CP,
    curvature_error=self._active_curvature_error, angle_limits=self._active_angle_limits)

# Anti-windup: undo integral step if rate limiter clipped in same direction
if lc_integral_step != 0.0:
  rl_clip = apply_curvature_pre_rl - self.apply_curvature_last
  if rl_clip * lc_integral_step > 0:
    self.lane_centering_integral -= lc_integral_step
    self.lane_centering_integral = float(np.clip(self.lane_centering_integral, -1.0, 1.0))
```

### 2.7 Curvature Rate Feed-Forward
```python
if not reset_steering:
  self.curvature_rate_deque.append(predicted_curvature)

if CS.out.vEgoRaw > 1.0 and len(self.curvature_rate_deque) > 1:
  delta_t = (self.curvature_rate_delta_t if len(self.curvature_rate_deque) == self.curvature_rate_deque.maxlen
             else (len(self.curvature_rate_deque) - 1) * 0.05)
  apply_curvature_rate = (self.curvature_rate_deque[-1] - self.curvature_rate_deque[0]) / delta_t / CS.out.vEgoRaw
else:
  apply_curvature_rate = 0.0

curv_factor = float(np.interp(abs(predicted_curvature), [0.0, 0.001, 0.002], [0.0, 0.0, 1.0]))
apply_curvature_rate *= curv_factor * self.curvature_rate_gain
apply_curvature_rate = float(np.clip(apply_curvature_rate, -0.001023, 0.001023))
```

### 2.8 CAN Message Send (THE NEGATION)
```python
apply_curv_send = self.apply_curvature_last

# Lane change: scale curvature, not rate limiter state
lane_change = self.model is not None and self.model.meta.laneChangeState in (
  LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
if lane_change:
  factor = float(np.interp(CS.out.vEgoRaw, [4.4, 40.23], [0.95, 0.85]))
  if self.model.meta.laneChangeDirection == LaneChangeDirection.left and apply_curv_send < 0:
    apply_curv_send *= factor
  elif self.model.meta.laneChangeDirection == LaneChangeDirection.right and apply_curv_send > 0:
    apply_curv_send *= factor
  apply_curvature_rate = 0.0

# CRITICAL: curvature and curvature_rate are NEGATED before CAN send
can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive,
                 0., 0., -apply_curv_send, -apply_curvature_rate,
                 ramp_type=ramp_type, precision_type=1))
```

---

## 3. Longitudinal Control (BluePilot Long)

### 3.1 Brake Pad Scaling
```python
op_accel = actuators.accel
if op_accel < 0:
  op_accel *= 0.82  # Explorer ST aftermarket pads deliver 1.32x. 0.82 * 1.32 = 1.08x
op_gas = actuators.accel
```

### 3.2 Lead Classification & Coast Logic
```python
# Speed-dependent deadband: ±0.2 at low speed, ±0.4 at highway.
# At 80 mph, ±0.2 m/s = ±0.45 mph — normal lead cruise variance crossed both
# thresholds constantly (170-280 transitions/min). ±0.4 at highway reduces chatter
# 80-90%. Safe: classification only gates gas, brakes always flow through planner.
follow_deadband = float(np.interp(v_ego, [13., 30.], [0.2, 0.4]))
if lead:
  if v_rel < -follow_deadband:  gaining = True
  elif v_rel > follow_deadband: trailing = True
  else:                         pacing = True

# Gaining: speed-dependent coast threshold
if gaining:
  coast_time = float(np.interp(CS.out.vEgoRaw, [13., 20., 27., 36.], [1.5, 2.5, 3.5, 4.0]))
  if lead_time_sec < coast_time:
    max_follow_gas = 0.0
    min_follow_gas = 0.0

# Pacing: coast when gap is stable
if pacing:
  max_follow_gas = 0.10 + accel_due_to_pitch  # was 0.07→0.10 — 0.07 too passive at highway
  min_follow_gas = 0.0
  if v_rel >= 0 and lead_time_sec > 1.0:
    max_follow_accel = 0.0  # coast
    min_follow_accel = 0.0
```

### 3.3 Rate-Limited Braking
```python
if ttc_sec > 10.0 and lead_time_sec > 0.5:
  bp_accel = float(np.clip(bp_accel,
                           self.bp_accel_last - self.following_accel_ROC,
                           self.bp_accel_last + self.following_accel_ROC))
```

### 3.4 Brake/Precharge Actuation (Hysteresis)
```python
if bp_accel < self.brake_actuate_target:    bp_brake_actuate = True     # -0.19
if bp_accel > self.brake_actuate_release:   bp_brake_actuate = False    # -0.08
if bp_accel < self.precharge_actuate_target: bp_precharge_actuate = True  # -0.15
if bp_accel > self.precharge_actuate_release: bp_precharge_actuate = False # -0.08
```

### 3.5 Asymmetric Gas EMA
```python
if CC.longActive and gas > CarControllerParams.MIN_GAS:
  if gas < self.gas_ema:
    gas_alpha = 0.010  # tau=2.0s — very gradual coast lift-off (~5s to near-zero)
  else:
    gas_alpha = 0.025  # tau=0.8s — cushioned gas onset (~1.5s to full)
  self.gas_ema = gas_alpha * gas + (1 - gas_alpha) * self.gas_ema
  gas = self.gas_ema
else:
  self.gas_ema = 0.0

# Ramp to 0 on brake instead of hard-cutting to INACTIVE_GAS (-5.0).
# The -5.0 hard cut bypassed the EMA entirely (below MIN_GAS threshold),
# making all EMA tuning ineffective. Now the EMA ramps down naturally.
if brake_actuate:
  gas = 0.0  # EMA ramps down from here; don't reset EMA state
```

---

## 4. fordcan.py — CAN Message Construction

### create_acc_msg (Longitudinal)
```python
def create_acc_msg(packer, CAN, long_active, gas, accel, stopping,
                   brake_actuate, precharge_actuate, v_ego_kph):
  values = {
    "AccBrkTot_A_Rq": accel,
    "Cmbb_B_Enbl": 1 if long_active else 0,
    "AccPrpl_A_Rq": gas,
    "AccPrpl_A_Pred": gas if long_active else -5.0,  # match stock IPMA
    "AccResumEnbl_B_Rq": 1 if long_active else 0,
    "AccVeh_V_Trg": v_ego_kph,
    "AccBrkPrchg_B_Rq": 1 if precharge_actuate else 0,
    "AccBrkDecel_B_Rq": 1 if brake_actuate else 0,
    "AccStopStat_B_Rq": 1 if stopping else 0,
  }
  return packer.make_can_msg("ACCDATA", CAN.main, values)
```

### create_lat_ctl_msg (Lateral)
Stock signature modified to accept `ramp_type` and `precision_type` parameters:
```python
def create_lat_ctl_msg(packer, CAN, lat_active, path_offset, path_angle, curvature,
                       curvature_rate, ramp_type=0, precision_type=1):
```

---

## 5. values.py — Constants

### CarControllerParams
```python
ACCEL_MAX = 1.5    # was 2.0 — reduced for tuned Explorer ST
ACCEL_MIN = -3.5
MIN_GAS = -0.5
INACTIVE_GAS = -5.0

CURVATURE_ERROR = 0.004  # was 0.002

# Mode 0 rate limits (relaxed from stock)
# Rate up at [5, 16, 25] m/s: [0.0025, 0.0015, 0.00018]
# Rate down at [5, 16, 25] m/s: [0.0025, 0.0018, 0.00028]
```

### CarSpecs
```python
CarSpecs(mass=2050, wheelbase=3.025, steerRatio=17.2)  # was 16.8
```

---

## 6. interface.py — Vehicle Parameters

```python
ret.steerActuatorDelay = 0.25  # was 0.20

ret.longitudinalTuning.kiBP = [0., 13., 25., 27.]
ret.longitudinalTuning.kpV = [0.]     # integral-only
ret.longitudinalTuning.kiV = [0.3, 0.2, 0.2, 0.15]

# Hardcoded longitudinal control (AlphaLongitudinalEnabled clears on reboot)
ret.safetyConfigs[-1].safetyParam |= FordSafetyFlags.LONG_CONTROL.value
ret.openpilotLongitudinalControl = True
```

---

## 7. T_FOLLOW — Following Distance

**File**: `selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py`
**What**: Tightened following gap for all personality levels.
```python
def get_T_FOLLOW(personality):
  if personality == relaxed:    return 1.45  # was 1.75
  elif personality == standard: return 1.25  # was 1.45
  elif personality == aggressive: return 1.05  # was 1.25
```
**Why**: Driver confident in braking performance after brake scaling tuning. Tighter gaps feel more natural for the Explorer ST's capabilities.
**Impact**: Median following gap 1.36s on R36 (tracking 1.25 target). Reduced from 1.46s.

---

## 8. ford.h — Safety Layer

### Key Constants
```c
.max_angle_error = 300,        // was 100 (0.006 vs 0.002 curvature)
.angle_rate_up_lookup = { {5., 16., 25.}, {0.0026, 0.0020, 0.00025} },
.angle_rate_down_lookup = { {5., 16., 25.}, {0.0026, 0.0024, 0.00035} },
```

### Reset Bypass Latch
```c
static uint8_t reset_bypass_latch_counter = 0;
static const uint8_t RESET_BYPASS_LATCH_DURATION = 60;  // 3s at 20Hz

// In tx_hook: when curvature AND path_angle are both zero
if ((desired_curvature == 0) && (desired_path_angle == 0)) {
  reset_bypass_latch_counter = RESET_BYPASS_LATCH_DURATION;
  violation = false;
} else if (reset_bypass_latch_counter > 0) {
  reset_bypass_latch_counter--;
  violation = false;
}
```

### 4-Signal Value Range Checks
Added validation for curvature_rate, path_angle, path_offset with limits:
```c
#define FORD_CURVATURE_MIN -0.012f
#define FORD_CURVATURE_MAX 0.012f
#define FORD_PATH_OFFSET_MIN -1.0f
#define FORD_PATH_OFFSET_MAX 1.0f
#define FORD_PATH_ANGLE_MIN -0.25f
#define FORD_PATH_ANGLE_MAX 0.25f
```

---

## 8. Driver Monitoring

### selfdrive/monitoring/helpers.py
```python
self._PHONE_THRESH = 0.8    # was 0.5 — camera geometry causes P90=0.53 false positives
self._YAW_MIN_OFFSET = -0.7  # was -0.0246 — raw yaw -0.54 rad from camera mount position

# Cherry-picked from upstream (#37751): steering-aware yaw tolerance
# Widens DM yaw threshold when steering >30 deg, reducing false alerts in curves
self._POSE_YAW_MIN_STEER_DEG = 30
self._POSE_YAW_STEER_FACTOR = 0.15
self._POSE_YAW_STEER_MAX_OFFSET = 0.3927
```

---

## 9. Lane Change Gas Gating Fix

### selfdrive/controls/lib/longitudinal_planner.py
```python
from cereal import log
LaneChangeState = log.LaneChangeState

# In the throttle gating section:
lane_changing = sm['modelV2'].meta.laneChangeState in (
    LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
self.allow_throttle = (throttle_prob > ALLOW_THROTTLE_THRESHOLD or
                       v_ego <= MIN_ALLOW_THROTTLE_SPEED or lane_changing)
```

---

## 10. System Changes

### system/loggerd/deleter.py
```python
MIN_PERCENT = 15  # was 10 — prevent disk filling past build cache tolerance
```

### launch_chffrplus.sh — Scons Lock Cleanup
```bash
rm -f /data/scons_cache/config.lock  # cherry-picked from upstream #37734
```
Prevents stale scons lock from blocking builds after device crash.

---

## 11. Cherry-Picked Upstream Commits

1. **Camera odo delay** (#37543): `CAM_ODO_POSE_DELAY = 0.1` in `selfdrive/locationd/locationd.py`
2. **livePose timestamp** (#37704): Correct timestamp in `paramsd.py` and `torqued.py`
3. **Filter time** (#37697): Publish filter time in `locationd.py`
4. **DM fewer alerts during maneuvers** (#37751): Steering-aware yaw tolerance in `selfdrive/monitoring/helpers.py`
5. **Scons lock cleanup** (#37734): Remove stale lock on boot in `launch_chffrplus.sh`

---

## 12. Measured Performance

### Lateral Best: Route 36 (28.3 mi, offset 0.000035 + PI Phase 1)
| Metric | Value |
|--------|-------|
| Lane offset mean | **+0.005 m** (centered) |
| Lane offset >0.30m | **8.9%** |
| Lane offset >0.50m | **1.3%** |
| Steer overrides | **0** |
| Lateral engaged | **96.1%** (20.2 min continuous) |
| Smoothness ratio (65-74) | **1.02x** (near-perfect) |
| Curvature reversals/mile | **157** |

### Longitudinal Best: Route 37 (18.1 mi, deadband ±0.4 + gas cap 0.10)
| Metric | Value |
|--------|-------|
| Unnecessary accel | **0.0%** (0/54 events) |
| Brake onsets/min | **16/min** (was 34/min on R36) |
| Longitudinal jerk mean | **1.12 m/s^3** |
| Longitudinal jerk P95 | **2.79 m/s^3** |
| ISO "uncomfortable" | **14.7%** |
| Gas/brake cycling (±0.1) | **0.1/min** (essentially zero) |
| Following gap median | **1.39s** (target 1.25) |
| Below 1.0s (dangerous) | **0.0%** |
| Speed error | **-1.04 mph, std 0.20** |

### Combined Best: Route 37
| Metric | Value |
|--------|-------|
| Steer overrides | **0** |
| Brake overrides | **1** (0.0% time) |
| Lateral engaged | **60.3%** (mixed driving) |
| Longitudinal engaged | **53.2%** (mixed driving) |
| DM false distraction | **<10%** |

---

## 13. Key Lessons & Pitfalls

1. **CAN curvature negation**: `create_lat_ctl_msg` sends `-apply_curv_send`. To push car RIGHT, ADD to `apply_curvature` in code.
2. **Brake scaling placement**: Must be at PID output level, before actuation thresholds. Scaling on CAN wire causes PID feedback instability.
3. **Dual-controller conflict**: path_angle PID + curvature planner = two controllers on one actuator = hunting. Fold centering into curvature loop instead.
4. **DM camera geometry**: Each vehicle has unique camera-to-face angles. Stock thresholds won't work for all mounting positions.
5. **SSH fragility**: Comma 4 SSH daemon crashes with parallel connections. Single ControlMaster only.
6. **Never `sudo reboot`**: Causes EPAS alerts. Use `echo -n "1" > /data/params/d/DoReboot`.
7. **AccPrpl_A_Pred**: Stock IPMA mirrors current gas request. Sending -5.0 (inactive) is a protocol violation that may cause PCM hesitation.
8. **Panda auto-reflash**: When ford.h changes, the panda firmware signature mismatches and triggers auto-reflash on boot.
9. **Gas INACTIVE_GAS hard-cut bypasses EMA**: Setting gas to -5.0 (INACTIVE) on brake skips the EMA filter entirely (below MIN_GAS threshold). Set to 0.0 instead and let the EMA ramp down naturally.
10. **EPAS bias offset is non-linear**: The relationship between static offset and lane centering is approximately linear but varies by road crown. Data-driven interpolation across multiple routes is essential — don't extrapolate from a single data point.
11. **Left bias masks right-curve tracking**: When the car has a left straight-line bias, it enters right curves from the left side of the lane, hiding the EPAS under-delivery in curves. Fixing the straight-line centering reveals the curve issue.
12. **PI memory/code drift trap**: Code values can drift from the "tested baseline" through incremental commits if memory isn't updated each time. From 2026-03-29 through 2026-05-23, `lc_kp`, integral cap, and decay accumulated changes that were never validated on-road because PI was simultaneously disabled by an unrelated UI param. Re-enabling PI then unknowingly committed to the drifted controller. **Always cross-check `MEMORY.md` PI section against actual code before re-enabling after a disable period.**

---

## 14. PI Lane Centering Baseline Validation — 2026-05-23

After re-enabling PI (off since 2026-03-29 due to a UI toggle, then crash bug fixed 2026-05-03), the user reported **abrupt comfort/performance regression** during 2026-05-07 through 2026-05-23. Adversarial QA review of 15 drives found the running code's PI gains had drifted significantly from the documented memory baseline during a period when PI was disabled and the changes were never road-validated.

### Configuration WAS (drifted, 2026-04-06 → 2026-05-23)

| Parameter | Drifted value | Source commit |
|---|---|---|
| `lc_kp` | **0.0005** (5× memory) | `88d52dc0` 2026-04-08 "PI Phase 1: Kp 5x" |
| `int_cap` | **speed-interp ±0.3 → ±1.0** at highway (3.3× memory at top) | `952657ca` 2026-03-31 |
| Off-gate decay | **0.995** (per-step, retains 61% through 5s) | line 354 inline change |
| Zero-crossing decay | 0.97 (was 0.92) | inline change, not in revert |

Worst-case correction authority at 29 m/s with saturated integral: **~0.21 m/s²** lateral pull. ~3.6× more total authority than memory implied.

### Configuration IS (validated baseline, 2026-05-23 onward — route 9b)

| Parameter | Value | Notes |
|---|---|---|
| `lc_kp` | **0.0001** | reverted to memory baseline |
| `lc_ki` | 0.0002 | unchanged through drift period |
| `int_cap` | **fixed ±0.3** | reverted from speed-interp |
| Off-gate decay | **0.98** | reverted from 0.995 |
| Zero-crossing decay | 0.97 | left at current (memory didn't speak to it) |
| EMA tau on offset | 1.5s | unchanged |

Worst-case correction authority at 29 m/s with saturated integral: **~0.059 m/s²** lateral pull. Gentle but proven.

Reverted by commits:
- opendbc `6ca2e16e` "revert PI lane centering to memory baseline"
- main `c719932278` "bump opendbc — PI revert to memory baseline"

### Validation evidence — Route 9b (2026-05-23, 42.8 min, 75.6% lateral engaged)

| Metric | PI-off period (8d/90/91/96/99) | Route 9b (PI on baseline) | Δ |
|---|---|---|---|
| Lateral engagement % | 28-67% avg ~40% | **75.6%** | ↑ best in dataset |
| Overrides / engaged min | 0.30-0.40 | **0.12** | **3-4× reduction** |
| Lane offset mean | +0.04 to +0.12 m | **-0.006 m** | centered |
| Lane offset >0.30m | 9-22% | **8.6%** | improvement |
| Moderate-curve peak lag | 284-974 ms | **230 ms** | dramatic improvement |
| Lateral jerk RMS | 2.24 m/s³ | 1.95 m/s³ | -13% |
| RMS lat accel | 0.42-0.51 m/s² | 0.52 m/s² | ~same |
| Pipeline anti-windup hits | 0% | 0% | safe |
| Pipeline rate-limit clips | 0% | 0% | safe |

### Sign convention CONFIRMED on route 9b

The QA's S2 hazard (PI sign possibly inverted, would cause wrong-way pull) is **resolved by observation**. Speed-binned PI state shows the correct relationship:

| Speed | Mean offset | Mean integral | Behavior |
|---|---|---|---|
| 15-30 mph | -0.049m (right) | -0.064 | correcting left ✓ |
| 30-45 mph | -0.046m | -0.020 | correcting left ✓ |
| 45-65 mph | -0.016m | +0.013 | near zero ✓ |
| **65-80 mph** | **+0.099m (left)** | **+0.163** | **correcting right ✓** |

Positive offset (car left of center) ⇒ positive integral ⇒ positive curvature (turn right). Sign chain is correct from `mv_path_y` through `lane_centering_integral` through `apply_curvature`. The `SIGN VERIFICATION NEEDED` comment in carcontroller.py can be retired.

### Open items (next iteration targets)

1. **Highway-speed left drift residual** — at 65-80 mph, offset still averages +0.10m left despite integral saturating at +0.163. Memory baseline I-term max is 6e-5 curvature ≈ 0.03 m/s² at 30 m/s, insufficient to fully counter structural left bias at highway. **Targeted small bump** (e.g., Kp 0.0001 → 0.0002 OR int_cap ±0.3 → ±0.5) — not back to the drift-period values.
2. **Sharp curve overshoot still 100%** — PI doesn't address this. Mean overshoot +0.00282, exit overshoot +0.000902. This is the **Path 4 (asymmetric release tau) target**. With PI now validated as safe, Path 4 A/B test (toggle mid-drive) is the natural next step.
3. **Smoothness ratio 1.35x at >25 m/s** — controller still adds ~35% jitter beyond planner. Most is disturbance-driven (83% per analyzer); not a tuning-friendly target.

### What "validated baseline" means going forward

- This configuration is the **default reference point** for future tuning experiments.
- Any change to PI gains, integral cap, decay, or related lateral parameters must be:
  - Documented in this section with a new entry (WAS → IS)
  - Validated on a >20-min mixed-driving route before being claimed as a new baseline
  - Cross-checked against MEMORY.md PI section in the same commit
- If PI ever gets disabled again for an extended period, **re-validate this baseline before resuming tuning work** — the drift trap is real (see Lesson 12 above).

---

### Iteration 1: Ki bump (2026-05-23, pending validation)

**Target:** Residual +0.099m left drift at 65-80 mph (open item from baseline validation).

**Analysis path that selected this change:**
- Considered int_cap raise 0.3→0.5: rejected — equilibrium analysis showed integral is decay-bound at +0.163 regardless of cap. Cap raise yields only ~1.0-1.3× at the mean.
- Considered decay 0.98→0.99: gives 2× I-term but introduces curve-persistence concerns (Scenario 2: sustained bias direction change takes 20-30s to flip vs 10-15s).
- Selected Ki bump because Ki appears in exactly ONE line (carcontroller.py:367, `pi_i = self.lc_ki × integral`) — no dynamics, no curve-path, no decay interaction.

**WAS (validated baseline):**
| Parameter | Value |
|---|---|
| `lc_ki` | 0.0002 |
| I-term at 65-80mph (integral=+0.163) | 3.3e-5 curvature → 0.043 m/s² @ 29 m/s |
| Worst-case wrong-way pull (cap × Ki) | 0.0002 × 0.3 = 6e-5 → 0.050 m/s² @ 29 m/s |

**IS (this iteration):**
| Parameter | Value | Δ |
|---|---|---|
| `lc_ki` | **0.0003** | 1.5× |
| Predicted I-term at 65-80mph (integral unchanged at +0.163) | 4.9e-5 curvature → 0.060 m/s² @ 29 m/s | 1.5× |
| Worst-case wrong-way pull | 0.0003 × 0.3 = 9e-5 → 0.076 m/s² @ 29 m/s, 0.130 m/s² @ 36 m/s | 1.5× |

**No other change.** lc_kp, int_cap, off-gate decay, zero-crossing decay, gate condition, EMA tau all unchanged.

**QA review (independent agent):** APPROVE. Verified Ki appears only in pi_i, no curve-path interaction. Cross-speed-bin impact analysis confirms only 65-80 mph bin sees meaningful effect (+0.017 m/s²); other bins shift by <0.001 m/s².

**Predicted result (don't oversell):**
- 65-80 mph mean offset: +0.099m → roughly **+0.06 to +0.08m** (not zero — drift force is partially structural)
- Other speed bins: no meaningful change
- Lateral comfort: ISO RMS ≤0.55 m/s² (no regression vs 0.520 baseline), jerk RMS ≤2.1 m/s³

**Success criteria for validation drive:**
1. **Primary**: 65-80 mph mean offset in +0.04 to +0.08m band on >20-min highway with ≥200 PI samples in bin
2. **No regression**: 45-65 mph offset stays within ±0.03m of -0.016m
3. **Comfort**: RMS measured lat accel ≤0.55 m/s², lateral jerk RMS ≤2.1 m/s³

**Fallback if Ki bump under-delivers (mean stays >+0.08m):**
- Next iteration: int_cap raise 0.3→0.5 (NOT decay change, per curve-persistence analysis)
- If convergence rate stays ~51% (was 51% on baseline), drift is structural — consider re-enabling FF bias instead of more PI

**Commits:** opendbc `e3235c24`, main `a924b219e8`.

### Iteration 1 — RESULT: REVERTED (2026-05-29)

3 validation drives (9c, 9d, 9e):
- **Zero 65-80 mph engagement** across all drives — primary Ki bump target was untested
- User reported "hunting and jerkiness of the wheel in moderate to sharp curves" leading to disable/override
- Initial regression narrative ("Ki amplifies I-term at curve exit causing exit overshoot") was DISPROVEN by adversarial QA — quantitative mechanism contributes 0.7% of observed effect; PI is applied unconditionally not gated; comfort metrics improved 2/3 drives

Reverted Ki 0.0003 → 0.0002 because: (1) primary 65-80 mph target unvalidated, (2) user-reported feel regression at speeds where data doesn't clearly implicate Ki, (3) return to known-good baseline pending dedicated highway drive.

**Commits (revert):** see Iteration 2.

---

### Iteration 2: steerRatio bump (2026-05-29, pending validation)

**Target:** "Hunting and jerkiness in moderate to sharp curves" (user-reported, primary complaint).

**Analysis path:**
- Comprehensive curve performance deep-dive across 19 engaged drives (5-07 to 5-29)
- Original findings highlighted exit overshoot (Path 4 territory) and peak lag (lookahead territory) as Tier 1/2
- Adversarial QA review found:
  - "100% overshoot rate" was a definitional artifact (metric dominated by EPAS bias + entry/exit transients, not transient peak)
  - Peak lag has non-negative floor by construction — bumping lookahead would INCREASE the reported metric not decrease it
  - Path 4 preflight math predicts only 16% drive-wide reduction if 30% per-trip, not 25-40% as originally claimed
  - **Strongest signal was steerRatio mismatch I had dismissed as "too systemic"**
- Measured effective steerRatio across 19 drives:
  - **Median 18.55** (assumed 17.2, +7.9% off)
  - 25-45 mph: median **17.95** (surface street range, where user complaint occurs)
  - 45-65 mph: median **19.91** (highway range)
- A 7.9% SR under-rotation mechanistically explains:
  - Moderate-curve entry under-rotation (-0.000270 mean entry bias in pi_off era)
  - Late peak alignment (controller commands less wheel angle than needed for given curvature)

**WAS (validated baseline, restored):**
| Parameter | Value |
|---|---|
| `lc_ki` | 0.0002 (reverted from Iteration 1's 0.0003) |
| `steerRatio` (Explorer MK6) | 17.2 |

**IS (this iteration):**
| Parameter | Value | Δ |
|---|---|---|
| `lc_ki` | **0.0002** | reverted from 0.0003 |
| `steerRatio` (Explorer MK6) | **18.0** | +4.7% (closer to measured 25-45mph median 17.95) |

**Predicted effect:**
- Surface-street curves (25-45 mph): SR mismatch drops from -7.9% (17.95/17.2) to +0.3% (17.95/18.0) — controller commands ~5% more wheel angle for given curvature
- Highway (45-65 mph): SR mismatch drops from -15.8% to -9.6% — partial improvement but still off (single SR can't perfectly fit variable-ratio EPAS)
- Moderate curve entry bias should shift toward zero
- Peak alignment should improve at apex (peak_bias closer to zero)

**Risk profile:**
- steerRatio affects multiple subsystems (planner kinematics, MPC, lane keeping)
- A 4.7% bump is small relative to the +7.9% measured mismatch — conservative step
- Highest measured uncertainty (sr_std 2-3 across drives) means individual road samples may behave differently

**Success criteria for next drives (no specific gate metric — observe broadly):**
1. **User subjective**: "hunting and jerkiness in moderate to sharp curves" should improve (primary success criterion since this is the user's complaint)
2. **Moderate curve peak_bias** should shift toward zero (from current ~-0.0002 in pi_off era)
3. **Comfort RMS** should not regress above baseline 0.55 m/s²
4. **No new override clustering** at any speed bin

**Adversarial QA review status:** Reviewed the analysis; QA verdict was MODIFY with steerRatio elevated from "not recommended" to Tier 1 candidate. This iteration follows QA's revised tier order.

**Commits (Iteration 1 revert + Iteration 2 in single change):** opendbc `8e2b2534`, main `27f2daf834`.

### Iteration 2 — RESULT: REVERTED (2026-05-30)

Validation drives a0, a1 (2 drives, 27.6 min engaged combined) with confirmed `steerRatio=18.0` in CarParams:

**Wins claimed (initial reading):**
- Sharp mean overshoot 0.00282 → 0.00210 (-25.5%) ← held up under QA
- Moderate mean overshoot 0.00244 → 0.00138 (-43.6%) ← **statistically empty per QA: N=2 events on a1 only**
- Sharp exit bias healed iter1's regression ← **misleading framing per QA: still +72% worse than baseline**
- Effective SR mismatch +17.3% → +8.1% ← **per-drive variance (a0=17.48 vs a1=21.44) is 5× the claimed improvement; 45-65mph bin actually worsened (21.29 → 23.05)**
- Jerk RMS 1.952 → 1.727 (-11.5%) ← held up

**Regressions (real, held up under QA):**
- **RMS lateral accel 0.520 → 0.660 (+26.9%)** — more relevant comfort metric than jerk for "feels rough in curves"
- **Lane offset std 0.188 → 0.235 (+25%)**, P95 0.371 → 0.556 (+50%)
- **Override rate per engaged min 0.123 → 0.811 (6.6×)**
- **a1 ISO rating "Fairly uncomfortable"** (first since route 96, which had narrow lane width excuse; a1's 3.25m lane width has no such excuse)

**User subjective feedback:** "I didn't really notice any improvement."

**Adversarial QA verdict on findings:** MODIFY with hard pushback on the overshoot/SR-mismatch claims; the regressions are robust, the wins are mostly small-sample artifacts.

**Decision:** Revert steerRatio 18.0 → 17.2. The 7.9% measured SR mismatch is NOT the dominant cause of user-felt curve issues — must look elsewhere.

**Lessons learned for §14:**
1. **N=2 drives is not enough validation** for a parameter that affects all subsystems. Plan for ≥4 drives minimum before declaring success.
2. **Subjective feedback should be weighted heavily** when objective metrics show mixed signals. "Hunting and jerkiness in curves" is the felt symptom; objective curve overshoot metrics don't capture this.
3. **steerRatio mismatch was a red herring** — the +7.9% measured mismatch is real but is NOT what makes curves feel jerky to the driver.
4. **Adversarial QA caught the overclaiming** (sample-size-empty improvements, misleading "healed" framing, distribution artifacts in override clustering). Apply this scrutiny BEFORE deploying, not just before recommending.

**Commits (revert):** opendbc `<pending>`, main `<pending>`.

---

### Iteration 3: TBD — fundamentally different approach to curve issues

Current state: back at PI baseline (Ki=0.0002, steerRatio=17.2, FF=0.000035). User reports curve issues persist — but neither PI gain nor steerRatio is the right lever.

Need to investigate what the data has NOT covered:
- Planner-side curvature signal quality (is the planner itself producing jerky commands?)
- CAN message construction / ramp_type / EPAS-side dynamics
- Frequency-domain analysis of the wheel during curves (what frequency is "jerky"?)
- Predicted curvature blend behavior in curves (is the 0.30 blend at apex causing oscillation?)
- Path 4 A/B test (still dormant — was deferred per QA tier reordering)

Next iteration is open — analytical work needed before tuning.

---

## 15. Pipeline-Origin Investigation — 2026-06-01/02 (simulator-validated)

After Iter 2 revert, conducted a deep methodology-first analysis to identify the actual source of curve hunting. Investigation went through 4 adversarial QA rounds catching 15+ real bugs in early analyzer iterations; surfaced the true picture only after building a production-faithful pipeline simulator.

### Question
Where in the curvature pipeline (model → controlsd → carcontroller → EPAS) do the zero-crossings that the user perceives as "hunting" originate?

### Analyzer evolution and bugs caught by QA

| Round | Tool | Bugs caught |
|---|---|---|
| 1 | `analyze_curves_v2.py` first cross-route compare | Pipeline gain methodology (relative vs absolute power), EPAS CI ddof+t-stat, override threshold fragility, had_override skip, override rate normalization missing |
| 2 | Same + new mechanism metrics | Hunting score sampling-mode bias, quantization threshold floor, integral-flicker outliers, EMA lag at integer-sample resolution |
| 3 | `planner_trace.py` (within-CX1) | Threshold scaled to wrong precision, boxcar bias, pred fallback unflagged, pmd unused; clip_curvature mistakenly thought to filter (it's a no-op) |
| 4 | `des_source_trace.py` (modelV2 vs CX1) | CX1 sampling sparsity (2-3Hz avg) created phantom crossings when interp'd to 20Hz — most apparent filtering effects were sampling artifacts |
| 5-8 | `pipeline_simulator.py` (production replay) | 17 bugs across 4 rounds (STEER_STEP, SMOOTH_TAU, EMA cadence, sign-aware RL, ZOH model_des, PI=0, cold-start contamination, inactive cmd handling, warmup-press, integral warm-start, end-to-end CX1 validation) |

QA round 8 APPROVED simulator for Mode 0 tuning analysis with end-to-end validation: median |sim - cx1_cmd| = 0-2e-5 1/m, std 4-8e-5 1/m (well below CAN quantization 2e-5).

### Validated findings (post-simulator, 3 routes × ~10 moderate curves each)

| Stage | Crossings/sec (median) | Role |
|---|---|---|
| **Driving model `desiredCurvature`** | **4.2-5.0 cps** | Primary jitter source |
| Model `orientationRate` (→ pred) | 8-10 cps | Secondary source — much jittier than des |
| controlsd `clip_curvature` | (no change, no-op) | ISO jerk rate-limit triggers 0.006% of frames |
| `carControl.actuators.curvature` (= des in CX1) | 3.7-4.3 cps | Same as model — confirming clip is no-op |
| Blend (pred + des) | (varies) | Adds pred's high-freq into the stream |
| **Deadband + EMA stabilizer** | ema 1.7-2.3 cps | **THE dominant filter** — removes 1.6-2.2 cps median |
| PI (lc_kp×offset + lc_ki×integral) | preRL ~ema (median delta ~0) | Bidirectional — adds on some events, removes on others |
| Rate limit (sign-aware) | (no change net) | Almost never engaged (0% clip on most events) |
| Curvature safety (current_curv ± 0.004) | (no change net) | Only active above 9 m/s; rarely hit |
| **Final cmd to EPAS** | **2.0-2.8 cps** | 38-60% reduction from model |

### Mechanism-level conclusions (validated mechanistically AND empirically)

1. **`clip_curvature` is essentially a no-op.** model→des delta ≈ 0. Mechanistically: rate limit at MAX_LATERAL_JERK=5 m/s³ translates to per-cycle Δ that exceeds typical model output changes. Empirically: only triggers 5/82893 ticks in test data.

2. **PI cannot mechanistically cause cmd sign-flips.** Kp×offset + Ki×integral ≈ 7e-5 1/m max contribution, which is ~3.5× one cmd quantum (2e-5). PI can shift cmd magnitude but cannot create rapid direction reversals.

3. **Rate limiter is rarely engaged.** Across all tested events, RL clipped 0-0.3% of samples. Not a meaningful filter at observed magnitudes.

4. **The EMA stabilizer (`smooth_tau` speed-interp 0.12→0.04s) does ~all the filtering work.** Removes 1.6-2.2 cps median.

5. **The model output is the primary source of zero-crossings.** Both `desiredCurvature` (4-5 cps) and `orientationRate` (8-10 cps via pred) emit substantially more crossings than any downstream stage adds.

### Refuted hypotheses
- ❌ PI causes hunting (mechanism: too small to flip cmd)
- ❌ Rate limiter cycles cause hunting (RL not engaged in normal driving)
- ❌ clip_curvature filters meaningful content (no-op at observed magnitudes)
- ❌ Lower pc_blend_ratio 0.30→0.20 would meaningfully help (direct simulation showed 0.000 median benefit in QA round 3)
- ❌ Cross-config metric differences from controller tuning iterations are statistically defensible at current N (8-22 events/route, IQR overlaps everywhere)

### Confirmed regressions (separate from origin analysis)
- **Iter 2 (steerRatio=18) override rate per minute**: 0.32-0.42/min vs baseline 0.077/min (4-5×). Real regression. Already reverted to 17.2.

### Available levers for the "model emits jitter" finding

| Lever | Effect | Cost | Defensibility |
|---|---|---|---|
| Increase modeld `LAT_SMOOTH_SECONDS` (current 0.1s) | Smooths source directly | Model-bundle setting; not car-side code; sub-linear ROI on crossings | Plausible but requires bundle change |
| **Increase carcontroller `smooth_tau`** | Strengthens dominant filter | More apex lag (currently 200-400ms timing offset; could add ~50-100ms) | Car-side, simulator-testable |
| Lower MAX_LATERAL_JERK in clip_curvature | Makes clip start mattering | Safety implication; controlsd-side change | Less actionable, less impact |
| PMD-gated blending (de-weight pred when pred-des disagrees) | Reduces pred's high-freq contribution | Added complexity; needs new code | Promising but speculative |

### Analysis tools (kept for future investigation)
- `pipeline_simulator.py` — QA-approved production-faithful replay, comparable to actual CX1 cmd within rounding
- `planner_trace.py` — within-CX1 stage-by-stage breakdown (deprecated for absolute numbers, useful for within-route relative)
- `des_source_trace.py` — modelV2 vs CX1 trace with sample-rate masking
- `analyze_curves_v2.py` — event detection, phase metrics, cross-route comparison

### Investigation status
Findings validated, ready to test stronger `smooth_tau` empirically via simulator sweep. No deployment yet — sweep first, then propose specific tau value based on predicted cps reduction vs added lag.

---

## 16. Iteration 3 — smooth_tau bump (2026-06-02, PENDING VALIDATION)

> **⚠️ SUPERSEDED 2026-06-07:** iter3 (0.25,0.12) WAS driven and showed **NO reliable center-hunt benefit** (the
> predicted gain was a pooled-variance/outlier artifact; the EMA-lag cost was real). **Reverted on-device to
> (0.12,0.04).** Reverted in the repo too (2026-06-14 reconcile): `values.py:40` is back to (0.12,0.04), matching the
> device. Read the "predicted effect / validation plan" below as the original proposal, not the outcome.

After §15's pipeline-origin investigation identified EMA as the dominant filter, ran a tau sweep with rich metrics (cmd_band 0.5-3 Hz, lag, peak ratio, integral envelope), then validated the metrics against actual CX1 aLat via Phase 1 analysis. Fresh-QA round (no prior baggage) prescribed speed-controlled re-analysis before any deployment.

### Phase 1 — speed-controlled validation results (the key finding)

**Partial correlation (Spearman R², controlling for speed) across 70 events on 8 routes:**

| Metric pair | R²_raw | R²_partial (speed-controlled) | Verdict |
|---|---|---|---|
| cmd_cps → aLat_band 0.5-3 Hz | 0.034 | **0.012** | useless — collapses to noise |
| **cmd_band → aLat_band 0.5-3 Hz** | 0.272 | **0.652** | **STRONG predictor after speed-control** |
| cmd_cps → jerk_RMS | 0.023 | 0.038 | useless |
| cmd_band → jerk_RMS | 0.084 | 0.257 | moderate |
| speed → aLat_band | 0.521 | n/a | speed dominates raw aLat |

Fresh QA predicted cmd_band→aLat would collapse to noise after partialing speed. Instead it strengthened from 0.272 → 0.652. **cmd_band IS a defensible predictor of subjective aLat oscillation**, and speed was actually a SUPPRESSOR of the controller-side correlation (because both signals share a speed-baseline; removing it sharpens the controller effect).

### Speed-binned comparison — refuted the "iter2 was controller-worse" premise

| Speed bin | BASELINE_OK (route 9b) aLat_band | ITER2_JERKY (a0+a1) aLat_band | Δ |
|---|---|---|---|
| 20-35 mph | N=4, med=10.0 | N=2, med=4.9 | **iter2 LOWER (-5.1)** |
| 45-55 mph | N=4, med=48.3 | N=9, med=39.4 | **iter2 LOWER (-8.9)** |

**Speed distributions by label:**
- BASELINE_OK: median 33.7 mph
- ITER2_JERKY: median **48.9 mph** (15 mph higher)

**At matched speed, iter2 actually had LOWER aLat oscillation than baseline.** The 120% aLat difference in raw cross-route medians was entirely from speed distribution: iter2 was driven on highway, baseline was mixed surface+highway.

**Implication**: the user's subjective "jerky on iter2" perception was likely speed-confounded, not controller-confounded. Iter2's override-rate regression (4-5x baseline) was real but is a separate signal. **The current production tune already performs well at matched speed.**

### Predicted effect of smooth_tau (0.12, 0.04) → (0.25, 0.12)

Per simulator sweep across 6 routes × moderate+sharp curves, translated via the speed-controlled cmd_band → aLat regression (power-law exponent recalibrated by the higher partial R²):

| Curve class | cmd_band reduction | Predicted aLat reduction | Lag cost | Peak loss |
|---|---|---|---|---|
| Moderate | -21% median | **~15-20%** | +24-49 ms (median-to-P90) | 1.5-2.5% |
| Sharp | -32% median | **~25-30%** | +8-20 ms | 1-2.4% |

These predicted aLat reductions are above typical JND (~15-25%) for steering-induced lateral acceleration. Should be subjectively perceptible on a drive.

### Pre-deployment QA history

11 QA rounds across the whole investigation caught 30+ real bugs. Key tools after fixes:
- `pipeline_simulator.py` — QA round 8 APPROVED for Mode 0 use; median |sim - cx1_cmd| < 1e-5 1/m
- `phase1_validation.py` — speed-partialed correlation, speed-binned subjective comparison, jerk_RMS correlation
- 17 simulator structural bugs fixed: STEER_STEP, SMOOTH_TAU shape, EMA cadence, sign-aware RL, ZOH model_des, PI offline reconstruction, cold-start contamination, inactive cmd handling, warmup-press filter, integral warm-start from CX1 lInt, end-to-end CX1 validation

### Unverified risks (per fresh QA round)

1. **Anti-windup interaction at higher tau** — not modeled in sweep. Higher tau slows EMA → preRL may diverge from cmd_last in transients → rate limit may clip more → anti-windup fires → integral oscillation regime changes. Watch override behavior on first drive.
2. **Mode 0 only** — sweep assumes constant Mode 0. Verified for the 6 test routes via CX1 `cx1_last_smooth_tau` (logged as p4Tau field).
3. **Sub-linear power-law extrapolation** — regression is over baseline-tau events only; behavior at higher tau is extrapolated, not measured.
4. **Lane-offset EMA (1.5s) not swept** — only curvature EMA changes. If hunting source were PI's lane offset filter, this change wouldn't help.

### Change

**File**: `opendbc_repo/opendbc/car/ford/values.py` line 40
- WAS: `'smooth_tau': (0.12, 0.04),`
- IS:  `'smooth_tau': (0.25, 0.12),  # Iter 3 bump per §16 — stronger EMA on cmd, predicted -15-30% aLat`

Single-line, easily reverted. No safety-layer impact (`ford.h` unaffected).

### Validation plan

On next drive:
1. Drive normal route mix (paired baseline+new tau ideal but optional)
2. Note any subjective change at moderate/sharp curves — "smoother" vs "delayed/lazy"
3. Watch for any new override clustering (anti-windup risk #1)
4. Pull rlog and CX1; rerun phase1_validation.py with new drive included to verify cmd_band actually dropped per prediction at matched speed bins

### Revert criteria

Any of:
- Override rate per engaged minute > 0.2/min (current baseline ~0.08/min)
- User subjectively reports "delayed", "lazy", or "felt worse"
- aLat_band 0.5-3 Hz at matched speed bin INCREASED vs baseline

If revert: change line back to `'smooth_tau': (0.12, 0.04)`, commit, reboot device.

### Commits

Will be: `<pending>`.

---

## 17. Upstream Repos & Update Workflow

We don't merge from upstream wholesale — we surgically cherry-pick infrastructure updates and helpful changes while preserving Explorer ST customizations. This section documents what's tracked and how.

### Remotes configured (as of 2026-06-03)

**Sunnypilot main repo** (`/Users/dregilley/Documents/GitHub/sunnypilot`):
| Remote | URL | Tracked branch | Purpose |
|---|---|---|---|
| `origin` | `https://github.com/DreCode3/sunnypilot.git` | `2021_explorer_st-mici` | User's fork (this is what device pulls from) |
| `upstream` | `https://github.com/sunnypilot/sunnypilot` | **`dev`** | Sunnypilot's dev branch — **primary infrastructure source** |
| `bluepilot` | `https://github.com/BluePilotDev/bluepilot.git` | **`bp-6.0`** | BluePilot's Comma 4 branch — **primary Ford-feature source** |
| `openpilot` | `https://github.com/commaai/openpilot.git` | `master` | Upstream openpilot (reference; rarely cherry-pick directly) |

**opendbc submodule** (`/opendbc_repo`):
| Remote | URL | Tracked branch | Purpose |
|---|---|---|---|
| `origin` | `https://github.com/DreCode3/opendbc.git` | `2021_explorer_st-mici` | User's opendbc fork |
| `upstream` | `https://github.com/commaai/opendbc.git` | `master` | Upstream commaai opendbc |
| `sunnypilot` (alias `sunnypilot_upstream`) | `https://github.com/sunnypilot/opendbc.git` | `master` | sunnypilot's opendbc fork |

### Workflow: surgical updates, not merges

**The pattern is selective cherry-picks**, not merges. Why: we have 970+ commits ahead on sunnypilot with deep Explorer ST customization across carcontroller, ford.h safety, locationd, controlsd. A merge would create huge conflicts and risk silently overriding tuning. Cherry-pick lets us pull specific improvements (infrastructure cleanups, model updates, ALC behavior, etc.) while keeping our PI controller, smooth_tau values, blend ratios, rate limits, and ford.h safety values untouched.

### Step-by-step

```bash
# 1. Fetch everything
cd /Users/dregilley/Documents/GitHub/sunnypilot
git fetch --all

cd opendbc_repo
git fetch --all
cd ..

# 2. See what's new in tracked branches
# Sunnypilot infrastructure (typical target: selfdrive/, system/, common/)
git log HEAD..upstream/dev --oneline -- selfdrive/ system/ common/ | head -30

# BluePilot Ford features (typical target: opendbc-side already, but BP touches main repo too)
git log HEAD..bluepilot/bp-6.0 --oneline -- opendbc_repo/ selfdrive/car/ | head -30

# opendbc upstream (Ford-specific changes)
cd opendbc_repo
git log HEAD..upstream/master --oneline -- opendbc/car/ford/ opendbc/car/lateral.py opendbc/safety/ | head -30

# 3. Review a candidate commit
git show <sha>                  # full diff
git log -1 --stat <sha>          # files changed summary

# 4. Cherry-pick if useful
git cherry-pick <sha>

# 5. If conflict, resolve preserving Explorer ST tuning
#    Reference customizations.md §14/§15/§16 for what NOT to overwrite
git cherry-pick --continue       # after resolution

# 6. Commit message convention: use "cherry-pick" + upstream PR # if applicable
#    Example existing cherry-picks (memory baseline):
#    - locationd cam odo delay (commaai #37543) — 100ms camera pipeline delay compensation
#    - livePose timestamp (commaai #37704) — paramsd/torqued use correct livePose timestamp
#    - filter time (commaai #37697) — locationd publish filter time
#    See customizations.md §11 for these cherry-picks documented
```

### What to look for / what NOT to pull

**Pull preferentially**:
- Infrastructure refactors (process_config, messaging schemas, modeld constants)
- Logging/diagnostic improvements (CX1-style telemetry hooks)
- Model pipeline updates (modeld, locationd, paramsd, torqued)
- BluePilot bp-6.0 Ford-specific feature additions (anti_overshoot variants, new CAN handlers)
- Safety layer additions IF they're additive (new range checks, new violation types)

**Do NOT pull without careful review**:
- `opendbc/car/ford/values.py` — contains our tuning (steerRatio, smooth_tau, rate limits, etc.)
- `opendbc/car/ford/carcontroller.py` — contains our PI controller, blend logic, override handling, anti-windup
- `opendbc/safety/modes/ford.h` — contains our rate limits, reset bypass latch, 4-signal validation
- Anything touching: `_smooth_tau`, `lc_kp`, `lc_ki`, `_apply_curve_mode`, `LANE_OFFSET_EMA_TAU`, `INTEGRAL_CAP`

If an upstream commit changes our customized files, **review carefully**: extract just the non-tuning parts (e.g., signature changes, new param threading, refactor mechanics) and leave the tuning-related lines untouched. Often easier to manually apply the relevant changes than to cherry-pick.

### Position as of 2026-06-03 (snapshot — re-check after each fetch)

| Repo | vs upstream/dev | vs bluepilot/bp-6.0 |
|---|---|---|
| sunnypilot | upstream 1 ahead, we 970 ahead | bluepilot 326 ahead, we 869 ahead |

| Repo | vs upstream/master | vs sunnypilot/master |
|---|---|---|
| opendbc_repo | upstream 113 ahead, we 267 ahead | sunnypilot 141 ahead, we 101 ahead |

**326 commits behind bluepilot/bp-6.0** is the largest gap and most likely to have useful Ford-feature updates. Worth a periodic survey.

### Sanity check after cherry-picks

After any cherry-pick that touches steering, longitudinal, or safety:

1. Run analysis tools on existing routes to confirm baseline behavior preserved:
   ```bash
   .venv311/bin/python explorer_st_logs/phase1_validation.py --routes route_9b,route_aa
   # baseline metrics should match prior runs (within noise)
   ```
2. Diff `values.py`, `carcontroller.py`, `ford.h` against last-known-good HEAD to confirm tuning values untouched:
   ```bash
   git diff HEAD~5 -- opendbc_repo/opendbc/car/ford/values.py opendbc_repo/opendbc/car/ford/carcontroller.py opendbc_repo/opendbc/safety/modes/ford.h | head -50
   ```
3. Test the device build before driving: deploy and reboot, verify `cx1_last_smooth_tau` (p4Tau field) reads expected value.

### History of cherry-picks

Already in baseline (per §11):
1. `commaai/openpilot#37543` — locationd cam odo delay (100ms camera pipeline delay compensation)
2. `commaai/openpilot#37704` — livePose timestamp (paramsd/torqued use correct livePose timestamp)
3. `commaai/openpilot#37697` — filter time (locationd publish filter time)

Future cherry-picks should be added to §11 with: upstream repo, PR/SHA, what it does, why we wanted it, any local modifications.

