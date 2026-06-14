# BluePilot 6.0.1 vs Our Implementation — Diff Analysis
*Analyzed: 2026-03-19 against branch `bp-6.0.1-beta`*
*Reference: https://github.com/BluePilotDev/bluepilot/tree/bp-6.0.1-beta*

---

## Overview

BluePilot 6.0.1 represents a significant architectural expansion of Ford lateral control compared to stock openpilot. The most important differences are in `ford.h` (safety layer) and `carcontroller.py`. Their `values.py` and `interface.py` are nearly identical to ours for lateral-relevant parameters.

---

## `ford.h` — Safety Layer

### 1. reset_bypass_latch (BP has it, we don't)

**What it does**: When both `desired_curvature == 0` AND `desired_path_angle == 0` are sent simultaneously, the panda activates a 3-second bypass window that clears all safety violations. This allows the controller to ramp curvature freely after a driver override without triggering rate-limit violations during the transition.

```c
static uint8_t reset_bypass_latch_counter = 0;
static const uint8_t RESET_BYPASS_LATCH_DURATION = 60;  // ~3.0 seconds at 20Hz

// Inside ford_tx_hook, after all violation checks:
if ((desired_curvature == 0) && (desired_path_angle == 0)) {
    reset_bypass_latch_counter = RESET_BYPASS_LATCH_DURATION;
    violation = false;
} else if (reset_bypass_latch_counter > 0) {
    reset_bypass_latch_counter--;
    violation = false;
}
```

**Why this matters**: After a driver override (steeringPressed), the controller snaps to measured curvature (or zero). When the driver releases and the system re-engages, the normal rate limits prevent the curvature command from climbing quickly back to the desired value — the PSCM interprets blocked messages as faults and can show "steering unavailable." The bypass window lets the post-reset ramp happen freely.

**How BP uses it**: Their carcontroller.py keeps `path_angle = 0` during `post_reset_ramp_active`. Combined with zeroing curvature during the reset itself, the latch fires on the first frame of re-engagement. The bypass stays active for 3 seconds — long enough for any curvature transition.

**Our current state**: We have `post_reset_ramp_active` in software, but the safety layer still enforces rate limits during ramp-up. We always send `path_angle = 0`, which means if we adopt their ford.h, our latch would fire automatically on override resets with no additional carcontroller.py changes needed — because we already zero curvature AND path_angle is always zero.

**To adopt**: Add the `reset_bypass_latch_counter` state variable and the latch logic at the end of the `FORD_LateralMotionControl` tx check block in our `ford.h`.

---

### 2. Four Independent Signal Validators (BP has them, we have one)

BP's ford.h validates all four IPMA signals independently. We only validate curvature.

#### Our current state: 1 checker
We send a CAN message with four signal fields (curvature, curvature_rate, path_angle, path_offset) but the panda only validates curvature. The other three fields are hardcoded to zero in our carcontroller.py and pass through unchecked — *except* curvature_rate, which we do send non-zero values for. This is a gap.

#### BP's four checkers and their limits:

**A) `FORD_PATH_ANGLE_LIMITS`** — validates `path_angle` signal
```c
static const AngleSteeringLimits FORD_PATH_ANGLE_LIMITS = {
  .max_angle = 1000,
  .angle_deg_to_can = 2000,       // 1 / (5e-4) rad to CAN
  .max_angle_error = 4,           // ±0.002 rad
  .angle_rate_up_lookup = {
    .x = {5., 15., 25.},
    .y = {0.003, 0.0015, 0.002}   // rad/s
  },
  .angle_rate_down_lookup = {
    .x = {5., 15., 25.},
    .y = {0.003, 0.0015, 0.002}
  },
  .angle_error_min_speed = 9.9,
  .frequency = 100U,              // 100Hz (LKA_STEP = 3 → 33Hz, so headroom exists)
  .enforce_angle_error = true,
  .inactive_angle_is_zero = true,
};
```
Path_angle is in radians. The max ±0.25 rad limit (from `FORD_PATH_ANGLE_MAX = 0.25f`) represents a meaningful steering correction. At 0.002 rad max_angle_error, the PSCM can only deviate 0.002 rad from the last commanded value before the panda blocks it. Rate is symmetric (same up/down).

**B) `FORD_PATH_OFFSET_LIMITS`** — validates `path_offset` signal
```c
static const AngleSteeringLimits FORD_PATH_OFFSET_LIMITS = {
  .max_angle = 100,               // ±1.0 meter
  .angle_deg_to_can = 100,        // 1 / 0.01 meter to CAN
  .max_angle_error = 2,           // ±0.02 meter
  .angle_rate_up_lookup = {
    .x = {5., 15., 25.},
    .y = {0.05, 0.025, 0.01}     // meters/s
  },
  .angle_rate_down_lookup = {
    .x = {5., 15., 25.},
    .y = {0.05, 0.025, 0.01}
  },
  .angle_error_min_speed = 5.0,
  .frequency = 20U,               // 20Hz
  .enforce_angle_error = true,
  .inactive_angle_is_zero = true,
};
```
Path_offset is in meters (lateral distance from path center). Rate limit of 0.01 m/s at 25 m/s means the path offset can change at most 0.01 m per second at highway speed — very conservative. The 0.05 m/s at low speed allows quicker recovery.

**C) `FORD_CURVATURE_RATE_LIMITS_CAN`** — validates `curvature_rate` signal (CAN vehicles)
```c
static const AngleSteeringLimits FORD_CURVATURE_RATE_LIMITS_CAN = {
  .max_angle = 100,
  .angle_deg_to_can = 4000000,    // 1 / (1e-6) — very high resolution
  .max_angle_error = 2,
  .angle_rate_up_lookup = {
    .x = {5., 15., 25.},
    .y = {0.05, 0.025, 0.01}
  },
  // ...same up/down
  .angle_error_min_speed = 5.0,
  .frequency = 20U,
};
```
Note: same rate values as path_offset but different `angle_deg_to_can` scale.

**D) `FORD_CURVATURE_RATE_LIMITS_CANFD`** — same structure, different CAN scale factor
```c
.angle_deg_to_can = 1000000,    // CANFD uses different encoding
```

#### The gap in our implementation
We send curvature_rate (computed as `d(predicted_curvature)/dt`, clipped to ±0.001023 m⁻¹/s) but the panda doesn't validate it. A software bug generating runaway curvature_rate values would not be caught. **Adding the `FORD_CURVATURE_RATE_LIMITS_CAN` checker is the highest-priority safety layer addition for our branch**, independent of whether we adopt path_angle centering.

---

### 3. Curvature Limits — Identical to Ours

Both use the same `FORD_LIMITS` macro:
```c
.max_angle = 1000,          // 0.02 curvature
.angle_deg_to_can = 50000,  // 1 / 2e-5
.max_angle_error = 100,     // 0.002 * 50000 = CURVATURE_ERROR 0.002
.angle_rate_up_lookup   = { {5., 16., 25.}, {0.0026, 0.0013, 0.0001} }
.angle_rate_down_lookup = { {5., 16., 25.}, {0.0026, 0.0015, 0.0002} }
.angle_error_min_speed = 10.0
```

Our plan to increase `max_angle_error` to 200 (CURVATURE_ERROR 0.004) and relax high-speed rate limits is NOT something BP has done — it's our own modification and still valid as a separate change.

---

## `carcontroller.py` — Key Differences

### 1. Lane Centering: path_angle PID vs Our Curvature Offset

**BP approach**: Uses a `PIDController` from `common.pid` applied to a computed path_angle correction. The `compute_dm_msg_values()` helper in `helpers.py` calculates the path_angle value from lane offset. This correction is sent as the `path_angle` field in the `LateralMotionControl` CAN message.

The path_angle signal goes directly into the PSCM's lateral planner — it's the "steering correction angle to reach the ideal path" signal that the PSCM was designed to receive. It's far more powerful than a curvature offset because the PSCM acts on it differently.

**Their PID for Explorer-class vehicles**:
- `Low Curvature PID Gain: 1.0` (reduced from 3.0 default for smaller vehicles)
- `k_i = 0.05` (integral term)
- Activation: `|curvature| < threshold`, `speed > gate`

**Our approach**: Curvature offset — `apply_curvature += lane_offset * centering_gain`. Simpler, safer, doesn't require ford.h changes, but less effective (53% convergence observed vs ~80%+ expected with path_angle).

**Why path_angle is more effective**: The PSCM interprets path_angle as the angular error it needs to steer out. A 0.01 rad path_angle correction directly commands the PSCM to adjust steering by 0.01 rad. A 0.0001 m⁻¹ curvature correction only indirectly influences path through the curvature tracking loop.

**Safety prerequisite for path_angle**: Must add `FORD_PATH_ANGLE_LIMITS` checker to `ford.h` before sending non-zero path_angle. Without it, a runaway path_angle (max 0.25 rad) could cause a ditch-entry scenario as documented in BP's articles.

---

### 2. post_reset_ramp_active Integration with the Safety Latch

**BP's design**: During `post_reset_ramp_active`, `path_angle` is held at zero. This is intentional — it keeps the "both signals zero" condition for the safety latch active longer, extending the bypass window.

```python
# BP carcontroller.py pseudocode:
if self.post_reset_ramp_active:
    path_angle = 0.0           # keeps latch active
    curvature = ramp_up_value  # ramps freely under latch
```

**Our design**: We always send `path_angle = 0`, so the latch condition (`curvature == 0 AND path_angle == 0`) would fire naturally during our override snap. This means adopting their `ford.h` latch gives us the benefit without changing carcontroller.py.

---

### 3. Predicted Curvature Blend Ratio — Tunable vs Hardcoded

**BP**: User-configurable via params. Default 0.4, recommended 0.30 for Explorer/MachE/Maverick class.
```python
self.pc_blend_ratio_high = float(params.get("PredictedCurvatureBlendRatioHigh") or 0.4)
self.pc_blend_ratio_low = float(params.get("PredictedCurvatureBlendRatioLow") or 0.4)
```

**Ours**: Hardcoded at 0.30 (`self.pc_blend_ratio = 0.30`). We already match their recommended value for Explorer-class. No functional gap, just less flexibility.

---

### 4. anti_overshoot() — Identical

Both implement the same function:
```python
def anti_overshoot(apply_curvature, apply_curvature_last, v_ego):
  diff = 0.1
  tau = 5  # 5s smooths over the overshoot
  dt = DT_CTRL * CarControllerParams.STEER_STEP
  alpha = 1 - np.exp(-dt / tau)
  lataccel = apply_curvature * (v_ego ** 2)
  last_lataccel = apply_curvature_last * (v_ego ** 2)
  last_lataccel = apply_hysteresis(lataccel, last_lataccel, diff)
  last_lataccel = alpha * lataccel + (1 - alpha) * last_lataccel
  output_curvature = last_lataccel / (max(v_ego, 1) ** 2)
  return float(np.interp(v_ego, [5, 10], [apply_curvature, output_curvature]))
```

BP only applies it to Bronco Sport and F-150. We inherited the function but it's not active for Explorer.

---

### 5. handle_post_lane_change_transition()

BP has a dedicated method for smoothing the transition after a lane change completes. Details not fully retrieved but likely applies a short ramp or dampening period after `laneChangeState` returns to `idle`. We handle lane changes with a static factor (0.85–0.95) but no explicit post-change transition.

---

### 6. calculate_lateral_uncertainty()

BP computes some uncertainty metric for lateral control, likely used to modulate blend ratios or confidence weighting. Details not retrieved. May relate to their "Confidence Ball" UI feature.

---

## `values.py` — Largely Identical

| Parameter | BluePilot 6.0.1 | Ours |
|---|---|---|
| CURVATURE_ERROR | 0.002 | 0.002 |
| Rate up [5,16,25 m/s] | [0.0025, 0.0012, 0.00008] | [0.0025, 0.0012, 0.00008] |
| Rate down [5,16,25 m/s] | [0.0025, 0.0014, 0.00018] | [0.0025, 0.0014, 0.00018] |
| Explorer MK6 steerRatio | **16.8** | **16.8** |
| Explorer MK6 wheelbase | 3.025 m | 3.025 m |

**Note on steerRatio**: Both BP and we use 16.8 for the Explorer MK6. Our telemetry analysis shows a measured median effective SR of 17.23 at highway speed. The discrepancy is 2.5% — minor but worth updating to 17.2 for accuracy. Our earlier concern about 15.0 was a mistake — 15.0 was the analysis script constant (`STEER_RATIO_ASSUMED`), not the actual car parameter.

---

## `interface.py` — Minor Differences

- BP enables `alphaLongitudinalAvailable = True` for all Ford (including CAN). We do too.
- BP's CAN longitudinal in `ford.h` is gated behind `#ifdef ALLOW_DEBUG`. We don't have that gate — CAN longitudinal can be enabled on our branch without a debug build.
- `steerActuatorDelay = 0.22` — identical.
- BP has `_get_params_sp()` enabling ICBM (Intelligent Cruise Button Management) for Ford. We don't have ICBM.

---

## Adoption Priority

### Adopt Now (independent of path_angle work)

| Change | File | Effort | Why |
|---|---|---|---|
| `reset_bypass_latch_counter` | `ford.h` | ~20 lines | Cleaner override recovery; our `path_angle=0` already triggers it |
| `curvature_rate` safety check | `ford.h` | ~30 lines | Closes existing gap — we send curvature_rate but panda doesn't validate it |
| steerRatio 16.8 → 17.2 | `values.py` | 1 line | Matches telemetry-measured median SR |

### Adopt Later (if/when upgrading to path_angle centering)

| Change | File | Effort | Why |
|---|---|---|---|
| `FORD_PATH_ANGLE_LIMITS` checker | `ford.h` | ~30 lines | Required safety prerequisite |
| `FORD_PATH_OFFSET_LIMITS` checker | `ford.h` | ~30 lines | Required if using path_offset signal |
| `compute_dm_msg_values` helper | `helpers.py` (new) | Medium | Computes path_angle from lane offset |
| `PIDController` for lane centering | `carcontroller.py` | Medium | Replaces curvature-offset PI with path_angle PID |

### Not Needed

| Item | Reason |
|---|---|
| Tunable pc_blend_ratio | Already at 0.30 (their recommended value for Explorer-class) |
| anti_overshoot for Explorer | BP only applies to Bronco/F-150; not validated for Explorer |
| ICBM | Longitudinal feature, out of scope |

---

## Path to path_angle Lane Centering (Future Reference)

If curvature-offset PI centering proves insufficient after testing, here is the complete migration path:

1. **ford.h**: Add `FORD_PATH_ANGLE_LIMITS` struct (as above)
2. **ford.h**: Add `path_angle_cmd_checks()` function (as above)
3. **ford.h**: Add `desired_path_angle_last` state variable and call `path_angle_cmd_checks()` in `ford_tx_hook` for `FORD_LateralMotionControl`
4. **ford.h**: Add `reset_bypass_latch_counter` (should already be done by then)
5. **carcontroller.py**: Port `compute_dm_msg_values()` from BP's `helpers.py` — this converts lane offset to path_angle using vehicle geometry
6. **carcontroller.py**: Replace `apply_curvature += lane_offset * centering_gain` with path_angle PID:
   ```python
   # Replace centering_gain with a PIDController
   self.lane_centering_pid = PIDController(k_p=1.0, k_i=0.05, k_f=0.0)
   path_angle = self.lane_centering_pid.update(lane_offset, speed=CS.out.vEgoRaw)
   path_angle = float(np.clip(path_angle, -0.05, 0.05))  # conservative limit initially
   ```
7. **carcontroller.py**: Send non-zero path_angle in `create_lat_ctl_msg()` call
8. **Test**: Start with `k_p=0.5, k_i=0.0` (P-only first), validate no oscillation or windup before enabling I term

**Key safety note from BP articles**: Path_angle "winds up quickly but fails to unwind properly" if limits are wrong. Test conservatively — start at half their gain (k_p=0.5 vs their 1.0 for Explorer-class) and verify the FORD_PATH_ANGLE_LIMITS rate constraints are sufficient to prevent runaway.

---

*Last updated: 2026-03-19*
