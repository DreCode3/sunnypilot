# Explorer ST Longitudinal Customizations

Documents every longitudinal change beyond stock sunnypilot on the `2021_explorer_st-mici` branch.

## 1. Hardcoded openpilotLongitudinalControl

- **File**: `opendbc_repo/opendbc/car/ford/interface.py` (lines 57-59)
- **Before**: `ret.openpilotLongitudinalControl` set via `AlphaLongitudinalEnabled` param (development-only, clears on reboot)
- **After**: Hardcoded `ret.safetyConfigs[-1].safetyParam |= FordSafetyFlags.LONG_CONTROL.value` and `ret.openpilotLongitudinalControl = True`
- **Why**: The AlphaLongitudinalEnabled param is DEVELOPMENT_ONLY and resets every reboot, making it unusable for daily driving
- **Result**: Works reliably -- longitudinal control persists across reboots

## 2. ACCEL_MAX reduced from 2.0 to 1.5

- **File**: `opendbc_repo/opendbc/car/ford/values.py` (line 70)
- **Before**: `ACCEL_MAX = 2.0` (stock sunnypilot)
- **After**: `ACCEL_MAX = 1.5`
- **Why**: The Explorer ST's 3.0L twin-turbo V6 produces significantly more power than typical Ford vehicles. At 2.0 m/s^2 the acceleration felt overly aggressive and jerky
- **Result**: Smoother acceleration that feels appropriate for the ST's powertrain

## 3. PID Tuning: kpV zeroed, kiV reduced

- **File**: `opendbc_repo/opendbc/car/ford/interface.py` (lines 40-42)
- **Before**: `kpV = [0.]` (stock was already zero in some versions), `kiV = [0.5]`
- **After**: `kpV = [0.]`, `kiV = [0.3]`
- **Why**: `kpV` caused longitudinal hunting (100+ reversals per 5 seconds). `kiV` at 0.5 caused overshoot from the ST's aggressive brakes and engine response
- **Result**: Integral-only control eliminates hunting. Lower Ki reduces overshoot while still maintaining adequate response

## 4. Speed-Error-Based Soft Gas Ramp + Coast Zone

- **File**: `opendbc_repo/opendbc/car/ford/carcontroller.py` (lines 460-485)
- **File**: `opendbc_repo/opendbc/car/ford/values.py` (lines 72-75)
- **Before**: Stock logic sends full PID gas output at all times
- **After**: Three-tier speed-error ramp:
  - `planner_accel < COAST_ZONE_MIN (-0.5)`: full braking (pass through)
  - `planner_accel` between -0.5 and 0: coast (gas = INACTIVE_GAS, accel = 0)
  - `planner_accel >= 0` with `speed_error < 0.22 m/s` (~0.5 mph): coast
  - `planner_accel >= 0` with `speed_error 0.22-0.90 m/s`: ramped gas (0-100%)
  - `planner_accel >= 0` with `speed_error > 0.90 m/s` (~2 mph): full PID gas
- **Constants added**: `COAST_ZONE_MIN = -0.5`, `COAST_ZONE_MAX = +0.25`
- **Why**: Stock PID sends constant small gas pulses near cruise speed, causing surging and unnecessary fuel consumption. The coast zone lets the car glide when close to target
- **Result**: Much smoother highway cruising. Eliminated gas pulsing near set speed. However, this approach lacks lead-awareness -- it coasts the same whether a lead car is present or not

## 5. Brake Request Hysteresis

- **File**: `opendbc_repo/opendbc/car/ford/carcontroller.py` (lines 496-500)
- **Before**: Single `brake_request` with thresholds 0.3 (release) / 0.0 (engage) on pitch-compensated accel
- **After**: Same single `brake_request` variable used for both `AccBrkPrchg_B_Rq` and `AccBrkDecel_B_Rq`
- **Why**: Stock approach from upstream sunnypilot. Both brake and precharge use the same boolean
- **Result**: Works but is not as refined as BluePilot's separate brake/precharge with independent hysteresis

## Commit History (longitudinal-specific)

1. `dbd0c3aa7a` - Initial longitudinal tuning: coasting zone, PID changes, brake softening
2. `51b133caf6` - Fix braking safety: restore jerk limit, fix coast zone thresholds
3. `5070cc4e3c` - Revert Kp to 0, fix hunting
4. `73fb817a49` - Lead-aware coasting (attempted)
5. `6f1fde4281` - Remove lead-aware coasting, keep simple coast zone (reverted lead logic)
6. `8a5fe1d369` - Hardcode longitudinal control on
7. `8e73070fd8` - Widen coast zone to reduce gas pulsing
8. `8fa40a30e2` - Speed-error soft gas ramp
9. `033eeae39d` - Narrow coast deadband for highway speed

## Known Issues

- **No lead awareness**: The coast zone operates purely on speed error and planner accel, with no knowledge of whether a lead car is present or how far away it is. This means the car coasts identically whether following traffic or on an open road.
- **No separate brake/precharge control**: Both `AccBrkPrchg_B_Rq` and `AccBrkDecel_B_Rq` use the same boolean, losing the benefit of pre-charging brakes slightly before full braking engagement for smoother deceleration.
- **No downhill compensation toggle**: The stock pitch compensation is always applied in both directions, which can cause harsh braking on downhill grades where the Ford PCM already has its own compensation.
