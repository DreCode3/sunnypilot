# Ford CAN Bus Reference — 2021 Explorer ST (Q3/CAN)

## Architecture

### CGEA 1.3 CAN Bus Topology
The 2021 Explorer ST uses Ford's **CGEA 1.3** (Common Global Electronic Architecture). Key buses:

| Bus | Type | Speed | Purpose |
|-----|------|-------|---------|
| HS-CAN1 (Bus 0) | High-Speed | 500 Kbps | Powertrain, chassis, ADAS |
| HS-CAN2 (Bus 2) | High-Speed | 500 Kbps | Camera, body modules |
| Private CAN | High-Speed | 500 Kbps | Radar (CCM) ↔ IPMA only |

### Key Modules

| Module | Abbrev | CAN ID (UDS) | Role |
|--------|--------|-------------|------|
| Power Steering Control Module | PSCM | 0x730 | Executes steering curvature commands |
| Image Processing Module A | IPMA | 0x706 | Camera (Mobileye Q3), sends LCA/ACC commands |
| ABS/ESC Module | ABS_ESC | 0x760 | Vehicle speed, braking, stability |
| Powertrain Control Module | PCM | 0x7E0 | Engine, throttle, cruise state |
| Cruise Control Module | CCM | 0x764 | Front radar (Delphi MRR), on private bus |
| Gateway Module | GWM | — | Routes messages across all CAN buses |
| Restraints Control Module | RCM | — | Yaw rate sensor (via GWM) |

### Q3 vs Q4 (Explorer ST is Q3)
- **Q3 (CAN)**: Uses `LateralMotionControl` (0x3D3), 12-pin Molex harness
- **Q4 (CAN FD)**: Uses `LateralMotionControl2` (0x3D6), 20-pin harness, counter+checksum
- **Explorer ST is Q3** — `MAX_LATERAL_ACCEL` clipping does NOT apply

---

## CRITICAL: Sign Conventions & The Curvature Negation

### The Negation That Bit Us
**The code NEGATES curvature before sending to CAN** (carcontroller.py lines 480-484):
```python
can_sends.append(fordcan.create_lat_ctl_msg(packer, CAN, CC.latActive,
                 0., 0., -apply_curv_send, -apply_curvature_rate, ...))
#                        ^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^
#                        NEGATED!           NEGATED!
```

### Convention Summary

| Value | Code Convention | CAN Wire Convention |
|-------|----------------|---------------------|
| Curvature | **Positive = LEFT** | **Positive = RIGHT** (negated) |
| Curvature rate | **Positive = LEFT** | **Positive = RIGHT** (negated) |
| Path offset | Positive = RIGHT | Positive = RIGHT (not negated, sent as 0) |
| Path angle | Positive = RIGHT | Positive = RIGHT (not negated, sent as 0) |
| `current_curvature` | `-yawRate / vEgo` → **Positive = LEFT** | N/A (internal only) |
| `yawRate` (Ford) | **Positive = clockwise = RIGHT turn** | N/A |

### How to Read the EPAS Bias
Analysis script computes: `measured_curv = -yawRate / vEgo` (positive = LEFT)
```
Bias = measured - applied
  +0.000245 = EPAS over-delivers LEFTWARD
```
To compensate: `apply_curvature += 0.000245` (adds rightward in code, which after CAN negation becomes leftward reduction on wire).

---

## Lateral Control Messages

### LateralMotionControl (0x3D3 / 979) — Primary Steering
- **Bus**: 0 (main)
- **Sender**: IPMA_ADAS (us)
- **Receiver**: PSCM
- **Frequency**: 20 Hz (`STEER_STEP = 5` at 100 Hz base)
- **Length**: 8 bytes

| Signal | Bits | Scale | Offset | Physical Range | Units | Description |
|--------|------|-------|--------|----------------|-------|-------------|
| `LatCtlCurv_No_Actl` | 7\|11 | 2E-5 | -0.02 | [-0.02, 0.02094] | 1/m | **Curvature** (negated in code) |
| `LatCtlCurv_NoRate_Actl` | 12\|13 | 2.5E-7 | -0.001024 | [-0.001024, 0.00102375] | 1/m² | **Curvature rate** (negated) |
| `LatCtlPath_An_Actl` | 31\|11 | 0.0005 | -0.5 | [-0.5, 0.5235] | rad | Path angle (sent as 0) |
| `LatCtlPathOffst_L_Actl` | 47\|10 | 0.01 | -5.12 | [-5.12, 5.11] | m | Path offset (sent as 0) |
| `LatCtl_D_Rq` | 36\|3 | 1 | 0 | [0, 7] | enum | Mode (1=ContinuousPathFollowing) |
| `LatCtlRampType_D_Rq` | 53\|2 | 1 | 0 | [0, 3] | enum | 0=Slow, 1=Medium, 2=Fast, 3=Immediate |
| `LatCtlPrecision_D_Rq` | 33\|2 | 1 | 0 | [0, 3] | enum | 0=Comfortable, 1=Precise |
| `LatCtlRng_L_Max` | 63\|6 | 2 | 0 | [0, 126] | m | Max range |
| `HandsOffCnfm_B_Rq` | 51\|1 | 1 | 0 | [0, 1] | bool | Hands-off confirmation |

**PSCM Polynomial Model**: The 4 signals form a 3rd-order polynomial describing the road centerline:
- c0 = path_offset (lateral distance)
- c1 = path_angle (heading error)
- c2 = curvature
- c3 = curvature_rate

We currently only use curvature + curvature_rate (path_offset and path_angle sent as 0).

**CAN Inactive Values** (from ford.h):
| Signal | CAN Raw Inactive | Physical |
|--------|-----------------|----------|
| Curvature | 1000 | 0.0 1/m |
| Curvature rate | 4096 (CAN) / 1024 (CANFD) | 0.0 1/m² |
| Path offset | 512 | 0.0 m |
| Path angle | 1000 | 0.0 rad |

### LateralMotionControl2 (0x3D6 / 982) — CAN FD Only
Not used on Explorer ST (Q3). Same signals but with:
- Counter: `LatCtlPath_No_Cnt` (4-bit, increments per message)
- Checksum: `LatCtlPath_No_Cs` (8-bit, `0xFF - sum`)
- Different curvature rate scale: 1E-6 vs 2.5E-7
- Mode field: `LatCtl_D2_Rq` (0=None, 1=LimitedMode, 2=ExtendedMode, 3=SafeRampOut)

### Lane_Assist_Data1 (0x3CA / 970) — Required Companion
**Must be sent alongside LateralMotionControl** or PSCM ignores curvature commands.
- **Frequency**: 33 Hz (`LKA_STEP = 3`)
- **Safety constraint**: `LkaActvStats_D2_Req` must be 0 (panda blocks non-zero)
- This is the LKA (Lane Keeping Aid) message — subject to 10s lockout if used directly
- We send it empty; actual control uses LateralMotionControl (LCA/TJA API, no lockout)

---

## Longitudinal Control Messages

### ACCDATA (0x186 / 390) — Primary Longitudinal
- **Bus**: 0 (main)
- **Sender**: IPMA_ADAS (us)
- **Receiver**: GWM, ABS_ESC, PCM
- **Frequency**: 50 Hz (`ACC_CONTROL_STEP = 2`)
- **Length**: 8 bytes

| Signal | Bits | Scale | Offset | Physical Range | Units | Description |
|--------|------|-------|--------|----------------|-------|-------------|
| `AccBrkTot_A_Rq` | 4\|13 | 0.0039 | -20 | [-20, 11.9449] | m/s² | **Brake/decel request** |
| `AccPrpl_A_Rq` | 49\|10 | 0.01 | -5 | [-5, 5.23] | m/s² | **Gas/accel request** |
| `AccPrpl_A_Pred` | 17\|10 | 0.01 | -5 | [-5, 5.23] | m/s² | Predicted accel (sent as -5.0) |
| `AccVeh_V_Trg` | 32\|9 | 0.5 | 0 | [0, 255.5] | kph | Target speed |
| `Cmbb_B_Enbl` | 50\|1 | — | — | [0, 1] | bool | **ACC enabled** |
| `AccBrkPrchg_B_Rq` | 54\|1 | — | — | [0, 1] | bool | **Pre-charge brakes** |
| `AccBrkDecel_B_Rq` | 55\|1 | — | — | [0, 1] | bool | **Deceleration active** |
| `AccStopStat_B_Rq` | 34\|1 | — | — | [0, 1] | bool | Stopping status |
| `AccResumEnbl_B_Rq` | 33\|1 | — | — | [0, 1] | bool | Resume enabled |
| `CmbbDeny_B_Actl` | 37\|1 | — | — | [0, 1] | bool | **Must be 0** (panda blocks; would deny stock AEB) |

**Safety Limits** (ford.h):

| Parameter | CAN Raw | Physical | Description |
|-----------|---------|----------|-------------|
| max_accel | 5641 | +2.0 m/s² | Max acceleration command |
| min_accel | 4231 | -3.5 m/s² | Max braking command |
| inactive_accel | 5128 | -0.0008 m/s² | Inactive/coast value |
| max_gas | 700 | +2.0 m/s² | Max gas pedal command |
| min_gas | 450 | -0.5 m/s² | Min gas (engine braking) |
| inactive_gas | 0 | -5.0 m/s² | Inactive gas |

**Our Limits** (values.py):
| Parameter | Value | Description |
|-----------|-------|-------------|
| `ACCEL_MAX` | 1.5 m/s² | Reduced from stock 2.0 |
| `ACCEL_MIN` | -3.5 m/s² | Max braking authority |
| `MIN_GAS` | -0.5 m/s² | Engine braking floor |
| `INACTIVE_GAS` | -5.0 m/s² | Gas off |

**Brake Actuation Thresholds** (hysteresis):
| Threshold | Actuate (On) | Release (Off) |
|-----------|-------------|---------------|
| Brake | -0.19 m/s² | -0.08 m/s² |
| Pre-charge | -0.15 m/s² | -0.08 m/s² |

**Gas Signal Notes**:
- Gas and accel are independent channels on the CAN wire
- `AccBrkTot_A_Rq` controls PCM deceleration
- `AccPrpl_A_Rq` controls PCM acceleration/throttle
- When `AccBrkDecel_B_Rq = 1`, gas is set to INACTIVE (-5.0)
- Gas EMA filter: asymmetric tau (0.3s up, 0.8s down) for smooth coast lift-off
- Brake output scaled 0.82x to compensate for 1.32x aftermarket pad over-delivery

### ACCDATA_2 (0x187 / 391) — Collision Mitigation Braking
| Signal | Description |
|--------|-------------|
| `CmbbBrkDecel_A_Rq` | CMBB brake decel request |
| `CmbbBrkDecel_B_Rq` | CMBB brake active |
| `CmbbBrkPrchg_D_Rq` | CMBB pre-charge |

### ACCDATA_3 (0x18A / 394) — ACC/TJA Dashboard UI
- **Frequency**: 5 Hz
- Controls dashboard icons: ACC status, TJA status, FCW alerts, distance bars
- `Tja_D_Stat`: 0=Off, 1=Standby, 2=Active, 3-6=Intervention/Warning Left/Right

---

## Vehicle State Messages (Inputs)

### Speed & Motion

| Message | ID | Signal | Scale | Offset | Range | Units | Freq |
|---------|-----|--------|-------|--------|-------|-------|------|
| BrakeSysFeatures | 0x415 | `Veh_V_ActlBrk` | 0.01 | 0 | [0, 655] | kph | 50 Hz |
| EngVehicleSpThrottle2 | 0x202 | `Veh_V_ActlEng` | 0.01 | 0 | — | kph | — |
| Yaw_Data_FD1 | 0x091 | `VehYaw_W_Actl` | 0.0002 | -6.5 | [-6.5, 6.6] | rad/s | 100 Hz |
| Yaw_Data_FD1 | 0x091 | `VehRol_W_Actl` | 0.0002 | -6.5 | [-6.5, 6.6] | rad/s | 100 Hz |

**Yaw Rate Convention**: Ford uses **positive yawRate = clockwise = RIGHT turn**.
Code computes: `current_curvature = -yawRate / vEgo` → positive = LEFT turn.

### Steering

| Message | ID | Signal | Scale | Offset | Range | Units |
|---------|-----|--------|-------|--------|-------|-------|
| SteeringPinion_Data | 0x07E | `StePinComp_An_Est` | 0.1 | -3200 | deg | Steering angle |
| EPAS_INFO | 0x082 | `SteeringColumnTorque` | 0.0625 | -8.0 | Nm | Driver torque |
| EPAS_INFO | 0x082 | `EPAS_Failure` | — | — | [0, 3] | 0=OK, 1=Temp, 2-3=Perm |

### Pedals & Cruise

| Message | ID | Signal | Description |
|---------|-----|--------|-------------|
| EngBrakeData | 0x165 | `CcStat_D_Actl` | Cruise: 3=Standby, 4=Active, 5=Suspended |
| EngBrakeData | 0x165 | `BpedDrvAppl_D_Actl` | Brake pedal: 2=Pressed |
| EngBrakeData | 0x165 | `Veh_V_DsplyCcSet` | Cruise set speed |
| EngVehicleSpThrottle | 0x204 | `ApedPos_Pc_ActlArb` | Accel pedal 0-102.3% |
| DesiredTorqBrk | 0x213 | `VehStop_D_Stat` | Standstill: 1=Stopped |

### BSM (Blind Spot Monitoring)
| Message | ID | Signal | Description |
|---------|-----|--------|-------------|
| Side_Detect_L_Stat | 0x3A6 | — | Left blind spot |
| Side_Detect_R_Stat | 0x3A7 | — | Right blind spot |

### Checksum/Counter Protection
Several messages use 4-bit counter + 8-bit checksum (0xFF - rolling sum):
- `BrakeSysFeatures`: counter `VehVActlBrk_No_Cnt`, quality `VehVActlBrk_D_Qf` (must be 0x3)
- `Yaw_Data_FD1`: counter `VehRollYaw_No_Cnt`, quality `VehYawWActl_D_Qf` (must be 0x3)

---

## Safety Layer (ford.h)

### Message Addresses
```c
FORD_LateralMotionControl   = 0x3D3  // Steering (CAN)
FORD_LateralMotionControl2  = 0x3D6  // Steering (CAN FD)
FORD_ACCDATA                = 0x186  // Longitudinal
FORD_ACCDATA_3              = 0x18A  // ACC UI
FORD_Lane_Assist_Data1      = 0x3CA  // LKA companion
FORD_IPMA_Data              = 0x3D8  // LKAS status
FORD_Steering_Data_FD1      = 0x083  // Buttons
```

### Curvature Rate Limits (speed-dependent, per 0.05s frame)

| Speed (m/s) | Rate Up | Rate Down |
|-------------|---------|-----------|
| 5 | 0.0026 | 0.0026 |
| 16 | 0.0020 | 0.0024 |
| 25 | 0.00025 | 0.00035 |

### BluePilot 4-Signal Validation
| Signal | Physical Min | Physical Max |
|--------|-------------|-------------|
| Curvature | -0.012 1/m | +0.012 1/m |
| Curvature rate | -0.001024 1/m² | +0.00102375 1/m² |
| Path offset | -1.0 m | +1.0 m |
| Path angle | -0.25 rad | +0.25 rad |

### Custom Safety Modifications
- `max_angle_error = 300` (0.006 curvature — ceiling raised from stock 250)
- `reset_bypass_latch`: 60 frames (3s) bypass after steering reset for smooth ramp-up
- Rate limits relaxed to match application code Mode 0

---

## Timing & Frequencies

| Function | Frequency | Step | Description |
|----------|-----------|------|-------------|
| Base loop | 100 Hz | 1 | carcontroller update rate |
| Steering | 20 Hz | STEER_STEP=5 | LateralMotionControl |
| ACC | 50 Hz | ACC_CONTROL_STEP=2 | ACCDATA |
| LKA | 33 Hz | LKA_STEP=3 | Lane_Assist_Data1 |
| Buttons | 10 Hz | BUTTONS_STEP=5 | Steering_Data_FD1 |
| ACC UI | 5 Hz | — | ACCDATA_3 |

---

## PSCM (Steering Module) Notes

- **Does NOT accept angle commands** — only curvature-based polynomial
- Independently calculates steering angle from curvature + vehicle speed + yaw rate
- Requires FORScan configuration: PSCM block `730-02-02` bytes 7 (TJA) and 8 (LCA) = 0xFF
- LCA (our method) has **no lockout, no timeout**, works to 0 mph
- LKA (Lane_Assist_Data1) has 10s lockout — we only send empty messages

---

## Value Pipeline Summary

### Lateral (Curvature → EPAS)
```
actuators.curvature (from planner, positive=LEFT)
  → predicted curvature blend (30/70)
  → EMA smoothing
  → PI lane centering correction
  → EPAS bias compensation (+0.000245)
  → driver override handling
  → rate limiting (apply_std_steer_angle_limits)
  → apply_curv_send
  → NEGATED: -apply_curv_send → CAN wire (positive=RIGHT on wire)
  → PSCM interprets and steers
```

### Longitudinal (Accel → PCM)
```
carControl.accel (from planner)
  → 0.82x brake scaling (if negative, for aftermarket pads)
  → creep compensation (low speed)
  → pitch compensation
  → BP Long gas/accel classification (gaining/pacing/trailing)
  → brake/precharge actuation thresholds (hysteresis)
  → gas EMA (asymmetric: 0.3s up, 0.8s down)
  → AccBrkTot_A_Rq (accel) + AccPrpl_A_Rq (gas) → CAN wire
  → PCM controls throttle/brakes
```

---

## Key Files

| File | Purpose |
|------|---------|
| `opendbc/dbc/ford_lincoln_base_pt.dbc` | DBC signal definitions |
| `opendbc/car/ford/fordcan.py` | CAN message construction (THE NEGATION LIVES HERE) |
| `opendbc/car/ford/carcontroller.py` | Control logic, pipeline, all customizations |
| `opendbc/car/ford/carstate.py` | CAN signal parsing (vehicle state input) |
| `opendbc/car/ford/interface.py` | Vehicle parameters (steerRatio, delays, Ki) |
| `opendbc/car/ford/values.py` | Constants (ACCEL_MAX, rate limits, curve modes) |
| `opendbc/safety/modes/ford.h` | Panda safety layer (rate limits, value ranges) |
| `opendbc/car/lateral.py` | `apply_std_steer_angle_limits` (rate limiting) |

---

## External Resources

- [Ford Wiki — commaai/openpilot](https://github.com/commaai/openpilot/wiki/Ford) — Harness info, Q3 vs Q4
- [Ford Initial LCA Support — PR #23331](https://github.com/commaai/openpilot/pull/23331) — Original lateral implementation
- [Ford Higher Curvature Rate Limits — PR #33846](https://github.com/commaai/openpilot/pull/33846) — Rate limit tuning
- [FORScan Explorer Spreadsheet](https://www.explorerst.org/threads/updated-forscan-sheet-for-2020-24-ford-explorer-all-trims.7271/) — As-Built modifications
- [FORScan PSCM LCA Enable](https://forscan.org/forum/viewtopic.php?t=11371) — PSCM configuration
- [BluePilot FAQ](https://bluepilot.dev/faq/) — Ford-specific openpilot fork
- [CyanLabs CGEA 1.3 Database](https://cyanlabs.net/asbuilt-db/) — As-Built data reference

---

*Last updated: 2026-04-06. Based on sunnypilot branch `2021_explorer_st-mici`.*
