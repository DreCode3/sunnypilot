# Customizations vs. base sunnypilot / openpilot

How this vehicle's software differs from a stock install, stated **factually**. Two of these differences
are **controlled variables across the provided drives** (see `drives/METADATA.csv`); the rest are common
to all drives unless noted. Nothing here states which configuration performs better — that is for your
analysis to determine.

## Vehicle / platform
- 2021 Ford Explorer ST; comma 4 device; CAN ("Q3") harness.
- Base software: a sunnypilot fork of openpilot (a vehicle-specific development branch).

## ★ Variable 1 across drives — driving-model version
Two versions of the vision driving model appear in the dataset, labeled **`CD210`** and **`OPM7`** in
`METADATA.csv`. These are different neural-net checkpoints; each produces the `modelV2` path/lane-line
outputs the lateral controller follows. The active model is a device parameter (a model swap, not a code
change). `METADATA.csv` lists which drives used which model.

## ★ Variable 2 across drives — lateral-controller (PI lane-centering) parameters
A custom **PI lane-centering trim** was added to the Ford steering controller, on top of the stock
model-following control. Its purpose is to bias the commanded path curvature toward lane center.
Mechanism:
- A **lane-position-error signal** is derived each control step from the driving model's lane-line / path
  outputs (confidence-gated; EMA-smoothed with ~1.5 s time constant).
- A **proportional** term (`lc_kp` × error) reacts to the current error.
- An **integral** term (`lc_ki` × accumulated error) cancels persistent bias. The integral **accumulates
  only on straights and only when the driver is not overriding**, is **clamped** to a cap (`int_cap`), is
  **decayed** when not accumulating and on sign zero-crossings, and is **persisted across drives**
  (warm-start).
- The resulting proportional + integral curvature is **added** to the model-following command.

Parameters that **differ by configuration** (per drive, in `METADATA.csv`):

| param | meaning |
|---|---|
| `lc_kp` | proportional gain |
| `lc_ki` | integral gain |
| `int_cap` | integral clamp (a fixed value, or a speed-interpolated range) |
| `offgate_decay` | per-control-step integral decay factor when not accumulating |

The configurations in this dataset use these parameter sets (see `METADATA.csv` for which drive is which):
- **Set 1:** `lc_kp=0.0001`, `lc_ki=0.0002`, `int_cap=fixed 0.30`, `offgate_decay=0.98`.
- **Set 2:** `lc_kp=0.0005`, `lc_ki=0.0002`, `int_cap=speed-interpolated 0.30→1.00` (interpolated over
  vEgo 20→30 m/s), `offgate_decay=0.995`.

For configurations where it is enabled, the controller emits periodic **debug text** (in `logMessage`)
containing some of its internal values; you may parse it if useful, or ignore it.

## Other lateral settings (common to all drives unless noted — verify exact values in `carParams`)
- `steerActuatorDelay` ≈ 0.25 s; `steerRatio` ≈ 17.2.
- A speed-dependent curvature look-ahead time.
- A blend of the model's *predicted* vs *desired* curvature.
- An increased curvature-error tolerance and custom steering **rate limits**.

## Panda safety firmware (Ford) modifications
- Raised the curvature / angle-error ceiling; custom command rate limits; additional command-signal
  range validation. (These are safety-layer clamps bounding what the controller may command.)

## Cherry-picked upstream commits (localization)
- Camera-pipeline odometry **delay compensation** in `locationd`.
- `livePose` timestamp correction.
- `locationd` filter-time publishing.

## Hardware note
- The camera has a known mounting **yaw offset of ≈ −3.05°**, accounted for by online calibration
  (`liveCalibration.rpyCalib`).
