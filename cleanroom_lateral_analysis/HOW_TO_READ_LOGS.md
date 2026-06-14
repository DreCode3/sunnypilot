# How to Read the Logs

## Format
Each drive is in `drives/drive_NN/` as a sequence of segment folders `seg_000/`, `seg_001/`, … Each
segment contains one file `rlog.zst`. These are **openpilot "rlog" files**: a zstd-compressed stream of
[Cap'n Proto](https://capnproto.org)-serialized log messages. Segments are ~1 minute each and are
time-ordered; concatenate a drive's segments in numeric order for a continuous timeline.

## Reading them
The straightforward way is openpilot's own `LogReader`:
```bash
git clone https://github.com/commaai/openpilot.git
cd openpilot && git submodule update --init   # 'cereal' (the message schema) is a submodule
# set up the python environment per the repo's current instructions
```
```python
from openpilot.tools.lib.logreader import LogReader
for msg in LogReader('drives/drive_02/seg_000/rlog.zst'):
    typ = msg.which()                 # message type (a capnp union tag)
    t   = msg.logMonoTime * 1e-9      # monotonic timestamp, seconds
    if typ == 'carState':
        v = msg.carState.vEgo
```
The complete, authoritative schema for **every** message type and field (names, types, units) is
**`cereal/log.capnp`** in the openpilot repo — treat it as the source of truth.

## Message types relevant to lateral behavior (non-exhaustive — consult `log.capnp` for the rest)
Messages are interleaved in time; align by `logMonoTime`. Rates below are approximate.

- **`carState`** (~100 Hz) — measured vehicle state: `vEgo` (m/s, forward speed),
  `steeringAngleDeg` (deg, measured steering angle), `steeringRateDeg` (deg/s), `steeringTorque`,
  `steeringPressed` (bool — driver applying torque to the wheel), `vEgoRaw`, …
- **`carControl`** (~100 Hz) — commanded control: `latActive` (bool — lateral control engaged this frame),
  `actuators.curvature` (1/m, commanded path curvature), `actuators.steeringAngleDeg`, …
- **`modelV2`** (~20 Hz) — vision driving-model output, in the calibrated vehicle frame:
  `position` (predicted path, with `x`/`y`/`z` arrays), `laneLines` (a list of detected lane-line
  polylines, each with `x`/`y` arrays), `laneLineProbs`, `orientation`, `velocity`, lead info, …
- **`liveLocationKalman`** and/or **`livePose`** (~20 Hz) — fused localization:
  `angularVelocityCalibrated` (rad/s; the z component is yaw rate), `positionGeodetic` (`value` =
  [lat, lon, alt]), `calibratedOrientationNED`, `velocityCalibrated`. Each sub-field has a `.valid` flag.
- **`liveCalibration`** — camera extrinsic calibration: `rpyCalib` ([roll, pitch, yaw] in rad), `calStatus`.
- **`carParams`** (in the log's `initData`, once per route) — vehicle parameters (steer ratio, wheelbase,
  tuning). Useful ground-truth for the vehicle model.
- **`logMessage`** — text/JSON debug strings from various daemons. Some may contain controller-internal
  debug values; parse at your discretion.

## Notes
- Different message types have different rates and timestamps — resample/interpolate onto a common time
  base for any joint analysis.
- `steeringAngleDeg` is the *measured* steering; `carControl.actuators.curvature` is the *commanded*
  curvature — related through the vehicle's steering geometry (see `carParams`).
- `drives/METADATA.csv` gives each drive's configuration, duration, engaged %, speed distribution, and
  GPS bounding box, so you can assess comparability yourself.
- A benign `Corrupted events detected` warning can appear from the reader on some segments; it does not
  prevent reading the valid messages.

*(No analysis approach is suggested here on purpose — choosing the signals, metrics, and methods is your task.)*
