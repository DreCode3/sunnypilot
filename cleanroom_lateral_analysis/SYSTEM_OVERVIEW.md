# System Overview (background)

This vehicle runs **sunnypilot**, a community fork of **openpilot** (comma.ai's open-source
driver-assistance system). The following is general background; see `REFERENCES.md` for authoritative detail.

## Lateral (steering) pipeline
1. **Road camera → driving model.** A vision neural network ("the driving model") processes the forward
   camera and outputs, in the car's calibrated reference frame, a **predicted desired path**, **lane
   lines**, road edges, and lead-vehicle info — published as `modelV2` (~20 Hz). Different *versions* of
   this neural net exist and can produce different paths / lane-lines and a different steering "feel."
2. **Planning.** openpilot converts the model's path into a **desired trajectory / desired path
   curvature** for the car to follow.
3. **Lateral control.** A lateral controller computes the steering command needed to track the desired
   curvature. (openpilot has used several controller types over time — PID, INDI, LQR, "torque," and
   curvature-based.) Depending on the car, it commands a steering angle/torque or a **path curvature**.
4. **Actuation & safety.** The command goes to the car's steering system. A separate safety
   microcontroller ("panda") enforces hard limits on what may be commanded.

Longitudinal control (gas/brake) is a separate subsystem and not the focus here.

## Engagement and override
Lateral control acts only when **engaged** (`carControl.latActive == true`). The driver can override at
any time by applying torque to the wheel (`carState.steeringPressed`), which typically suspends automated
lateral control. (Consequently, "engaged" and "driver-steering" are largely mutually exclusive states.)

## Calibration
The camera's mounting orientation is estimated online (`liveCalibration.rpyCalib`); the model's outputs
are expressed in a calibrated, road-aligned vehicle frame. A mount offset or miscalibration shifts where
the model perceives the lane to be.

## This vehicle
A 2021 Ford Explorer ST on a comma 4 device. For Ford, the lateral command is expressed as a **path
curvature** (1/m) sent to the production lane-centering steering system. See `CUSTOMIZATIONS.md` for how
this install differs from stock and for the two variables that differ across the provided drives.

## Reasoning aids (general, not specific to this case)
- A feedback steering controller tracking a slightly noisy or biased reference can exhibit
  **oscillation / limit-cycle** behavior; gains, delays, saturation/clamping, and the plant (vehicle)
  dynamics all influence whether and at what frequency it oscillates.
- Steering *angle* needed to produce a given path scales strongly with **speed** (roughly with the
  inverse of speed-squared for a fixed path curvature), so steering-amplitude comparisons across drives
  at different speeds require care.
These are offered only as physical context; how (or whether) they matter here is for your analysis.
