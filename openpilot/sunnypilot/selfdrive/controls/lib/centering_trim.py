"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Slow adaptive lane-centering trim.

Every constant below is a numeric twin of
docs/superpowers/specs/2026-08-30-sd28-lateral-remediation-design.md section 3 -- change the
spec first. All of them are measured on the sd28 corpus, which was recorded on THIS build
(device commit 9eab5aab, 14 of 14 routes). No constant is inherited from any other build.

WHY THIS EXISTS
  On a Ford, LatControlAngle is pure feedforward and its ANGLE output is never transmitted:
  opendbc/car/ford/carcontroller.py reads actuators.curvature. So steerRatio, stiffnessFactor
  and angleOffsetDeg cannot reach the command, and the only place a lateral correction can
  enter is as an addition to the model's curvature.

  Correction (QA F-04): roll does NOT belong in that list. clip_curvature takes roll and
  forms roll_compensation = roll * G into its accel clamp, whose output IS the transmitted
  actuators.curvature -- so roll is structurally on the command path. It simply never bound
  on sd28 (duty 0.0000 % on 8/8 drives, worst-case margin 0.098 m/s^2). That is a
  corpus-scoped observation, not a structural guarantee.

WHAT IT CORRECTS
  On sd28 the car sits +0.172 m left of the model's own perceived lane centre (7/7 drives),
  and +0.124 m left of where the HUMAN drives on the same drives with the same instrument
  (paired, 7/7). TARGET_OFFSET is the human reference, not zero, because the instrument
  itself carries a +0.044 m bias.

WHY ACCELERATION SPACE
  The bias is ~constant in metres across speed bins (+0.146 at 8-15 m/s to +0.175 at 30-40)
  while the model's restoring gain scales as 1/v^2, so a constant lateral ACCELERATION is the
  speed-invariant parameterisation and k_trim = a_trim / v^2.

NO ROLL FEEDFORWARD. The roll-to-offset relation measured on sd28 has ZERO lag (drive median
0.00 s), which is a common cause, not a drift the loop regulates -- so the usual
feedforward derivation does not apply on this build.

DEPLOYMENT CONSTRAINT. LateralCenteringTrim is registered in common/params_keys.h, which is
C++ compiled into libparams_c. On a prebuilt release branch that library is a shipped binary
and cannot be rebuilt in-tree, so the key will not exist and get_params() will fall back to
MODE_OFF -- the feature is INERT there, by design, not broken. Enabling it requires a branch
that still carries the build system, or regenerated build artifacts.
"""
import math

MODE_OFF = 0
MODE_SHADOW = 1      # integrates and logs, applies nothing
MODE_ACTIVE = 2      # integrates and applies

TARGET_OFFSET = 0.044   # m, + = car LEFT. sd28 human median on this instrument (11 drives).
A_MAX = 0.40            # m/s^2. Sized from SAFETY, not reach. The natural parameter is
                        # C*v^2 (restoring lat-accel per metre of error, 1/s^2), which is
                        # speed-invariant since C ~ 1/v^2 -- so the displacement bound
                        # A_MAX/(C*v^2) holds at ALL speeds. Its square root is 0.234 Hz at
                        # the central C, i.e. the measured loop crossover. Reach needs
                        # A_MAX >= C*v^2*e0 (binds at the STIFFEST C: 0.476); safety needs
                        # A_MAX <= C*v^2*d_max (binds at the MOST COMPLIANT C: 0.405 at
                        # d_max=0.35). Across the measured C range 1.13e-3..3.63e-3 those
                        # windows do NOT overlap, so one must give. Under-reach is benign;
                        # over-displacement is not. 0.40 is the largest value whose worst
                        # case (0.346 m at C=1.13e-3) stays inside the 0.35 m abort trigger,
                        # and it still delivers 100 % of the target on 6 of 7 drives and
                        # 84 % on the stiffest. Re-derive after C_DC is measured UNDER
                        # AUTHORITY -- shadow CANNOT measure it: with nothing applied the
                        # car is never displaced, so no dose-response exists. The C floor
                        # itself is unestablished; MODE_ACTIVE is blocked until it is.
RATE_MAX = 0.010        # (m/s^2)/s. Full authority takes >=40 s (A_MAX/RATE_MAX); 500x below
                        # clip_curvature's 5/v^2 = 4.9e-3 1/m/s allowance, so it cannot
                        # perturb that limiter.
KI = 7.0e-3             # (m/s^2) per (m*s). ~300 s convergence, which bounds wind-up over the
                        # longest sustained curve in sd28 (41.3 s) to 20 % of authority = 36 mm.
V_APPLY_MIN = 8.0       # m/s. Below this update() returns exactly 0.0 -- see the docstring
                        # of update(). Same threshold as the integrate gate.
V_APPLY_FULL = 12.0     # m/s. Output ramps 0->1 over V_APPLY_MIN..V_APPLY_FULL (change 2).
MIN_SPEED = 1.0         # m/s. Belt-and-braces against divide-by-zero.
PROB_MIN = 0.6          # min(laneLineProbs[1], laneLineProbs[2])
OFFSET_MAX = 1.5        # m. Beyond this the midpoint is not a lane.
WIDTH_MIN = 2.6         # m. Lane-width plausibility (change 3). Measured median on sd28: 3.14 m.
WIDTH_MAX = 4.4         # m. Above this the two "ego" lines bound a fork, a merge or a gore,
                        # so their midpoint is not this lane's centre either.
LEAK_AFTER_S = 10.0     # s of CONTINUOUS gating before the trim starts decaying (change 4).
LEAK_TAU = 300.0        # s. Decay time constant once leaking.
SUPPORTED_FINGERPRINT = "FORD_EXPLORER_MK6"   # change 5; every constant here was measured on this car


class CenteringTrim:
  def __init__(self, params, car_fingerprint: str = ""):
    self.params = params
    self.car_fingerprint = car_fingerprint
    self.mode = MODE_OFF
    self.accel_trim = 0.0
    self.gated_frames = 0
    self.gated_seconds = 0.0
    self.params_read_failed = False    # latched; see get_params
    self.get_params()

  def get_params(self) -> None:
    """Read the mode, defaulting to OFF on ANY failure.

    This is deliberately a bare except. Params.get raises UnknownKeyName for a key that is
    not compiled into libparams_c, and it raises RuntimeError on an internal params error.
    On a PREBUILT release branch -- which is what this vehicle runs -- libparams_c.so is a
    tracked aarch64 binary, a tracked `prebuilt` marker makes launch_chffrplus.sh skip the
    build, and there is no SConstruct, so a newly registered key CANNOT be compiled in. An
    unguarded read therefore kills controlsd at construction on every boot, taking
    longitudinal control with it. A feature that cannot read its own switch must be inert,
    never fatal.
    """
    try:
      mode = self.params.get("LateralCenteringTrim", return_default=True)
      self.mode = int(mode) if mode is not None else MODE_OFF
    except Exception:                                        # deliberately broad -- see docstring
      self.mode = MODE_OFF
      # Log ONCE, not on every ~3 s refresh. Without this a genuine params failure is
      # indistinguishable from OFF-by-choice except via flat shadow telemetry (QA F-16).
      # The import is lazy and the whole thing is nested in its own try so that neither a
      # missing swaglog nor a logging error can turn an inert feature into a fatal one --
      # which is the entire point of the broad except above. It also keeps this module's
      # import surface at  alone, so the tests still run in an unbuilt tree.
      if not self.params_read_failed:
        self.params_read_failed = True
        try:
          from openpilot.common.swaglog import cloudlog
          cloudlog.exception("centering_trim: LateralCenteringTrim unreadable, degrading to MODE_OFF")
        except Exception:
          pass

    # Gate on the PLATFORM, not the brand. Every constant above -- TARGET_OFFSET, A_MAX, KI,
    # the stiffness C they are derived from, the lane-width band -- was measured on ONE car
    # (a 2021 Ford Explorer) on ONE corridor. A different platform has a different C, a
    # different instrument bias and a different lane geometry, so the same numbers would
    # regulate it toward a centre nobody measured. This runs AFTER the read so an unreadable
    # param still degrades to OFF rather than raising.
    if self.car_fingerprint != SUPPORTED_FINGERPRINT:
      self.mode = MODE_OFF

  def reset(self) -> None:
    self.accel_trim = 0.0
    self.gated_seconds = 0.0

  @staticmethod
  def _lane(model_v2):
    """(midpoint of the ego lane lines [+ = car LEFT], min lane-line probability, lane width).
    Returns (None, 0.0, 0.0) when the model has not produced usable lane lines. In this
    frame y is positive to the RIGHT, so laneLines[1] (left) is negative and [2] (right)
    positive, and width = right - left is positive. Measured median on sd28: 3.14 m."""
    ll = model_v2.laneLines
    probs = model_v2.laneLineProbs
    if len(ll) < 3 or len(probs) < 3:
      return None, 0.0, 0.0
    if not len(ll[1].y) or not len(ll[2].y):
      return None, 0.0, 0.0
    left, right = ll[1].y[0], ll[2].y[0]
    return (left + right) / 2.0, min(probs[1], probs[2]), right - left

  def update(self, CS, model_v2, lat_active: bool, dt: float) -> float:
    """Returns the curvature trim [1/m], + = RIGHT.

    Returns exactly 0.0 when inactive or below V_APPLY_MIN, while HOLDING the integrator:
      - controlsd sets new_desired_curvature = self.curvature when latActive is false,
        specifically so clip_curvature does not slew from a stale value on engage. Adding a
        trim there would defeat that reset.
      - k = a_trim / max(v, 1)^2 at A_MAX and 1 m/s is 0.40 1/m: 2x clip_curvature's
        MAX_CURVATURE and 20x Ford's CURVATURE_MAX. It would be clipped, but the command
        would sit pinned at the limit through every parking manoeuvre.
    """
    if self.mode == MODE_OFF:
      self.accel_trim = 0.0
      return 0.0

    offset, prob, width = self._lane(model_v2)
    integrate = (bool(lat_active)
                 and not CS.steeringPressed
                 and CS.canValid
                 and CS.vEgo > V_APPLY_MIN
                 and prob >= PROB_MIN
                 and model_v2.meta.laneChangeState.raw == 0
                 and not (CS.leftBlinker or CS.rightBlinker)
                 and offset is not None
                 and math.isfinite(offset)
                 and abs(offset) < OFFSET_MAX
                 and math.isfinite(width)
                 and WIDTH_MIN <= width <= WIDTH_MAX)

    if integrate:
      self.gated_seconds = 0.0
      a_dot = KI * (offset - TARGET_OFFSET)
      a_dot = max(-RATE_MAX, min(RATE_MAX, a_dot))
      self.accel_trim = max(-A_MAX, min(A_MAX, self.accel_trim + a_dot * dt))
    else:
      self.gated_frames += 1
      self.gated_seconds += dt
      # Decay only after SUSTAINED gating: 33-86 % of engaged frames are gated in normal
      # driving, so a per-frame leak would never converge. After LEAK_AFTER_S of continuous
      # gating the feedback signal is genuinely absent and holding a value learned on another
      # road is not justified.
      if self.gated_seconds > LEAK_AFTER_S:
        self.accel_trim -= self.accel_trim * dt / LEAK_TAU

    if self.mode != MODE_ACTIVE or not lat_active or CS.vEgo <= V_APPLY_MIN:
      return 0.0
    # Ramp rather than step. openpilot's own idiom at speed boundaries (ford
    # carcontroller's anti_overshoot blends 5->10 m/s, apply_creep_compensation likewise).
    # A 4 m/s ramp traversed at 2 m/s^2 gives 0.14 m/s^3, against the ISO limit of 5.0.
    # Note blend * a / v^2 * v^2 = blend * a, so the applied lateral acceleration is still
    # bounded by A_MAX everywhere in the ramp.
    blend = min(1.0, (CS.vEgo - V_APPLY_MIN) / (V_APPLY_FULL - V_APPLY_MIN))
    return blend * self.accel_trim / max(CS.vEgo, MIN_SPEED) ** 2
