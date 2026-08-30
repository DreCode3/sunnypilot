"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Unit tests for CenteringTrim. Constants are numeric twins of
docs/superpowers/specs/2026-08-30-sd28-lateral-remediation-design.md -- change the spec first.
"""
import math

from types import SimpleNamespace

from openpilot.sunnypilot.selfdrive.controls.lib.centering_trim import (
    CenteringTrim, MODE_OFF, MODE_SHADOW, MODE_ACTIVE, TARGET_OFFSET, A_MAX, RATE_MAX,
    V_APPLY_MIN, V_APPLY_FULL, KI, OFFSET_MAX, PROB_MIN, MIN_SPEED, WIDTH_MIN, WIDTH_MAX,
    LEAK_AFTER_S, LEAK_TAU, SUPPORTED_FINGERPRINT)

DT = 0.01


class FakeParams:
  def __init__(self, mode=MODE_OFF):
    self.store = {"LateralCenteringTrim": mode}

  def get(self, key, return_default=False):
    return self.store.get(key)


def make_cs(v_ego=32.0, pressed=False, can_valid=True, left=False, right=False):
  return SimpleNamespace(vEgo=v_ego, steeringPressed=pressed, canValid=can_valid,
                         leftBlinker=left, rightBlinker=right)


def make_model(offset=0.20, prob=0.9, lcs=0, width=3.14):
  h = width / 2.0
  ll = [SimpleNamespace(y=[0.0]), SimpleNamespace(y=[offset - h]),
        SimpleNamespace(y=[offset + h]), SimpleNamespace(y=[0.0])]
  return SimpleNamespace(laneLines=ll, laneLineProbs=[0.1, prob, prob, 0.1],
                         meta=SimpleNamespace(laneChangeState=SimpleNamespace(raw=lcs)))


def run(trim, n, cs=None, model=None, lat_active=True):
  cs = cs or make_cs()
  model = model or make_model()
  out = 0.0
  for _ in range(n):
    out = trim.update(cs, model, lat_active, DT)
  return out


def test_lane_reports_the_midpoint_probability_and_width():
  """_lane replaced _offset: the width is the third return value and gates integration."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  offset, prob, width = t._lane(make_model(offset=0.37, prob=0.8, width=3.14))
  assert abs(offset - 0.37) < 1e-9
  assert prob == 0.8
  assert abs(width - 3.14) < 1e-9


def test_lane_returns_zero_width_when_there_are_no_lane_lines():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  empty = SimpleNamespace(laneLines=[], laneLineProbs=[],
                          meta=SimpleNamespace(laneChangeState=SimpleNamespace(raw=0)))
  assert t._lane(empty) == (None, 0.0, 0.0)


def test_positive_offset_error_gives_positive_right_trim():
  """offset + = car LEFT; the corrective curvature is + = RIGHT."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 500, model=make_model(offset=TARGET_OFFSET + 0.2))
  assert t.accel_trim > 0.0


def test_negative_offset_error_gives_negative_trim():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 500, model=make_model(offset=TARGET_OFFSET - 0.2))
  assert t.accel_trim < 0.0


def test_mode_off_returns_zero_and_holds_no_state():
  t = CenteringTrim(FakeParams(MODE_OFF), SUPPORTED_FINGERPRINT)
  assert run(t, 500) == 0.0
  assert t.accel_trim == 0.0


class RaisingParams:
  """Params.get raises UnknownKeyName when the key is not compiled into libparams_c."""

  def __init__(self, exc):
    self.exc = exc

  def get(self, key, return_default=False):
    raise self.exc


def test_unreadable_param_is_inert_not_fatal():
  """A key absent from the prebuilt libparams_c must degrade to OFF, not kill controlsd."""
  for exc in (RuntimeError("params_last_error"), Exception("UnknownKeyName")):
    t = CenteringTrim(RaisingParams(exc), SUPPORTED_FINGERPRINT)
    assert t.mode == MODE_OFF
    assert run(t, 200, model=make_model(offset=TARGET_OFFSET + 0.3)) == 0.0
    assert t.accel_trim == 0.0


def test_get_params_survives_a_raise_after_construction():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 300, model=make_model(offset=TARGET_OFFSET + 0.3))
  t.params = RaisingParams(RuntimeError("boom"))
  t.get_params()
  assert t.mode == MODE_OFF
  assert t.update(make_cs(), make_model(), True, DT) == 0.0


def test_shadow_integrates_but_applies_nothing():
  t = CenteringTrim(FakeParams(MODE_SHADOW), SUPPORTED_FINGERPRINT)
  out = run(t, 500, model=make_model(offset=TARGET_OFFSET + 0.2))
  assert out == 0.0
  assert t.accel_trim > 0.0


def test_inactive_returns_zero_but_holds_the_integrator():
  """controlsd sets new_desired_curvature = self.curvature when inactive, specifically to
  reset on engage. Adding a trim there would defeat that reset."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 500, model=make_model(offset=TARGET_OFFSET + 0.2))
  held = t.accel_trim
  assert t.update(make_cs(), make_model(), False, DT) == 0.0
  assert t.accel_trim == held


def test_below_apply_speed_returns_zero_but_holds():
  """k = a/max(v,1)^2 at A_MAX and 1 m/s would be 0.40 1/m -- 2x MAX_CURVATURE and 20x
  Ford's limit. The output gate at V_APPLY_MIN removes that entirely."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 500, model=make_model(offset=TARGET_OFFSET + 0.2))
  held = t.accel_trim
  assert t.update(make_cs(v_ego=V_APPLY_MIN - 0.1), make_model(), True, DT) == 0.0
  assert t.accel_trim == held


def test_each_gate_freezes_integration_but_keeps_applying():
  base = {"offset": TARGET_OFFSET + 0.3}
  cases = [
      ("steeringPressed", make_cs(pressed=True), make_model(**base)),
      ("canValid", make_cs(can_valid=False), make_model(**base)),
      ("blinker", make_cs(left=True), make_model(**base)),
      ("low speed", make_cs(v_ego=V_APPLY_MIN - 0.1), make_model(**base)),
      ("low prob", make_cs(), make_model(prob=0.4, **base)),
      ("lane change", make_cs(), make_model(lcs=2, **base)),
      ("implausible offset", make_cs(), make_model(offset=2.0)),
  ]
  for name, cs, model in cases:
    t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
    run(t, 300, model=make_model(**base))
    held = t.accel_trim
    assert held > 0.0, name
    for _ in range(300):
      out = t.update(cs, model, True, DT)
    assert abs(t.accel_trim - held) < 1e-12, f"{name}: integrator moved while gated"
    if cs.vEgo > V_APPLY_MIN:
      assert out != 0.0, f"{name}: stopped applying the held trim"


def test_missing_or_short_lane_lines_freeze_without_nan():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 300, model=make_model(offset=TARGET_OFFSET + 0.3))
  held = t.accel_trim
  empty = SimpleNamespace(laneLines=[], laneLineProbs=[],
                          meta=SimpleNamespace(laneChangeState=SimpleNamespace(raw=0)))
  out = t.update(make_cs(), empty, True, DT)
  assert t.accel_trim == held
  assert out == out                                   # not NaN


def test_reset_zeroes():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 500, model=make_model(offset=TARGET_OFFSET + 0.2))
  t.reset()
  assert t.accel_trim == 0.0


def test_authority_never_exceeds_a_max():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 200000, model=make_model(offset=TARGET_OFFSET + 1.4))
  # saturating, but INSIDE OFFSET_MAX -- past it the plausibility gate rejects every frame and the test becomes vacuous
  assert abs(t.accel_trim) <= A_MAX + 1e-12


def test_saturated_authority_is_bounded_by_the_measured_stiffness():
  """A_MAX is sized so a SATURATED trim cannot displace the car past the abort trigger, at
  the MOST COMPLIANT end of the measured stiffness range -- that is the binding case, since
  displacement = (A_MAX/v^2)/C grows as C falls. C_MIN = 1.13e-3 1/m per m is the low end of
  the gate-swept |S| estimate on sd28. This is the only test that can catch a change to
  A_MAX itself, because it compares against measured physics, not the constant."""
  C_MIN, v, ABORT_M = 1.13e-3, 32.0, 0.35
  displacement = (A_MAX / v ** 2) / C_MIN
  assert displacement < ABORT_M, f"a saturated trim would displace {displacement:.3f} m"


def test_rate_never_exceeds_rate_max():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  prev = t.accel_trim
  model = make_model(offset=TARGET_OFFSET + 1.4)
  # saturating, but INSIDE OFFSET_MAX -- past it the plausibility gate rejects every frame and the test becomes vacuous
  for _ in range(5000):
    t.update(make_cs(), model, True, DT)
    assert abs(t.accel_trim - prev) <= RATE_MAX * DT + 1e-12
    prev = t.accel_trim


def test_full_authority_takes_at_least_forty_seconds():
  """A_MAX / RATE_MAX = 0.40 / 0.010 = 40 s is the floor, down from 60 s when A_MAX was 0.60.
  ABSOLUTE, not derived from the constants under test: 40 s is the number the ramp rate is
  justified against. The real curve-windup guard is test_curve_windup_is_bounded, which is
  driven at the curve's MEASURED 0.27 m error rather than at the saturating one."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  model = make_model(offset=TARGET_OFFSET + 1.4)
  # saturating, but INSIDE OFFSET_MAX -- past it the plausibility gate rejects every frame and the test becomes vacuous
  n = 0
  while abs(t.accel_trim) < A_MAX - 1e-9 and n < 100000:
    t.update(make_cs(), model, True, DT)
    n += 1
  assert n * DT >= 39.9      # 40 s by design; tolerance for float accumulation


def test_k_trim_scales_as_one_over_v_squared():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 2000, model=make_model(offset=TARGET_OFFSET + 0.2))
  model = make_model(offset=TARGET_OFFSET + 0.2)
  k32 = t.update(make_cs(v_ego=32.0), model, True, 0.0)
  k15 = t.update(make_cs(v_ego=15.0), model, True, 0.0)
  assert abs(k15 / k32 - (32.0 / 15.0) ** 2) < 1e-6


def test_curve_windup_is_bounded():
  """The longest sustained curve in sd28 is 41.3 s. At the curve's own offset error (0.27 m
  beyond target) the trim must not wind past ~13 % of authority."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  model = make_model(offset=TARGET_OFFSET + 0.27)
  for _ in range(int(41.3 / DT)):
    t.update(make_cs(), model, True, DT)
  # ABSOLUTE, not a fraction of A_MAX: a threshold that scales with the constant under test
  # cannot detect a change in it. 0.09 m/s^2 is 36 mm of induced displacement at the
  # measured stiffness, which is the number the spec actually justifies.
  assert t.accel_trim < 0.09


def test_convergence_is_about_three_hundred_seconds():
  """At the measured 0.128 m error the trim must reach the 0.276 m/s^2 the sd28 stiffness
  says is needed, in roughly 300 s -- slow enough that a curve cannot wind it."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  model = make_model(offset=TARGET_OFFSET + 0.128)
  for _ in range(int(300.0 / DT)):
    t.update(make_cs(), model, True, DT)
  assert 0.22 < t.accel_trim < 0.33


def test_deterministic():
  a = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  b = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  m = make_model(offset=TARGET_OFFSET + 0.1)
  for _ in range(1000):
    assert a.update(make_cs(), m, True, DT) == b.update(make_cs(), m, True, DT)


def test_constants_are_the_spec_values():
  """Numeric twins of spec section 3. Changing a number here changes how the car drives;
  it must go through the spec first."""
  assert (TARGET_OFFSET, A_MAX, RATE_MAX, V_APPLY_MIN) == (0.044, 0.40, 0.010, 8.0)
  assert KI == 7.0e-3
  assert OFFSET_MAX == 1.5
  assert PROB_MIN == 0.6
  assert MIN_SPEED == 1.0
  assert V_APPLY_FULL == 12.0
  assert (WIDTH_MIN, WIDTH_MAX) == (2.6, 4.4)
  assert (LEAK_AFTER_S, LEAK_TAU) == (10.0, 300.0)
  assert SUPPORTED_FINGERPRINT == "FORD_EXPLORER_MK6"


# ---------------------------------------------------------------- change 2: the speed blend

def _saturate(t):
  """Wind the trim to exactly +A_MAX. 1.444 m error is inside OFFSET_MAX, so every frame
  integrates. Loops to saturation rather than for a fixed frame count, so the blend tests
  below stay independent of A_MAX and RATE_MAX."""
  m = make_model(offset=TARGET_OFFSET + 1.4)
  for _ in range(200000):
    t.update(make_cs(), m, True, DT)
    if t.accel_trim == A_MAX:
      break
  assert t.accel_trim == A_MAX
  return t


def test_output_is_exactly_zero_at_the_bottom_of_the_speed_ramp():
  t = _saturate(CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT))
  assert t.update(make_cs(v_ego=V_APPLY_MIN), make_model(), True, 0.0) == 0.0


def test_output_blends_in_monotonically_across_the_speed_ramp():
  """The old code STEPPED from 0.0 to full at V_APPLY_MIN, where k = a/v^2 is 16x its
  highway value, and clip_curvature then slewed it at the ISO 5.0 m/s^3 ceiling -- 12 gate
  transitions were measured in 12 s of stop-and-go dithering across 8 m/s."""
  t = _saturate(CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT))
  m = make_model()
  prev = 0.0
  for i in range(1, 41):                       # 8.1 .. 12.0 m/s
    v = V_APPLY_MIN + 0.1 * i
    out = t.update(make_cs(v_ego=v), m, True, 0.0)
    assert out > prev, f"output is not strictly increasing at {v:.1f} m/s"
    prev = out


def test_the_blend_never_lets_the_applied_lat_accel_exceed_a_max():
  """blend * a / v^2 * v^2 = blend * a <= A_MAX. The ramp cannot smuggle authority in at
  low speed, which is the whole reason a blend is safe here."""
  t = _saturate(CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT))
  m = make_model()
  for i in range(41):                          # 8.0 .. 12.0 m/s, inclusive
    v = V_APPLY_MIN + 0.1 * i
    out = t.update(make_cs(v_ego=v), m, True, 0.0)
    assert out * v ** 2 <= A_MAX + 1e-12, f"lat accel {out * v ** 2:.4f} m/s^2 at {v:.1f} m/s"


def test_output_is_full_authority_at_and_above_v_apply_full():
  t = _saturate(CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT))
  m = make_model()
  for v in (V_APPLY_FULL, 15.0, 20.0, 32.0):
    out = t.update(make_cs(v_ego=v), m, True, 0.0)
    assert abs(out - A_MAX / v ** 2) < 1e-15, f"blend is not 1.0 at {v} m/s"


# ------------------------------------------------------- change 3: lane-width plausibility

def test_the_measured_lane_width_is_accepted():
  """3.14 m is the sd28 median. It must not be gated, or the trim never converges."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 300, model=make_model(offset=TARGET_OFFSET + 0.3, width=3.14))
  assert t.accel_trim > 0.0


def test_implausible_lane_width_freezes_integration():
  """A left line collapsed onto the lane centre implies a 1.8 m lane; the old code accepted
  it and bought ~100-175 mm of bias per minute. 5.0 m is the other end (fork/merge/gore)."""
  for width in (1.8, 5.0):
    t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
    run(t, 300, model=make_model(offset=TARGET_OFFSET + 0.3))
    held = t.accel_trim
    assert held > 0.0
    bad = make_model(offset=TARGET_OFFSET + 0.3, width=width)
    out = 0.0
    for _ in range(300):                       # 3 s -- below LEAK_AFTER_S, so no decay either
      out = t.update(make_cs(), bad, True, DT)
    assert abs(t.accel_trim - held) < 1e-12, f"integrated at an implausible width {width} m"
    assert out != 0.0, f"stopped applying the held trim at width {width} m"


def test_the_width_band_is_bounded_on_both_sides():
  """Just inside each edge integrates, just outside does not. Offset by 0.05 m rather than
  testing the edge exactly, because make_model builds the width from two float half-widths
  and the reconstructed value is not bit-exact."""
  for width, should_integrate in ((WIDTH_MIN - 0.05, False), (WIDTH_MIN + 0.05, True),
                                  (WIDTH_MAX - 0.05, True), (WIDTH_MAX + 0.05, False)):
    t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
    run(t, 300, model=make_model(offset=TARGET_OFFSET + 0.3, width=width))
    assert (t.accel_trim > 0.0) is should_integrate, f"width {width} m"


# ----------------------------------------------------------------- change 4: integrator leak

def test_no_leak_while_gating_is_shorter_than_the_threshold():
  """33-86 % of engaged frames are gated in normal driving. A leak that started immediately
  would stop the trim ever converging, so nothing may decay below LEAK_AFTER_S."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 3000, model=make_model(offset=TARGET_OFFSET + 0.2))
  held = t.accel_trim
  assert held > 0.0
  # 9.5 s is ABSOLUTE, not LEAK_AFTER_S - 0.5: a duration derived from the constant under
  # test cannot detect a change to it (at LEAK_AFTER_S = 0 the loop count would go negative
  # and the test would pass vacuously).
  for _ in range(950):
    t.update(make_cs(pressed=True), make_model(), True, DT)
  assert t.accel_trim == held


def test_the_trim_leaks_toward_zero_after_sustained_gating():
  """10 minutes at laneLineProbs = 0 used to apply the full held trim open-loop. An
  integrator whose feedback signal is gone must decay."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 3000, model=make_model(offset=TARGET_OFFSET + 0.2))
  held = t.accel_trim
  for _ in range(int(60.0 / DT)):                       # 60 s gated => 50 s of leaking
    t.update(make_cs(pressed=True), make_model(), True, DT)
  assert 0.0 < t.accel_trim < held
  # 60 s of gating = 10 s held + 50 s decaying at tau = 300 s. Both numbers are ABSOLUTE
  # rather than read from LEAK_AFTER_S / LEAK_TAU, so this also pins WHEN the leak starts:
  # leaking from the first frame instead would give exp(-60/300) = 0.819, 2.8 % away.
  expected = held * math.exp(-50.0 / 300.0)
  assert abs(t.accel_trim - expected) < 0.01 * held, f"{t.accel_trim} vs {expected}"


def test_a_burst_of_gating_shorter_than_the_threshold_leaves_the_trim_untouched():
  """The counter is CONTINUOUS gating, not cumulative: one clean frame resets it. 200 s of
  gating delivered in 5 s bursts must not move the trim at all."""
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  run(t, 3000, model=make_model(offset=TARGET_OFFSET + 0.2))
  held = t.accel_trim
  at_target = make_model(offset=TARGET_OFFSET)           # integrates, but the error is zero
  for _ in range(40):
    for _ in range(500):                                 # 5 s gated
      t.update(make_cs(pressed=True), make_model(), True, DT)
    t.update(make_cs(), at_target, True, DT)             # one clean frame resets the counter
  assert abs(t.accel_trim - held) < 1e-9


def test_the_leak_is_toward_zero_from_both_signs():
  for sign in (1.0, -1.0):
    t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
    run(t, 3000, model=make_model(offset=TARGET_OFFSET + sign * 0.2))
    held = t.accel_trim
    assert held * sign > 0.0
    for _ in range(int(60.0 / DT)):
      t.update(make_cs(pressed=True), make_model(), True, DT)
    assert abs(t.accel_trim) < abs(held), f"no decay for sign {sign}"
    assert t.accel_trim * sign > 0.0, f"the leak crossed zero for sign {sign}"


def test_reset_clears_the_gating_clock():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  for _ in range(int(30.0 / DT)):
    t.update(make_cs(pressed=True), make_model(), True, DT)
  t.reset()
  assert t.gated_seconds == 0.0


# --------------------------------------------------------------- change 5: platform gate

def test_an_unsupported_platform_is_off_even_with_the_param_set_active():
  """Every constant here was measured on ONE car on ONE corridor: a different platform has
  a different stiffness, bias and lane geometry, so gate on the fingerprint, not the brand.
  Another Ford is as wrong as another marque -- FORD_F_150_MK14 must be OFF."""
  for fingerprint in ("", "FORD_F_150_MK14", "FORD_ESCAPE_MK4", "TOYOTA_RAV4", "MOCK"):
    t = CenteringTrim(FakeParams(MODE_ACTIVE), fingerprint)
    assert t.mode == MODE_OFF, fingerprint
    assert run(t, 500, model=make_model(offset=TARGET_OFFSET + 0.3)) == 0.0, fingerprint
    assert t.accel_trim == 0.0, fingerprint


def test_an_unsupported_platform_stays_off_across_a_param_refresh():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), "FORD_F_150_MK14")
  t.get_params()
  assert t.mode == MODE_OFF


def test_the_supported_platform_is_not_gated():
  t = CenteringTrim(FakeParams(MODE_ACTIVE), SUPPORTED_FINGERPRINT)
  assert t.mode == MODE_ACTIVE
  assert run(t, 500, model=make_model(offset=TARGET_OFFSET + 0.3)) != 0.0
