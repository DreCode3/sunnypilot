"""
Unit spec for AolSafeguardMonitor (F3 alert-only stock port, 2026-07-03).

Pure-stdlib tests (no numpy/capnp/params) so they run in the analysis venv and on-device.

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD \
      .venv311/bin/python -m pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py \
      -v --noconftest -o addopts=""

Timing arithmetic behind the assertion bounds (constants: CONF_EMA_TAU=0.5, CONF_THRESHOLD=0.30,
CONF_QUALIFIER=1.0, MANEUVER_BLIND_GRACE=2.0, DEPART_CONF_MIN=0.6, DEPART_OFFSET=0.40,
DEPART_RATE=0.15, DEPART_SUSTAIN=0.5, LINE_DIST_MIN=0.35, MANEUVER_DEPART_GRACE=3.0,
MIN_SPEED=10.0, RATE_WINDOW=11, dt=0.05):
  - EMA alpha = 0.05/(0.5+0.05) = 0.0909. From ~0.95, with inner_prob=0.02 the EMA crosses
    0.30 after ~13 frames ((0.93)*(0.909)^k < 0.28 => k >= 13), then the 1.0 s qualifier
    (20 frames) => first possible low-conf fire ~33 frames after collapse onset.
  - Departure needs the 11-sample offset window full (0.5 s), |offset| > 0.40 m, moving
    away from center at |rate| > 0.15 m/s, sustained 0.5 s (10 more frames).
  - Post-maneuver departure grace 3.0 s + 0.5 s deque refill + 0.5 s sustain => a drift
    already qualifying at maneuver end first fires ~4.0 s after the maneuver ends.
If a constant changes (Task 10 tuning), recompute these bounds BY HAND and update them.
"""
import math

from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor

DT = 0.05  # 20 Hz modelV2 cadence


def run(mon, seq):
  """seq: list of dicts of step kwargs; returns list of (low_conf, departure)."""
  out = []
  t = 0.0
  for s in seq:
    kw = dict(t=t, lat_active=True, v_ego=25.0, inner_prob=0.95, lane_offset=0.0,
              left_line_dist=1.7, right_line_dist=1.7, maneuver=False)
    kw.update(s)
    out.append(mon.update(**kw))
    t += DT
  return out


def steps(n, **kw):
  return [dict(kw) for _ in range(n)]


class TestLowConfidence:
  def test_sustained_collapse_alerts_after_qualifier(self):
    mon = AolSafeguardMonitor()
    seq = steps(40) + steps(60, inner_prob=0.02)  # 2s good, 3s blind
    res = run(mon, seq)
    fired = [i for i, r in enumerate(res) if r[0]]
    assert fired, "low-conf alert never fired on a 3 s collapse"
    # EMA (tau .5) needs ~0.6 s to fall below 0.30 from 0.95, then 1.0 s qualifier:
    # expect first fire between 1.2 s and 2.2 s after collapse onset (frames 64..84)
    assert 64 <= fired[0] <= 84, f"first fire at frame {fired[0]}, expected 64..84"

  def test_short_flicker_no_alert(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(8, inner_prob=0.02) + steps(40))  # 0.4 s dip
    assert not any(r[0] for r in res)

  def test_suppressed_first_2s_of_maneuver_then_fires(self):
    mon = AolSafeguardMonitor()
    # maneuver starts, goes blind immediately, blindness persists 4 s into maneuver
    seq = steps(40) + steps(80, inner_prob=0.02, maneuver=True)
    res = run(mon, seq)
    fired = [i for i, r in enumerate(res) if r[0]]
    assert fired, "persistent blindness in a maneuver must still alert"
    # suppressed for 2.0 s of maneuver (frames 40..79); EMA+qualifier already
    # elapsed by then, so first fire right after suppression lifts (frames 80..86)
    assert fired[0] >= 80
    assert fired[0] <= 86

  def test_inactive_or_slow_never_alerts_and_resets(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(60, inner_prob=0.02, lat_active=False))
    assert not any(r[0] for r in res)
    res = run(mon, steps(60, inner_prob=0.02, v_ego=5.0))
    assert not any(r[0] for r in res)


class TestDeparture:
  def test_drift_with_rate_alerts_before_07m(self):
    mon = AolSafeguardMonitor()
    # replicate the decisive event: offset ramps 0 -> -1.0 m at -0.24 m/s (car
    # drifting RIGHT of center; midpoint convention + = left of center)
    drift = [dict(lane_offset=-0.24 * i * DT) for i in range(84)]
    res = run(mon, steps(40) + drift)
    fired = [i - 40 for i, r in enumerate(res) if r[1]]
    assert fired, "departure alert never fired on a 0.24 m/s drift"
    t_fire = fired[0] * DT
    # |off|>0.40 at ~1.67 s, +0.5 s sustain -> ~2.17 s; offset then is ~0.52 m,
    # well before the 0.70 m the driver reacted to in the real event
    assert 1.9 <= t_fire <= 2.6, f"fired at {t_fire:.2f}s"
    assert abs(-0.24 * (fired[0] * DT)) < 0.70

  def test_static_offset_no_alert(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(100, lane_offset=-0.55))  # parked off-center, no rate
    assert not any(r[1] for r in res)

  def test_approach_to_center_no_alert(self):
    mon = AolSafeguardMonitor()
    # drift OUT to -0.8 m at 0.10 m/s (below DEPART_RATE, never fires), then recover
    # TOWARD center at 0.30 m/s: |offset| > 0.40 with a big rate, but moving toward
    # center (rate * offset < 0) -> must never fire
    out = [dict(lane_offset=-0.10 * i * DT) for i in range(160)]           # 0 -> -0.8 over 8 s
    back = [dict(lane_offset=-0.8 + 0.30 * i * DT) for i in range(40)]     # -0.8 -> -0.2 over 2 s
    res = run(mon, steps(40) + out + back + steps(40, lane_offset=-0.2))
    assert not any(r[1] for r in res)

  def test_inner_line_proximity_backstop(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(30, right_line_dist=0.25, lane_offset=-1.2))
    assert any(r[1] for r in res)

  def test_lane_change_sweep_suppressed(self):
    mon = AolSafeguardMonitor()
    # commanded lane change: offset sweeps to -1.6 m and back over 4 s, maneuver=True,
    # then settles; no departure alert anywhere (3 s post-maneuver grace covers the tail)
    sweep = [dict(lane_offset=-1.6 * math.sin(math.pi * i / 80), maneuver=True) for i in range(80)]
    settle = [dict(lane_offset=-0.3 + 0.3 * min(1.0, i / 40)) for i in range(60)]
    res = run(mon, steps(40) + sweep + settle)
    assert not any(r[1] for r in res)

  def test_post_maneuver_departure_grace_then_fires(self):
    mon = AolSafeguardMonitor()
    # 2 s clean, 1 s maneuver, then a drift that already qualifies the moment the
    # maneuver ends (|offset| > 0.40, moving away at 0.20 m/s). Grace suppresses
    # appends for 3.0 s after maneuver end; deque refill 0.5 s; sustain 0.5 s
    # => first fire ~4.0 s after maneuver end (maneuver ends at frame 60).
    pre = steps(40) + steps(20, maneuver=True)
    drift = [dict(lane_offset=-(0.45 + 0.20 * i * DT)) for i in range(100)]  # 5 s
    res = run(mon, pre + drift)
    fired = [i for i, r in enumerate(res) if r[1]]
    assert fired, "post-grace qualifying drift must fire"
    t_after_end = (fired[0] - 60) * DT
    assert t_after_end >= 3.9, f"fired {t_after_end:.2f}s after maneuver end (inside grace/refill)"
    assert t_after_end <= 4.3, f"fired too late: {t_after_end:.2f}s after maneuver end"

  def test_blind_never_fires_departure(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(60, inner_prob=0.05, lane_offset=-1.0))
    assert not any(r[1] for r in res)  # low-conf monitor owns the blind case
