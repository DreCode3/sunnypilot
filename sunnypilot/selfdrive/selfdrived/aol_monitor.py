"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

AOL/MADS lateral safeguard monitors (2026-07-01 incident, handoff v3 §5).

Two pure monitors stepped at modelV2 rate (20 Hz):
  low-confidence: model blind while steering -> alert instead of silent continuation
  departure:      sighted but drifting off-center with velocity -> alert ~1.5 s
                  before the takeover point observed in the incident

Sign conventions (verified in the investigation, handoff v3 §1.2):
  lane_offset (lane-line midpoint y): + = car LEFT of lane center
  a drift "away from center" means sign(d(offset)/dt) == sign(offset)
"""
from collections import deque


class AolSafeguardMonitor:
  # low-confidence monitor
  CONF_EMA_TAU = 0.5          # s
  CONF_THRESHOLD = 0.30
  CONF_QUALIFIER = 1.0        # s below threshold before alerting
  MANEUVER_BLIND_GRACE = 2.0  # s of maneuver during which blindness is expected

  # departure monitor
  DEPART_CONF_MIN = 0.6
  DEPART_OFFSET = 0.40        # m
  DEPART_RATE = 0.15          # m/s, away from center
  DEPART_SUSTAIN = 0.5        # s
  LINE_DIST_MIN = 0.35        # m, absolute backstop to the nearest inner line
  MANEUVER_DEPART_GRACE = 3.0 # s after a maneuver ends

  MIN_SPEED = 10.0            # m/s
  RATE_WINDOW = 11            # samples (0.5 s at 20 Hz)

  def __init__(self, dt: float = 0.05):
    self.dt = dt
    self._reset()

  def _reset(self):
    self.conf_ema = 1.0
    self.low_conf_since = None
    self.depart_since = None
    self.offsets = deque(maxlen=self.RATE_WINDOW)
    self.maneuver_start_t = None
    self.maneuver_end_t = -1e9

  def update(self, t: float, lat_active: bool, v_ego: float, inner_prob: float,
             lane_offset: float, left_line_dist: float, right_line_dist: float,
             maneuver: bool) -> tuple[bool, bool]:
    """Step once per modelV2 frame. Returns (low_confidence_alert, departure_alert)."""
    if not lat_active or v_ego < self.MIN_SPEED:
      self._reset()
      return False, False

    # maneuver edge tracking
    if maneuver and self.maneuver_start_t is None:
      self.maneuver_start_t = t
    elif not maneuver and self.maneuver_start_t is not None:
      self.maneuver_end_t = t
      self.maneuver_start_t = None

    # ---- low-confidence monitor ----------------------------------------
    alpha = self.dt / (self.CONF_EMA_TAU + self.dt)
    self.conf_ema = alpha * inner_prob + (1.0 - alpha) * self.conf_ema

    blind_suppressed = (self.maneuver_start_t is not None
                        and (t - self.maneuver_start_t) < self.MANEUVER_BLIND_GRACE)
    low_conf = False
    if self.conf_ema < self.CONF_THRESHOLD:
      if self.low_conf_since is None:
        self.low_conf_since = t
      low_conf = (t - self.low_conf_since) >= self.CONF_QUALIFIER and not blind_suppressed
    else:
      self.low_conf_since = None

    # ---- departure monitor ----------------------------------------------
    departure = False
    depart_suppressed = (self.maneuver_start_t is not None
                         or (t - self.maneuver_end_t) < self.MANEUVER_DEPART_GRACE)
    if self.conf_ema >= self.DEPART_CONF_MIN and not depart_suppressed:
      self.offsets.append(lane_offset)
      cond = min(left_line_dist, right_line_dist) < self.LINE_DIST_MIN
      if not cond and len(self.offsets) == self.RATE_WINDOW:
        rate = (self.offsets[-1] - self.offsets[0]) / ((self.RATE_WINDOW - 1) * self.dt)
        moving_away = rate * lane_offset > 0
        cond = abs(lane_offset) > self.DEPART_OFFSET and moving_away and abs(rate) > self.DEPART_RATE
      if cond:
        if self.depart_since is None:
          self.depart_since = t
        departure = (t - self.depart_since) >= self.DEPART_SUSTAIN
      else:
        self.depart_since = None
    else:
      self.offsets.clear()
      self.depart_since = None

    return low_conf, departure
