import json
import math

import numpy as np
import pytest

from retrospective_lateral.code.metrics import Episode, detect_low_speed_wheel_swing


def synthetic_low_speed_route(duration_s=20.0):
  fs = 20.0
  t = np.arange(0.0, duration_s, 1.0 / fs)
  steer = 6.0 * np.sin(2 * np.pi * 0.25 * t)
  return {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 3.0, dtype=np.float32),
    "steering_angle_deg": steer.astype(np.float32),
    "steering_rate_deg": np.gradient(steer, 1.0 / fs).astype(np.float32),
    "steering_pressed": np.zeros_like(t, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
    "blinker": np.zeros_like(t, dtype=np.float32),
    "lane_change_state": np.zeros_like(t, dtype=np.float32),
    "act_curvature": (0.001 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32),
  }


def test_detect_low_speed_wheel_swing_finds_large_angle_episode():
  episodes = detect_low_speed_wheel_swing("route_test", synthetic_low_speed_route())
  assert len(episodes) >= 1
  worst = episodes[0]
  assert worst.symptom == "low_speed_wheel_swing"
  assert worst.route_id == "route_test"
  assert worst.speed_mph_median < 10.0
  assert worst.steering_peak_to_peak_deg >= 11.0
  assert worst.command_peak_to_peak_curvature > 0.0015


def test_episode_to_row_normalizes_missing_numeric_values_for_strict_json():
  episode = Episode(
    symptom="low_speed_wheel_swing",
    route_id="route_test",
    start_s=0.0,
    end_s=1.0,
    peak_s=0.5,
    speed_mph_median=5.0,
    steering_peak_to_peak_deg=12.0,
    steering_rate_rms_deg_s=math.nan,
    steering_band_rms_deg=4.0,
    command_peak_to_peak_curvature=math.nan,
    path_curvature_band_rms_1e4=math.nan,
    stage_first_growth="steering_wheel_only",
    confidence="steering_only",
  )

  row = episode.to_row()

  assert row["steering_rate_rms_deg_s"] is None
  assert row["command_peak_to_peak_curvature"] is None
  assert row["path_curvature_band_rms_1e4"] is None
  json.dumps(row, allow_nan=False)


def test_detect_low_speed_wheel_swing_requires_lat_active():
  arrays = synthetic_low_speed_route()
  arrays.pop("lat_active")

  with pytest.raises(ValueError, match="lat_active"):
    detect_low_speed_wheel_swing("route_test", arrays)


def test_detect_low_speed_wheel_swing_documents_boundary_erosion_conservatism():
  episodes = detect_low_speed_wheel_swing("route_test", synthetic_low_speed_route(duration_s=9.0))

  assert episodes == []


def test_detect_low_speed_wheel_swing_marks_steering_only_confidence_without_corroboration():
  arrays = synthetic_low_speed_route()
  arrays.pop("act_curvature")

  episodes = detect_low_speed_wheel_swing("route_test", arrays)

  assert len(episodes) >= 1
  assert episodes[0].stage_first_growth == "steering_wheel_only"
  assert episodes[0].confidence == "steering_only"
  assert episodes[0].confidence != "supported"
