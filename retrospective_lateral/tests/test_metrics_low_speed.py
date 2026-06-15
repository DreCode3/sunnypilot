import numpy as np

from retrospective_lateral.code.metrics import detect_low_speed_wheel_swing


def synthetic_low_speed_route():
  fs = 20.0
  t = np.arange(0.0, 20.0, 1.0 / fs)
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
