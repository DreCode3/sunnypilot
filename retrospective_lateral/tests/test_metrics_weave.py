import numpy as np

from retrospective_lateral.code.metrics import detect_weave_windows


def synthetic_weave_route(stage: str):
  fs = 20.0
  t = np.arange(0.0, 60.0, 1.0 / fs)
  wave = np.sin(2 * np.pi * 0.18 * t)
  des = 0.00025 * wave if stage == "desired" else np.zeros_like(t)
  cmd = 0.00030 * wave if stage in ("desired", "command") else np.zeros_like(t)
  yaw = 18.0 * (0.00035 * wave)
  return {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 18.0, dtype=np.float32),
    "steering_angle_deg": (0.7 * wave).astype(np.float32),
    "steering_rate_deg": np.gradient(0.7 * wave, 1.0 / fs).astype(np.float32),
    "yaw_rate": yaw.astype(np.float32),
    "steering_pressed": np.zeros_like(t, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
    "blinker": np.zeros_like(t, dtype=np.float32),
    "lane_change_state": np.zeros_like(t, dtype=np.float32),
    "act_curvature": cmd.astype(np.float32),
    "desired_curvature": des.astype(np.float32),
    "model_y20": np.zeros_like(t, dtype=np.float32),
    "lane_center_y20": np.zeros_like(t, dtype=np.float32),
  }


def test_detect_weave_windows_finds_path_weave():
  windows = detect_weave_windows("route_test", synthetic_weave_route("command"))
  assert len(windows) >= 1
  worst = windows[0]
  assert worst.symptom == "weave_10_70"
  assert worst.path_curvature_band_rms_1e4 > 2.0
  assert worst.steering_band_rms_deg > 0.3
  assert worst.stage_first_growth == "controller_or_command"


def test_detect_weave_windows_falls_back_to_can_yaw_when_calibrated_yaw_missing():
  arrays = synthetic_weave_route("command")
  arrays["yaw_rate_calibrated"] = np.full_like(arrays["t"], np.nan)

  windows = detect_weave_windows("route_test", arrays)

  assert len(windows) >= 1
  worst = windows[0]
  assert worst.path_curvature_band_rms_1e4 > 2.0
  assert worst.stage_first_growth == "controller_or_command"


def test_detect_weave_windows_identifies_desired_stage():
  windows = detect_weave_windows("route_test", synthetic_weave_route("desired"))
  assert windows[0].stage_first_growth == "model_or_desired"
