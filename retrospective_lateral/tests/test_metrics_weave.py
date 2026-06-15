import numpy as np
import pytest

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


def _add_command_stage(arrays, key, amplitude=0.00030):
  t = arrays["t"].astype(float)
  wave = np.sin(2 * np.pi * 0.18 * t)
  arrays["act_curvature"] = np.zeros_like(t, dtype=np.float32)
  arrays[key] = (amplitude * wave).astype(np.float32)
  return arrays


def synthetic_excluded_burst_route(flag_name: str):
  fs = 20.0
  t = np.arange(0.0, 40.0, 1.0 / fs)
  wave = np.zeros_like(t)
  burst = (t >= 12.0) & (t < 18.0)
  wave[burst] = np.sin(2 * np.pi * 0.18 * t[burst])
  path_curv = 0.012 * wave
  arrays = {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 18.0, dtype=np.float32),
    "steering_angle_deg": (25.0 * wave).astype(np.float32),
    "steering_rate_deg": np.gradient(25.0 * wave, 1.0 / fs).astype(np.float32),
    "yaw_rate": (18.0 * path_curv).astype(np.float32),
    "steering_pressed": np.zeros_like(t, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
    "blinker": np.zeros_like(t, dtype=np.float32),
    "lane_change_state": np.zeros_like(t, dtype=np.float32),
    "act_curvature": path_curv.astype(np.float32),
    "desired_curvature": np.zeros_like(t, dtype=np.float32),
    "model_y20": np.zeros_like(t, dtype=np.float32),
  }
  if flag_name not in arrays:
    arrays[flag_name] = np.zeros_like(t, dtype=np.float32)
  arrays[flag_name][burst] = 1.0
  return arrays


def synthetic_edge_excluded_weave_route():
  fs = 20.0
  t = np.arange(0.0, 30.0, 1.0 / fs)
  wave = np.sin(2 * np.pi * 0.18 * t)
  path_curv = 0.00035 * wave
  blinker = np.zeros_like(t, dtype=np.float32)
  blinker[(t < 3.0) | (t >= 28.0)] = 1.0
  return {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 18.0, dtype=np.float32),
    "steering_angle_deg": (0.7 * wave).astype(np.float32),
    "steering_rate_deg": np.gradient(0.7 * wave, 1.0 / fs).astype(np.float32),
    "yaw_rate": (18.0 * path_curv).astype(np.float32),
    "steering_pressed": np.zeros_like(t, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
    "blinker": blinker,
    "lane_change_state": np.zeros_like(t, dtype=np.float32),
    "act_curvature": path_curv.astype(np.float32),
    "desired_curvature": np.zeros_like(t, dtype=np.float32),
    "model_y20": np.zeros_like(t, dtype=np.float32),
  }


def synthetic_internal_excluded_weave_route():
  fs = 20.0
  t = np.arange(0.0, 60.0, 1.0 / fs)
  wave = np.sin(2 * np.pi * 0.18 * t)
  path_curv = 0.00035 * wave
  blinker = np.zeros_like(t, dtype=np.float32)
  blinker[(t >= 15.0) & (t < 16.0)] = 1.0
  return {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 18.0, dtype=np.float32),
    "steering_angle_deg": (0.7 * wave).astype(np.float32),
    "steering_rate_deg": np.gradient(0.7 * wave, 1.0 / fs).astype(np.float32),
    "yaw_rate": (18.0 * path_curv).astype(np.float32),
    "steering_pressed": np.zeros_like(t, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
    "blinker": blinker,
    "lane_change_state": np.zeros_like(t, dtype=np.float32),
    "act_curvature": path_curv.astype(np.float32),
    "desired_curvature": np.zeros_like(t, dtype=np.float32),
    "model_y20": np.zeros_like(t, dtype=np.float32),
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


def test_detect_weave_windows_uses_cp_final_command_when_act_curvature_is_flat():
  windows = detect_weave_windows("route_test", _add_command_stage(synthetic_weave_route("plant"), "cp_final_command"))

  assert len(windows) >= 1
  worst = windows[0]
  assert worst.stage_first_growth == "cp_final_command"
  assert worst.command_band_rms_1e4 > 1.0
  assert "CP final_command" in worst.evidence_note


def test_detect_weave_windows_uses_cx1_command_when_act_curvature_is_flat():
  windows = detect_weave_windows("route_test", _add_command_stage(synthetic_weave_route("plant"), "cx1_command_curvature"))

  assert len(windows) >= 1
  worst = windows[0]
  assert worst.stage_first_growth == "cx1_command_curvature"
  assert worst.command_band_rms_1e4 > 1.0
  assert "CX1 command_curvature" in worst.evidence_note


@pytest.mark.parametrize("flag_name", ["blinker", "lane_change_state", "steering_pressed"])
def test_detect_weave_windows_does_not_let_excluded_bursts_leak_into_detection(flag_name):
  windows = detect_weave_windows("route_test", synthetic_excluded_burst_route(flag_name))

  assert windows == []


@pytest.mark.parametrize("flag_name", ["cp_override", "cx1_override", "cx1_lane_change"])
def test_detect_weave_windows_honors_controller_telemetry_exclusion_flags(flag_name):
  windows = detect_weave_windows("route_test", synthetic_excluded_burst_route(flag_name))

  assert windows == []


def test_detect_weave_windows_reports_eligible_span_when_fixed_window_edges_are_excluded():
  windows = detect_weave_windows("route_test", synthetic_edge_excluded_weave_route())

  assert len(windows) == 1
  assert 3.9 <= windows[0].start_s <= 4.1
  assert 26.8 <= windows[0].end_s <= 27.0


def test_detect_weave_windows_does_not_span_internal_excluded_samples():
  arrays = synthetic_internal_excluded_weave_route()
  windows = detect_weave_windows("route_test", arrays)

  assert windows
  for window in windows:
    in_reported_span = (arrays["t"] >= window.start_s) & (arrays["t"] <= window.end_s)
    assert not arrays["blinker"][in_reported_span].any()
