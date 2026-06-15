from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.signal_utils import (
  contiguous_regions,
  dilate_flags,
  erode_true,
  filter_continuous,
  peak_to_peak_masked,
  rms_masked,
  spectral_peak_hz,
)


@dataclass(frozen=True)
class Episode:
  symptom: str
  route_id: str
  start_s: float
  end_s: float
  peak_s: float
  speed_mph_median: float
  steering_peak_to_peak_deg: float
  steering_rate_rms_deg_s: float
  steering_band_rms_deg: float
  command_peak_to_peak_curvature: float
  path_curvature_band_rms_1e4: float
  stage_first_growth: str
  confidence: str

  def to_row(self) -> dict[str, object]:
    return {
      key: None if isinstance(value, float) and not math.isfinite(value) else value
      for key, value in asdict(self).items()
    }


@dataclass(frozen=True)
class WeaveWindow:
  symptom: str
  route_id: str
  start_s: float
  end_s: float
  speed_mph_median: float
  path_curvature_band_rms_1e4: float
  steering_band_rms_deg: float
  command_band_rms_1e4: float
  desired_band_rms_1e4: float
  model_y20_band_rms_m: float
  steer_per_path: float
  spectral_peak_hz: float
  stage_first_growth: str
  evidence_note: str

  def to_row(self) -> dict[str, object]:
    return {
      key: None if isinstance(value, float) and not math.isfinite(value) else value
      for key, value in asdict(self).items()
    }


def _flag_radius(seconds: float) -> int:
  return int(round(seconds * C.FS_HZ))


def _base_clean_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
  n = len(arrays["t"])
  if "lat_active" not in arrays:
    raise ValueError("lat_active is required for low-speed wheel-swing eligibility")
  lat_active = erode_true(arrays["lat_active"] > 0.5, _flag_radius(C.ENGAGE_ERODE_S))
  no_override = ~dilate_flags(arrays.get("steering_pressed", np.zeros(n)) > 0.5, _flag_radius(C.OVERRIDE_BUFFER_S))
  no_blinker = ~dilate_flags(arrays.get("blinker", np.zeros(n)) > 0.5, _flag_radius(C.BLINKER_BUFFER_S))
  no_lane_change = ~dilate_flags(arrays.get("lane_change_state", np.zeros(n)) > 0.5, _flag_radius(C.LANE_CHANGE_BUFFER_S))
  return lat_active & no_override & no_blinker & no_lane_change


def _path_curvature(arrays: dict[str, np.ndarray]) -> np.ndarray:
  v = arrays.get("v_ego", np.array([], dtype=float)).astype(float)
  yaw = arrays.get("yaw_rate_calibrated", arrays.get("yaw_rate", np.full_like(v, np.nan))).astype(float)
  return np.divide(yaw, v, out=np.full_like(v, np.nan), where=(v > 1.0) & np.isfinite(yaw))


def _stage_label(path_rms: float, steer_rms: float, cmd_rms: float, des_rms: float, model_rms: float) -> tuple[str, str]:
  if np.isfinite(model_rms) and model_rms > 0.02:
    return "model_or_desired", "model path y20 contains slow-band motion"
  if np.isfinite(des_rms) and des_rms >= 0.5 * max(path_rms, 1e-9):
    return "model_or_desired", "desiredCurvature contains comparable slow-band motion"
  if np.isfinite(cmd_rms) and cmd_rms >= 0.5 * max(path_rms, 1e-9):
    return "controller_or_command", "final command contains comparable slow-band motion"
  if np.isfinite(path_rms) and path_rms > 0.5:
    return "actual_path_or_plant", "actual path contains slow-band motion not obvious in command"
  if np.isfinite(steer_rms) and steer_rms > 0.2:
    return "steering_wheel_only", "steering motion exceeds path motion"
  return "unknown", "slow-band energy below stage thresholds"


def detect_low_speed_wheel_swing(route_id: str, arrays: dict[str, np.ndarray]) -> list[Episode]:
  t = arrays["t"].astype(float)
  speed_mph = arrays["v_ego"].astype(float) * 2.23694
  clean = _base_clean_mask(arrays)
  speed_gate = (speed_mph >= C.LOW_SPEED_MPH[0]) & (speed_mph <= C.LOW_SPEED_MPH[1])
  eligible = clean & speed_gate & np.isfinite(arrays["steering_angle_deg"])
  min_len = int(round(C.MIN_LOW_SPEED_WINDOW_S * C.FS_HZ))
  steer = arrays["steering_angle_deg"].astype(float)
  steer_rate = arrays.get("steering_rate_deg", np.gradient(steer, 1.0 / C.FS_HZ)).astype(float)
  command = arrays.get("act_curvature", np.full_like(steer, np.nan)).astype(float)
  path_curv = _path_curvature(arrays)
  steer_band = filter_continuous(steer, C.FS_HZ, band=C.LOW_SPEED_INSPECT_BAND_HZ)
  path_band = filter_continuous(path_curv, C.FS_HZ, band=C.LOW_SPEED_INSPECT_BAND_HZ)

  episodes: list[Episode] = []
  for start, end in contiguous_regions(eligible, min_len=min_len):
    mask = np.zeros(len(t), dtype=bool)
    mask[start:end] = True
    steer_ptp = peak_to_peak_masked(steer, mask)
    if not np.isfinite(steer_ptp) or steer_ptp < 6.0:
      continue
    local_abs = np.abs(np.where(mask, steer_band, np.nan))
    peak_idx = int(np.nanargmax(local_abs)) if np.isfinite(local_abs).any() else start
    cmd_ptp = peak_to_peak_masked(command, mask)
    path_rms = rms_masked(path_band, mask) * 1e4
    if np.isfinite(cmd_ptp) and cmd_ptp > 0.0005:
      stage = "final_command_or_before"
      confidence = "supported"
    elif np.isfinite(path_rms) and path_rms > 0.5:
      stage = "actual_path_or_plant"
      confidence = "supported"
    else:
      stage = "steering_wheel_only"
      confidence = "steering_only"
    episodes.append(Episode(
      symptom="low_speed_wheel_swing",
      route_id=route_id,
      start_s=float(t[start]),
      end_s=float(t[end - 1]),
      peak_s=float(t[peak_idx]),
      speed_mph_median=float(np.nanmedian(speed_mph[mask])),
      steering_peak_to_peak_deg=float(steer_ptp),
      steering_rate_rms_deg_s=rms_masked(steer_rate, mask),
      steering_band_rms_deg=rms_masked(steer_band, mask),
      command_peak_to_peak_curvature=float(cmd_ptp) if np.isfinite(cmd_ptp) else math.nan,
      path_curvature_band_rms_1e4=float(path_rms) if np.isfinite(path_rms) else math.nan,
      stage_first_growth=stage,
      confidence=confidence,
    ))
  return sorted(episodes, key=lambda e: e.steering_peak_to_peak_deg, reverse=True)


def detect_weave_windows(route_id: str, arrays: dict[str, np.ndarray]) -> list[WeaveWindow]:
  t = arrays["t"].astype(float)
  speed_mph = arrays["v_ego"].astype(float) * 2.23694
  clean = _base_clean_mask(arrays)
  speed_gate = (speed_mph >= C.WEAVE_SPEED_MPH[0]) & (speed_mph <= C.WEAVE_SPEED_MPH[1])
  path_curv = _path_curvature(arrays)
  road_lp = filter_continuous(path_curv, C.FS_HZ, lowpass_hz=C.ROAD_LP_HZ)
  gentle = np.abs(road_lp) <= C.ROAD_CURV_ABS_MAX_1PM
  eligible = clean & speed_gate & gentle & np.isfinite(path_curv)
  window_len = int(round(C.WEAVE_WINDOW_S * C.FS_HZ))
  min_eligible = int(round(C.MIN_WEAVE_ELIGIBLE_S * C.FS_HZ))

  path_band = filter_continuous(path_curv, C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  steer_band = filter_continuous(arrays["steering_angle_deg"].astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  cmd_band = filter_continuous(arrays.get("act_curvature", np.full_like(path_curv, np.nan)).astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  des_band = filter_continuous(arrays.get("desired_curvature", np.full_like(path_curv, np.nan)).astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)
  model_band = filter_continuous(arrays.get("model_y20", np.full_like(path_curv, np.nan)).astype(float), C.FS_HZ, band=C.DEFAULT_WEAVE_BAND_HZ)

  windows: list[WeaveWindow] = []
  for start in range(0, max(0, len(t) - window_len + 1), window_len):
    end = start + window_len
    mask = np.zeros(len(t), dtype=bool)
    mask[start:end] = eligible[start:end]
    if mask.sum() < min_eligible:
      continue
    path_rms = rms_masked(path_band, mask) * 1e4
    steer_rms = rms_masked(steer_band, mask)
    if not np.isfinite(path_rms) or path_rms < 0.2:
      continue
    cmd_rms = rms_masked(cmd_band, mask) * 1e4
    des_rms = rms_masked(des_band, mask) * 1e4
    model_rms = rms_masked(model_band, mask)
    stage, note = _stage_label(path_rms, steer_rms, cmd_rms, des_rms, model_rms)
    peak_hz = spectral_peak_hz(np.where(mask, path_band, np.nan), C.FS_HZ, C.DEFAULT_WEAVE_BAND_HZ)
    windows.append(WeaveWindow(
      symptom="weave_10_70",
      route_id=route_id,
      start_s=float(t[start]),
      end_s=float(t[end - 1]),
      speed_mph_median=float(np.nanmedian(speed_mph[mask])),
      path_curvature_band_rms_1e4=float(path_rms),
      steering_band_rms_deg=float(steer_rms) if np.isfinite(steer_rms) else math.nan,
      command_band_rms_1e4=float(cmd_rms) if np.isfinite(cmd_rms) else math.nan,
      desired_band_rms_1e4=float(des_rms) if np.isfinite(des_rms) else math.nan,
      model_y20_band_rms_m=float(model_rms) if np.isfinite(model_rms) else math.nan,
      steer_per_path=float(steer_rms / path_rms) if np.isfinite(steer_rms) and path_rms > 0 else math.nan,
      spectral_peak_hz=float(peak_hz) if np.isfinite(peak_hz) else math.nan,
      stage_first_growth=stage,
      evidence_note=note,
    ))
  return sorted(windows, key=lambda w: w.path_curvature_band_rms_1e4, reverse=True)
