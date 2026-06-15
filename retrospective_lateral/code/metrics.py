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
