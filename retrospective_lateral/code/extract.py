from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "opendbc_repo")

from openpilot.tools.lib.logreader import LogReader

from retrospective_lateral.code import config as C
from retrospective_lateral.code.routes import RouteRef, discover_routes
from retrospective_lateral.code.signal_utils import fill_guarded
from retrospective_lateral.code.telemetry import parse_cp_line, parse_cx1_line, parse_lc_line, recover_pi_config


@dataclass
class RawChannels:
  times: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
  values: dict[str, dict[str, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
  log_messages: list[tuple[float, str]] = field(default_factory=list)
  init: dict[str, Any] = field(default_factory=dict)
  car_params: dict[str, Any] = field(default_factory=dict)

  def add(self, family: str, t: float, row: dict[str, float]) -> None:
    self.times[family].append(float(t))
    for key, value in row.items():
      self.values[family][key].append(float(value) if value is not None else np.nan)


def _interp_numeric(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray, fs_hz: float) -> np.ndarray:
  if len(t_src) < 2:
    return np.full_like(t_grid, np.nan, dtype=np.float32)
  order = np.argsort(t_src)
  t = t_src[order]
  y = y_src[order]
  out = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
  sample_gap = np.full_like(t_grid, np.nan, dtype=float)
  nearest = np.clip(np.searchsorted(t, t_grid), 0, len(t) - 1)
  prev = np.clip(nearest - 1, 0, len(t) - 1)
  sample_gap = np.minimum(np.abs(t_grid - t[nearest]), np.abs(t_grid - t[prev]))
  out[sample_gap > C.MAX_INTERP_GAP_S] = np.nan
  return out.astype(np.float32)


def _nearest_flag(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
  if len(t_src) < 1:
    return np.zeros_like(t_grid, dtype=np.float32)
  order = np.argsort(t_src)
  t = t_src[order]
  y = y_src[order]
  idx = np.clip(np.searchsorted(t, t_grid), 0, len(t) - 1)
  return (y[idx] > 0.5).astype(np.float32)


def resample_channels(raw: RawChannels, fs_hz: float = C.FS_HZ) -> dict[str, np.ndarray]:
  all_times: list[float] = []
  for ts in raw.times.values():
    all_times.extend(ts)
  if len(all_times) < 2:
    return {}
  t0 = float(np.nanmin(all_times))
  t1 = float(np.nanmax(all_times))
  t_grid = np.arange(t0, t1 + 0.5 / fs_hz, 1.0 / fs_hz, dtype=float)
  out: dict[str, np.ndarray] = {"t": (t_grid - t0).astype(np.float32), "mono_time": t_grid.astype(np.float64)}

  mapping = {
    "carState": ["v_ego", "v_ego_raw", "a_ego", "steering_angle_deg", "steering_rate_deg", "steering_torque", "yaw_rate"],
    "carControl": ["act_curvature", "current_curvature"],
    "controlsState": ["desired_curvature", "controls_curvature"],
    "modelV2": ["model_y0", "model_y20", "lane_center_y0", "lane_center_y20", "lane_width_y0", "lane_width_y20", "lane_prob_left", "lane_prob_right", "orientation_rate_z0"],
    "liveLocationKalman": ["lat", "lon", "yaw_rate_calibrated", "roll", "pitch"],
    "liveCalibration": ["cal_roll", "cal_pitch", "cal_yaw"],
  }
  flag_mapping = {
    "carState": ["steering_pressed", "left_blinker", "right_blinker", "can_valid"],
    "carControl": ["lat_active", "long_active"],
    "modelV2": ["lane_change_state"],
  }
  for family, keys in mapping.items():
    t = np.asarray(raw.times.get(family, []), dtype=float)
    for key in keys:
      y = np.asarray(raw.values.get(family, {}).get(key, []), dtype=float)
      out[key] = _interp_numeric(t, y, t_grid, fs_hz) if len(y) else np.full_like(t_grid, np.nan, dtype=np.float32)
  for family, keys in flag_mapping.items():
    t = np.asarray(raw.times.get(family, []), dtype=float)
    for key in keys:
      y = np.asarray(raw.values.get(family, {}).get(key, []), dtype=float)
      out[key] = _nearest_flag(t, y, t_grid) if len(y) else np.zeros_like(t_grid, dtype=np.float32)
  out["blinker"] = ((out.get("left_blinker", 0) > 0.5) | (out.get("right_blinker", 0) > 0.5)).astype(np.float32)
  return out


def route_cache_metadata(route_id: str, schema_version: str, segments: int,
                         config_confidence: str, notes: list[str]) -> dict[str, Any]:
  return {
    "route_id": route_id,
    "schema_version": schema_version,
    "segments": int(segments),
    "config_confidence": config_confidence,
    "notes": list(notes),
  }
