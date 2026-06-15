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
  signal_times: dict[str, dict[str, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
  values: dict[str, dict[str, list[float]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(list)))
  log_messages: list[tuple[float, str]] = field(default_factory=list)
  init: dict[str, Any] = field(default_factory=dict)
  car_params: dict[str, Any] = field(default_factory=dict)

  def add(self, family: str, t: float, row: dict[str, float]) -> None:
    t = float(t)
    self.times[family].append(t)
    for key, value in row.items():
      self.signal_times[family][key].append(t)
      self.values[family][key].append(float(value) if value is not None else np.nan)


def _interp_numeric(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray, fs_hz: float) -> np.ndarray:
  valid = np.isfinite(t_src) & np.isfinite(y_src)
  t_src = t_src[valid]
  y_src = y_src[valid]
  if len(t_src) < 2:
    return np.full_like(t_grid, np.nan, dtype=np.float32)
  order = np.argsort(t_src, kind="stable")
  t = t_src[order]
  y = y_src[order]
  _, last_idx = np.unique(t[::-1], return_index=True)
  keep = np.sort(len(t) - 1 - last_idx)
  t = t[keep]
  y = y[keep]
  if len(t) < 2:
    return np.full_like(t_grid, np.nan, dtype=np.float32)
  out = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
  right = np.searchsorted(t, t_grid, side="left")
  exact_right = (right < len(t)) & (t[right.clip(max=len(t) - 1)] == t_grid)
  left = right - 1
  left_idx = np.clip(left, 0, len(t) - 1)
  right_idx = np.clip(right, 0, len(t) - 1)
  bracket_gap = t[right_idx] - t[left_idx]
  bracketed = (left >= 0) & (right < len(t))
  out[bracketed & (bracket_gap > C.MAX_INTERP_GAP_S) & ~exact_right] = np.nan
  return out.astype(np.float32)


def _nearest_flag(t_src: np.ndarray, y_src: np.ndarray, t_grid: np.ndarray,
                  max_hold_s: float = C.MAX_INTERP_GAP_S) -> np.ndarray:
  if len(t_src) < 1:
    return np.zeros_like(t_grid, dtype=np.float32)
  order = np.argsort(t_src)
  t = t_src[order]
  y = y_src[order]
  right = np.searchsorted(t, t_grid, side="left")
  left = right - 1
  right_idx = np.clip(right, 0, len(t) - 1)
  left_idx = np.clip(left, 0, len(t) - 1)
  right_dist = np.abs(t_grid - t[right_idx])
  left_dist = np.abs(t_grid - t[left_idx])
  right_dist[right >= len(t)] = np.inf
  left_dist[left < 0] = np.inf
  idx = np.where(left_dist <= right_dist, left_idx, right_idx)
  nearest_dist = np.minimum(left_dist, right_dist)
  out = (y[idx] > 0.5).astype(np.float32)
  out[nearest_dist > max_hold_s] = 0.0
  return out


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
    for key in keys:
      t = np.asarray(raw.signal_times.get(family, {}).get(key, []), dtype=float)
      y = np.asarray(raw.values.get(family, {}).get(key, []), dtype=float)
      out[key] = _interp_numeric(t, y, t_grid, fs_hz) if len(y) else np.full_like(t_grid, np.nan, dtype=np.float32)
  for family, keys in flag_mapping.items():
    for key in keys:
      t = np.asarray(raw.signal_times.get(family, {}).get(key, []), dtype=float)
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


def _as_log_text(log_message: Any) -> str:
  text = str(log_message)
  if text.startswith("{"):
    try:
      decoded = json.loads(text)
      return str(decoded.get("msg", text))
    except Exception:
      return text
  return text


def _extract_message(raw: RawChannels, msg: Any) -> None:
  which = msg.which()
  t = msg.logMonoTime * 1e-9
  if which == "initData" and not raw.init:
    init = msg.initData
    raw.init = {"commit": str(init.gitCommit), "branch": str(init.gitBranch), "dirty": bool(init.dirty)}
  elif which == "carParams" and not raw.car_params:
    cp = msg.carParams
    raw.car_params = {
      "carFingerprint": str(cp.carFingerprint),
      "wheelbase": float(cp.wheelbase),
      "steerRatio": float(cp.steerRatio),
      "steerActuatorDelay": float(cp.steerActuatorDelay),
    }
  elif which == "carState":
    cs = msg.carState
    raw.add("carState", t, {
      "v_ego": cs.vEgo,
      "v_ego_raw": cs.vEgoRaw,
      "a_ego": cs.aEgo,
      "steering_angle_deg": cs.steeringAngleDeg,
      "steering_rate_deg": cs.steeringRateDeg,
      "steering_torque": cs.steeringTorque,
      "yaw_rate": cs.yawRate,
      "steering_pressed": 1.0 if cs.steeringPressed else 0.0,
      "left_blinker": 1.0 if cs.leftBlinker else 0.0,
      "right_blinker": 1.0 if cs.rightBlinker else 0.0,
      "can_valid": 1.0 if cs.canValid else 0.0,
    })
  elif which == "carControl":
    cc = msg.carControl
    raw.add("carControl", t, {
      "lat_active": 1.0 if cc.latActive else 0.0,
      "long_active": 1.0 if cc.longActive else 0.0,
      "act_curvature": float(cc.actuators.curvature),
      "current_curvature": float(cc.currentCurvature),
    })
  elif which == "controlsState":
    st = msg.controlsState
    raw.add("controlsState", t, {
      "desired_curvature": float(st.desiredCurvature),
      "controls_curvature": float(st.curvature),
    })
  elif which == "modelV2":
    m = msg.modelV2
    lane_change = 0 if str(m.meta.laneChangeState) == "off" else 1
    raw.add("modelV2", t, {
      "model_y0": _interp_model_xy(m.position.x, m.position.y, 0.0),
      "model_y20": _interp_model_xy(m.position.x, m.position.y, 20.0),
      "lane_center_y0": _lane_center(m, 0.0)[0],
      "lane_center_y20": _lane_center(m, 20.0)[0],
      "lane_width_y0": _lane_center(m, 0.0)[1],
      "lane_width_y20": _lane_center(m, 20.0)[1],
      "lane_prob_left": float(m.laneLineProbs[1]) if len(m.laneLineProbs) > 2 else np.nan,
      "lane_prob_right": float(m.laneLineProbs[2]) if len(m.laneLineProbs) > 2 else np.nan,
      "orientation_rate_z0": float(m.orientationRate.z[0]) if len(m.orientationRate.z) else np.nan,
      "lane_change_state": float(lane_change),
    })
  elif which == "liveLocationKalman":
    loc = msg.liveLocationKalman
    lat = lon = yaw_cal = roll = pitch = np.nan
    if loc.positionGeodetic.valid and len(loc.positionGeodetic.value) >= 2:
      lat = float(loc.positionGeodetic.value[0])
      lon = float(loc.positionGeodetic.value[1])
    if loc.angularVelocityCalibrated.valid and len(loc.angularVelocityCalibrated.value) >= 3:
      yaw_cal = float(loc.angularVelocityCalibrated.value[2])
    if loc.orientationNED.valid and len(loc.orientationNED.value) >= 2:
      roll = float(loc.orientationNED.value[0])
      pitch = float(loc.orientationNED.value[1])
    raw.add("liveLocationKalman", t, {"lat": lat, "lon": lon, "yaw_rate_calibrated": yaw_cal, "roll": roll, "pitch": pitch})
  elif which == "liveCalibration":
    rpy = list(msg.liveCalibration.rpyCalib)
    raw.add("liveCalibration", t, {
      "cal_roll": rpy[0] if len(rpy) > 0 else np.nan,
      "cal_pitch": rpy[1] if len(rpy) > 1 else np.nan,
      "cal_yaw": rpy[2] if len(rpy) > 2 else np.nan,
    })
  elif which == "logMessage":
    text = _as_log_text(msg.logMessage)
    raw.log_messages.append((t, text))


def extract_route_raw(route: RouteRef) -> tuple[RawChannels, list[str]]:
  raw = RawChannels()
  notes: list[str] = []
  for segment in route.segments:
    try:
      for msg in LogReader(str(segment.rlog_path)):
        try:
          _extract_message(raw, msg)
        except Exception as exc:
          notes.append(f"{segment.rlog_path}: message: {type(exc).__name__}: {exc}")
    except Exception as exc:
      notes.append(f"{segment.rlog_path}: {type(exc).__name__}: {exc}")
  return raw, notes


def _interp_model_xy(xs: Any, ys: Any, xq: float) -> float:
  x = np.asarray(list(xs), dtype=float)
  y = np.asarray(list(ys), dtype=float)
  ok = np.isfinite(x) & np.isfinite(y)
  if ok.sum() < 2 or xq < np.nanmin(x[ok]) or xq > np.nanmax(x[ok]):
    return np.nan
  return float(np.interp(xq, x[ok], y[ok]))


def _lane_center(model: Any, xq: float) -> tuple[float, float]:
  if len(model.laneLines) <= 2:
    return np.nan, np.nan
  left = _interp_model_xy(model.laneLines[1].x, model.laneLines[1].y, xq)
  right = _interp_model_xy(model.laneLines[2].x, model.laneLines[2].y, xq)
  if not np.isfinite(left) or not np.isfinite(right):
    return np.nan, np.nan
  return float(0.5 * (left + right)), float(abs(right - left))


def _cached_sample_count(npz_path: Path) -> int:
  try:
    with np.load(npz_path) as data:
      return int(len(data["t"])) if "t" in data else 0
  except Exception:
    return 0


def _normalize_cache_metadata(meta: dict[str, Any], out_npz: Path, out_json: Path) -> dict[str, Any]:
  if "success" in meta and "sample_count" in meta and "npz_path" in meta:
    return meta
  sample_count = _cached_sample_count(out_npz)
  meta.update({
    "success": sample_count > 0,
    "sample_count": sample_count,
    "npz_path": str(out_npz) if sample_count > 0 else None,
  })
  out_json.write_text(json.dumps(meta, indent=2, sort_keys=True))
  return meta


def _write_route_cache(route: RouteRef, cache_root: Path, force: bool = False) -> dict[str, Any]:
  cache_root.mkdir(parents=True, exist_ok=True)
  out_npz = cache_root / f"{route.route_id}.npz"
  out_json = cache_root / f"{route.route_id}.json"
  if out_npz.exists() and out_json.exists() and not force:
    return _normalize_cache_metadata(json.loads(out_json.read_text()), out_npz, out_json)
  raw, notes = extract_route_raw(route)
  arrays = resample_channels(raw, fs_hz=C.FS_HZ)
  lc_rows = [parse_lc_line(text, t) for t, text in raw.log_messages]
  lc_rows = [row for row in lc_rows if row is not None]
  evidence = recover_pi_config(
    offsets=[row.offset_m for row in lc_rows],
    p_terms=[row.p_term for row in lc_rows],
    integrals=[row.integral for row in lc_rows],
    i_terms=[row.i_term for row in lc_rows],
  )
  if arrays:
    np.savez_compressed(out_npz, **arrays)
  elif out_npz.exists():
    out_npz.unlink()
  sample_count = int(len(arrays["t"])) if arrays else 0
  npz_path = str(out_npz) if arrays else None
  meta = route_cache_metadata(route.route_id, C.CACHE_SCHEMA_VERSION, len(route.segments), evidence.confidence, notes)
  meta.update({
    "success": bool(arrays),
    "sample_count": sample_count,
    "npz_path": npz_path,
    "pi_set": evidence.pi_set,
    "lc_kp": evidence.lc_kp,
    "lc_ki": evidence.lc_ki,
    "init": raw.init,
    "car_params": raw.car_params,
  })
  out_json.write_text(json.dumps(meta, indent=2, sort_keys=True))
  return meta


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-root", type=Path, default=C.DEFAULT_LOG_ROOT)
  parser.add_argument("--cache-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
  parser.add_argument("--route", action="append", default=[])
  parser.add_argument("--list-routes", action="store_true")
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args(argv)
  routes = discover_routes(args.log_root)
  if args.route:
    wanted = set(args.route)
    found = {r.route_id for r in routes}
    missing = wanted - found
    if missing:
      parser.error(f"missing route(s): {', '.join(sorted(missing))}")
    routes = [r for r in routes if r.route_id in wanted]
  if args.list_routes:
    for route in routes:
      print(json.dumps({"route_id": route.route_id, "segments": len(route.segments), "layout": route.layout}, sort_keys=True))
    return 0
  rows = []
  for route in routes:
    rows.append(_write_route_cache(route, args.cache_root, force=args.force))
    print(json.dumps(rows[-1], sort_keys=True))
  args.cache_root.mkdir(parents=True, exist_ok=True)
  (args.cache_root / "manifest.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
