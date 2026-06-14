#!/usr/bin/env python3
"""
Extract raw lateral-analysis signals from the clean-room rlogs.

This intentionally uses only the Cap'n Proto schema and zstd stream format,
not the full openpilot LogReader, because the latter imports unrelated device
runtime modules. The fields extracted here are documented in the package and
in cereal/log.capnp / cereal/car.capnp.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import capnp
import numpy as np
import zstandard as zstd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from cereal import log as capnp_log


DRIVES_DIR = ROOT / "drives"
OUT_DIR = ROOT / "analysis" / "cache"


LANE_CHANGE_STATE = {
  "off": 0,
  "preLaneChange": 1,
  "laneChangeStarting": 2,
  "laneChangeFinishing": 3,
}


def read_rlog(path: Path) -> Iterable[capnp._DynamicStructReader]:
  with path.open("rb") as fh:
    with zstd.ZstdDecompressor().stream_reader(fh) as reader:
      data = reader.read()

  events = capnp_log.Event.read_multiple_bytes(data)
  try:
    yield from events
  except capnp.KjException as exc:
    print(f"warning: truncated/corrupt tail in {path}: {exc}")


def interp_at_x(xs: Iterable[float], ys: Iterable[float], xq: float) -> float:
  x = np.asarray(list(xs), dtype=np.float64)
  y = np.asarray(list(ys), dtype=np.float64)
  ok = np.isfinite(x) & np.isfinite(y)
  if ok.sum() < 2:
    return np.nan
  x = x[ok]
  y = y[ok]
  order = np.argsort(x)
  x = x[order]
  y = y[order]
  if xq < x[0] or xq > x[-1]:
    return np.nan
  return float(np.interp(xq, x, y))


def path_curvature(xs: Iterable[float], ys: Iterable[float], min_x: float = 3.0, max_x: float = 35.0) -> float:
  x = np.asarray(list(xs), dtype=np.float64)
  y = np.asarray(list(ys), dtype=np.float64)
  ok = np.isfinite(x) & np.isfinite(y) & (x >= min_x) & (x <= max_x)
  if ok.sum() < 6:
    return np.nan
  # Small-angle local path approximation: y ~= a*x^2 + b*x + c, curvature ~= 2a.
  a, _b, _c = np.polyfit(x[ok], y[ok], deg=2)
  return float(2.0 * a)


def lane_features(model) -> dict[str, float]:
  out = {
    "lane_center_y0": np.nan,
    "lane_center_y20": np.nan,
    "lane_width_y0": np.nan,
    "lane_width_y20": np.nan,
    "lane_prob_left": np.nan,
    "lane_prob_right": np.nan,
  }
  if len(model.laneLines) < 3 or len(model.laneLineProbs) < 3:
    return out

  left = model.laneLines[1]
  right = model.laneLines[2]
  left_y0 = interp_at_x(left.x, left.y, 0.0)
  right_y0 = interp_at_x(right.x, right.y, 0.0)
  left_y20 = interp_at_x(left.x, left.y, 20.0)
  right_y20 = interp_at_x(right.x, right.y, 20.0)

  out["lane_prob_left"] = float(model.laneLineProbs[1])
  out["lane_prob_right"] = float(model.laneLineProbs[2])
  if np.isfinite(left_y0) and np.isfinite(right_y0):
    out["lane_center_y0"] = 0.5 * (left_y0 + right_y0)
    out["lane_width_y0"] = abs(right_y0 - left_y0)
  if np.isfinite(left_y20) and np.isfinite(right_y20):
    out["lane_center_y20"] = 0.5 * (left_y20 + right_y20)
    out["lane_width_y20"] = abs(right_y20 - left_y20)
  return out


def as_float_array(values: list[float]) -> np.ndarray:
  return np.asarray(values, dtype=np.float32)


def as_bool_array(values: list[bool]) -> np.ndarray:
  return np.asarray(values, dtype=np.bool_)


def extract_drive(drive_dir: Path, out_dir: Path) -> dict[str, object]:
  drive_id = drive_dir.name
  arrays: dict[str, list] = defaultdict(list)
  car_params: dict[str, float | str] = {}
  counts: dict[str, int] = defaultdict(int)
  log_contexts: list[dict[str, object]] = []

  for rlog in sorted(drive_dir.glob("seg_*/rlog.zst")):
    for msg in read_rlog(rlog):
      typ = msg.which()
      counts[typ] += 1
      t = msg.logMonoTime * 1e-9

      if typ == "carState":
        cs = msg.carState
        arrays["cs_t"].append(t)
        arrays["cs_v_ego"].append(cs.vEgo)
        arrays["cs_v_ego_raw"].append(cs.vEgoRaw)
        arrays["cs_a_ego"].append(cs.aEgo)
        arrays["cs_steer_angle_deg"].append(cs.steeringAngleDeg)
        arrays["cs_steer_rate_deg"].append(cs.steeringRateDeg)
        arrays["cs_steer_torque"].append(cs.steeringTorque)
        arrays["cs_steering_pressed"].append(cs.steeringPressed)
        arrays["cs_yaw_rate"].append(cs.yawRate)
        arrays["cs_left_blinker"].append(cs.leftBlinker)
        arrays["cs_right_blinker"].append(cs.rightBlinker)
        arrays["cs_can_valid"].append(cs.canValid)

      elif typ == "carControl":
        cc = msg.carControl
        arrays["cc_t"].append(t)
        arrays["cc_lat_active"].append(cc.latActive)
        arrays["cc_long_active"].append(cc.longActive)
        arrays["cc_enabled"].append(cc.enabled)
        arrays["cc_act_curvature"].append(cc.actuators.curvature)
        arrays["cc_current_curvature"].append(cc.currentCurvature)
        arrays["cc_act_steer_angle_deg"].append(cc.actuators.steeringAngleDeg)
        arrays["cc_left_blinker"].append(cc.leftBlinker)
        arrays["cc_right_blinker"].append(cc.rightBlinker)

      elif typ == "controlsState":
        st = msg.controlsState
        arrays["ctl_t"].append(t)
        arrays["ctl_curvature"].append(st.curvature)
        arrays["ctl_desired_curvature"].append(st.desiredCurvature)
        state = st.lateralControlState
        arrays["ctl_lat_state"].append(str(state.which()))
        active = saturated = angle_error = desired_angle = np.nan
        which = str(state.which())
        try:
          if which == "angleState":
            s = state.angleState
            active = float(s.active)
            saturated = float(s.saturated)
            angle_error = s.angleError
            desired_angle = s.steeringAngleDesiredDeg
          elif which == "pidState":
            s = state.pidState
            active = float(s.active)
            saturated = float(s.saturated)
            angle_error = s.angleError
            desired_angle = s.steeringAngleDesiredDeg
        except Exception:
          pass
        arrays["ctl_lat_active_state"].append(active)
        arrays["ctl_saturated"].append(saturated)
        arrays["ctl_angle_error"].append(angle_error)
        arrays["ctl_steer_angle_desired_deg"].append(desired_angle)

      elif typ == "modelV2":
        m = msg.modelV2
        arrays["model_t"].append(t)
        arrays["model_path_y10"].append(interp_at_x(m.position.x, m.position.y, 10.0))
        arrays["model_path_y20"].append(interp_at_x(m.position.x, m.position.y, 20.0))
        arrays["model_path_y30"].append(interp_at_x(m.position.x, m.position.y, 30.0))
        arrays["model_path_curvature"].append(path_curvature(m.position.x, m.position.y))
        feats = lane_features(m)
        for key, value in feats.items():
          arrays[f"model_{key}"].append(value)
        arrays["model_lane_change_state"].append(LANE_CHANGE_STATE.get(str(m.meta.laneChangeState), -1))
        arrays["model_lane_change_direction"].append(str(m.meta.laneChangeDirection))
        arrays["model_confidence"].append(str(m.confidence))

      elif typ == "liveLocationKalman":
        llk = msg.liveLocationKalman
        arrays["loc_t"].append(t)
        if llk.positionGeodetic.valid and len(llk.positionGeodetic.value) >= 2:
          arrays["loc_lat"].append(llk.positionGeodetic.value[0])
          arrays["loc_lon"].append(llk.positionGeodetic.value[1])
          arrays["loc_alt"].append(llk.positionGeodetic.value[2] if len(llk.positionGeodetic.value) > 2 else np.nan)
        else:
          arrays["loc_lat"].append(np.nan)
          arrays["loc_lon"].append(np.nan)
          arrays["loc_alt"].append(np.nan)
        if llk.angularVelocityCalibrated.valid and len(llk.angularVelocityCalibrated.value) >= 3:
          arrays["loc_yaw_rate_calibrated"].append(llk.angularVelocityCalibrated.value[2])
        else:
          arrays["loc_yaw_rate_calibrated"].append(np.nan)
        if llk.velocityCalibrated.valid and len(llk.velocityCalibrated.value) >= 2:
          arrays["loc_vx_calibrated"].append(llk.velocityCalibrated.value[0])
          arrays["loc_vy_calibrated"].append(llk.velocityCalibrated.value[1])
        else:
          arrays["loc_vx_calibrated"].append(np.nan)
          arrays["loc_vy_calibrated"].append(np.nan)
        arrays["loc_status"].append(str(llk.status))

      elif typ == "livePose":
        lp = msg.livePose
        arrays["pose_t"].append(t)
        arrays["pose_yaw_rate"].append(lp.angularVelocityDevice.z if lp.angularVelocityDevice.valid else np.nan)
        arrays["pose_vx"].append(lp.velocityDevice.x if lp.velocityDevice.valid else np.nan)
        arrays["pose_vy"].append(lp.velocityDevice.y if lp.velocityDevice.valid else np.nan)

      elif typ == "liveCalibration":
        lc = msg.liveCalibration
        arrays["cal_t"].append(t)
        rpy = list(lc.rpyCalib)
        arrays["cal_roll"].append(rpy[0] if len(rpy) > 0 else np.nan)
        arrays["cal_pitch"].append(rpy[1] if len(rpy) > 1 else np.nan)
        arrays["cal_yaw"].append(rpy[2] if len(rpy) > 2 else np.nan)
        arrays["cal_status"].append(str(lc.calStatus))

      elif typ == "carParams" and not car_params:
        cp = msg.carParams
        car_params = {
          "brand": str(cp.brand),
          "carFingerprint": str(cp.carFingerprint),
          "mass": float(cp.mass),
          "wheelbase": float(cp.wheelbase),
          "steerRatio": float(cp.steerRatio),
          "steerActuatorDelay": float(cp.steerActuatorDelay),
          "steerControlType": str(cp.steerControlType),
        }

      elif typ == "logMessage" and len(log_contexts) < 20:
        text = str(msg.logMessage)
        if "\"ctx\"" in text and ("\"commit\"" in text or "\"branch\"" in text):
          try:
            log_contexts.append(json.loads(text).get("ctx", {}))
          except Exception:
            pass

  out: dict[str, np.ndarray] = {}
  for key, values in arrays.items():
    if not values:
      continue
    if key.endswith("_pressed") or key.endswith("_active") or key.endswith("_enabled") or key.endswith("_blinker") or key.endswith("_valid"):
      out[key] = as_bool_array(values)
    elif key in {"ctl_lat_state", "model_lane_change_direction", "model_confidence", "loc_status", "cal_status"}:
      out[key] = np.asarray(values)
    else:
      out[key] = as_float_array(values)

  out_path = out_dir / f"{drive_id}_signals.npz"
  np.savez_compressed(out_path, **out)

  summary = {
    "drive_id": drive_id,
    "out_file": str(out_path.relative_to(ROOT)),
    "message_counts": dict(sorted(counts.items())),
    "car_params": car_params,
    "log_context_examples": log_contexts[:5],
  }
  return summary


def load_metadata(path: Path) -> list[dict[str, str]]:
  with path.open(newline="") as fh:
    return list(csv.DictReader(fh))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--drive", action="append", help="Drive id to extract, e.g. drive_01. Defaults to all.")
  parser.add_argument("--force", action="store_true", help="Overwrite existing extracted npz files.")
  args = parser.parse_args()

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  metadata = load_metadata(DRIVES_DIR / "METADATA.csv")
  drive_ids = args.drive or [row["drive_id"] for row in metadata]
  summaries = []

  for drive_id in drive_ids:
    drive_dir = DRIVES_DIR / drive_id
    out_path = OUT_DIR / f"{drive_id}_signals.npz"
    if out_path.exists() and not args.force:
      print(f"{drive_id}: exists, skipping")
      continue
    print(f"{drive_id}: extracting")
    summaries.append(extract_drive(drive_dir, OUT_DIR))

  if summaries:
    summary_path = OUT_DIR / "extract_summary.json"
    existing = []
    if summary_path.exists():
      try:
        existing = json.loads(summary_path.read_text())
      except Exception:
        existing = []
    by_drive = {row["drive_id"]: row for row in existing}
    for row in summaries:
      by_drive[row["drive_id"]] = row
    summary_path.write_text(json.dumps([by_drive[k] for k in sorted(by_drive)], indent=2) + "\n")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
  main()
