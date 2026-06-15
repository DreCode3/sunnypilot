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


CURVATURE_STAGE_CANDIDATES = (
  ("desired_curvature", "model_or_desired", "desiredCurvature"),
  ("cp_desired_curvature", "cp_desired_curvature", "CP desired_curvature"),
  ("cx1_desired_curvature", "cx1_desired_curvature", "CX1 desired_curvature"),
  ("cp_predicted_curvature", "cp_predicted_curvature", "CP predicted_curvature"),
  ("cx1_predicted_curvature", "cx1_predicted_curvature", "CX1 predicted_curvature"),
  ("cp_ema_curvature", "cp_ema_curvature", "CP ema_curvature"),
  ("cx1_ema_curvature", "cx1_ema_curvature", "CX1 ema_curvature"),
  ("cp_pre_rate_limit", "cp_pre_rate_limit", "CP pre_rate_limit"),
  ("cx1_pre_rate_limit", "cx1_pre_rate_limit", "CX1 pre_rate_limit"),
  ("cp_rate_limited", "cp_rate_limited", "CP rate_limited"),
  ("cx1_rate_limited", "cx1_rate_limited", "CX1 rate_limited"),
  ("cp_final_command", "cp_final_command", "CP final_command"),
  ("cx1_command_curvature", "cx1_command_curvature", "CX1 command_curvature"),
  ("act_curvature", "controller_or_command", "final command"),
)

COMMAND_STAGE_KEYS = ("cp_final_command", "cx1_command_curvature", "act_curvature")
DESIRED_STAGE_KEYS = ("desired_curvature", "cp_desired_curvature", "cx1_desired_curvature")


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


SYMPTOM_CATALOG_FIELDS = (
  "route_id",
  "symptom",
  "status",
  "error_type",
  "error_message",
  "start_s",
  "end_s",
  "peak_s",
  "speed_mph_median",
  "steering_peak_to_peak_deg",
  "steering_rate_rms_deg_s",
  "steering_band_rms_deg",
  "command_peak_to_peak_curvature",
  "path_curvature_band_rms_1e4",
  "stage_first_growth",
  "confidence",
  "command_band_rms_1e4",
  "desired_band_rms_1e4",
  "model_y20_band_rms_m",
  "steer_per_path",
  "spectral_peak_hz",
  "evidence_note",
)


def _flag_radius(seconds: float) -> int:
  return int(round(seconds * C.FS_HZ))


def _base_clean_mask(arrays: dict[str, np.ndarray]) -> np.ndarray:
  n = len(arrays["t"])
  if "lat_active" not in arrays:
    raise ValueError("lat_active is required for low-speed wheel-swing eligibility")
  lat_active = erode_true(arrays["lat_active"] > 0.5, _flag_radius(C.ENGAGE_ERODE_S))
  override = (
    (arrays.get("steering_pressed", np.zeros(n)) > 0.5)
    | (arrays.get("cp_override", np.zeros(n)) > 0.5)
    | (arrays.get("cx1_override", np.zeros(n)) > 0.5)
  )
  lane_change = (
    (arrays.get("lane_change_state", np.zeros(n)) > 0.5)
    | (arrays.get("cx1_lane_change", np.zeros(n)) > 0.5)
  )
  no_override = ~dilate_flags(override, _flag_radius(C.OVERRIDE_BUFFER_S))
  no_blinker = ~dilate_flags(arrays.get("blinker", np.zeros(n)) > 0.5, _flag_radius(C.BLINKER_BUFFER_S))
  no_lane_change = ~dilate_flags(lane_change, _flag_radius(C.LANE_CHANGE_BUFFER_S))
  return lat_active & no_override & no_blinker & no_lane_change


def _path_curvature(arrays: dict[str, np.ndarray]) -> np.ndarray:
  v = arrays.get("v_ego", np.array([], dtype=float)).astype(float)
  raw_yaw = arrays.get("yaw_rate", np.full_like(v, np.nan)).astype(float)
  calibrated_yaw = arrays.get("yaw_rate_calibrated", np.full_like(v, np.nan)).astype(float)
  yaw = np.where(np.isfinite(calibrated_yaw), calibrated_yaw, raw_yaw)
  return np.divide(yaw, v, out=np.full_like(v, np.nan), where=(v > 1.0) & np.isfinite(yaw))


def _filter_metric_signal(x: np.ndarray, eligible: np.ndarray, *, band: tuple[float, float] | None = None,
                          lowpass_hz: float | None = None) -> np.ndarray:
  src = np.asarray(x, dtype=float).copy()
  src[~np.asarray(eligible, dtype=bool)] = np.nan
  return filter_continuous(src, C.FS_HZ, band=band, lowpass_hz=lowpass_hz)


def _stage_bands(arrays: dict[str, np.ndarray], eligible: np.ndarray) -> dict[str, np.ndarray]:
  bands: dict[str, np.ndarray] = {}
  for key, _, _ in CURVATURE_STAGE_CANDIDATES:
    if key in arrays:
      bands[key] = _filter_metric_signal(arrays[key].astype(float), eligible, band=C.DEFAULT_WEAVE_BAND_HZ)
  if "model_y20" in arrays:
    bands["model_y20"] = _filter_metric_signal(arrays["model_y20"].astype(float), eligible, band=C.DEFAULT_WEAVE_BAND_HZ)
  return bands


def _stage_rms_values(stage_bands: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
  values: dict[str, float] = {}
  for key, band in stage_bands.items():
    scale = 1.0 if key == "model_y20" else 1e4
    values[key] = rms_masked(band, mask) * scale
  return values


def _best_rms(values: dict[str, float], keys: tuple[str, ...]) -> float:
  finite = [values[key] for key in keys if key in values and np.isfinite(values[key])]
  if not finite:
    return math.nan
  return float(max(finite))


def _stage_label(path_rms: float, steer_rms: float, stage_rms: dict[str, float]) -> tuple[str, str]:
  model_rms = stage_rms.get("model_y20", math.nan)
  if np.isfinite(model_rms) and model_rms > 0.02:
    return "model_or_desired", "model path y20 contains slow-band motion"
  for key, label, note_name in CURVATURE_STAGE_CANDIDATES:
    rms = stage_rms.get(key, math.nan)
    if np.isfinite(rms) and rms >= 0.5 * max(path_rms, 1e-9):
      return label, f"{note_name} contains comparable slow-band motion"
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
  steer_band = _filter_metric_signal(steer, eligible, band=C.LOW_SPEED_INSPECT_BAND_HZ)
  path_band = _filter_metric_signal(path_curv, eligible, band=C.LOW_SPEED_INSPECT_BAND_HZ)

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
      confidence = "command_correlated"
    elif np.isfinite(path_rms) and path_rms > 0.5:
      stage = "actual_path_or_plant"
      confidence = "path_correlated"
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
  prelim_eligible = clean & speed_gate & np.isfinite(path_curv)
  road_lp = _filter_metric_signal(path_curv, prelim_eligible, lowpass_hz=C.ROAD_LP_HZ)
  gentle = np.abs(road_lp) <= C.ROAD_CURV_ABS_MAX_1PM
  eligible = prelim_eligible & gentle
  window_len = int(round(C.WEAVE_WINDOW_S * C.FS_HZ))
  min_eligible = int(round(C.MIN_WEAVE_ELIGIBLE_S * C.FS_HZ))

  path_band = _filter_metric_signal(path_curv, eligible, band=C.DEFAULT_WEAVE_BAND_HZ)
  steer_band = _filter_metric_signal(arrays["steering_angle_deg"].astype(float), eligible, band=C.DEFAULT_WEAVE_BAND_HZ)
  stage_bands = _stage_bands(arrays, eligible)

  windows: list[WeaveWindow] = []
  for start in range(0, max(0, len(t) - window_len + 1), window_len):
    end = start + window_len
    bucket_mask = np.zeros(len(t), dtype=bool)
    bucket_mask[start:end] = eligible[start:end]
    for sub_start, sub_end in contiguous_regions(bucket_mask, min_len=min_eligible):
      mask = np.zeros(len(t), dtype=bool)
      mask[sub_start:sub_end] = True
      path_rms = rms_masked(path_band, mask) * 1e4
      steer_rms = rms_masked(steer_band, mask)
      if not np.isfinite(path_rms) or path_rms < 0.2:
        continue
      stage_rms = _stage_rms_values(stage_bands, mask)
      cmd_rms = _best_rms(stage_rms, COMMAND_STAGE_KEYS)
      des_rms = _best_rms(stage_rms, DESIRED_STAGE_KEYS)
      model_rms = stage_rms.get("model_y20", math.nan)
      stage, note = _stage_label(path_rms, steer_rms, stage_rms)
      peak_hz = spectral_peak_hz(np.where(mask, path_band, np.nan), C.FS_HZ, C.DEFAULT_WEAVE_BAND_HZ)
      windows.append(WeaveWindow(
        symptom="weave_10_70",
        route_id=route_id,
        start_s=float(t[sub_start]),
        end_s=float(t[sub_end - 1]),
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


def _load_npz(path: str) -> dict[str, np.ndarray]:
  with np.load(path) as data:
    return {key: data[key] for key in data.files}


def _catalog_success_row(row: dict[str, object]) -> dict[str, object]:
  return {
    **row,
    "status": "ok",
    "error_type": "",
    "error_message": "",
  }


def _catalog_failure_row(route_id: str, exc: Exception) -> dict[str, object]:
  return {
    "route_id": route_id,
    "symptom": "route_error",
    "status": "failed",
    "error_type": type(exc).__name__,
    "error_message": str(exc),
  }


def write_symptom_catalog(cache_root: str, out_dir: str) -> list[dict[str, object]]:
  import csv
  from pathlib import Path

  cache = Path(cache_root)
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  rows: list[dict[str, object]] = []
  for npz in sorted(cache.glob("route_*.npz")):
    route_id = npz.stem
    try:
      arrays = _load_npz(str(npz))
      route_rows: list[dict[str, object]] = []
      route_rows.extend(_catalog_success_row(ep.to_row()) for ep in detect_low_speed_wheel_swing(route_id, arrays))
      route_rows.extend(_catalog_success_row(win.to_row()) for win in detect_weave_windows(route_id, arrays))
      rows.extend(route_rows)
    except Exception as exc:
      rows.append(_catalog_failure_row(route_id, exc))
  catalog = out / "symptom_catalog.csv"
  with catalog.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=SYMPTOM_CATALOG_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  return rows


def main(argv: list[str] | None = None) -> int:
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--cache-root", required=True)
  parser.add_argument("--out", required=True)
  args = parser.parse_args(argv)
  rows = write_symptom_catalog(args.cache_root, args.out)
  print(f"wrote {len(rows)} symptom rows to {args.out}/symptom_catalog.csv")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
