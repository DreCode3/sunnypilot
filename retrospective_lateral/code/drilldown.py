from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from retrospective_lateral.code import config as C
from retrospective_lateral.code.signal_utils import (
  contiguous_regions,
  filter_continuous,
  gps_cells,
  heading_bin_deg,
  peak_to_peak_masked,
  rms_masked,
)


STAGE_ORDER: tuple[tuple[str, str, str, float], ...] = (
  ("model_y20", "model_or_desired", "model path y20", 1.0),
  ("orientation_rate_curvature", "model_or_desired", "orientationRate/vEgo", 1e4),
  ("desired_curvature", "model_or_desired", "desiredCurvature", 1e4),
  ("cp_desired_curvature", "model_or_desired", "CP desired_curvature", 1e4),
  ("cx1_desired_curvature", "model_or_desired", "CX1 desired_curvature", 1e4),
  ("cp_predicted_curvature", "model_or_desired", "CP predicted_curvature", 1e4),
  ("cx1_predicted_curvature", "model_or_desired", "CX1 predicted_curvature", 1e4),
  ("cp_ema_curvature", "controller_or_filter", "CP ema_curvature", 1e4),
  ("cx1_ema_curvature", "controller_or_filter", "CX1 ema_curvature", 1e4),
  ("cp_pre_rate_limit", "controller_or_filter", "CP pre_rate_limit", 1e4),
  ("cx1_pre_rate_limit", "controller_or_filter", "CX1 pre_rate_limit", 1e4),
  ("cp_rate_limited", "final_command", "CP rate_limited", 1e4),
  ("cx1_rate_limited", "final_command", "CX1 rate_limited", 1e4),
  ("cp_final_command", "final_command", "CP final_command", 1e4),
  ("cx1_command_curvature", "final_command", "CX1 command_curvature", 1e4),
  ("act_curvature", "final_command", "actuator curvature", 1e4),
  ("path_curvature", "actual_path_or_plant", "yaw/vEgo path curvature", 1e4),
  ("steering_angle_deg", "steering_response", "steering angle", 1.0),
)

CURVATURE_STAGES = {name for name, _, _, scale in STAGE_ORDER if scale == 1e4}
LOW_SPEED_BAND = C.LOW_SPEED_INSPECT_BAND_HZ
WEAVE_BAND = C.DEFAULT_WEAVE_BAND_HZ
LEAD_EVENT_STAGE_NAMES = tuple(name for name, _, _, _ in STAGE_ORDER if name != "steering_angle_deg")
LEAD_EVENT_WINDOW_S = 10.0
LEAD_EVENT_MIN_RUN_S = 5.0


@dataclass(frozen=True)
class DrilldownOutputPaths:
  enriched_catalog: Path
  stage_metrics: Path
  stage_gain_lag_summary: Path
  lead_event_alignment: Path
  lane_geometry_audit: Path
  episode_stage_summary: Path
  context_summary: Path
  pi_comparisons: Path
  robustness_summary: Path


def _flatten_manifest_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
  return pd.json_normalize(rows, sep=".")


def load_manifest(manifest_path: Path) -> pd.DataFrame:
  return _flatten_manifest_rows(json.loads(manifest_path.read_text()))


def load_npz(path: Path) -> dict[str, np.ndarray]:
  with np.load(path) as data:
    return {key: data[key] for key in data.files}


def _path_curvature(arrays: dict[str, np.ndarray], yaw_source: str = "best") -> np.ndarray:
  v = arrays.get("v_ego", np.array([], dtype=float)).astype(float)
  raw = arrays.get("yaw_rate", np.full_like(v, np.nan)).astype(float)
  calibrated = arrays.get("yaw_rate_calibrated", np.full_like(v, np.nan)).astype(float)
  if yaw_source == "can":
    yaw = raw
  elif yaw_source == "calibrated":
    yaw = calibrated
  else:
    yaw = np.where(np.isfinite(calibrated), calibrated, raw)
  return np.divide(yaw, v, out=np.full_like(v, np.nan), where=(v > 1.0) & np.isfinite(yaw))


def _orientation_rate_curvature(arrays: dict[str, np.ndarray]) -> np.ndarray:
  v = arrays.get("v_ego", np.array([], dtype=float)).astype(float)
  orientation_rate = arrays.get("orientation_rate_z0", np.full_like(v, np.nan)).astype(float)
  return np.divide(orientation_rate, v, out=np.full_like(v, np.nan), where=(v > 1.0) & np.isfinite(orientation_rate))


def _heading_deg_from_gps(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
  lat = np.asarray(lat, dtype=float)
  lon = np.asarray(lon, dtype=float)
  ok = np.isfinite(lat) & np.isfinite(lon)
  out = np.full(lat.shape, np.nan)
  if ok.sum() < 3:
    return out
  lat_rad = np.radians(lat)
  x = lon * C.M_PER_DEG_LON_AT_EQUATOR * np.cos(lat_rad)
  y = lat * C.M_PER_DEG_LAT
  dx = np.gradient(x)
  dy = np.gradient(y)
  heading = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
  out[ok] = heading[ok]
  return out


def _window_mask(arrays: dict[str, np.ndarray], start_s: float, end_s: float) -> np.ndarray:
  t = arrays["t"].astype(float)
  return (t >= float(start_s)) & (t <= float(end_s))


def _mode(values: Iterable[object]) -> object:
  series = pd.Series(list(values)).dropna()
  if series.empty:
    return np.nan
  modes = series.mode(dropna=True)
  return modes.iloc[0] if not modes.empty else np.nan


def _speed_bin_label(speed_mph: float, bin_mph: float = 5.0) -> str:
  if not math.isfinite(float(speed_mph)):
    return "unknown"
  lo = int(math.floor(float(speed_mph) / bin_mph) * bin_mph)
  hi = int(lo + bin_mph)
  return f"{lo}-{hi}"


def _route_context(row: pd.Series, arrays: dict[str, np.ndarray]) -> dict[str, object]:
  mask = _window_mask(arrays, float(row["start_s"]), float(row["end_s"]))
  lat = arrays.get("lat", np.full(len(arrays["t"]), np.nan)).astype(float)
  lon = arrays.get("lon", np.full(len(arrays["t"]), np.nan)).astype(float)
  cells = gps_cells(lat, lon, C.GPS_CELL_M)
  heading_bins = heading_bin_deg(_heading_deg_from_gps(lat, lon), C.HEADING_BIN_DEG)
  lane_left = arrays.get("lane_prob_left", np.full(len(arrays["t"]), np.nan)).astype(float)
  lane_right = arrays.get("lane_prob_right", np.full(len(arrays["t"]), np.nan)).astype(float)
  lane_min = np.fmin(lane_left, lane_right)
  cp_samples = int(np.isfinite(arrays.get("cp_final_command", np.array([], dtype=float))).sum())
  cx1_samples = int(np.isfinite(arrays.get("cx1_command_curvature", np.array([], dtype=float))).sum())
  lead_prob = arrays.get("lead_prob", np.full(len(arrays["t"]), np.nan)).astype(float)
  lead_d_rel = arrays.get("lead_d_rel", np.full(len(arrays["t"]), np.nan)).astype(float)
  lead_time_headway = arrays.get("lead_time_headway_s", np.full(len(arrays["t"]), np.nan)).astype(float)
  has_lead_channel = "lead_prob" in arrays and "lead_d_rel" in arrays
  lead_prob_median = float(np.nanmedian(lead_prob[mask])) if np.isfinite(lead_prob[mask]).any() else math.nan
  lead_d_rel_median = float(np.nanmedian(lead_d_rel[mask])) if np.isfinite(lead_d_rel[mask]).any() else math.nan
  lead_time_headway_min = float(np.nanmin(lead_time_headway[mask])) if np.isfinite(lead_time_headway[mask]).any() else math.nan
  near = (lead_prob > 0.5) & (lead_time_headway < C.LEAD_HEADWAY_S)
  near_fraction = float(np.mean(near[mask])) if mask.any() else math.nan
  if not has_lead_channel:
    lead_status = "not_extracted"
  elif np.isfinite(lead_prob[mask]).sum() == 0:
    lead_status = "lead_unknown"
  elif near_fraction > 0.0:
    lead_status = "lead_near"
  elif np.nanmax(lead_prob[mask]) > 0.5:
    lead_status = "lead_far"
  else:
    lead_status = "no_lead"
  return {
    "gps_cell": _mode(cells[mask]),
    "heading_bin": _mode(heading_bins[mask]),
    "speed_bin_mph": _speed_bin_label(float(row.get("speed_mph_median", math.nan))),
    "lane_prob_min_median": float(np.nanmedian(lane_min[mask])) if np.isfinite(lane_min[mask]).any() else math.nan,
    "has_cp": bool(cp_samples > 0),
    "has_cx1": bool(cx1_samples > 0),
    "cp_samples": cp_samples,
    "cx1_samples": cx1_samples,
    "lead_status": lead_status,
    "lead_prob_median": lead_prob_median,
    "lead_d_rel_median": lead_d_rel_median,
    "lead_time_headway_min_s": lead_time_headway_min,
    "lead_near_fraction": near_fraction,
  }


def enrich_symptom_catalog(
  catalog: pd.DataFrame,
  manifest: pd.DataFrame,
  arrays_by_route: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
  meta = manifest.copy()
  if "init.commit" not in meta.columns and "init" in meta.columns:
    meta = pd.json_normalize(meta.to_dict("records"), sep=".")
  for col in [
    "pi_set",
    "config_confidence",
    "schema_version",
    "sample_count",
    "init.branch",
    "init.commit",
    "init.dirty",
    "car_params.steerRatio",
    "car_params.steerActuatorDelay",
  ]:
    if col not in meta.columns:
      meta[col] = np.nan
  keep = [
    "route_id",
    "pi_set",
    "config_confidence",
    "schema_version",
    "sample_count",
    "init.branch",
    "init.commit",
    "init.dirty",
    "car_params.steerRatio",
    "car_params.steerActuatorDelay",
  ]
  out = catalog.merge(meta[keep], on="route_id", how="left")
  context_rows: list[dict[str, object]] = []
  for _, row in out.iterrows():
    arrays = arrays_by_route.get(str(row["route_id"]))
    if arrays is None or pd.isna(row.get("start_s")) or pd.isna(row.get("end_s")):
      context_rows.append({
        "gps_cell": np.nan,
        "heading_bin": np.nan,
        "speed_bin_mph": "unknown",
        "lane_prob_min_median": math.nan,
        "has_cp": False,
        "has_cx1": False,
        "cp_samples": 0,
        "cx1_samples": 0,
        "lead_status": "missing_route_cache",
        "lead_prob_median": math.nan,
        "lead_d_rel_median": math.nan,
        "lead_time_headway_min_s": math.nan,
        "lead_near_fraction": math.nan,
      })
      continue
    context_rows.append(_route_context(row, arrays))
  context = pd.DataFrame(context_rows)
  enriched = pd.concat([out.reset_index(drop=True), context], axis=1)
  enriched["commit_short"] = enriched["init.commit"].astype(str).str[:8]
  enriched["is_dirty"] = enriched["init.dirty"].fillna(False).astype(bool)
  for col in ("has_cp", "has_cx1", "is_dirty"):
    enriched[col] = enriched[col].map(bool).astype(object)
  return enriched


def _band_for_symptom(symptom: str, band_override: tuple[float, float] | None = None) -> tuple[float, float]:
  if band_override is not None:
    return band_override
  return LOW_SPEED_BAND if symptom == "low_speed_wheel_swing" else WEAVE_BAND


def _stage_source(arrays: dict[str, np.ndarray], stage_name: str, yaw_source: str) -> np.ndarray | None:
  if stage_name == "path_curvature":
    return _path_curvature(arrays, yaw_source=yaw_source)
  if stage_name == "orientation_rate_curvature":
    return _orientation_rate_curvature(arrays)
  if stage_name in arrays:
    return arrays[stage_name].astype(float)
  return None


def _build_band_cache(
  arrays: dict[str, np.ndarray],
  symptom: str,
  *,
  yaw_source: str = "best",
  band_override: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
  band = _band_for_symptom(symptom, band_override)
  cache: dict[str, np.ndarray] = {}
  for stage_name, _, _, _ in STAGE_ORDER:
    source = _stage_source(arrays, stage_name, yaw_source)
    if source is None or len(source) == 0:
      continue
    cache[stage_name] = filter_continuous(source, C.FS_HZ, band=band)
  return cache


def _best_lag_and_corr(stage: np.ndarray, reference: np.ndarray, mask: np.ndarray, max_lag_s: float) -> tuple[float, float]:
  idx = np.flatnonzero(mask)
  if len(idx) < 10:
    return math.nan, math.nan
  lo, hi = int(idx[0]), int(idx[-1]) + 1
  x = np.asarray(stage[lo:hi], dtype=float)
  y = np.asarray(reference[lo:hi], dtype=float)
  finite = np.isfinite(x) & np.isfinite(y)
  if finite.sum() < 10:
    return math.nan, math.nan
  x = np.where(finite, x, np.nan)
  y = np.where(finite, y, np.nan)
  max_lag = int(round(max_lag_s * C.FS_HZ))
  best_corr = math.nan
  best_lag = math.nan
  for lag in range(-max_lag, max_lag + 1):
    if lag < 0:
      xs = x[-lag:]
      ys = y[:len(y) + lag]
    elif lag > 0:
      xs = x[:-lag]
      ys = y[lag:]
    else:
      xs = x
      ys = y
    ok = np.isfinite(xs) & np.isfinite(ys)
    if ok.sum() < 10:
      continue
    xs = xs[ok] - np.nanmean(xs[ok])
    ys = ys[ok] - np.nanmean(ys[ok])
    denom = float(np.sqrt(np.sum(xs ** 2) * np.sum(ys ** 2)))
    if denom <= 0.0:
      continue
    corr = float(np.sum(xs * ys) / denom)
    if not math.isfinite(best_corr) or abs(corr) > abs(best_corr):
      best_corr = corr
      best_lag = float(lag / C.FS_HZ)
  return best_lag, best_corr


def stage_metrics_for_episode(
  episode: pd.Series,
  arrays: dict[str, np.ndarray],
  *,
  band_cache: dict[str, np.ndarray] | None = None,
  yaw_source: str = "best",
  band_override: tuple[float, float] | None = None,
  include_lag: bool = True,
) -> list[dict[str, object]]:
  symptom = str(episode["symptom"])
  mask = _window_mask(arrays, float(episode["start_s"]), float(episode["end_s"]))
  cache = band_cache if band_cache is not None else _build_band_cache(
    arrays,
    symptom,
    yaw_source=yaw_source,
    band_override=band_override,
  )
  reference_name = "steering_angle_deg" if symptom == "low_speed_wheel_swing" else "path_curvature"
  reference = cache.get(reference_name)
  rows: list[dict[str, object]] = []
  for order, (stage_name, family, description, scale) in enumerate(STAGE_ORDER):
    band = cache.get(stage_name)
    if band is None:
      continue
    raw_scale = scale
    lag_s = corr = math.nan
    if include_lag and reference is not None:
      lag_s, corr = _best_lag_and_corr(band, reference, mask, max_lag_s=2.0 if symptom == "low_speed_wheel_swing" else 3.0)
    stage_rms = rms_masked(band, mask)
    stage_ptp = peak_to_peak_masked(band, mask)
    rows.append({
      "route_id": episode["route_id"],
      "symptom": symptom,
      "start_s": episode["start_s"],
      "end_s": episode["end_s"],
      "stage_order": order,
      "stage_name": stage_name,
      "stage_family": family,
      "stage_description": description,
      "stage_rms": stage_rms,
      "stage_ptp": stage_ptp,
      "stage_rms_1e4": stage_rms * raw_scale if raw_scale == 1e4 and math.isfinite(stage_rms) else math.nan,
      "stage_ptp_1e4": stage_ptp * raw_scale if raw_scale == 1e4 and math.isfinite(stage_ptp) else math.nan,
      "reference_stage": reference_name,
      "lag_s_positive_stage_leads_reference": lag_s,
      "corr_to_reference": corr,
      "finite_samples": int(np.isfinite(band[mask]).sum()),
      "yaw_source": yaw_source,
      "band_hz": str(_band_for_symptom(symptom, band_override)),
    })
  return rows


def _supported_stage(episode: pd.Series, stage: dict[str, object], all_rows: list[dict[str, object]]) -> bool:
  symptom = str(episode["symptom"])
  corr = float(stage.get("corr_to_reference", math.nan))
  if math.isfinite(corr) and abs(corr) < 0.35:
    return False
  stage_name = str(stage["stage_name"])
  if stage_name == "model_y20":
    threshold = 0.03 if symptom == "low_speed_wheel_swing" else 0.02
    return float(stage.get("stage_rms", math.nan)) >= threshold
  if stage_name in CURVATURE_STAGES:
    value = float(stage.get("stage_rms_1e4", math.nan))
    if not math.isfinite(value):
      return False
    if symptom == "low_speed_wheel_swing":
      act_values = [
        float(row.get("stage_rms_1e4", math.nan))
        for row in all_rows
        if row.get("stage_name") in {"act_curvature", "cp_final_command", "cx1_command_curvature"}
      ]
      finite = [v for v in act_values if math.isfinite(v)]
      reference = max(finite) if finite else 0.0
      return value >= max(2.0, 0.50 * reference)
    path_metric = float(episode.get("path_curvature_band_rms_1e4", math.nan))
    return value >= max(0.25, 0.50 * path_metric)
  if stage_name == "path_curvature":
    return symptom == "weave_10_70" and float(stage.get("stage_rms_1e4", math.nan)) >= 0.25
  if stage_name == "steering_angle_deg":
    return float(stage.get("stage_rms", math.nan)) >= (0.2 if symptom == "weave_10_70" else 0.5)
  return False


def summarize_episode_stage(episode: pd.Series, stage_rows: list[dict[str, object]]) -> dict[str, object]:
  ordered = sorted(stage_rows, key=lambda row: int(row["stage_order"]))
  first = next((row for row in ordered if _supported_stage(episode, row, ordered)), None)
  max_corr = max(
    (abs(float(row.get("corr_to_reference", math.nan))) for row in ordered if math.isfinite(float(row.get("corr_to_reference", math.nan)))),
    default=math.nan,
  )
  return {
    "route_id": episode["route_id"],
    "symptom": episode["symptom"],
    "start_s": episode["start_s"],
    "end_s": episode["end_s"],
    "speed_mph_median": episode.get("speed_mph_median", math.nan),
    "catalog_stage_first_growth": episode.get("stage_first_growth", ""),
    "first_supported_stage": first["stage_name"] if first is not None else "unknown",
    "first_supported_family": first["stage_family"] if first is not None else "unknown",
    "first_supported_corr": first.get("corr_to_reference", math.nan) if first is not None else math.nan,
    "first_supported_lag_s": first.get("lag_s_positive_stage_leads_reference", math.nan) if first is not None else math.nan,
    "max_abs_corr_any_stage": max_corr,
    "stage_rows": len(ordered),
  }


def location_speed_context_summary(enriched_catalog: pd.DataFrame) -> pd.DataFrame:
  if enriched_catalog.empty:
    return pd.DataFrame()
  rows = []
  group_cols = ["symptom", "gps_cell", "heading_bin", "speed_bin_mph"]
  for key, group in enriched_catalog.groupby(group_cols, dropna=True):
    rows.append({
      "symptom": key[0],
      "gps_cell": key[1],
      "heading_bin": key[2],
      "speed_bin_mph": key[3],
      "rows": int(len(group)),
      "routes": int(group["route_id"].nunique()),
      "mode_stage": _mode(group.get("stage_first_growth", pd.Series(dtype=object))),
      "pi_sets": ",".join(sorted(str(x) for x in group["pi_set"].dropna().unique())) if "pi_set" in group else "",
      "median_speed_mph": float(group["speed_mph_median"].median()) if "speed_mph_median" in group else math.nan,
      "median_path_rms_1e4": float(group["path_curvature_band_rms_1e4"].median()) if "path_curvature_band_rms_1e4" in group else math.nan,
      "median_steering_band_rms_deg": float(group["steering_band_rms_deg"].median()) if "steering_band_rms_deg" in group else math.nan,
      "lead_statuses": ",".join(sorted(str(x) for x in group["lead_status"].dropna().unique())) if "lead_status" in group else "",
    })
  return pd.DataFrame(rows).sort_values(["symptom", "rows"], ascending=[True, False])


def _pct(candidate: float, baseline: float) -> float:
  if not math.isfinite(candidate) or not math.isfinite(baseline) or baseline == 0:
    return math.nan
  return float(100.0 * (candidate - baseline) / baseline)


def location_speed_pi_comparisons(
  enriched_catalog: pd.DataFrame,
  *,
  baseline: str | None = None,
  candidate: str | None = None,
) -> pd.DataFrame:
  needed = {
    "symptom",
    "gps_cell",
    "heading_bin",
    "speed_bin_mph",
    "route_id",
    "pi_set",
    "path_curvature_band_rms_1e4",
    "steering_band_rms_deg",
  }
  missing = needed - set(enriched_catalog.columns)
  if missing:
    raise ValueError(f"missing columns: {sorted(missing)}")
  rows: list[dict[str, object]] = []
  group_cols = ["symptom", "gps_cell", "heading_bin", "speed_bin_mph"]
  for key, group in enriched_catalog.dropna(subset=["gps_cell", "heading_bin"]).groupby(group_cols, dropna=True):
    groups = {
      str(name): pi_group
      for name, pi_group in group.groupby("pi_set", dropna=True)
      if str(name) not in {"", "nan", "unknown"}
    }
    pairs: list[tuple[str, str]]
    if baseline is not None and candidate is not None:
      pairs = [(baseline, candidate)]
    else:
      names = sorted(groups)
      pairs = [(a, b) for idx, a in enumerate(names) for b in names[idx + 1:]]
    for base_name, cand_name in pairs:
      base = groups.get(base_name)
      cand = groups.get(cand_name)
      if base is None or cand is None or base.empty or cand.empty:
        continue
      base_path = float(base["path_curvature_band_rms_1e4"].median())
      cand_path = float(cand["path_curvature_band_rms_1e4"].median())
      base_steer = float(base["steering_band_rms_deg"].median())
      cand_steer = float(cand["steering_band_rms_deg"].median())
      rows.append({
        "comparison": f"{cand_name}_minus_{base_name}",
        "symptom": key[0],
        "gps_cell": key[1],
        "heading_bin": key[2],
        "speed_bin_mph": key[3],
        "baseline": base_name,
        "candidate": cand_name,
        "n_baseline": int(len(base)),
        "n_candidate": int(len(cand)),
        "routes_baseline": int(base["route_id"].nunique()),
        "routes_candidate": int(cand["route_id"].nunique()),
        "path_baseline": base_path,
        "path_candidate": cand_path,
        "path_effect_pct": _pct(cand_path, base_path),
        "steer_baseline": base_steer,
        "steer_candidate": cand_steer,
        "steer_effect_pct": _pct(cand_steer, base_steer),
        "evidence_tier": "location_speed_matched_descriptive",
      })
  return pd.DataFrame(rows)


def _severity_metric_for_symptom(symptom: str, columns: set[str]) -> str | None:
  if symptom == "low_speed_wheel_swing":
    if "steering_peak_to_peak_deg" in columns:
      return "steering_peak_to_peak_deg"
    if "steering_band_rms_deg" in columns:
      return "steering_band_rms_deg"
    return None
  if "path_curvature_band_rms_1e4" in columns:
    return "path_curvature_band_rms_1e4"
  return None


def _episode_context_for_stage_summary(enriched: pd.DataFrame) -> pd.DataFrame:
  keep = [
    "route_id",
    "symptom",
    "start_s",
    "end_s",
    "lead_status",
    "pi_set",
    "first_supported_stage",
    "path_curvature_band_rms_1e4",
    "steering_peak_to_peak_deg",
    "steering_band_rms_deg",
  ]
  present = [col for col in keep if col in enriched.columns]
  return enriched[present].drop_duplicates(["route_id", "symptom", "start_s", "end_s"])


def _quantile_or_nan(values: pd.Series, q: float) -> float:
  finite = pd.to_numeric(values, errors="coerce").dropna()
  return float(finite.quantile(q)) if not finite.empty else math.nan


def _numeric_column(df: pd.DataFrame, col: str) -> pd.Series:
  if col not in df.columns:
    return pd.Series(np.nan, index=df.index, dtype=float)
  return pd.to_numeric(df[col], errors="coerce")


def _finite_median(values: np.ndarray) -> float:
  vals = np.asarray(values, dtype=float)
  vals = vals[np.isfinite(vals)]
  return float(np.median(vals)) if len(vals) else math.nan


def _range_mask(t: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
  return (t >= float(start_s)) & (t < float(end_s))


def _lead_near_mask(arrays: dict[str, np.ndarray]) -> np.ndarray | None:
  if "lead_prob" not in arrays or "lead_time_headway_s" not in arrays:
    return None
  lead_prob = arrays["lead_prob"].astype(float)
  headway = arrays["lead_time_headway_s"].astype(float)
  return (lead_prob > 0.5) & (headway < C.LEAD_HEADWAY_S) & np.isfinite(lead_prob) & np.isfinite(headway)


def _finite_fraction(mask: np.ndarray) -> float:
  return float(np.mean(mask)) if len(mask) else math.nan


def _corr_for_mask(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
  x = np.asarray(a, dtype=float)[mask]
  y = np.asarray(b, dtype=float)[mask]
  ok = np.isfinite(x) & np.isfinite(y)
  if ok.sum() < 10:
    return math.nan
  xs = x[ok] - np.mean(x[ok])
  ys = y[ok] - np.mean(y[ok])
  denom = float(np.sqrt(np.sum(xs ** 2) * np.sum(ys ** 2)))
  return float(np.sum(xs * ys) / denom) if denom > 0.0 else math.nan


def _ratio(numerator: float, denominator: float) -> float:
  if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator == 0.0:
    return math.nan
  return float(numerator / denominator)


def _geometry_band_cache(arrays: dict[str, np.ndarray], symptom: str) -> dict[str, np.ndarray]:
  cache = _build_band_cache(arrays, symptom, yaw_source="best")
  band = _band_for_symptom(symptom)
  for x_m in C.MODEL_LOOKAHEAD_X_M:
    for prefix in (
      "model_y",
      "lane_left_y",
      "lane_right_y",
      "lane_center_y",
      "lane_width_y",
      "road_edge_left_y",
      "road_edge_right_y",
      "road_edge_width_y",
    ):
      key = f"{prefix}{x_m}"
      if key in arrays:
        cache[key] = filter_continuous(arrays[key].astype(float), C.FS_HZ, band=band)
  return cache


def _geometry_source_row(
  *,
  base: dict[str, object],
  arrays: dict[str, np.ndarray],
  band_cache: dict[str, np.ndarray],
  symptom: str,
  start_s: float,
  end_s: float,
  lookahead_m: int,
) -> dict[str, object]:
  t = arrays["t"].astype(float)
  mask = _range_mask(t, start_s, end_s)
  n = len(t)
  v_mph = arrays.get("v_ego", np.full(n, np.nan)).astype(float) * 2.2369362920544
  lane_left_prob = arrays.get("lane_prob_left", np.full(n, np.nan)).astype(float)
  lane_right_prob = arrays.get("lane_prob_right", np.full(n, np.nan)).astype(float)
  lane_prob_min = np.fmin(lane_left_prob, lane_right_prob)
  near = _lead_near_mask(arrays)
  if near is None:
    near = np.zeros(n, dtype=bool)
  reference_name = "steering_angle_deg" if symptom == "low_speed_wheel_swing" else "path_curvature"
  reference = band_cache.get(reference_name)
  model_key = f"model_y{lookahead_m}"
  lane_left_key = f"lane_left_y{lookahead_m}"
  lane_right_key = f"lane_right_y{lookahead_m}"
  lane_center_key = f"lane_center_y{lookahead_m}"
  lane_width_key = f"lane_width_y{lookahead_m}"
  road_left_key = f"road_edge_left_y{lookahead_m}"
  road_right_key = f"road_edge_right_y{lookahead_m}"
  road_width_key = f"road_edge_width_y{lookahead_m}"

  def rms(name: str) -> float:
    sig = band_cache.get(name)
    return rms_masked(sig, mask) if sig is not None else math.nan

  def ptp(name: str) -> float:
    sig = band_cache.get(name)
    return peak_to_peak_masked(sig, mask) if sig is not None else math.nan

  model = band_cache.get(model_key, np.full(n, np.nan))
  lane_center = band_cache.get(lane_center_key, np.full(n, np.nan))
  lane_width = band_cache.get(lane_width_key, np.full(n, np.nan))
  road_width = band_cache.get(road_width_key, np.full(n, np.nan))
  model_rms = rms(model_key)
  lane_center_rms = rms(lane_center_key)
  lane_width_rms = rms(lane_width_key)
  road_width_rms = rms(road_width_key)
  model_lag = model_corr_to_reference = math.nan
  lane_center_lag = lane_center_corr_to_reference = math.nan
  if reference is not None:
    model_lag, model_corr_to_reference = _best_lag_and_corr(model, reference, mask, max_lag_s=3.0)
    lane_center_lag, lane_center_corr_to_reference = _best_lag_and_corr(lane_center, reference, mask, max_lag_s=3.0)
  desired_rms = rms("desired_curvature")
  cp_final_rms = rms("cp_final_command")
  row = {
    **base,
    "lookahead_m": int(lookahead_m),
    "window_start_s": float(start_s),
    "window_end_s": float(end_s),
    "finite_samples": int(mask.sum()),
    "speed_mph_median": _finite_median(v_mph[mask]),
    "lead_near_fraction": _finite_fraction(near[mask]),
    "lane_prob_min_median": _finite_median(lane_prob_min[mask]),
    "lane_prob_left_median": _finite_median(lane_left_prob[mask]),
    "lane_prob_right_median": _finite_median(lane_right_prob[mask]),
    "road_edge_std_left_median": _finite_median(arrays.get("road_edge_std_left", np.full(n, np.nan)).astype(float)[mask]),
    "road_edge_std_right_median": _finite_median(arrays.get("road_edge_std_right", np.full(n, np.nan)).astype(float)[mask]),
    "model_y_rms": model_rms,
    "model_y_ptp": ptp(model_key),
    "lane_left_y_rms": rms(lane_left_key),
    "lane_left_y_ptp": ptp(lane_left_key),
    "lane_right_y_rms": rms(lane_right_key),
    "lane_right_y_ptp": ptp(lane_right_key),
    "lane_center_y_rms": lane_center_rms,
    "lane_center_y_ptp": ptp(lane_center_key),
    "lane_width_y_rms": lane_width_rms,
    "lane_width_y_ptp": ptp(lane_width_key),
    "road_edge_left_y_rms": rms(road_left_key),
    "road_edge_left_y_ptp": ptp(road_left_key),
    "road_edge_right_y_rms": rms(road_right_key),
    "road_edge_right_y_ptp": ptp(road_right_key),
    "road_edge_width_y_rms": road_width_rms,
    "road_edge_width_y_ptp": ptp(road_width_key),
    "model_lane_center_corr": _corr_for_mask(model, lane_center, mask),
    "model_lane_width_corr": _corr_for_mask(model, lane_width, mask),
    "model_road_edge_width_corr": _corr_for_mask(model, road_width, mask),
    "model_lag_to_reference_s": model_lag,
    "model_corr_to_reference": model_corr_to_reference,
    "lane_center_lag_to_reference_s": lane_center_lag,
    "lane_center_corr_to_reference": lane_center_corr_to_reference,
    "lane_center_over_model_rms": _ratio(lane_center_rms, model_rms),
    "lane_width_over_model_rms": _ratio(lane_width_rms, model_rms),
    "road_edge_width_over_model_rms": _ratio(road_width_rms, model_rms),
    "path_curvature_rms_1e4": rms("path_curvature") * 1e4 if math.isfinite(rms("path_curvature")) else math.nan,
    "desired_curvature_rms_1e4": desired_rms * 1e4 if math.isfinite(desired_rms) else math.nan,
    "cp_final_command_rms_1e4": cp_final_rms * 1e4 if math.isfinite(cp_final_rms) else math.nan,
    "cp_final_over_desired_rms": _ratio(cp_final_rms, desired_rms),
    "steering_angle_ptp_deg": ptp("steering_angle_deg"),
  }
  return row


def lane_geometry_audit(
  enriched: pd.DataFrame,
  arrays_by_route: dict[str, dict[str, np.ndarray]],
  lead_event_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
  rows: list[dict[str, object]] = []
  ok_rows = enriched[enriched["status"].fillna("ok") == "ok"] if "status" in enriched.columns else enriched
  for route_id, route_rows in ok_rows.groupby("route_id", sort=True):
    arrays = arrays_by_route.get(str(route_id))
    if arrays is None:
      continue
    for _, episode in route_rows.iterrows():
      if pd.isna(episode.get("start_s")) or pd.isna(episode.get("end_s")):
        continue
      symptom = str(episode.get("symptom", "weave_10_70"))
      band_cache = _geometry_band_cache(arrays, symptom)
      base = {
        "source_type": "episode",
        "slice": "episode",
        "route_id": str(route_id),
        "symptom": symptom,
        "start_s": float(episode["start_s"]),
        "end_s": float(episode["end_s"]),
        "event_id": "",
        "event_type": "",
        "event_time_s": math.nan,
        "lead_status": episode.get("lead_status", ""),
        "pi_set": episode.get("pi_set", ""),
      }
      for x_m in C.MODEL_LOOKAHEAD_X_M:
        rows.append(_geometry_source_row(
          base=base,
          arrays=arrays,
          band_cache=band_cache,
          symptom=symptom,
          start_s=float(episode["start_s"]),
          end_s=float(episode["end_s"]),
          lookahead_m=int(x_m),
        ))

  if lead_event_df is not None and not lead_event_df.empty:
    event_rows = lead_event_df[
      (lead_event_df.get("stage_name") == "model_y20")
      & (lead_event_df.get("analysis_eligible") == True)  # noqa: E712
      & (pd.to_numeric(lead_event_df.get("overlapping_weave_rows"), errors="coerce") > 0)
    ].copy()
    for route_id, route_events in event_rows.groupby("route_id", sort=True):
      arrays = arrays_by_route.get(str(route_id))
      if arrays is None:
        continue
      band_cache = _geometry_band_cache(arrays, "weave_10_70")
      for _, event in route_events.iterrows():
        for slice_name, start_col, end_col in (
          ("pre_event", "pre_start_s", "pre_end_s"),
          ("post_event", "post_start_s", "post_end_s"),
        ):
          base = {
            "source_type": "lead_event",
            "slice": slice_name,
            "route_id": str(route_id),
            "symptom": "weave_10_70",
            "start_s": math.nan,
            "end_s": math.nan,
            "event_id": event.get("event_id", ""),
            "event_type": event.get("event_type", ""),
            "event_time_s": float(event.get("event_time_s", math.nan)),
            "lead_status": "lead_transition",
            "pi_set": "",
          }
          for x_m in C.MODEL_LOOKAHEAD_X_M:
            rows.append(_geometry_source_row(
              base=base,
              arrays=arrays,
              band_cache=band_cache,
              symptom="weave_10_70",
              start_s=float(event[start_col]),
              end_s=float(event[end_col]),
              lookahead_m=int(x_m),
            ))
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows).sort_values(
    ["source_type", "route_id", "event_time_s", "start_s", "slice", "lookahead_m"],
    kind="stable",
  )


def _context_for_mask(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
  n = len(arrays["t"])
  v_mph = arrays.get("v_ego", np.full(n, np.nan)).astype(float) * 2.2369362920544
  lat = arrays.get("lat", np.full(n, np.nan)).astype(float)
  lon = arrays.get("lon", np.full(n, np.nan)).astype(float)
  heading_bins = heading_bin_deg(_heading_deg_from_gps(lat, lon), C.HEADING_BIN_DEG)
  lane_left = arrays.get("lane_prob_left", np.full(n, np.nan)).astype(float)
  lane_right = arrays.get("lane_prob_right", np.full(n, np.nan)).astype(float)
  lane_min = np.fmin(lane_left, lane_right)
  return {
    "speed_mph_median": _finite_median(v_mph[mask]),
    "heading_bin_mode": _mode(heading_bins[mask]),
    "lane_prob_min_median": _finite_median(lane_min[mask]),
  }


def _same_heading_bin(a: object, b: object) -> bool:
  try:
    ai = int(a)
    bi = int(b)
  except (TypeError, ValueError):
    return False
  return ai == bi and ai >= 0


def _event_context_match(pre_context: dict[str, float | int], post_context: dict[str, float | int]) -> tuple[bool, float, float]:
  pre_speed = float(pre_context.get("speed_mph_median", math.nan))
  post_speed = float(post_context.get("speed_mph_median", math.nan))
  pre_lane = float(pre_context.get("lane_prob_min_median", math.nan))
  post_lane = float(post_context.get("lane_prob_min_median", math.nan))
  speed_delta = post_speed - pre_speed if math.isfinite(pre_speed) and math.isfinite(post_speed) else math.nan
  lane_delta = post_lane - pre_lane if math.isfinite(pre_lane) and math.isfinite(post_lane) else math.nan
  speed_ok = math.isfinite(speed_delta) and abs(speed_delta) <= 5.0
  lane_ok = math.isfinite(lane_delta) and abs(lane_delta) <= 0.25
  heading_ok = _same_heading_bin(pre_context.get("heading_bin_mode"), post_context.get("heading_bin_mode"))
  return bool(speed_ok and lane_ok and heading_ok), speed_delta, lane_delta


def _overlapping_weave_rows(route_rows: pd.DataFrame, start_s: float, end_s: float) -> int:
  if route_rows.empty or "start_s" not in route_rows.columns or "end_s" not in route_rows.columns:
    return 0
  weave = route_rows[route_rows["symptom"] == "weave_10_70"] if "symptom" in route_rows.columns else route_rows
  overlaps = (pd.to_numeric(weave["start_s"], errors="coerce") <= end_s) & (pd.to_numeric(weave["end_s"], errors="coerce") >= start_s)
  return int(overlaps.sum())


def _append_lead_event_stage_rows(
  rows: list[dict[str, object]],
  *,
  arrays: dict[str, np.ndarray],
  band_cache: dict[str, np.ndarray],
  route_rows: pd.DataFrame,
  route_id: str,
  event_index: int,
  event_type: str,
  event_time_s: float,
  window_s: float,
  pre_mask: np.ndarray,
  post_mask: np.ndarray,
  near: np.ndarray,
) -> None:
  t = arrays["t"].astype(float)
  n = len(t)
  lead_prob = arrays.get("lead_prob", np.full(n, np.nan)).astype(float)
  lead_d_rel = arrays.get("lead_d_rel", np.full(n, np.nan)).astype(float)
  headway = arrays.get("lead_time_headway_s", np.full(n, np.nan)).astype(float)
  pre_context = _context_for_mask(arrays, pre_mask)
  post_context = _context_for_mask(arrays, post_mask)
  context_match, speed_delta, lane_delta = _event_context_match(pre_context, post_context)
  pre_speed = float(pre_context.get("speed_mph_median", math.nan))
  post_speed = float(post_context.get("speed_mph_median", math.nan))
  speed_range_ok = (
    math.isfinite(pre_speed)
    and math.isfinite(post_speed)
    and C.WEAVE_SPEED_MPH[0] <= pre_speed <= C.WEAVE_SPEED_MPH[1]
    and C.WEAVE_SPEED_MPH[0] <= post_speed <= C.WEAVE_SPEED_MPH[1]
  )
  analysis_eligible = bool(context_match and speed_range_ok)
  event_window_start = float(t[pre_mask][0]) if pre_mask.any() else math.nan
  event_window_end = float(t[post_mask][-1]) if post_mask.any() else math.nan
  overlapping_weave_rows = _overlapping_weave_rows(route_rows, event_window_start, event_window_end)
  stage_lookup = {name: (order, family, description, scale) for order, (name, family, description, scale) in enumerate(STAGE_ORDER)}
  for stage_name in LEAD_EVENT_STAGE_NAMES:
    band = band_cache.get(stage_name)
    if band is None:
      continue
    order, family, description, scale = stage_lookup[stage_name]
    pre_rms = rms_masked(band, pre_mask)
    post_rms = rms_masked(band, post_mask)
    pre_ptp = peak_to_peak_masked(band, pre_mask)
    post_ptp = peak_to_peak_masked(band, post_mask)
    delta_rms = post_rms - pre_rms if math.isfinite(pre_rms) and math.isfinite(post_rms) else math.nan
    rows.append({
      "route_id": route_id,
      "event_id": f"{route_id}:{event_type}:{event_index}",
      "event_index": event_index,
      "event_type": event_type,
      "event_time_s": event_time_s,
      "pre_start_s": event_time_s - window_s,
      "pre_end_s": event_time_s,
      "post_start_s": event_time_s,
      "post_end_s": event_time_s + window_s,
      "pre_lead_near_fraction": float(np.mean(near[pre_mask])) if pre_mask.any() else math.nan,
      "post_lead_near_fraction": float(np.mean(near[post_mask])) if post_mask.any() else math.nan,
      "pre_lead_prob_median": _finite_median(lead_prob[pre_mask]),
      "post_lead_prob_median": _finite_median(lead_prob[post_mask]),
      "pre_lead_d_rel_median": _finite_median(lead_d_rel[pre_mask]),
      "post_lead_d_rel_median": _finite_median(lead_d_rel[post_mask]),
      "pre_lead_time_headway_median_s": _finite_median(headway[pre_mask]),
      "post_lead_time_headway_median_s": _finite_median(headway[post_mask]),
      "pre_speed_mph_median": pre_context["speed_mph_median"],
      "post_speed_mph_median": post_context["speed_mph_median"],
      "speed_delta_mph": speed_delta,
      "pre_heading_bin": pre_context["heading_bin_mode"],
      "post_heading_bin": post_context["heading_bin_mode"],
      "pre_lane_prob_min_median": pre_context["lane_prob_min_median"],
      "post_lane_prob_min_median": post_context["lane_prob_min_median"],
      "lane_prob_delta": lane_delta,
      "context_match": context_match,
      "speed_range_ok": bool(speed_range_ok),
      "analysis_eligible": analysis_eligible,
      "overlapping_weave_rows": overlapping_weave_rows,
      "stage_order": order,
      "stage_name": stage_name,
      "stage_family": family,
      "stage_description": description,
      "pre_stage_rms": pre_rms,
      "post_stage_rms": post_rms,
      "post_minus_pre_rms": delta_rms,
      "post_over_pre_rms": post_rms / pre_rms if math.isfinite(pre_rms) and pre_rms != 0.0 and math.isfinite(post_rms) else math.nan,
      "post_minus_pre_pct": _pct(post_rms, pre_rms),
      "pre_stage_ptp": pre_ptp,
      "post_stage_ptp": post_ptp,
      "pre_stage_rms_1e4": pre_rms * scale if scale == 1e4 and math.isfinite(pre_rms) else math.nan,
      "post_stage_rms_1e4": post_rms * scale if scale == 1e4 and math.isfinite(post_rms) else math.nan,
      "post_minus_pre_rms_1e4": delta_rms * scale if scale == 1e4 and math.isfinite(delta_rms) else math.nan,
      "pre_finite_samples": int(np.isfinite(band[pre_mask]).sum()),
      "post_finite_samples": int(np.isfinite(band[post_mask]).sum()),
    })


def lead_event_alignment(
  enriched: pd.DataFrame,
  arrays_by_route: dict[str, dict[str, np.ndarray]],
  *,
  window_s: float = LEAD_EVENT_WINDOW_S,
  min_run_s: float = LEAD_EVENT_MIN_RUN_S,
) -> pd.DataFrame:
  rows: list[dict[str, object]] = []
  route_source = enriched[enriched["symptom"] == "weave_10_70"] if "symptom" in enriched.columns else enriched
  min_len = max(1, int(round(min_run_s * C.FS_HZ)))
  for route_id, route_rows in route_source.groupby("route_id", sort=True):
    arrays = arrays_by_route.get(str(route_id))
    if arrays is None or "t" not in arrays:
      continue
    near = _lead_near_mask(arrays)
    if near is None:
      continue
    t = arrays["t"].astype(float)
    if len(t) == 0 or len(near) != len(t):
      continue
    band_cache = _build_band_cache(arrays, "weave_10_70", yaw_source="best")
    event_index = 0
    for start, end in contiguous_regions(near, min_len=min_len):
      if start > 0:
        event_time = float(t[start])
        pre_mask = _range_mask(t, event_time - window_s, event_time)
        post_mask = _range_mask(t, event_time, event_time + window_s)
        if pre_mask.any() and post_mask.any() and float(np.mean(near[pre_mask])) <= 0.20 and float(np.mean(near[post_mask])) >= 0.80:
          event_index += 1
          _append_lead_event_stage_rows(
            rows,
            arrays=arrays,
            band_cache=band_cache,
            route_rows=route_rows,
            route_id=str(route_id),
            event_index=event_index,
            event_type="onset",
            event_time_s=event_time,
            window_s=window_s,
            pre_mask=pre_mask,
            post_mask=post_mask,
            near=near,
          )
      if end < len(t):
        event_time = float(t[end])
        pre_mask = _range_mask(t, event_time - window_s, event_time)
        post_mask = _range_mask(t, event_time, event_time + window_s)
        if pre_mask.any() and post_mask.any() and float(np.mean(near[pre_mask])) >= 0.80 and float(np.mean(near[post_mask])) <= 0.20:
          event_index += 1
          _append_lead_event_stage_rows(
            rows,
            arrays=arrays,
            band_cache=band_cache,
            route_rows=route_rows,
            route_id=str(route_id),
            event_index=event_index,
            event_type="exit",
            event_time_s=event_time,
            window_s=window_s,
            pre_mask=pre_mask,
            post_mask=post_mask,
            near=near,
          )
  if not rows:
    return pd.DataFrame()
  out = pd.DataFrame(rows)
  for col in ("context_match", "speed_range_ok", "analysis_eligible"):
    out[col] = out[col].map(bool).astype(object)
  return out.sort_values(["route_id", "event_time_s", "event_type", "stage_order"], kind="stable")


def _append_stage_gain_rows(
  rows: list[dict[str, object]],
  data: pd.DataFrame,
  *,
  subset: str,
  group_name: str,
  group_value: str,
  severity_metric: str,
  severity_threshold: float,
) -> None:
  if data.empty:
    return
  stage_cols = [
    "stage_order",
    "stage_name",
    "stage_family",
    "stage_description",
    "reference_stage",
  ]
  for col in stage_cols:
    if col not in data.columns:
      data = data.assign(**{col: ""})
  for key, group in data.groupby(stage_cols, dropna=False, sort=True):
    stage_rms = _numeric_column(group, "stage_rms")
    stage_ptp = _numeric_column(group, "stage_ptp")
    stage_rms_1e4 = _numeric_column(group, "stage_rms_1e4")
    stage_ptp_1e4 = _numeric_column(group, "stage_ptp_1e4")
    corr = _numeric_column(group, "corr_to_reference")
    lag = _numeric_column(group, "lag_s_positive_stage_leads_reference")
    finite_samples = _numeric_column(group, "finite_samples")
    episodes = group[["route_id", "symptom", "start_s", "end_s"]].drop_duplicates()
    rows.append({
      "symptom": group["symptom"].iloc[0],
      "subset": subset,
      "group_name": group_name,
      "group_value": group_value,
      "severity_metric": severity_metric,
      "severity_threshold": severity_threshold,
      "stage_order": key[0],
      "stage_name": key[1],
      "stage_family": key[2],
      "stage_description": key[3],
      "reference_stage": key[4],
      "rows": int(len(group)),
      "episodes": int(len(episodes)),
      "routes": int(group["route_id"].nunique()),
      "median_stage_rms": float(stage_rms.median()) if stage_rms.notna().any() else math.nan,
      "median_stage_ptp": float(stage_ptp.median()) if stage_ptp.notna().any() else math.nan,
      "median_stage_rms_1e4": float(stage_rms_1e4.median()) if stage_rms_1e4.notna().any() else math.nan,
      "p75_stage_rms_1e4": _quantile_or_nan(stage_rms_1e4, 0.75),
      "p90_stage_rms_1e4": _quantile_or_nan(stage_rms_1e4, 0.90),
      "p95_stage_rms_1e4": _quantile_or_nan(stage_rms_1e4, 0.95),
      "median_stage_ptp_1e4": float(stage_ptp_1e4.median()) if stage_ptp_1e4.notna().any() else math.nan,
      "median_corr_to_reference": float(corr.median()) if corr.notna().any() else math.nan,
      "median_abs_corr_to_reference": float(corr.abs().median()) if corr.notna().any() else math.nan,
      "median_lag_s": float(lag.median()) if lag.notna().any() else math.nan,
      "p10_lag_s": _quantile_or_nan(lag, 0.10),
      "p90_lag_s": _quantile_or_nan(lag, 0.90),
      "median_finite_samples": float(finite_samples.median()) if finite_samples.notna().any() else math.nan,
    })


def stage_gain_lag_summary(enriched: pd.DataFrame, stage_metrics: pd.DataFrame) -> pd.DataFrame:
  if enriched.empty or stage_metrics.empty:
    return pd.DataFrame()
  context = _episode_context_for_stage_summary(enriched)
  episode_key = ["route_id", "symptom", "start_s", "end_s"]
  joined = stage_metrics.merge(
    context,
    on=episode_key,
    how="left",
  )
  rows: list[dict[str, object]] = []
  for symptom, symptom_rows in joined.groupby("symptom", sort=True):
    symptom_context = context[context["symptom"] == symptom]
    severity_metric = _severity_metric_for_symptom(str(symptom), set(symptom_context.columns) | set(symptom_rows.columns))
    if severity_metric is None:
      severity_threshold = math.nan
      subsets = [("all", symptom_rows)]
    else:
      severity = pd.to_numeric(symptom_context[severity_metric], errors="coerce")
      severity_threshold = float(severity.quantile(0.90)) if severity.notna().any() else math.nan
      subsets = [("all", symptom_rows)]
      if math.isfinite(severity_threshold):
        top_episode_keys = symptom_context.loc[severity >= severity_threshold, episode_key].drop_duplicates()
        subsets.append(("top_decile", symptom_rows.merge(top_episode_keys, on=episode_key, how="inner")))
    for subset_name, subset_rows in subsets:
      _append_stage_gain_rows(
        rows,
        subset_rows,
        subset=subset_name,
        group_name="overall",
        group_value="all",
        severity_metric=severity_metric or "",
        severity_threshold=severity_threshold,
      )
      if "lead_status" in subset_rows.columns:
        for lead_status, lead_group in subset_rows.groupby("lead_status", dropna=False, sort=True):
          _append_stage_gain_rows(
            rows,
            lead_group,
            subset=subset_name,
            group_name="lead_status",
            group_value=str(lead_status),
            severity_metric=severity_metric or "",
            severity_threshold=severity_threshold,
          )
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows).sort_values(
    ["symptom", "subset", "group_name", "group_value", "stage_order"],
    kind="stable",
  )


def robustness_summary(enriched: pd.DataFrame, episode_stage_summary: pd.DataFrame) -> pd.DataFrame:
  rows: list[dict[str, object]] = []
  for symptom, group in enriched.groupby("symptom", dropna=False):
    rows.append({
      "check": "row_count",
      "symptom": symptom,
      "value": len(group),
      "note": "input enriched catalog rows",
    })
    for status, count in group["lead_status"].value_counts(dropna=False).items():
      rows.append({
        "check": "lead_status",
        "symptom": symptom,
        "value": int(count),
        "note": str(status),
      })
    if "lane_prob_min_median" in group:
      rows.append({
        "check": "lane_quality_ge_0.5",
        "symptom": symptom,
        "value": int((group["lane_prob_min_median"] >= 0.5).sum()),
        "note": "rows surviving lane_prob_min_median >= 0.5",
      })
      rows.append({
        "check": "lane_quality_ge_0.8",
        "symptom": symptom,
        "value": int((group["lane_prob_min_median"] >= 0.8).sum()),
        "note": "rows surviving lane_prob_min_median >= 0.8",
      })
  for symptom, group in episode_stage_summary.groupby("symptom", dropna=False):
    for family, count in group["first_supported_family"].value_counts(dropna=False).items():
      rows.append({
        "check": "first_supported_family",
        "symptom": symptom,
        "value": int(count),
        "note": str(family),
      })
  return pd.DataFrame(rows)


def _variant_specs_for_symptom(symptom: str) -> list[tuple[str, str, tuple[float, float]]]:
  if symptom == "low_speed_wheel_swing":
    return [
      ("low_default_best", "best", LOW_SPEED_BAND),
      ("low_slow_best", "best", (0.08, 0.50)),
      ("low_wide_best", "best", (0.05, 0.80)),
    ]
  return [
    ("weave_default_best", "best", WEAVE_BAND),
    ("weave_default_can_yaw", "can", WEAVE_BAND),
    ("weave_default_calibrated_yaw", "calibrated", WEAVE_BAND),
    ("weave_narrow_best", "best", (0.12, 0.30)),
    ("weave_wide_best", "best", (0.08, 0.45)),
  ]


def robustness_variant_summary(
  enriched: pd.DataFrame,
  arrays_by_route: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
  rows: list[dict[str, object]] = []
  ok_rows = enriched[enriched["status"].fillna("ok") == "ok"]
  for symptom, symptom_all_routes in ok_rows.groupby("symptom", sort=True):
    for variant_name, yaw_source, band in _variant_specs_for_symptom(str(symptom)):
      summaries: list[dict[str, object]] = []
      for route_id, route_rows in symptom_all_routes.groupby("route_id", sort=True):
        arrays = arrays_by_route.get(str(route_id))
        if arrays is None:
          continue
        band_cache = _build_band_cache(arrays, str(symptom), yaw_source=yaw_source, band_override=band)
        for _, episode in route_rows.iterrows():
          if pd.isna(episode.get("start_s")) or pd.isna(episode.get("end_s")):
            continue
          stage_rows = stage_metrics_for_episode(
            episode,
            arrays,
            band_cache=band_cache,
            yaw_source=yaw_source,
            band_override=band,
            include_lag=False,
          )
          summaries.append(summarize_episode_stage(episode, stage_rows))
      if not summaries:
        rows.append({
          "check": "variant_first_supported_family",
          "symptom": symptom,
          "value": 0,
          "note": f"{variant_name}:no_rows",
        })
        continue
      summary_df = pd.DataFrame(summaries)
      for family, count in summary_df["first_supported_family"].value_counts(dropna=False).items():
        rows.append({
          "check": "variant_first_supported_family",
          "symptom": symptom,
          "value": int(count),
          "note": f"{variant_name}:{family}",
        })
      for stage, count in summary_df["first_supported_stage"].value_counts(dropna=False).items():
        rows.append({
          "check": "variant_first_supported_stage",
          "symptom": symptom,
          "value": int(count),
          "note": f"{variant_name}:{stage}",
        })
  return pd.DataFrame(rows)


def _load_arrays_for_catalog(cache_root: Path, catalog: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
  arrays_by_route: dict[str, dict[str, np.ndarray]] = {}
  for route_id in sorted(str(route) for route in catalog["route_id"].dropna().unique()):
    path = cache_root / f"{route_id}.npz"
    if path.exists():
      arrays_by_route[route_id] = load_npz(path)
  return arrays_by_route


def _build_stage_outputs(enriched: pd.DataFrame, arrays_by_route: dict[str, dict[str, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
  stage_rows: list[dict[str, object]] = []
  summary_rows: list[dict[str, object]] = []
  for route_id, route_rows in enriched.groupby("route_id", sort=True):
    arrays = arrays_by_route.get(str(route_id))
    if arrays is None:
      continue
    for symptom, symptom_rows in route_rows.groupby("symptom", sort=True):
      band_cache = _build_band_cache(arrays, str(symptom), yaw_source="best")
      for _, episode in symptom_rows.iterrows():
        if pd.isna(episode.get("start_s")) or pd.isna(episode.get("end_s")):
          continue
        rows = stage_metrics_for_episode(episode, arrays, band_cache=band_cache, yaw_source="best")
        stage_rows.extend(rows)
        summary_rows.append(summarize_episode_stage(episode, rows))
  return pd.DataFrame(stage_rows), pd.DataFrame(summary_rows)


def build_drilldown_outputs(cache_root: str | Path, report_root: str | Path) -> DrilldownOutputPaths:
  cache = Path(cache_root)
  reports = Path(report_root)
  reports.mkdir(parents=True, exist_ok=True)
  catalog = pd.read_csv(reports / "symptom_catalog.csv")
  manifest = load_manifest(cache / "manifest.json")
  arrays_by_route = _load_arrays_for_catalog(cache, catalog)
  enriched = enrich_symptom_catalog(catalog, manifest, arrays_by_route)
  stage_df, episode_stage_df = _build_stage_outputs(enriched[enriched["status"].fillna("ok") == "ok"], arrays_by_route)
  stage_gain_df = stage_gain_lag_summary(
    enriched.merge(
      episode_stage_df[["route_id", "symptom", "start_s", "end_s", "first_supported_stage"]],
      on=["route_id", "symptom", "start_s", "end_s"],
      how="left",
    ),
    stage_df,
  )
  lead_event_df = lead_event_alignment(enriched, arrays_by_route)
  lane_geometry_df = lane_geometry_audit(enriched, arrays_by_route, lead_event_df)
  context_df = location_speed_context_summary(enriched)
  pi_compare_df = location_speed_pi_comparisons(enriched)
  robust_df = pd.concat([
    robustness_summary(enriched, episode_stage_df),
    robustness_variant_summary(enriched, arrays_by_route),
  ], ignore_index=True)

  outputs = DrilldownOutputPaths(
    enriched_catalog=reports / "drilldown_enriched_catalog.csv",
    stage_metrics=reports / "drilldown_stage_metrics.csv",
    stage_gain_lag_summary=reports / "drilldown_stage_gain_lag_summary.csv",
    lead_event_alignment=reports / "drilldown_lead_event_alignment.csv",
    lane_geometry_audit=reports / "drilldown_lane_geometry_audit.csv",
    episode_stage_summary=reports / "drilldown_episode_stage_summary.csv",
    context_summary=reports / "drilldown_location_speed_context.csv",
    pi_comparisons=reports / "drilldown_location_speed_pi_comparisons.csv",
    robustness_summary=reports / "drilldown_robustness_summary.csv",
  )
  enriched.to_csv(outputs.enriched_catalog, index=False)
  stage_df.to_csv(outputs.stage_metrics, index=False)
  stage_gain_df.to_csv(outputs.stage_gain_lag_summary, index=False)
  lead_event_df.to_csv(outputs.lead_event_alignment, index=False)
  lane_geometry_df.to_csv(outputs.lane_geometry_audit, index=False)
  episode_stage_df.to_csv(outputs.episode_stage_summary, index=False)
  context_df.to_csv(outputs.context_summary, index=False)
  pi_compare_df.to_csv(outputs.pi_comparisons, index=False)
  robust_df.to_csv(outputs.robustness_summary, index=False)
  return outputs


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--cache-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
  parser.add_argument("--report-root", type=Path, default=C.DEFAULT_REPORT_ROOT)
  args = parser.parse_args(argv)
  outputs = build_drilldown_outputs(args.cache_root, args.report_root)
  print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
