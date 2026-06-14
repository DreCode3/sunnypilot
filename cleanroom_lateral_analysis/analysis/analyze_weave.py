#!/usr/bin/env python3
"""
Clean-room lateral weave analysis.

Primary symptom definition:
  Low-frequency steering/path oscillation while lateral control is engaged,
  on straight or very gentle road geometry.

The script:
  1. aligns extracted streams to a 20 Hz grid,
  2. gates to stable lateral engagement, no steering override, highway speed,
     and low slowly-varying road curvature,
  3. computes non-overlapping 30 s window metrics in a low-frequency band,
  4. summarizes by drive/config and compares configs with drive-aware
     bootstraps plus matched speed/curvature/GPS subsets.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "analysis" / "cache"
RESULTS = ROOT / "analysis" / "results"
FIGURES = ROOT / "analysis" / "figures"
FS = 20.0


@dataclass(frozen=True)
class Setting:
  name: str
  band_lo: float
  band_hi: float
  min_speed: float
  max_speed: float
  curve_abs_max: float


SETTINGS = [
  Setting("primary", 0.10, 0.35, 20.0, 35.0, 0.0015),
  Setting("strict_curve", 0.10, 0.35, 20.0, 35.0, 0.0010),
  Setting("loose_curve", 0.10, 0.35, 20.0, 35.0, 0.0020),
  Setting("broad_band", 0.08, 0.50, 20.0, 35.0, 0.0015),
  Setting("narrow_band", 0.12, 0.30, 20.0, 35.0, 0.0015),
  Setting("lower_speed", 0.10, 0.35, 15.0, 35.0, 0.0015),
]


PRIMARY_METRICS = [
  "steer_rms_deg",
  "actual_curv_rms_1pm",
  "lat_accel_rms_mps2",
  "cmd_curv_rms_1pm",
  "desired_curv_rms_1pm",
  "model_path_y20_rms_m",
  "lane_center_y20_rms_m",
]


def load_metadata() -> pd.DataFrame:
  return pd.read_csv(ROOT / "drives" / "METADATA.csv")


def finite_interp(src_t: np.ndarray, src_y: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
  src_t = np.asarray(src_t, dtype=np.float64)
  src_y = np.asarray(src_y, dtype=np.float64)
  ok = np.isfinite(src_t) & np.isfinite(src_y)
  if ok.sum() < 2:
    return np.full_like(grid_t, np.nan, dtype=np.float64)
  order = np.argsort(src_t[ok])
  t = src_t[ok][order]
  y = src_y[ok][order]
  return np.interp(grid_t, t, y, left=np.nan, right=np.nan)


def nearest_bool(src_t: np.ndarray, src_y: np.ndarray, grid_t: np.ndarray, max_gap: float = 0.2) -> np.ndarray:
  src_t = np.asarray(src_t, dtype=np.float64)
  src_y = np.asarray(src_y, dtype=bool)
  if len(src_t) == 0:
    return np.zeros_like(grid_t, dtype=bool)
  order = np.argsort(src_t)
  t = src_t[order]
  y = src_y[order]
  idx = np.searchsorted(t, grid_t)
  idx0 = np.clip(idx - 1, 0, len(t) - 1)
  idx1 = np.clip(idx, 0, len(t) - 1)
  choose1 = np.abs(t[idx1] - grid_t) < np.abs(t[idx0] - grid_t)
  nearest = np.where(choose1, idx1, idx0)
  gap = np.abs(t[nearest] - grid_t)
  return np.where(gap <= max_gap, y[nearest], False)


def nearest_int(src_t: np.ndarray, src_y: np.ndarray, grid_t: np.ndarray, max_gap: float = 0.3) -> np.ndarray:
  src_t = np.asarray(src_t, dtype=np.float64)
  src_y = np.asarray(src_y, dtype=np.float64)
  if len(src_t) == 0:
    return np.full_like(grid_t, -1, dtype=np.int16)
  order = np.argsort(src_t)
  t = src_t[order]
  y = src_y[order]
  idx = np.searchsorted(t, grid_t)
  idx0 = np.clip(idx - 1, 0, len(t) - 1)
  idx1 = np.clip(idx, 0, len(t) - 1)
  choose1 = np.abs(t[idx1] - grid_t) < np.abs(t[idx0] - grid_t)
  nearest = np.where(choose1, idx1, idx0)
  gap = np.abs(t[nearest] - grid_t)
  return np.where(gap <= max_gap, y[nearest], -1).astype(np.int16)


def fill_for_filter(x: np.ndarray) -> np.ndarray:
  x = np.asarray(x, dtype=np.float64)
  ok = np.isfinite(x)
  if ok.sum() < 10:
    return np.zeros_like(x)
  idx = np.arange(len(x))
  return np.interp(idx, idx[ok], x[ok])


def butter_filter(x: np.ndarray, fs: float, kind: str, cutoff: float | tuple[float, float], order: int = 3) -> np.ndarray:
  x_filled = fill_for_filter(x)
  sos = butter(order, cutoff, btype=kind, fs=fs, output="sos")
  if len(x_filled) < 60:
    return np.full_like(x_filled, np.nan)
  return sosfiltfilt(sos, x_filled)


def rms(x: np.ndarray) -> float:
  x = np.asarray(x, dtype=np.float64)
  x = x[np.isfinite(x)]
  if len(x) == 0:
    return np.nan
  return float(np.sqrt(np.mean(x * x)))


def peak_frequency(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
  x = fill_for_filter(x)
  if len(x) < fs * 10:
    return np.nan
  f, pxx = welch(x - np.mean(x), fs=fs, nperseg=min(512, len(x)), noverlap=None)
  band = (f >= lo) & (f <= hi)
  if not np.any(band) or np.nanmax(pxx[band]) <= 0:
    return np.nan
  return float(f[band][np.argmax(pxx[band])])


def rolling_all(mask: np.ndarray, radius_s: float) -> np.ndarray:
  radius = int(round(radius_s * FS))
  if radius <= 0:
    return mask.astype(bool)
  size = 2 * radius + 1
  return uniform_filter1d(mask.astype(float), size=size, mode="nearest") >= 1.0


def rolling_any(mask: np.ndarray, radius_s: float) -> np.ndarray:
  radius = int(round(radius_s * FS))
  if radius <= 0:
    return mask.astype(bool)
  size = 2 * radius + 1
  return uniform_filter1d(mask.astype(float), size=size, mode="nearest") > 0.0


def load_drive_grid(drive_id: str) -> pd.DataFrame:
  d = np.load(CACHE / f"{drive_id}_signals.npz", allow_pickle=True)
  start = np.nanmin([np.nanmin(d["cs_t"]), np.nanmin(d["cc_t"]), np.nanmin(d["ctl_t"])])
  end = np.nanmax([np.nanmax(d["cs_t"]), np.nanmax(d["cc_t"]), np.nanmax(d["ctl_t"])])
  grid_t = np.arange(start, end, 1.0 / FS)

  out = pd.DataFrame({"t": grid_t, "t_rel": grid_t - grid_t[0]})
  for dst, src in [
    ("speed_mps", "cs_v_ego"),
    ("steer_angle_deg", "cs_steer_angle_deg"),
    ("steer_rate_deg", "cs_steer_rate_deg"),
    ("yaw_rate_carstate", "cs_yaw_rate"),
    ("cmd_curvature", "cc_act_curvature"),
    ("current_curvature", "cc_current_curvature"),
    ("controls_curvature", "ctl_curvature"),
    ("desired_curvature", "ctl_desired_curvature"),
    ("angle_error_deg", "ctl_angle_error"),
    ("model_path_y20", "model_path_y20"),
    ("model_path_curvature", "model_path_curvature"),
    ("lane_center_y20", "model_lane_center_y20"),
    ("lane_width_y20", "model_lane_width_y20"),
    ("lane_prob_left", "model_lane_prob_left"),
    ("lane_prob_right", "model_lane_prob_right"),
    ("lat", "loc_lat"),
    ("lon", "loc_lon"),
    ("yaw_rate_loc", "loc_yaw_rate_calibrated"),
  ]:
    prefix = src.split("_")[0]
    if src in d and f"{prefix}_t" in d:
      out[dst] = finite_interp(d[f"{prefix}_t"], d[src], grid_t)
    else:
      out[dst] = np.nan

  out["lat_active"] = nearest_bool(d["cc_t"], d["cc_lat_active"], grid_t)
  out["steering_pressed"] = nearest_bool(d["cs_t"], d["cs_steering_pressed"], grid_t)
  out["left_blinker"] = nearest_bool(d["cs_t"], d["cs_left_blinker"], grid_t) | nearest_bool(d["cc_t"], d["cc_left_blinker"], grid_t)
  out["right_blinker"] = nearest_bool(d["cs_t"], d["cs_right_blinker"], grid_t) | nearest_bool(d["cc_t"], d["cc_right_blinker"], grid_t)
  out["model_lane_change_state"] = nearest_int(d["model_t"], d["model_lane_change_state"], grid_t) if "model_lane_change_state" in d else -1
  return out


def add_derived_signals(df: pd.DataFrame, setting: Setting) -> pd.DataFrame:
  out = df.copy()
  speed = out["speed_mps"].to_numpy()
  yaw = out["yaw_rate_carstate"].to_numpy()
  actual_curv = np.divide(yaw, speed, out=np.full_like(yaw, np.nan, dtype=np.float64), where=np.isfinite(speed) & (speed > 5.0))
  out["actual_curvature"] = actual_curv
  out["road_curvature_lp"] = butter_filter(actual_curv, FS, "lowpass", 0.035, order=2)

  band = (setting.band_lo, setting.band_hi)
  for col, out_col in [
    ("steer_angle_deg", "steer_bp"),
    ("actual_curvature", "actual_curv_bp"),
    ("cmd_curvature", "cmd_curv_bp"),
    ("desired_curvature", "desired_curv_bp"),
    ("controls_curvature", "controls_curv_bp"),
    ("model_path_y20", "model_path_y20_bp"),
    ("model_path_curvature", "model_curv_bp"),
    ("lane_center_y20", "lane_center_y20_bp"),
  ]:
    out[out_col] = butter_filter(out[col].to_numpy(), FS, "bandpass", band, order=3)

  out["lat_accel_bp"] = out["actual_curv_bp"].to_numpy() * np.square(speed)
  stable_active = rolling_all(out["lat_active"].to_numpy(dtype=bool), 2.0)
  no_press = ~rolling_any(out["steering_pressed"].to_numpy(dtype=bool), 1.0)
  no_blinker = ~rolling_any((out["left_blinker"] | out["right_blinker"]).to_numpy(dtype=bool), 1.0)
  lane_keep = out["model_lane_change_state"].to_numpy() <= 0
  speed_gate = (speed >= setting.min_speed) & (speed <= setting.max_speed)
  curve_gate = np.abs(out["road_curvature_lp"].to_numpy()) <= setting.curve_abs_max
  finite_gate = np.isfinite(out["steer_bp"]) & np.isfinite(out["actual_curv_bp"]) & np.isfinite(out["cmd_curv_bp"])
  out["eligible"] = stable_active & no_press & no_blinker & lane_keep & speed_gate & curve_gate & finite_gate
  return out


def latlon_to_xy(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  lat0 = np.nanmedian(lat)
  lon0 = np.nanmedian(lon)
  x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
  y = (lat - lat0) * 110_540.0
  return x, y


def window_metrics(df: pd.DataFrame, drive_meta: dict[str, object], setting: Setting) -> pd.DataFrame:
  rows = []
  duration = 30.0
  min_eligible = 20.0
  t_rel = df["t_rel"].to_numpy()
  max_start = t_rel[-1] - duration
  starts = np.arange(0.0, max_start + 1e-6, duration)
  eligible = df["eligible"].to_numpy(dtype=bool)

  for start in starts:
    in_win = (t_rel >= start) & (t_rel < start + duration)
    if not np.any(in_win):
      continue
    use = in_win & eligible
    eligible_s = float(use.sum() / FS)
    if eligible_s < min_eligible:
      continue

    full = df.loc[in_win]
    sub = df.loc[use]
    f_peak = peak_frequency(full["steer_angle_deg"].to_numpy(), FS, setting.band_lo, setting.band_hi)
    lat_accel_rms = rms(sub["lat_accel_bp"].to_numpy())
    disp_pp_cm = np.nan
    if np.isfinite(f_peak) and f_peak > 0:
      # Sinusoid approximation: y_rms = a_rms / omega^2; peak-to-peak = 2*sqrt(2)*rms.
      disp_pp_cm = 100.0 * 2.0 * math.sqrt(2.0) * lat_accel_rms / ((2.0 * math.pi * f_peak) ** 2)

    row = {
      "setting": setting.name,
      "drive_id": drive_meta["drive_id"],
      "config_id": drive_meta["config_id"],
      "driving_model": drive_meta["driving_model"],
      "controller_set": "set2" if float(drive_meta["lat_pi_lc_kp"]) > 0.0001 else "set1",
      "window_start_s": start,
      "eligible_s": eligible_s,
      "eligible_frac": eligible_s / duration,
      "speed_median_mps": float(np.nanmedian(sub["speed_mps"])),
      "speed_p25_mps": float(np.nanpercentile(sub["speed_mps"], 25)),
      "speed_p75_mps": float(np.nanpercentile(sub["speed_mps"], 75)),
      "road_curv_median": float(np.nanmedian(sub["road_curvature_lp"])),
      "road_curv_abs_median": float(np.nanmedian(np.abs(sub["road_curvature_lp"]))),
      "lat_median": float(np.nanmedian(sub["lat"])),
      "lon_median": float(np.nanmedian(sub["lon"])),
      "steer_rms_deg": rms(sub["steer_bp"].to_numpy()),
      "steer_pp_sine_deg": 2.0 * math.sqrt(2.0) * rms(sub["steer_bp"].to_numpy()),
      "actual_curv_rms_1pm": 1e4 * rms(sub["actual_curv_bp"].to_numpy()),
      "lat_accel_rms_mps2": lat_accel_rms,
      "lat_disp_pp_sine_cm": disp_pp_cm,
      "cmd_curv_rms_1pm": 1e4 * rms(sub["cmd_curv_bp"].to_numpy()),
      "desired_curv_rms_1pm": 1e4 * rms(sub["desired_curv_bp"].to_numpy()),
      "controls_curv_rms_1pm": 1e4 * rms(sub["controls_curv_bp"].to_numpy()),
      "model_path_y20_rms_m": rms(sub["model_path_y20_bp"].to_numpy()),
      "model_curv_rms_1pm": 1e4 * rms(sub["model_curv_bp"].to_numpy()),
      "lane_center_y20_rms_m": rms(sub["lane_center_y20_bp"].to_numpy()),
      "lane_good_frac": float(np.mean((sub["lane_prob_left"] > 0.5) & (sub["lane_prob_right"] > 0.5) & (sub["lane_width_y20"].between(2.4, 4.6)))),
      "steer_peak_freq_hz": f_peak,
    }
    rows.append(row)

  return pd.DataFrame(rows)


def summarize_windows(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  drive_rows = []
  for (setting, drive_id), g in windows.groupby(["setting", "drive_id"]):
    row = {
      "setting": setting,
      "drive_id": drive_id,
      "config_id": g["config_id"].iloc[0],
      "driving_model": g["driving_model"].iloc[0],
      "controller_set": g["controller_set"].iloc[0],
      "n_windows": len(g),
      "eligible_min": g["eligible_s"].sum() / 60.0,
      "speed_median_mph": np.nanmedian(g["speed_median_mps"]) * 2.236936,
      "road_curv_abs_median": np.nanmedian(g["road_curv_abs_median"]),
    }
    for metric in PRIMARY_METRICS + ["steer_pp_sine_deg", "lat_disp_pp_sine_cm", "steer_peak_freq_hz"]:
      row[metric] = np.nanmedian(g[metric])
    drive_rows.append(row)
  drive_summary = pd.DataFrame(drive_rows)

  config_rows = []
  for (setting, config_id), g in drive_summary.groupby(["setting", "config_id"]):
    row = {
      "setting": setting,
      "config_id": config_id,
      "driving_model": g["driving_model"].iloc[0],
      "controller_set": g["controller_set"].iloc[0],
      "n_drives": g["drive_id"].nunique(),
      "n_windows": int(g["n_windows"].sum()),
      "eligible_min": float(g["eligible_min"].sum()),
      "speed_median_mph": float(np.nanmedian(g["speed_median_mph"])),
      "road_curv_abs_median": float(np.nanmedian(g["road_curv_abs_median"])),
    }
    for metric in PRIMARY_METRICS + ["steer_pp_sine_deg", "lat_disp_pp_sine_cm", "steer_peak_freq_hz"]:
      row[metric] = float(np.nanmedian(g[metric]))
      row[f"{metric}_drive_iqr"] = float(np.nanpercentile(g[metric], 75) - np.nanpercentile(g[metric], 25)) if len(g) > 1 else np.nan
    config_rows.append(row)
  config_summary = pd.DataFrame(config_rows)
  return drive_summary, config_summary


def bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float]:
  values = np.asarray(values, dtype=np.float64)
  values = values[np.isfinite(values)]
  if len(values) <= 1:
    return np.nan, np.nan
  boots = np.empty(n_boot)
  for i in range(n_boot):
    boots[i] = np.median(rng.choice(values, size=len(values), replace=True))
  return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def exact_permutation_p(values: pd.DataFrame, metric: str, a: str, b: str) -> float:
  sub = values[values["config_id"].isin([a, b])][["drive_id", "config_id", metric]].dropna()
  drive_vals = sub.drop_duplicates("drive_id")
  n_a = int((drive_vals["config_id"] == a).sum())
  n_b = int((drive_vals["config_id"] == b).sum())
  if n_a == 0 or n_b == 0 or len(drive_vals) > 12:
    return np.nan
  vals = drive_vals[metric].to_numpy()
  observed = np.median(vals[drive_vals["config_id"].to_numpy() == b]) - np.median(vals[drive_vals["config_id"].to_numpy() == a])
  diffs = []
  idxs = range(len(vals))
  for a_idx in itertools.combinations(idxs, n_a):
    mask_a = np.zeros(len(vals), dtype=bool)
    mask_a[list(a_idx)] = True
    diffs.append(np.median(vals[~mask_a]) - np.median(vals[mask_a]))
  diffs = np.asarray(diffs)
  return float(np.mean(np.abs(diffs) >= abs(observed) - 1e-12))


def compare_configs(drive_summary: pd.DataFrame, windows: pd.DataFrame, metric: str, setting: str, subset_label: str) -> pd.DataFrame:
  rng = np.random.default_rng(20260613)
  rows = []
  pairs = [("A", "B"), ("A", "C"), ("B", "C")]
  ds = drive_summary[drive_summary["setting"] == setting]
  for a, b in pairs:
    ga = ds[ds["config_id"] == a][metric].dropna().to_numpy()
    gb = ds[ds["config_id"] == b][metric].dropna().to_numpy()
    if len(ga) == 0 or len(gb) == 0:
      continue
    diff = float(np.median(gb) - np.median(ga))
    boots = []
    for _ in range(5000):
      ba = rng.choice(ga, size=len(ga), replace=True)
      bb = rng.choice(gb, size=len(gb), replace=True)
      boots.append(np.median(bb) - np.median(ba))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = exact_permutation_p(ds, metric, a, b)
    rows.append({
      "setting": setting,
      "subset": subset_label,
      "metric": metric,
      "comparison": f"{b}-minus-{a}",
      "config_low_reference": a,
      "config_high_reference": b,
      "a_n_drives": len(ga),
      "b_n_drives": len(gb),
      "a_median": float(np.median(ga)),
      "b_median": float(np.median(gb)),
      "diff": diff,
      "diff_pct_of_a": 100.0 * diff / float(np.median(ga)) if np.median(ga) != 0 else np.nan,
      "drive_boot_ci_low": float(lo),
      "drive_boot_ci_high": float(hi),
      "exact_drive_perm_p": p,
      "a_n_windows": int(windows[(windows["setting"] == setting) & (windows["config_id"] == a)].shape[0]),
      "b_n_windows": int(windows[(windows["setting"] == setting) & (windows["config_id"] == b)].shape[0]),
    })
  return pd.DataFrame(rows)


def add_match_strata(w: pd.DataFrame) -> pd.DataFrame:
  out = w.copy()
  x, y = latlon_to_xy(out["lat_median"].to_numpy(), out["lon_median"].to_numpy())
  out["x_m"] = x
  out["y_m"] = y
  out["speed_bin"] = np.floor(out["speed_median_mps"] / 2.5).astype("Int64")
  out["curv_bin"] = np.floor(out["road_curv_median"] / 0.0005).astype("Int64")
  out["x_cell_500m"] = np.floor(out["x_m"] / 500.0).astype("Int64")
  out["y_cell_500m"] = np.floor(out["y_m"] / 500.0).astype("Int64")
  out["speed_curve_stratum"] = out["speed_bin"].astype(str) + "|" + out["curv_bin"].astype(str)
  out["gps_speed_curve_stratum"] = (
    out["x_cell_500m"].astype(str) + "|" +
    out["y_cell_500m"].astype(str) + "|" +
    out["speed_bin"].astype(str) + "|" +
    out["curv_bin"].astype(str)
  )
  return out


def matched_subset(windows: pd.DataFrame, setting: str, a: str, b: str, stratum_col: str) -> pd.DataFrame:
  sub = windows[(windows["setting"] == setting) & (windows["config_id"].isin([a, b]))].copy()
  if sub.empty:
    return sub
  counts = sub.groupby([stratum_col, "config_id"]).size().unstack(fill_value=0)
  if a not in counts.columns or b not in counts.columns:
    return sub.iloc[0:0]
  common = counts[(counts[a] > 0) & (counts[b] > 0)].index
  return sub[sub[stratum_col].isin(common)]


def comparison_tables(windows: pd.DataFrame, drive_summary: pd.DataFrame) -> pd.DataFrame:
  all_rows = []
  for setting in sorted(windows["setting"].unique()):
    for metric in ["steer_rms_deg", "actual_curv_rms_1pm", "lat_accel_rms_mps2", "cmd_curv_rms_1pm", "desired_curv_rms_1pm"]:
      all_rows.append(compare_configs(drive_summary, windows, metric, setting, "all_eligible"))

  windows = add_match_strata(windows)
  for setting in sorted(windows["setting"].unique()):
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
      for stratum_col, label in [
        ("speed_curve_stratum", "speed_curve_matched"),
        ("gps_speed_curve_stratum", "gps_speed_curve_matched"),
      ]:
        sub = matched_subset(windows, setting, a, b, stratum_col)
        if sub.empty:
          continue
        ds, _cfg = summarize_windows(sub)
        for metric in ["steer_rms_deg", "actual_curv_rms_1pm", "lat_accel_rms_mps2", "cmd_curv_rms_1pm", "desired_curv_rms_1pm"]:
          c = compare_configs(ds, sub, metric, setting, label)
          c = c[c["comparison"] == f"{b}-minus-{a}"]
          c["matched_strata"] = sub[stratum_col].nunique()
          all_rows.append(c)

  return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
  values = np.asarray(values, dtype=np.float64)
  weights = np.asarray(weights, dtype=np.float64)
  ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
  if ok.sum() == 0:
    return np.nan
  values = values[ok]
  weights = weights[ok]
  order = np.argsort(values)
  values = values[order]
  weights = weights[order]
  cdf = np.cumsum(weights) / np.sum(weights)
  return float(np.interp(q, cdf, values))


def matched_stratum_differences(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  rng = np.random.default_rng(20260613)
  detail_rows = []
  summary_rows = []
  windows = add_match_strata(windows)
  metrics = ["steer_rms_deg", "actual_curv_rms_1pm", "lat_accel_rms_mps2", "cmd_curv_rms_1pm", "desired_curv_rms_1pm"]

  for setting in sorted(windows["setting"].unique()):
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
      for stratum_col, label in [
        ("speed_curve_stratum", "speed_curve_matched"),
        ("gps_speed_curve_stratum", "gps_speed_curve_matched"),
      ]:
        sub = matched_subset(windows, setting, a, b, stratum_col)
        if sub.empty:
          continue
        for metric in metrics:
          diffs = []
          weights = []
          for stratum, sg in sub.groupby(stratum_col):
            va = sg[sg["config_id"] == a][metric].dropna().to_numpy()
            vb = sg[sg["config_id"] == b][metric].dropna().to_numpy()
            if len(va) == 0 or len(vb) == 0:
              continue
            diff = float(np.median(vb) - np.median(va))
            weight = float(min(len(va), len(vb)))
            diffs.append(diff)
            weights.append(weight)
            detail_rows.append({
              "setting": setting,
              "subset": label,
              "metric": metric,
              "comparison": f"{b}-minus-{a}",
              "stratum": stratum,
              "a_n_windows": len(va),
              "b_n_windows": len(vb),
              "weight": weight,
              "a_median": float(np.median(va)),
              "b_median": float(np.median(vb)),
              "diff": diff,
            })
          if not diffs:
            continue
          diffs_a = np.asarray(diffs, dtype=np.float64)
          weights_a = np.asarray(weights, dtype=np.float64)
          weighted_mean = float(np.average(diffs_a, weights=weights_a))
          weighted_med = weighted_quantile(diffs_a, weights_a, 0.5)
          boots = []
          if len(diffs_a) > 1:
            for _ in range(5000):
              idx = rng.choice(np.arange(len(diffs_a)), size=len(diffs_a), replace=True)
              boots.append(np.average(diffs_a[idx], weights=weights_a[idx]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
          else:
            lo = hi = np.nan
          summary_rows.append({
            "setting": setting,
            "subset": label,
            "metric": metric,
            "comparison": f"{b}-minus-{a}",
            "n_strata": len(diffs_a),
            "a_n_windows": int(sub[sub["config_id"] == a].shape[0]),
            "b_n_windows": int(sub[sub["config_id"] == b].shape[0]),
            "weighted_mean_diff": weighted_mean,
            "weighted_median_diff": weighted_med,
            "stratum_boot_ci_low": float(lo),
            "stratum_boot_ci_high": float(hi),
            "share_strata_b_less_than_a": float(np.mean(diffs_a < 0.0)),
          })

  return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def comparability_summary(windows: pd.DataFrame) -> pd.DataFrame:
  rows = []
  primary = add_match_strata(windows[windows["setting"] == "primary"].copy())
  for config, g in primary.groupby("config_id"):
    rows.append({
      "config_id": config,
      "n_drives": g["drive_id"].nunique(),
      "n_windows": len(g),
      "eligible_min": g["eligible_s"].sum() / 60.0,
      "speed_median_mph": np.nanmedian(g["speed_median_mps"]) * 2.236936,
      "speed_p25_mph": np.nanpercentile(g["speed_median_mps"], 25) * 2.236936,
      "speed_p75_mph": np.nanpercentile(g["speed_median_mps"], 75) * 2.236936,
      "road_curv_abs_median": np.nanmedian(g["road_curv_abs_median"]),
      "gps_cells_500m": g[["x_cell_500m", "y_cell_500m"]].drop_duplicates().shape[0],
    })

  for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
    for stratum_col, label in [
      ("speed_curve_stratum", "speed_curve_matched"),
      ("gps_speed_curve_stratum", "gps_speed_curve_matched"),
    ]:
      sub = matched_subset(primary, "primary", a, b, stratum_col)
      rows.append({
        "config_id": f"{a}-{b} {label}",
        "n_drives": sub["drive_id"].nunique(),
        "n_windows": len(sub),
        "eligible_min": sub["eligible_s"].sum() / 60.0,
        "speed_median_mph": np.nanmedian(sub["speed_median_mps"]) * 2.236936 if len(sub) else np.nan,
        "speed_p25_mph": np.nanpercentile(sub["speed_median_mps"], 25) * 2.236936 if len(sub) else np.nan,
        "speed_p75_mph": np.nanpercentile(sub["speed_median_mps"], 75) * 2.236936 if len(sub) else np.nan,
        "road_curv_abs_median": np.nanmedian(sub["road_curv_abs_median"]) if len(sub) else np.nan,
        "gps_cells_500m": sub[["x_cell_500m", "y_cell_500m"]].drop_duplicates().shape[0] if len(sub) else 0,
      })
  return pd.DataFrame(rows)


def make_figures(windows: pd.DataFrame, drive_summary: pd.DataFrame) -> None:
  FIGURES.mkdir(parents=True, exist_ok=True)
  primary = windows[windows["setting"] == "primary"].copy()
  order = ["A", "B", "C"]

  plt.figure(figsize=(8, 4.8))
  data = [primary[primary["config_id"] == c]["speed_median_mps"] * 2.236936 for c in order]
  plt.boxplot(data, tick_labels=order, showfliers=False)
  plt.ylabel("30 s window median speed (mph)")
  plt.title("Primary eligible-window speed distributions")
  plt.tight_layout()
  plt.savefig(FIGURES / "primary_speed_by_config.png", dpi=160)
  plt.close()

  plt.figure(figsize=(8, 4.8))
  metric = "steer_rms_deg"
  for i, c in enumerate(order, 1):
    vals = drive_summary[(drive_summary["setting"] == "primary") & (drive_summary["config_id"] == c)][metric]
    plt.scatter(np.full(len(vals), i), vals, s=80, label=c)
  plt.xticks([1, 2, 3], order)
  plt.ylabel("Drive-median steering weave RMS (deg)")
  plt.title("Primary low-frequency steering metric by drive")
  plt.tight_layout()
  plt.savefig(FIGURES / "primary_steer_metric_by_drive.png", dpi=160)
  plt.close()

  plt.figure(figsize=(8, 4.8))
  metric = "actual_curv_rms_1pm"
  for i, c in enumerate(order, 1):
    vals = drive_summary[(drive_summary["setting"] == "primary") & (drive_summary["config_id"] == c)][metric]
    plt.scatter(np.full(len(vals), i), vals, s=80, label=c)
  plt.xticks([1, 2, 3], order)
  plt.ylabel("Drive-median actual curvature weave RMS (1e-4 1/m)")
  plt.title("Primary low-frequency path-curvature metric by drive")
  plt.tight_layout()
  plt.savefig(FIGURES / "primary_actual_curvature_metric_by_drive.png", dpi=160)
  plt.close()


def write_requirements() -> None:
  req = "\n".join([
    "matplotlib==3.11.0",
    "numpy==2.4.6",
    "pandas==3.0.3",
    "pycapnp==2.2.3",
    "scipy==1.17.1",
    "zstandard==0.25.0",
    "",
  ])
  (ROOT / "analysis" / "requirements.txt").write_text(req)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--settings", nargs="*", default=[s.name for s in SETTINGS], help="Settings to run.")
  args = parser.parse_args()

  RESULTS.mkdir(parents=True, exist_ok=True)
  metadata = load_metadata()
  selected = {s.name: s for s in SETTINGS if s.name in set(args.settings)}
  all_windows = []
  grid_summaries = []

  for setting in selected.values():
    for row in metadata.to_dict(orient="records"):
      drive_id = row["drive_id"]
      print(f"{setting.name}: {drive_id}")
      grid = load_drive_grid(drive_id)
      grid = add_derived_signals(grid, setting)
      windows = window_metrics(grid, row, setting)
      all_windows.append(windows)
      grid_summaries.append({
        "setting": setting.name,
        "drive_id": drive_id,
        "grid_duration_min": (grid["t_rel"].iloc[-1] - grid["t_rel"].iloc[0]) / 60.0,
        "lat_active_min": grid["lat_active"].sum() / FS / 60.0,
        "primary_eligible_min": grid["eligible"].sum() / FS / 60.0,
        "n_windows": len(windows),
      })

  windows = pd.concat(all_windows, ignore_index=True)
  windows = add_match_strata(windows)
  drive_summary, config_summary = summarize_windows(windows)
  comparisons = comparison_tables(windows, drive_summary)
  matched_detail, matched_summary = matched_stratum_differences(windows)
  comparability = comparability_summary(windows)

  windows.to_csv(RESULTS / "window_metrics.csv", index=False)
  pd.DataFrame(grid_summaries).to_csv(RESULTS / "grid_drive_summary.csv", index=False)
  drive_summary.to_csv(RESULTS / "drive_summary.csv", index=False)
  config_summary.to_csv(RESULTS / "config_summary.csv", index=False)
  comparisons.to_csv(RESULTS / "comparisons.csv", index=False)
  matched_detail.to_csv(RESULTS / "matched_strata_detail.csv", index=False)
  matched_summary.to_csv(RESULTS / "matched_strata_summary.csv", index=False)
  comparability.to_csv(RESULTS / "comparability.csv", index=False)

  make_figures(windows, drive_summary)
  write_requirements()

  manifest = {
    "fs_hz": FS,
    "window_s": 30.0,
    "min_eligible_s_per_window": 20.0,
    "settings": [s.__dict__ for s in selected.values()],
    "primary_metrics": PRIMARY_METRICS,
  }
  (RESULTS / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"wrote {RESULTS}")


if __name__ == "__main__":
  main()
