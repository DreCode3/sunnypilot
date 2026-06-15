from __future__ import annotations

import math

import pandas as pd


def evidence_tier(*, has_location: bool, has_speed: bool, same_corridor: bool) -> str:
  if same_corridor and has_location and has_speed:
    return "same_corridor_transition"
  if has_location and has_speed:
    return "location_matched"
  if has_speed:
    return "speed_matched"
  return "descriptive"


def _pct(new: float, base: float) -> float:
  if not math.isfinite(new) or not math.isfinite(base) or base == 0:
    return math.nan
  return float(100.0 * (new - base) / base)


def location_speed_scorecard(df: pd.DataFrame, *, config_a: str, config_b: str) -> list[dict[str, object]]:
  needed = {"config", "cell", "heading_bin", "speed_bin", "path_curvature_band_rms_1e4", "steering_band_rms_deg"}
  missing = needed - set(df.columns)
  if missing:
    raise ValueError(f"missing columns: {sorted(missing)}")
  rows: list[dict[str, object]] = []
  strata = ["cell", "heading_bin", "speed_bin"]
  for key, group in df.groupby(strata, dropna=True):
    a = group[group["config"] == config_a]
    b = group[group["config"] == config_b]
    if a.empty or b.empty:
      continue
    a_path = float(a["path_curvature_band_rms_1e4"].median())
    b_path = float(b["path_curvature_band_rms_1e4"].median())
    a_steer = float(a["steering_band_rms_deg"].median())
    b_steer = float(b["steering_band_rms_deg"].median())
    rows.append({
      "comparison": f"{config_b}_minus_{config_a}",
      "stratum": "|".join(str(x) for x in key),
      "config_a": config_a,
      "config_b": config_b,
      "n_a": int(len(a)),
      "n_b": int(len(b)),
      "evidence_tier": evidence_tier(has_location=True, has_speed=True, same_corridor=False),
      "path_a": a_path,
      "path_b": b_path,
      "path_effect_pct": _pct(b_path, a_path),
      "steer_a": a_steer,
      "steer_b": b_steer,
      "steer_effect_pct": _pct(b_steer, a_steer),
    })
  return rows
