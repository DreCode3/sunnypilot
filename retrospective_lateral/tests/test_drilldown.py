import json
from pathlib import Path

import numpy as np
import pandas as pd

import retrospective_lateral.code.drilldown as drilldown
from retrospective_lateral.code.drilldown import (
  build_drilldown_outputs,
  enrich_symptom_catalog,
  lane_geometry_audit,
  lead_event_alignment,
  location_speed_pi_comparisons,
  location_speed_context_summary,
  stage_metrics_for_episode,
  summarize_episode_stage,
)


def _synthetic_arrays(stage: str = "desired") -> dict[str, np.ndarray]:
  fs = 20.0
  t = np.arange(0.0, 60.0, 1.0 / fs)
  wave = np.sin(2.0 * np.pi * 0.18 * t)
  desired = 0.00030 * wave if stage in {"desired", "command"} else np.zeros_like(t)
  command = 0.00028 * wave if stage == "command" else desired.copy()
  yaw_rate = 18.0 * 0.00032 * wave
  lat = 34.0 + 0.00001 * t
  lon = -84.0 + 0.00002 * t
  out = {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 18.0, dtype=np.float32),
    "steering_angle_deg": (0.8 * wave).astype(np.float32),
    "yaw_rate": yaw_rate.astype(np.float32),
    "yaw_rate_calibrated": yaw_rate.astype(np.float32),
    "desired_curvature": desired.astype(np.float32),
    "act_curvature": command.astype(np.float32),
    "cp_desired_curvature": desired.astype(np.float32),
    "cp_final_command": command.astype(np.float32),
    "model_y20": (0.08 * wave).astype(np.float32),
    "lat": lat.astype(np.float32),
    "lon": lon.astype(np.float32),
    "lane_prob_left": np.full_like(t, 0.9, dtype=np.float32),
    "lane_prob_right": np.full_like(t, 0.85, dtype=np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
  }
  for x_m, factor in [(0, 0.25), (5, 0.45), (10, 0.60), (15, 0.80), (20, 1.00), (30, 1.25)]:
    out[f"model_y{x_m}"] = (factor * 0.08 * wave).astype(np.float32)
    out[f"lane_left_y{x_m}"] = (-1.8 + factor * 0.04 * wave).astype(np.float32)
    out[f"lane_right_y{x_m}"] = (1.8 + factor * 0.08 * wave).astype(np.float32)
    out[f"lane_center_y{x_m}"] = (0.5 * (out[f"lane_left_y{x_m}"] + out[f"lane_right_y{x_m}"])).astype(np.float32)
    out[f"lane_width_y{x_m}"] = np.abs(out[f"lane_right_y{x_m}"] - out[f"lane_left_y{x_m}"]).astype(np.float32)
    out[f"road_edge_left_y{x_m}"] = (-4.4 + factor * 0.03 * wave).astype(np.float32)
    out[f"road_edge_right_y{x_m}"] = (4.4 + factor * 0.07 * wave).astype(np.float32)
    out[f"road_edge_width_y{x_m}"] = np.abs(out[f"road_edge_right_y{x_m}"] - out[f"road_edge_left_y{x_m}"]).astype(np.float32)
  out["road_edge_std_left"] = np.full_like(t, 0.2, dtype=np.float32)
  out["road_edge_std_right"] = np.full_like(t, 0.3, dtype=np.float32)
  return out


def _synthetic_arrays_with_lead() -> dict[str, np.ndarray]:
  arrays = _synthetic_arrays()
  arrays["lead_prob"] = np.full_like(arrays["t"], 0.9, dtype=np.float32)
  arrays["lead_d_rel"] = np.full_like(arrays["t"], 18.0, dtype=np.float32)
  arrays["lead_time_headway_s"] = np.full_like(arrays["t"], 1.0, dtype=np.float32)
  arrays["radar_lead_one_status"] = np.ones_like(arrays["t"], dtype=np.float32)
  return arrays


def _synthetic_arrays_with_lead_transition() -> dict[str, np.ndarray]:
  fs = 20.0
  t = np.arange(0.0, 100.0, 1.0 / fs)
  near = (t >= 40.0) & (t < 70.0)
  wave = np.sin(2.0 * np.pi * 0.18 * t)
  amp = np.where(near, 0.00050, 0.00010)
  yaw_rate = 20.0 * amp * wave
  return {
    "t": t.astype(np.float32),
    "v_ego": np.full_like(t, 20.0, dtype=np.float32),
    "steering_angle_deg": (0.7 * wave).astype(np.float32),
    "yaw_rate": yaw_rate.astype(np.float32),
    "yaw_rate_calibrated": yaw_rate.astype(np.float32),
    "orientation_rate_z0": yaw_rate.astype(np.float32),
    "desired_curvature": (amp * wave).astype(np.float32),
    "act_curvature": (amp * wave).astype(np.float32),
    "cp_desired_curvature": (amp * wave).astype(np.float32),
    "cp_final_command": (amp * 0.90 * wave).astype(np.float32),
    "model_y20": (np.where(near, 0.10, 0.02) * wave).astype(np.float32),
    "lat": (34.0 + 0.00001 * t).astype(np.float32),
    "lon": (-84.0 + 0.00002 * t).astype(np.float32),
    "lane_prob_left": np.full_like(t, 0.92, dtype=np.float32),
    "lane_prob_right": np.full_like(t, 0.90, dtype=np.float32),
    "lead_prob": np.where(near, 0.95, 0.05).astype(np.float32),
    "lead_d_rel": np.where(near, 18.0, 90.0).astype(np.float32),
    "lead_time_headway_s": np.where(near, 1.0, 3.5).astype(np.float32),
    "radar_lead_one_status": near.astype(np.float32),
    "lat_active": np.ones_like(t, dtype=np.float32),
  }


def _catalog() -> pd.DataFrame:
  return pd.DataFrame([
    {
      "route_id": "route_synth",
      "symptom": "weave_10_70",
      "status": "ok",
      "start_s": 10.0,
      "end_s": 40.0,
      "speed_mph_median": 40.0,
      "path_curvature_band_rms_1e4": 2.0,
      "steering_band_rms_deg": 0.4,
      "stage_first_growth": "model_or_desired",
    }
  ])


def _manifest() -> pd.DataFrame:
  return pd.DataFrame([
    {
      "route_id": "route_synth",
      "success": True,
      "pi_set": "weak",
      "config_confidence": "proven",
      "schema_version": "retrolat-v2",
      "sample_count": 1200,
      "init.branch": "2021_explorer_st-mici",
      "init.commit": "abcdef123456",
      "init.dirty": False,
      "car_params.steerRatio": 17.2,
      "car_params.steerActuatorDelay": 0.25,
    }
  ])


def test_enrich_symptom_catalog_adds_context_and_provenance():
  enriched = enrich_symptom_catalog(_catalog(), _manifest(), {"route_synth": _synthetic_arrays()})

  row = enriched.iloc[0]
  assert row["pi_set"] == "weak"
  assert row["commit_short"] == "abcdef12"
  assert row["has_cp"] is True
  assert row["has_cx1"] is False
  assert row["lead_status"] == "not_extracted"
  assert row["gps_cell"] == row["gps_cell"]
  assert row["heading_bin"] >= 0
  assert row["speed_bin_mph"] == "40-45"
  assert row["lane_prob_min_median"] > 0.8


def test_enrich_symptom_catalog_classifies_near_lead_context():
  enriched = enrich_symptom_catalog(_catalog(), _manifest(), {"route_synth": _synthetic_arrays_with_lead()})

  row = enriched.iloc[0]
  assert row["lead_status"] == "lead_near"
  assert np.isclose(row["lead_prob_median"], 0.9)
  assert np.isclose(row["lead_d_rel_median"], 18.0)
  assert np.isclose(row["lead_time_headway_min_s"], 1.0)
  assert np.isclose(row["lead_near_fraction"], 1.0)


def test_stage_metrics_identify_model_and_desired_before_path():
  row = _catalog().iloc[0]
  stage_rows = stage_metrics_for_episode(row, _synthetic_arrays("desired"))
  names = {stage["stage_name"]: stage for stage in stage_rows}

  assert names["model_y20"]["stage_rms"] > 0.02
  assert names["desired_curvature"]["stage_rms_1e4"] > 1.0
  assert names["desired_curvature"]["corr_to_reference"] > 0.9
  summary = summarize_episode_stage(row, stage_rows)
  assert summary["first_supported_stage"] == "model_y20"
  assert summary["first_supported_family"] == "model_or_desired"


def test_location_speed_context_summary_groups_same_cell_speed_and_stage():
  enriched = enrich_symptom_catalog(_catalog(), _manifest(), {"route_synth": _synthetic_arrays()})
  summary = location_speed_context_summary(enriched)

  assert len(summary) == 1
  row = summary.iloc[0]
  assert row["symptom"] == "weave_10_70"
  assert row["rows"] == 1
  assert row["routes"] == 1
  assert row["mode_stage"] == "model_or_desired"


def test_location_speed_pi_comparisons_use_only_shared_contexts():
  rows = pd.DataFrame([
    {
      "symptom": "weave_10_70",
      "gps_cell": 1,
      "heading_bin": 2,
      "speed_bin_mph": "40-45",
      "route_id": "route_a",
      "pi_set": "weak",
      "path_curvature_band_rms_1e4": 4.0,
      "steering_band_rms_deg": 1.0,
    },
    {
      "symptom": "weave_10_70",
      "gps_cell": 1,
      "heading_bin": 2,
      "speed_bin_mph": "40-45",
      "route_id": "route_b",
      "pi_set": "golden",
      "path_curvature_band_rms_1e4": 2.0,
      "steering_band_rms_deg": 0.5,
    },
    {
      "symptom": "weave_10_70",
      "gps_cell": 9,
      "heading_bin": 2,
      "speed_bin_mph": "40-45",
      "route_id": "route_c",
      "pi_set": "weak",
      "path_curvature_band_rms_1e4": 99.0,
      "steering_band_rms_deg": 99.0,
    },
  ])

  comparisons = location_speed_pi_comparisons(rows, baseline="weak", candidate="golden")

  assert len(comparisons) == 1
  row = comparisons.iloc[0]
  assert row["comparison"] == "golden_minus_weak"
  assert row["path_effect_pct"] == -50.0
  assert row["steer_effect_pct"] == -50.0


def test_stage_gain_lag_summary_groups_overall_top_decile_and_lead_status():
  enriched = pd.DataFrame([
    {
      "route_id": "route_synth",
      "symptom": "weave_10_70",
      "start_s": 10.0,
      "end_s": 40.0,
      "path_curvature_band_rms_1e4": 1.0,
      "lead_status": "lead_far",
    },
    {
      "route_id": "route_synth",
      "symptom": "weave_10_70",
      "start_s": 45.0,
      "end_s": 55.0,
      "path_curvature_band_rms_1e4": 10.0,
      "lead_status": "lead_near",
    },
  ])
  stage_metrics = pd.DataFrame([
    {
      "route_id": "route_synth",
      "symptom": "weave_10_70",
      "start_s": 10.0,
      "end_s": 40.0,
      "stage_order": 2,
      "stage_name": "desired_curvature",
      "stage_family": "model_or_desired",
      "stage_description": "desiredCurvature",
      "stage_rms_1e4": 2.0,
      "stage_ptp_1e4": 4.0,
      "corr_to_reference": 0.80,
      "lag_s_positive_stage_leads_reference": 0.60,
      "finite_samples": 600,
    },
    {
      "route_id": "route_synth",
      "symptom": "weave_10_70",
      "start_s": 10.0,
      "end_s": 40.0,
      "stage_order": 13,
      "stage_name": "cp_final_command",
      "stage_family": "final_command",
      "stage_description": "CP final_command",
      "stage_rms_1e4": 1.5,
      "stage_ptp_1e4": 3.0,
      "corr_to_reference": 0.70,
      "lag_s_positive_stage_leads_reference": 0.50,
      "finite_samples": 600,
    },
    {
      "route_id": "route_synth",
      "symptom": "weave_10_70",
      "start_s": 45.0,
      "end_s": 55.0,
      "stage_order": 2,
      "stage_name": "desired_curvature",
      "stage_family": "model_or_desired",
      "stage_description": "desiredCurvature",
      "stage_rms_1e4": 8.0,
      "stage_ptp_1e4": 12.0,
      "corr_to_reference": 0.90,
      "lag_s_positive_stage_leads_reference": 0.40,
      "finite_samples": 200,
    },
  ])

  summary = drilldown.stage_gain_lag_summary(enriched, stage_metrics)

  overall = summary[
    (summary["subset"] == "all")
    & (summary["group_name"] == "overall")
    & (summary["stage_name"] == "desired_curvature")
  ].iloc[0]
  assert overall["episodes"] == 2
  assert overall["routes"] == 1
  assert overall["severity_metric"] == "path_curvature_band_rms_1e4"
  assert overall["median_stage_rms_1e4"] == 5.0
  assert overall["p90_stage_rms_1e4"] == 7.4
  assert np.isclose(overall["median_corr_to_reference"], 0.85)
  assert overall["median_lag_s"] == 0.5

  top_near = summary[
    (summary["subset"] == "top_decile")
    & (summary["group_name"] == "lead_status")
    & (summary["group_value"] == "lead_near")
    & (summary["stage_name"] == "desired_curvature")
  ].iloc[0]
  assert top_near["episodes"] == 1
  assert top_near["median_stage_rms_1e4"] == 8.0
  assert np.isclose(top_near["severity_threshold"], 9.1)


def test_lead_event_alignment_reports_onset_and_exit_stage_deltas():
  arrays = _synthetic_arrays_with_lead_transition()
  enriched = pd.DataFrame([
    {
      "route_id": "route_synth",
      "symptom": "weave_10_70",
      "status": "ok",
      "start_s": 35.0,
      "end_s": 65.0,
      "path_curvature_band_rms_1e4": 3.0,
    }
  ])

  events = lead_event_alignment(
    enriched,
    {"route_synth": arrays},
    window_s=10.0,
    min_run_s=5.0,
  )

  model_rows = events[events["stage_name"] == "model_y20"]
  assert set(model_rows["event_type"]) == {"onset", "exit"}
  onset = model_rows[model_rows["event_type"] == "onset"].iloc[0]
  assert np.isclose(onset["event_time_s"], 40.0)
  assert onset["post_stage_rms"] > onset["pre_stage_rms"] * 2.0
  assert onset["pre_lead_near_fraction"] == 0.0
  assert onset["post_lead_near_fraction"] == 1.0
  assert onset["context_match"] is True
  assert onset["speed_range_ok"] is True
  assert onset["analysis_eligible"] is True
  assert onset["overlapping_weave_rows"] == 1

  exit_row = model_rows[model_rows["event_type"] == "exit"].iloc[0]
  assert np.isclose(exit_row["event_time_s"], 70.0)
  assert exit_row["post_stage_rms"] < exit_row["pre_stage_rms"] * 0.75


def test_lane_geometry_audit_reports_per_lookahead_model_lane_coupling():
  arrays = _synthetic_arrays("command")
  enriched = enrich_symptom_catalog(_catalog(), _manifest(), {"route_synth": arrays})

  audit = lane_geometry_audit(enriched, {"route_synth": arrays})

  assert set(audit["lookahead_m"]) == {0, 5, 10, 15, 20, 30}
  row5 = audit[audit["lookahead_m"] == 5].iloc[0]
  assert row5["model_y_rms"] > 0.0
  assert row5["lane_center_y_rms"] > 0.0
  row15 = audit[audit["lookahead_m"] == 15].iloc[0]
  assert row15["model_lane_center_corr"] > 0.9
  row20 = audit[audit["lookahead_m"] == 20].iloc[0]
  assert row20["source_type"] == "episode"
  assert row20["slice"] == "episode"
  assert row20["model_y_rms"] > 0.02
  assert row20["lane_center_y_rms"] > 0.0
  assert row20["lane_width_y_ptp"] > 0.0
  assert row20["road_edge_width_y_ptp"] > 0.0
  assert row20["model_lane_center_corr"] > 0.9
  assert 0.0 < row20["lane_center_over_model_rms"] < 1.0
  assert row20["cp_final_over_desired_rms"] < 1.0


def test_build_drilldown_outputs_writes_expected_artifacts(tmp_path):
  cache = tmp_path / "cache"
  reports = tmp_path / "reports"
  cache.mkdir()
  reports.mkdir()
  arrays = _synthetic_arrays_with_lead_transition()
  np.savez_compressed(cache / "route_synth.npz", **arrays)
  (cache / "manifest.json").write_text(json.dumps([
    {
      "route_id": "route_synth",
      "success": True,
      "pi_set": "weak",
      "config_confidence": "proven",
      "schema_version": "retrolat-v2",
      "sample_count": 1200,
      "init": {"branch": "2021_explorer_st-mici", "commit": "abcdef123456", "dirty": False},
      "car_params": {"steerRatio": 17.2, "steerActuatorDelay": 0.25},
    }
  ]))
  _catalog().to_csv(reports / "symptom_catalog.csv", index=False)

  outputs = build_drilldown_outputs(cache, reports)

  assert outputs.enriched_catalog.exists()
  assert outputs.stage_metrics.exists()
  assert outputs.stage_gain_lag_summary.exists()
  assert outputs.lead_event_alignment.exists()
  assert outputs.lane_geometry_audit.exists()
  assert outputs.episode_stage_summary.exists()
  assert outputs.context_summary.exists()
  assert outputs.pi_comparisons.exists()
  assert outputs.robustness_summary.exists()
  stage_df = pd.read_csv(outputs.stage_metrics)
  assert {"route_id", "symptom", "stage_name", "stage_rms"}.issubset(stage_df.columns)
  gain_df = pd.read_csv(outputs.stage_gain_lag_summary)
  assert {"subset", "group_name", "group_value", "stage_name", "median_lag_s"}.issubset(gain_df.columns)
  lead_event_df = pd.read_csv(outputs.lead_event_alignment)
  assert {"event_type", "event_time_s", "stage_name", "post_minus_pre_rms"}.issubset(lead_event_df.columns)
  lane_geometry_df = pd.read_csv(outputs.lane_geometry_audit)
  assert {"lookahead_m", "model_lane_center_corr", "road_edge_width_y_ptp"}.issubset(lane_geometry_df.columns)
  robust_df = pd.read_csv(outputs.robustness_summary)
  assert "variant_first_supported_family" in set(robust_df["check"])
