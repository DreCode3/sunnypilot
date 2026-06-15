import pandas as pd

from retrospective_lateral.code.compare import evidence_tier, location_speed_scorecard


def test_evidence_tier_prefers_location_matched():
  assert evidence_tier(has_location=True, has_speed=True, same_corridor=False) == "location_matched"
  assert evidence_tier(has_location=True, has_speed=True, same_corridor=True) == "same_corridor_transition"
  assert evidence_tier(has_location=False, has_speed=True, same_corridor=False) == "speed_matched"


def test_location_speed_scorecard_uses_matched_cells():
  df = pd.DataFrame([
    {"route_id": "r1", "config": "A", "cell": 1, "heading_bin": 2, "speed_bin": 20, "path_curvature_band_rms_1e4": 2.0, "steering_band_rms_deg": 0.6},
    {"route_id": "r2", "config": "B", "cell": 1, "heading_bin": 2, "speed_bin": 20, "path_curvature_band_rms_1e4": 1.0, "steering_band_rms_deg": 0.3},
    {"route_id": "r3", "config": "A", "cell": 2, "heading_bin": 2, "speed_bin": 20, "path_curvature_band_rms_1e4": 9.0, "steering_band_rms_deg": 9.0},
  ])
  rows = location_speed_scorecard(df, config_a="A", config_b="B")
  assert len(rows) == 1
  row = rows[0]
  assert row["evidence_tier"] == "location_matched"
  assert row["path_effect_pct"] == -50.0
  assert row["steer_effect_pct"] == -50.0
