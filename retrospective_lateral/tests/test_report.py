import pandas as pd

from retrospective_lateral.code.report import write_markdown_report


def test_write_markdown_report_includes_evidence_language(tmp_path):
  symptom = pd.DataFrame([
    {"symptom": "weave_10_70", "route_id": "route_x", "path_curvature_band_rms_1e4": 3.0, "stage_first_growth": "controller_or_command"},
  ])
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": "location_matched", "path_effect_pct": -20.0, "steer_effect_pct": -10.0},
  ])
  path = write_markdown_report(tmp_path, symptom, scorecard)
  text = path.read_text()
  assert "# Retrospective Lateral Weave Analysis Report" in text
  assert "weave_10_70" in text
  assert "location_matched" in text
  assert "controlled_drive_needed" in text
