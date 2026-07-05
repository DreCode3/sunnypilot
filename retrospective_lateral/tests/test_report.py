import pandas as pd

from retrospective_lateral.code.report import write_markdown_report


def _section(text, title):
  start = text.index(title)
  next_section = text.find("\n## ", start + len(title))
  if next_section == -1:
    return text[start:]
  return text[start:next_section]


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


def test_write_markdown_report_escapes_table_cells(tmp_path):
  symptom = pd.DataFrame([
    {"symptom": "weave_10_70", "status": "ok", "error_message": r"a|b\c" + "\nnext"},
  ])
  scorecard = pd.DataFrame()

  path = write_markdown_report(tmp_path, symptom, scorecard)
  top = _section(path.read_text(), "## Top Symptom Episodes")
  data_rows = [line for line in top.splitlines() if line.startswith("| weave_10_70")]

  assert len(data_rows) == 1
  assert r"a\|b\\c next" in data_rows[0]
  assert "\nnext" not in data_rows[0]
  assert data_rows[0].count("|") == 5


def test_write_markdown_report_splits_processing_failures(tmp_path):
  symptom = pd.DataFrame([
    {"symptom": "weave_10_70", "status": "ok", "route_id": "route_ok"},
    {"symptom": "route_error", "status": "failed", "route_id": "route_bad", "error_message": "segment missing"},
  ])
  scorecard = pd.DataFrame()

  path = write_markdown_report(tmp_path, symptom, scorecard)
  text = path.read_text()
  top = _section(text, "## Top Symptom Episodes")
  failures = _section(text, "## Processing/Data Quality Failures")

  assert "weave_10_70" in top
  assert "route_ok" in top
  assert "route_error" not in top
  assert "route_bad" not in top
  assert "route_error" in failures
  assert "route_bad" in failures
  assert "segment missing" in failures


def test_write_markdown_report_handles_empty_inputs(tmp_path):
  symptom = pd.DataFrame()
  scorecard = pd.DataFrame()

  path = write_markdown_report(tmp_path, symptom, scorecard)
  text = path.read_text()

  assert text.count("_No rows._") == 3
  assert "## Top Symptom Episodes" in text
  assert "## Processing/Data Quality Failures" in text
  assert "## Historical Scorecard" in text
