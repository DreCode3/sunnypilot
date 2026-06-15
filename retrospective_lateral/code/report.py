from __future__ import annotations

from pathlib import Path

import pandas as pd


def _top_rows(df: pd.DataFrame, n: int = 10) -> str:
  if df.empty:
    return "_No rows._"
  view = df.head(n).fillna("")
  columns = list(view.columns)
  header = "| " + " | ".join(columns) + " |"
  sep = "| " + " | ".join(["---"] * len(columns)) + " |"
  body = ["| " + " | ".join(str(row[col]) for col in columns) + " |" for _, row in view.iterrows()]
  return "\n".join([header, sep] + body)


def write_markdown_report(out_dir: str | Path, symptom_catalog: pd.DataFrame, scorecard: pd.DataFrame) -> Path:
  out = Path(out_dir)
  out.mkdir(parents=True, exist_ok=True)
  report = out / "retrospective_lateral_report.md"
  text = "\n".join([
    "# Retrospective Lateral Weave Analysis Report",
    "",
    "## Evidence Rules",
    "",
    "Claims use evidence tiers: descriptive, speed_matched, location_matched, same_corridor_transition, and controlled_drive_needed.",
    "No root-cause claim should be promoted without cooperative and adversarial QA.",
    "",
    "## Top Symptom Episodes",
    "",
    _top_rows(symptom_catalog, 10),
    "",
    "## Historical Scorecard",
    "",
    _top_rows(scorecard, 20),
    "",
    "## Next Experiment Guidance",
    "",
    "Rows that remain confounded after location and speed matching should be labeled controlled_drive_needed.",
    "",
  ])
  report.write_text(text)
  return report
