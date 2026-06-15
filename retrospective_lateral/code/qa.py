from __future__ import annotations

import math

import pandas as pd


def _normalize_token(value: object) -> str:
  return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _safe_count(value: object) -> int:
  try:
    count = float(value)
  except (TypeError, ValueError):
    return 0
  if not math.isfinite(count):
    return 0
  return int(count)


def audit_scorecard_claims(scorecard: pd.DataFrame) -> list[dict[str, str]]:
  findings: list[dict[str, str]] = []
  if scorecard.empty:
    return findings
  metric_count_fields = {"path_n_a", "path_n_b", "steer_n_a", "steer_n_b"}
  has_metric_counts = bool(metric_count_fields.intersection(scorecard.columns))
  for idx, row in scorecard.iterrows():
    claim = _normalize_token(row.get("claim", ""))
    if claim in {"supported", "refuted", "root_cause_likely"}:
      tier = _normalize_token(row.get("evidence_tier", ""))
      if has_metric_counts:
        has_valid_counts = (
          (_safe_count(row.get("path_n_a", 0)) >= 2 and _safe_count(row.get("path_n_b", 0)) >= 2)
          or (_safe_count(row.get("steer_n_a", 0)) >= 2 and _safe_count(row.get("steer_n_b", 0)) >= 2)
        )
        if not has_valid_counts:
          findings.append({
            "severity": "blocker",
            "row": str(idx),
            "message": f"strong claim requires valid metric evidence for {row.get('comparison', '')}",
          })
      elif _safe_count(row.get("n_a", 0)) < 2 or _safe_count(row.get("n_b", 0)) < 2:
        findings.append({
          "severity": "blocker",
          "row": str(idx),
          "message": f"single-drive claim is not allowed for {row.get('comparison', '')}",
        })
      if tier not in {"location_matched", "same_corridor_transition"}:
        findings.append({
          "severity": "blocker",
          "row": str(idx),
          "message": f"strong claim requires location-matched evidence for {row.get('comparison', '')}",
        })
  return findings
