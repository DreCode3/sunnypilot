from __future__ import annotations

import pandas as pd


def audit_scorecard_claims(scorecard: pd.DataFrame) -> list[dict[str, str]]:
  findings: list[dict[str, str]] = []
  if scorecard.empty:
    return findings
  for idx, row in scorecard.iterrows():
    claim = str(row.get("claim", "")).lower()
    if claim in {"supported", "refuted", "root-cause likely", "root_cause_likely"}:
      n_a = int(row.get("n_a", 0))
      n_b = int(row.get("n_b", 0))
      tier = str(row.get("evidence_tier", ""))
      if n_a < 2 or n_b < 2:
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
