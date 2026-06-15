import pandas as pd

from retrospective_lateral.code.qa import audit_scorecard_claims


def test_audit_rejects_single_drive_supported_claim():
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": "location_matched", "n_a": 1, "n_b": 1, "claim": "supported"},
  ])
  findings = audit_scorecard_claims(scorecard)
  assert findings[0]["severity"] == "blocker"
  assert "single-drive" in findings[0]["message"]


def test_audit_accepts_conservative_claim():
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": "location_matched", "n_a": 3, "n_b": 3, "claim": "suggestive"},
  ])
  assert audit_scorecard_claims(scorecard) == []
