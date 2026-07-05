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


def test_audit_normalizes_strong_claim_whitespace():
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": "location_matched", "n_a": 1, "n_b": 1, "claim": " Supported "},
  ])
  findings = audit_scorecard_claims(scorecard)
  assert findings[0]["severity"] == "blocker"
  assert "single-drive" in findings[0]["message"]


def test_audit_normalizes_evidence_tier_whitespace():
  scorecard = pd.DataFrame([
    {"comparison": "B_minus_A", "evidence_tier": " Location_Matched ", "n_a": 3, "n_b": 3, "claim": "supported"},
  ])
  assert audit_scorecard_claims(scorecard) == []


def test_audit_rejects_strong_claim_without_valid_metric_evidence():
  scorecard = pd.DataFrame([
    {
      "comparison": "B_minus_A",
      "evidence_tier": "location_matched",
      "n_a": 3,
      "n_b": 3,
      "path_n_a": 0,
      "path_n_b": 3,
      "steer_n_a": 0,
      "steer_n_b": 3,
      "claim": "supported",
    },
  ])
  findings = audit_scorecard_claims(scorecard)
  assert findings[0]["severity"] == "blocker"
  assert "valid metric evidence" in findings[0]["message"]


def test_audit_handles_invalid_counts_without_crashing():
  scorecard = pd.DataFrame([
    {"comparison": "nan_counts", "evidence_tier": "location_matched", "n_a": float("nan"), "n_b": 3, "claim": "supported"},
    {"comparison": "missing_counts", "evidence_tier": "location_matched", "claim": "supported"},
    {"comparison": "malformed_counts", "evidence_tier": "location_matched", "n_a": "bad", "n_b": 3, "claim": "supported"},
  ])
  findings = audit_scorecard_claims(scorecard)
  assert len(findings) == 3
  assert all(finding["severity"] == "blocker" for finding in findings)
