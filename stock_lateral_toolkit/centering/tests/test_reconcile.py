import pytest


def test_verdict_pass_and_decomposition():
    from stock_lateral_toolkit.centering import reconcile as R
    v = R.verdict(m1_median_cam=0.02, m2_sp002_vs_m1=0.04, m2_sp002_median_own=0.10,
                  logged_median=0.103, consensus_pairs_max_abs=0.06)
    assert v["gate_m1_vs_m2"] == "PASS"          # 0.04 <= 0.06
    assert v["gate_m2_vs_logged"] == "PASS"      # |0.10-0.103| <= 0.04
    assert v["proceed_to_R"] is True
    # decomposition: L = P_cam + dmid  ->  dmid = 0.103 - 0.02
    assert v["dmid_logged_minus_video_m"] == pytest.approx(0.083)
    assert v["consensus_flag"] == "AGREE"


def test_verdict_stop_on_method_disagreement():
    from stock_lateral_toolkit.centering import reconcile as R
    v = R.verdict(m1_median_cam=0.02, m2_sp002_vs_m1=0.09, m2_sp002_median_own=0.10,
                  logged_median=0.103, consensus_pairs_max_abs=0.15)
    assert v["gate_m1_vs_m2"] == "FAIL"
    assert v["proceed_to_R"] is False
    assert v["consensus_flag"] == "DEFINITIONS_DIFFER"
