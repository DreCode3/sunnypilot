import pytest
from model_replay_sim.assets import anchor_candidates, require_same_model_anchor_asset, require_sanity_asset

def test_cd210_route_b5_is_a_usable_strict_anchor():     # FIX 4 — the window must now be found
    c = next(x for x in anchor_candidates("CD210") if x.route_id == "route_b5")
    assert c.local_fcamera_segments >= 1
    assert c.npz_exists is True
    assert c.active_bundle_match is True                  # route_b5 ran CD210
    assert c.eligible_aligned_windows_20s >= 1            # the 68.6s+45.9s windows are now extractable
    got = require_same_model_anchor_asset("CD210")
    assert got.route_id == "route_b5"

def test_nevada_strict_anchor_fails_needs_frame_pull():
    for c in anchor_candidates("Nevada"):
        assert c.local_fcamera_segments == 0 and c.needs_frame_pull is True
    with pytest.raises(AssertionError, match="missing local fcamera"):
        require_same_model_anchor_asset("Nevada")

def test_opm7_strict_anchor_fails_no_provenance():
    with pytest.raises(AssertionError, match="active-bundle provenance"):
        require_same_model_anchor_asset("OPM7")

def test_cd210_sanity_asset_available():
    c = require_sanity_asset("CD210")
    assert c.local_fcamera_segments > 0
