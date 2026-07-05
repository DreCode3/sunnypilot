import pytest
from model_replay_sim.pytest_helpers import require_real_asset
def test_require_real_asset_fails_by_default():
    with pytest.raises(AssertionError, match="missing required real asset"):
        require_real_asset(False, "route_c4 fcamera")
def test_require_real_asset_skips_in_unit_mode(monkeypatch):
    monkeypatch.setenv("MODEL_REPLAY_ALLOW_MISSING_ASSETS", "1")
    with pytest.raises(pytest.skip.Exception):     # FIX 5: SKIP (do not just return)
        require_real_asset(False, "route_c4 fcamera")
def test_require_real_asset_returns_when_present():
    require_real_asset(True, "x")  # no raise
