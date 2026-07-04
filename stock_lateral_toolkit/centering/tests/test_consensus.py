import numpy as np
import pytest


def test_lane_center_series_x0_and_interp():
    from stock_lateral_toolkit.centering import consensus as CO
    n = 7
    ll = np.zeros((n, 4, 33, 2), dtype=np.float32)
    ll[:, 1, :, 0] = -1.6   # inner-left y (calibrated +right)
    ll[:, 2, :, 0] = +1.8   # inner-right
    center0, width0 = CO.lane_center_series(ll, x_eval=0.0)
    assert center0.shape == (n,)
    assert center0[0] == pytest.approx(0.1)
    assert width0[0] == pytest.approx(3.4)
    # linear-in-x lane lines: interp at x=10 must match exactly
    from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
    xg = np.asarray(ModelConstants.X_IDXS)
    ll2 = ll.copy()
    ll2[:, 1, :, 0] = -1.6 + 0.01 * xg
    ll2[:, 2, :, 0] = +1.8 + 0.01 * xg
    c10, _ = CO.lane_center_series(ll2, x_eval=10.0)
    assert c10[0] == pytest.approx(0.1 + 0.01 * 10.0, abs=1e-6)


def test_pair_stats():
    from stock_lateral_toolkit.centering import consensus as CO
    a = np.array([0.10, 0.11, 0.09, 0.10])
    b = np.array([0.02, 0.03, 0.01, 0.02])
    s = CO.pair_stats(a, b)
    assert s["median_delta_m"] == pytest.approx(0.08)
    assert s["n"] == 4
