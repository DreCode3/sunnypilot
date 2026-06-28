import glob
import numpy as np
import pytest
from model_replay_sim.alignment import build_frame_timeline
from model_replay_sim.parse import logged_action_curvature


def _route():
    return "route_b5" if glob.glob("explorer_st_logs/route_b5/000000b5--*--16/rlog.zst") else None


@pytest.mark.skipif(_route() is None, reason="no route_b5 rlogs")
def test_logged_action_curvature_returns_finite_plausible_values():
    tl = build_frame_timeline("route_b5")
    # frame-present segments 16-18; pick a handful of real frame_ids
    fids = [r.frame_id for r in tl if r.segment_num in (16, 17, 18)]
    assert len(fids) >= 10
    sample = fids[::400][:6]                       # spread across segments
    out = logged_action_curvature("route_b5", sample)
    assert isinstance(out, np.ndarray)
    assert out.shape == (len(sample),)
    assert np.all(np.isfinite(out)), out
    assert np.all(np.abs(out) < 0.05), out         # plausible curvature magnitude


@pytest.mark.skipif(_route() is None, reason="no route_b5 rlogs")
def test_logged_action_curvature_matches_known_value():
    # verified ground truth on this route: frameId 19202 -> -0.0001947...
    out = logged_action_curvature("route_b5", [19202])
    assert out.shape == (1,)
    assert np.isfinite(out[0])
    assert abs(out[0] - (-0.0001947204873431474)) < 1e-9


@pytest.mark.skipif(_route() is None, reason="no route_b5 rlogs")
def test_logged_action_curvature_preserves_order():
    out_fwd = logged_action_curvature("route_b5", [19202, 19300, 20401])
    out_rev = logged_action_curvature("route_b5", [20401, 19300, 19202])
    assert np.allclose(out_fwd, out_rev[::-1], equal_nan=True)


@pytest.mark.skipif(_route() is None, reason="no route_b5 rlogs")
def test_bogus_frame_id_returns_nan_without_dropping():
    # missing frame_id must yield nan in place, not be silently dropped
    out = logged_action_curvature("route_b5", [19202, 10**9, 19300])
    assert out.shape == (3,)
    assert np.isfinite(out[0]) and np.isfinite(out[2])
    assert np.isnan(out[1])
