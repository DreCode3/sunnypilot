import math

import numpy as np

from retrospective_lateral.code import model_vs_loop as M


def _route(weave_dis_fraction: float):
    """60 s, 20 Hz, straight road at ~29 mph. First 30 s engaged, last 30 s disengaged.
    The model path weaves at 0.2 Hz with full amplitude engaged and weave_dis_fraction
    of that amplitude disengaged (1.0 => A-like intrinsic, ~0.0 => C-like closed-loop)."""
    fs = 20.0
    t = np.arange(0, 60, 1 / fs)
    n = len(t)
    v = np.full(n, 13.0)  # ~29 mph -> speed bin lo=25.0
    la = np.where(t < 30, 1.0, 0.0)
    amp = np.where(t < 30, 0.30, weave_dis_fraction * 0.30)
    model_y20 = amp * np.sin(2 * np.pi * 0.2 * t)
    return {
        "t": t.astype(np.float32),
        "v_ego": v.astype(np.float32),
        "yaw_rate": np.zeros(n, dtype=np.float32),          # straight road
        "lat_active": la.astype(np.float32),
        "model_y20": model_y20.astype(np.float32),
        "lane_center_y20": np.zeros(n, dtype=np.float32),
        "desired_curvature": (2.0 * model_y20 / (20.0 ** 2)).astype(np.float32),
        "orientation_rate_z0": np.zeros(n, dtype=np.float32),
    }


def test_weave_by_state_reports_both_engaged_and_disengaged():
    out = M.weave_by_state(_route(1.0))
    assert ("model_y20", 25.0, "eng") in out
    assert ("model_y20", 25.0, "dis") in out
    # equal-amplitude weave in both states -> similar RMS
    assert out[("model_y20", 25.0, "dis")] > 0
    assert abs(out[("model_y20", 25.0, "dis")] / out[("model_y20", 25.0, "eng")] - 1.0) < 0.2


def test_weave_by_state_empty_without_channels():
    assert M.weave_by_state({"v_ego": np.zeros(5)}) == {}


def test_classify_a_vs_c_thresholds():
    assert M.classify_a_vs_c({"model_y20": 1.5, "desired_curvature": 1.6, "model_minus_lane": 1.5})[0] == "model_artifact_A"
    assert M.classify_a_vs_c({"model_y20": 0.1, "desired_curvature": 0.2, "model_minus_lane": 0.15})[0] == "loop_limit_cycle_C"
    assert M.classify_a_vs_c({"model_y20": 0.5, "desired_curvature": 0.5, "model_minus_lane": 0.5})[0] == "ambiguous"
    assert M.classify_a_vs_c({})[0] == "insufficient_evidence"
    # verdict ignores non-native signals (e.g. orientation_rate_curv contaminated disengaged)
    assert M.classify_a_vs_c({"orientation_rate_curv": 0.1})[0] == "insufficient_evidence"


def test_build_model_vs_loop_parallel_smoke(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in ("route_x1", "route_x2"):
        np.savez_compressed(cache / f"{name}.npz", **_route(1.0))
    reports = tmp_path / "reports"
    result = M.build_model_vs_loop(cache, reports, pi_set=None, workers=1)
    assert result["routes"] == 2
    assert result["verdict"] == "model_artifact_A"   # weave present open-loop
    assert (reports / "model_vs_loop_signals.csv").exists()
