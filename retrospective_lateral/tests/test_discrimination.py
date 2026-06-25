import math
import warnings

import numpy as np
import pytest

from retrospective_lateral.code import discrimination as D


def test_offset_to_curvature_parabolic_relation():
    # A lateral offset of 0.5 m at 20 m lookahead implies curvature 2*y/L^2.
    assert math.isclose(D.offset_to_curvature(0.5, 20.0), 2 * 0.5 / (20.0 ** 2), rel_tol=1e-9)
    arr = D.offset_to_curvature(np.array([0.0, 0.2, np.nan]), 10.0)
    assert arr[0] == 0.0
    assert math.isclose(arr[1], 2 * 0.2 / 100.0, rel_tol=1e-9)
    assert np.isnan(arr[2])


def test_offset_to_curvature_rejects_nonpositive_lookahead():
    with pytest.raises(ValueError):
        D.offset_to_curvature(1.0, 0.0)


def test_gps_course_deg_cardinal_directions():
    # Heading north: lat increasing, lon constant -> ~0 deg. Heading east -> ~90 deg.
    n = 50
    lat_north = 34.0 + np.arange(n) * 1e-4
    lon_const = np.full(n, -84.0)
    course_n = D.gps_course_deg(lat_north, lon_const)
    assert abs(((np.nanmedian(course_n) + 180) % 360) - 180) < 5.0  # ~0 deg

    lat_const = np.full(n, 34.0)
    lon_east = -84.0 + np.arange(n) * 1e-4
    course_e = D.gps_course_deg(lat_const, lon_east)
    assert abs(np.nanmedian(course_e) - 90.0) < 5.0


def test_gps_course_deg_nan_gap_does_not_contaminate_far_samples():
    # An interior GPS gap must NaN the gap and its two immediate neighbors (np.gradient
    # poisons them), while samples far from the gap stay finite.
    n = 30
    lat = 34.0 + np.arange(n) * 1e-4
    lon = np.full(n, -84.0)
    gap = 15
    lat[gap] = np.nan
    course = D.gps_course_deg(lat, lon)
    assert np.isnan(course[gap])
    assert np.isnan(course[gap - 1]) and np.isnan(course[gap + 1])  # gradient-contaminated
    assert np.isfinite(course[5]) and np.isfinite(course[25])       # far away, unaffected


def test_gps_course_deg_all_nan_and_short_inputs_return_all_nan():
    course_all_nan = D.gps_course_deg(np.full(10, np.nan), np.full(10, np.nan))
    assert np.all(np.isnan(course_all_nan))
    course_len1 = D.gps_course_deg(np.array([34.0]), np.array([-84.0]))
    assert np.all(np.isnan(course_len1))


def test_path_curvature_from_rate_guards_low_speed():
    rate = np.array([0.1, 0.1, 0.1, -0.2])
    v = np.array([10.0, 1.0, np.nan, 10.0])  # 0.1/10 = 0.01; v=1 < 1.5 guard -> nan; nan -> nan
    out = D.path_curvature_from_rate(rate, v, min_speed_mps=1.5)
    assert math.isclose(out[0], 0.01, rel_tol=1e-9)
    assert np.isnan(out[1])
    assert np.isnan(out[2])
    assert math.isclose(out[3], -0.02, rel_tol=1e-9)  # negative rate (left curve) covers both signs


def test_path_curvature_from_rate_accepts_scalar_speed():
    # A single average speed must broadcast across the rate array (no IndexError).
    out = D.path_curvature_from_rate(np.array([0.1, 0.2]), 10.0)
    assert math.isclose(out[0], 0.01, rel_tol=1e-9)
    assert math.isclose(out[1], 0.02, rel_tol=1e-9)


def test_xcorr_best_recovers_known_lag_and_sign():
    fs = 20.0
    t = np.arange(0, 30, 1 / fs)
    f = 0.2
    a = np.sin(2 * np.pi * f * t)
    d = 6  # samples; b is a delayed by d samples (a leads b)
    b = np.concatenate([np.full(d, np.nan), a[:-d]])
    corr, lag_s = D.xcorr_best(a, b, fs, max_lag_s=2.0)
    assert corr > 0.95
    assert abs(lag_s - d / fs) < 1.5 / fs  # positive => a leads b


def test_xcorr_best_returns_nan_on_flat_signal():
    fs = 20.0
    a = np.ones(200)
    b = np.zeros(200)
    corr, lag_s = D.xcorr_best(a, b, fs, max_lag_s=2.0)
    assert math.isnan(corr) and math.isnan(lag_s)


def _synthetic_arrays(*, model_extra_wobble: bool):
    """Build a 30 s, 20 Hz route dict with a 0.2 Hz weave.

    Road case: lane center, model path, and realized yaw all carry the same 0.2 Hz wobble.
    Artifact case: the model path carries an EXTRA wobble not present in the lane center
    or the realized yaw (i.e. the planner invents motion).
    """
    fs = 20.0
    t = np.arange(0, 30, 1 / fs)
    L = 20.0
    f = 0.2
    v = np.full_like(t, 13.0)  # ~29 mph
    base_offset = 0.30 * np.sin(2 * np.pi * f * t)          # lane-center lateral offset (m)
    lane_curv = 2 * base_offset / (L ** 2)
    model_offset = base_offset.copy()
    if model_extra_wobble:
        model_offset = base_offset + 0.30 * np.sin(2 * np.pi * 0.27 * t + 1.0)
    yaw_rate = (2 * base_offset / (L ** 2)) * v              # realized curvature * v
    n = len(t)
    a = {
        "t": t.astype(np.float32),
        "v_ego": v.astype(np.float32),
        "yaw_rate": yaw_rate.astype(np.float32),
        "yaw_rate_calibrated": yaw_rate.astype(np.float32),
        "lat": (34.0 + np.cumsum(np.full(n, 1e-6))).astype(np.float32),
        "lon": np.full(n, -84.0, dtype=np.float32),
        "desired_curvature": lane_curv.astype(np.float32),
        "cp_final_command": (0.85 * lane_curv).astype(np.float32),
        "steering_angle_deg": (10.0 * base_offset).astype(np.float32),
        "lat_active": np.ones(n, dtype=np.float32),
        "steering_pressed": np.zeros(n, dtype=np.float32),
        "blinker": np.zeros(n, dtype=np.float32),
        "lane_change_state": np.zeros(n, dtype=np.float32),
        "lead_time_headway_s": np.full(n, 5.0, dtype=np.float32),
        "lane_prob_left": np.full(n, 0.9, dtype=np.float32),
        "lane_prob_right": np.full(n, 0.9, dtype=np.float32),
    }
    for key, off in (("model", model_offset), ("lane_center", base_offset)):
        a[f"{key}_y20"] = off.astype(np.float32)
    a["road_edge_left_y20"] = (base_offset - 1.8).astype(np.float32)
    a["road_edge_right_y20"] = (base_offset + 1.8).astype(np.float32)
    return a


def test_discriminate_window_labels_road_like_when_all_move_together():
    a = _synthetic_arrays(model_extra_wobble=False)
    rec = D.discriminate_window(a, "weave_10_70", "route_syn", 2.0, 28.0, 15.0)
    assert rec.tentative_label == "road_like"
    assert abs(rec.model_vs_lane_corr) >= 0.6
    assert abs(rec.lane_vs_independent_corr) >= 0.6


def test_discriminate_window_labels_artifact_like_when_model_adds_motion():
    a = _synthetic_arrays(model_extra_wobble=True)
    rec = D.discriminate_window(a, "weave_10_70", "route_syn", 2.0, 28.0, 15.0)
    assert rec.tentative_label == "artifact_like"
    assert rec.model_residual_over_lane >= 0.5


def test_discriminate_window_missing_optional_channels_no_warning():
    # A minimal arrays dict that drops optional channels (lane_prob_*, cp_final_command,
    # yaw_rate_calibrated) must still produce a valid record and emit no warnings.
    a = _synthetic_arrays(model_extra_wobble=False)
    for k in ("lane_prob_left", "lane_prob_right", "cp_final_command", "yaw_rate_calibrated"):
        a.pop(k, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an exception
        rec = D.discriminate_window(a, "weave_10_70", "route_syn", 2.0, 28.0, 15.0)
    assert rec.straight_clean in (0, 1)
    # lane_prob_* absent -> lane_ok cannot be confirmed -> not straight_clean.
    assert rec.straight_clean == 0


def test_spatial_curvature_profile_bins_by_gps_cell():
    a = _synthetic_arrays(model_extra_wobble=False)
    # Spread the route across multiple GPS cells by moving north steadily.
    n = len(a["t"])
    a["lat"] = (34.0 + np.arange(n) * 5e-5).astype(np.float32)
    prof = D.spatial_curvature_profile(a, 2.0, 28.0, D._band_for("weave_10_70"))
    assert len(prof) >= 2
    any_cell = next(iter(prof.values()))
    assert "lane" in any_cell and "model" in any_cell and "n" in any_cell


def test_cross_pass_reproducibility_high_for_identical_profiles():
    cells = [float(i) for i in range(10)]
    prof_a = {c: {"lane": math.sin(c), "model": 0.0, "n": 5} for c in cells}
    prof_b = {c: {"lane": math.sin(c), "model": 0.0, "n": 5} for c in cells}
    res = D.cross_pass_reproducibility([prof_a, prof_b], key="lane")
    assert res["n_shared_cells"] >= 4
    assert res["median_pairwise_corr"] > 0.95
    assert res["reproducible"] is True


def test_cross_pass_reproducibility_low_for_independent_noise():
    cells = [float(i) for i in range(12)]
    prof_a = {c: {"lane": math.sin(c), "model": 0.0, "n": 5} for c in cells}
    prof_b = {c: {"lane": math.cos(3 * c + 1.7), "model": 0.0, "n": 5} for c in cells}
    res = D.cross_pass_reproducibility([prof_a, prof_b], key="lane")
    assert res["reproducible"] is False
