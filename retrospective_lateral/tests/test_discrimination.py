import math

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
