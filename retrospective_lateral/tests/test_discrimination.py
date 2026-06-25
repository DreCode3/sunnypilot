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
