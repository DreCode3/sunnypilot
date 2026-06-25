import math

import numpy as np

from retrospective_lateral.code import corridor_repro as R


def _synthetic_route_arrays(*, heading_north=True, weave_sign=1.0):
    """A ~30 s, 20 Hz pass that drives in a straight line through many fine cells,
    carrying a 0.2 Hz weave on the lane center. Moving north -> course ~0 deg."""
    fs = 20.0
    t = np.arange(0, 30, 1 / fs)
    v = np.full_like(t, 13.0)  # ~29 mph
    weave = weave_sign * 0.30 * np.sin(2 * np.pi * 0.2 * t)  # lane-center lateral offset (m)
    # move ~390 m north over the window so we span ~26 cells of 15 m
    lat = 34.0 + np.cumsum(np.full_like(t, 13.0 / 111320.0 / fs))
    lon = np.full_like(t, -84.5)
    if not heading_north:
        lat = lat[::-1].copy()  # drive the same span southbound -> course ~180 deg
    n = len(t)
    return {
        "t": t.astype(np.float32),
        "v_ego": v.astype(np.float32),
        "lat": lat.astype(np.float32),
        "lon": lon.astype(np.float32),
        "lat_active": np.ones(n, dtype=np.float32),
        "lane_center_y20": weave.astype(np.float32),
    }


def test_profile_from_arrays_bins_weave_band_into_many_fine_cells():
    prof = R.profile_from_arrays(_synthetic_route_arrays())
    assert len(prof) >= 8
    lane_mean, course, n = next(iter(prof.values()))
    assert math.isfinite(lane_mean) and math.isfinite(course) and n >= 1
    # northbound -> course near 0/360
    courses = np.array([v[1] for v in prof.values()])
    assert np.nanmin(np.minimum(courses % 360, 360 - (courses % 360))) < 20


def test_profile_from_arrays_returns_empty_without_channels():
    assert R.profile_from_arrays({"t": np.zeros(3)}) == {}


def test_corridor_reproducibility_same_direction_roadlocked_is_reproducible():
    pa = {float(c): (math.sin(c), 10.0, 5) for c in range(20)}
    pb = {float(c): (math.sin(c), 12.0, 5) for c in range(20)}  # same weave, same heading
    res = R.corridor_reproducibility(pa, pb)
    assert res["direction"] == "same"
    assert res["corr_corrected"] > 0.95
    assert res["reproducible"] is True


def test_corridor_reproducibility_opposite_direction_signflip_is_reproducible():
    pa = {float(c): (math.sin(c), 10.0, 5) for c in range(20)}
    # opposite pass: travel-frame curvature negated, heading +180
    pb = {float(c): (-math.sin(c), 190.0, 5) for c in range(20)}
    res = R.corridor_reproducibility(pa, pb)
    assert res["direction"] == "opposite"
    assert res["corr_raw"] < -0.95          # raw anti-correlates
    assert res["corr_corrected"] > 0.95     # sign-corrected -> reproducible
    assert res["reproducible"] is True


def test_corridor_reproducibility_independent_is_not_reproducible():
    pa = {float(c): (math.sin(c), 10.0, 5) for c in range(24)}
    pb = {float(c): (math.cos(3 * c + 1.7), 10.0, 5) for c in range(24)}
    res = R.corridor_reproducibility(pa, pb)
    assert res["reproducible"] is False


def test_corridor_reproducibility_min_cells_guard_returns_none():
    pa = {float(c): (math.sin(c), 10.0, 5) for c in range(3)}
    pb = {float(c): (math.sin(c), 10.0, 5) for c in range(3)}
    assert R.corridor_reproducibility(pa, pb, min_cells=8) is None


def test_build_corridor_repro_parallel_smoke(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # two northbound passes over the same corridor with the same road-locked weave
    for name in ("route_aa", "route_bb"):
        a = _synthetic_route_arrays()
        np.savez_compressed(cache / f"{name}.npz", **a)
    reports = tmp_path / "reports"
    result = R.build_corridor_repro(cache, reports, workers=1)
    assert result["routes_profiled"] == 2
    assert result["pairs_scored"] >= 1
    assert (reports / "corridor_repro_pairs.csv").exists()
    assert (reports / "corridor_repro_summary.csv").exists()
