import numpy as np

from retrospective_lateral.code import signal_utils as S


def test_fill_guarded_keeps_long_gaps_nan():
  x = np.array([0.0, 1.0, np.nan, np.nan, np.nan, 5.0])
  y = S.fill_guarded(x, max_gap_samples=1)
  assert np.isfinite(y[2])
  assert np.isnan(y[3])
  assert np.isfinite(y[4])


def test_rms_and_peak_to_peak_ignore_nan():
  x = np.array([1.0, -1.0, np.nan, 1.0, -1.0])
  mask = np.array([True, True, True, False, False])
  assert S.rms_masked(x, mask) == 1.0
  assert S.peak_to_peak_masked(x, mask) == 2.0


def test_dilate_and_erode_boolean_flags():
  flags = np.array([False, False, True, False, False])
  assert S.dilate_flags(flags, radius=1).tolist() == [False, True, True, True, False]
  assert S.erode_true(np.ones(5, dtype=bool), radius=1).tolist() == [False, True, True, True, False]


def test_contiguous_regions_filters_short_runs():
  mask = np.array([False, True, True, False, True, True, True])
  assert S.contiguous_regions(mask, min_len=3) == [(4, 7)]


def test_gps_cell_and_heading_bin_are_stable():
  lat = np.array([34.0, 34.0001])
  lon = np.array([-84.0, -84.0001])
  cells = S.gps_cells(lat, lon, cell_m=80.0)
  assert cells.shape == (2,)
  assert np.isfinite(cells).all()
  assert S.heading_bin_deg(np.array([1.0, 44.0, 46.0]), bin_deg=45.0).tolist() == [0, 0, 1]
