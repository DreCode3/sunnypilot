import numpy as np

from retrospective_lateral.code import signal_utils as S


def test_fill_guarded_keeps_long_gaps_nan():
  x = np.array([0.0, 1.0, np.nan, np.nan, np.nan, 5.0])
  y = S.fill_guarded(x, max_gap_samples=1)
  assert np.isfinite(y[2])
  assert np.isnan(y[3])
  assert np.isfinite(y[4])


def test_fill_guarded_keeps_endpoint_nans():
  x = np.array([np.nan, 1.0, 2.0, np.nan])
  y = S.fill_guarded(x, max_gap_samples=1)
  assert np.isnan(y[0])
  assert y[1] == 1.0
  assert y[2] == 2.0
  assert np.isnan(y[3])


def test_filter_continuous_does_not_bridge_long_invalid_gap():
  fs_hz = 20.0
  x = np.ones(500)
  x[200:300] = np.nan

  y = S.filter_continuous(x, fs_hz, lowpass_hz=0.5, max_gap_s=0.5)

  assert np.allclose(y[80:190], 1.0, atol=1e-3)
  assert np.isnan(y[230:270]).all()
  assert np.allclose(y[310:420], 1.0, atol=1e-3)


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


def test_gps_cells_are_not_batch_dependent():
  fixed_lat = 34.0
  fixed_lon = -84.0
  alone = S.gps_cells(np.array([fixed_lat]), np.array([fixed_lon]), cell_m=80.0)[0]
  batched = S.gps_cells(np.array([fixed_lat, 60.0]), np.array([fixed_lon, -122.0]), cell_m=80.0)[0]
  assert alone == batched


def test_spectral_peak_hz_uses_continuous_finite_span():
  fs_hz = 20.0
  t = np.arange(int(fs_hz * 60.0)) / fs_hz
  x = np.sin(2.0 * np.pi * 0.20 * t)

  assert abs(S.spectral_peak_hz(x, fs_hz, band=(0.1, 0.35)) - 0.20) < 0.02

  short_spans = x[: int(fs_hz * 5.0)].copy()
  short_spans = np.r_[short_spans, np.full(int(fs_hz), np.nan), short_spans]
  assert np.isnan(S.spectral_peak_hz(short_spans, fs_hz, band=(0.1, 0.35)))
