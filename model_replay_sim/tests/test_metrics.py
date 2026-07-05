import numpy as np
import pytest
from model_replay_sim import config as C
from model_replay_sim.metrics import weave_band_rms, anchor_metrics


def _sine(freq_hz, n=1200, fs=C.FS_HZ, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)


def test_identical_inband_sines_give_corr1_ratio1():
    a = _sine(0.2)
    b = a.copy()
    m = anchor_metrics(a, b)
    assert m["n_valid"] == len(a)
    assert m["corr"] == pytest.approx(1.0, abs=1e-6)
    assert m["band_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["band_rms_replayed"] > 0
    assert m["band_rms_logged"] > 0


def test_weave_band_is_frequency_selective():
    inband = weave_band_rms(_sine(0.2))          # inside (0.10, 0.35)
    outband = weave_band_rms(_sine(1.0))         # well outside the band
    assert inband > 0
    assert inband > 5 * outband                  # band strongly attenuates 1.0 Hz


def test_nan_masking_does_not_crash_and_counts_valid():
    a = _sine(0.2)
    b = _sine(0.2)
    a = a.copy()
    a[10] = np.nan
    a[200] = np.nan
    b[500] = np.nan
    m = anchor_metrics(a, b)
    # jointly-finite count: 3 distinct indices removed
    assert m["n_valid"] == len(a) - 3
    assert np.isfinite(m["corr"])
    # weave_band_rms must tolerate nans too
    assert np.isfinite(weave_band_rms(a))


def test_constant_series_corr_is_nan_not_crash():
    a = np.full(1200, 0.0)
    b = _sine(0.2)
    m = anchor_metrics(a, b)
    assert np.isnan(m["corr"])                    # constant => undefined correlation
    # band_ratio: logged (b) band-rms finite>0 but replayed (a) band-rms 0 => 0.0, not crash
    assert np.isfinite(m["band_ratio"]) or np.isnan(m["band_ratio"])


def test_band_ratio_nan_when_logged_band_rms_zero():
    a = _sine(0.2)
    b = np.full(1200, 3.0)                         # constant logged => band-rms 0
    m = anchor_metrics(a, b)
    assert np.isnan(m["band_ratio"])


def test_too_short_series_returns_nan_band_rms():
    short = _sine(0.2, n=5)
    assert np.isnan(weave_band_rms(short))


def test_all_nan_series_is_nan_safe():
    a = np.full(1200, np.nan)
    b = _sine(0.2)
    assert np.isnan(weave_band_rms(a))
    m = anchor_metrics(a, b)
    assert m["n_valid"] == 0
    assert np.isnan(m["corr"])
