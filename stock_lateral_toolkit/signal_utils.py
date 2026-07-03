from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt, welch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def fill_guarded(x: np.ndarray, max_gap_samples: int) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  ok = np.isfinite(x)
  if ok.sum() < 2:
    return np.full_like(x, np.nan, dtype=float)
  idx = np.arange(len(x))
  out = np.interp(idx, idx[ok], x[ok])
  left = np.searchsorted(idx[ok], idx, side="right") - 1
  right = left + 1
  left = np.clip(left, 0, ok.sum() - 1)
  right = np.clip(right, 0, ok.sum() - 1)
  nearest_dist = np.minimum(np.abs(idx - idx[ok][left]), np.abs(idx - idx[ok][right]))
  out[nearest_dist > max_gap_samples] = np.nan
  out[idx < idx[ok][0]] = np.nan
  out[idx > idx[ok][-1]] = np.nan
  return out


def _sos_band(lo_hz: float, hi_hz: float, fs_hz: float) -> np.ndarray:
  return butter(3, [lo_hz, hi_hz], btype="band", fs=fs_hz, output="sos")


def _sos_low(cutoff_hz: float, fs_hz: float) -> np.ndarray:
  return butter(2, cutoff_hz, btype="low", fs=fs_hz, output="sos")


def filter_continuous(x: np.ndarray, fs_hz: float, *, band: tuple[float, float] | None = None,
                      lowpass_hz: float | None = None, max_gap_s: float = C.MAX_INTERP_GAP_S) -> np.ndarray:
  max_gap_samples = max(1, int(round(max_gap_s * fs_hz)))
  filled = fill_guarded(np.asarray(x, dtype=float), max_gap_samples=max_gap_samples)
  good = np.isfinite(filled)
  min_samples = max(12, int(fs_hz * 3))
  out = np.full(len(filled), np.nan)
  if band is not None:
    sos = _sos_band(band[0], band[1], fs_hz)
  elif lowpass_hz is not None:
    sos = _sos_low(lowpass_hz, fs_hz)
  else:
    raise ValueError("band or lowpass_hz is required")
  for start, end in contiguous_regions(good, min_len=min_samples):
    out[start:end] = sosfiltfilt(sos, filled[start:end])
  return out


def rms_masked(x: np.ndarray, mask: np.ndarray) -> float:
  vals = np.asarray(x, dtype=float)[np.asarray(mask, dtype=bool)]
  vals = vals[np.isfinite(vals)]
  if len(vals) == 0:
    return math.nan
  return float(np.sqrt(np.mean(vals ** 2)))


def peak_to_peak_masked(x: np.ndarray, mask: np.ndarray) -> float:
  vals = np.asarray(x, dtype=float)[np.asarray(mask, dtype=bool)]
  vals = vals[np.isfinite(vals)]
  if len(vals) == 0:
    return math.nan
  return float(np.nanmax(vals) - np.nanmin(vals))


def dilate_flags(flags: np.ndarray, radius: int) -> np.ndarray:
  flags = np.asarray(flags, dtype=bool)
  if radius <= 0:
    return flags.copy()
  return uniform_filter1d(flags.astype(float), 2 * radius + 1, mode="constant", cval=0.0) > 0


def erode_true(flags: np.ndarray, radius: int) -> np.ndarray:
  flags = np.asarray(flags, dtype=bool)
  if radius <= 0:
    return flags.copy()
  return uniform_filter1d(flags.astype(float), 2 * radius + 1, mode="constant", cval=0.0) >= 1.0


def contiguous_regions(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
  mask = np.asarray(mask, dtype=bool)
  padded = np.r_[False, mask, False]
  changes = np.flatnonzero(padded[1:] != padded[:-1])
  regions = [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]
  return [(a, b) for a, b in regions if b - a >= min_len]


def gps_cells(lat_deg: np.ndarray, lon_deg: np.ndarray, cell_m: float) -> np.ndarray:
  lat = np.asarray(lat_deg, dtype=float)
  lon = np.asarray(lon_deg, dtype=float)
  out = np.full(lat.shape, np.nan)
  ok = np.isfinite(lat) & np.isfinite(lon)
  if ok.sum() == 0:
    return out
  x = lon * C.M_PER_DEG_LON_AT_EQUATOR * np.cos(np.radians(lat))
  y = lat * C.M_PER_DEG_LAT
  gx = np.floor(x / cell_m)
  gy = np.floor(y / cell_m)
  out[ok] = gx[ok] * 10_000_000 + gy[ok]
  return out


def heading_bin_deg(heading_deg: np.ndarray, bin_deg: float) -> np.ndarray:
  heading = np.asarray(heading_deg, dtype=float)
  out = np.full(heading.shape, -1, dtype=int)
  ok = np.isfinite(heading)
  out[ok] = np.floor((heading[ok] % 360.0) / bin_deg).astype(int)
  return out


def spectral_peak_hz(x: np.ndarray, fs_hz: float, band: tuple[float, float]) -> float:
  vals = np.asarray(x, dtype=float)
  min_samples = int(fs_hz * 10)
  regions = contiguous_regions(np.isfinite(vals), min_len=min_samples)
  if len(regions) == 0:
    return math.nan
  start, end = max(regions, key=lambda region: region[1] - region[0])
  vals = vals[start:end]
  freqs, power = welch(vals - np.mean(vals), fs=fs_hz, nperseg=min(2048, len(vals)))
  band_mask = (freqs >= band[0]) & (freqs <= band[1])
  if not band_mask.any() or np.nanmax(power[band_mask]) <= 0:
    return math.nan
  return float(freqs[band_mask][np.argmax(power[band_mask])])
