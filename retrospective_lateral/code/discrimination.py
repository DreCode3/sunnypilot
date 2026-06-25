"""Offline model/path-vs-road discrimination for the Explorer ST lateral symptoms.

Analysis-only. Reads existing route NPZ caches and the symptom catalog; writes only
to retrospective_lateral/results/. Imports nothing from the vehicle-control stack.
"""

from __future__ import annotations

import math

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.signal_utils import (
    contiguous_regions,
    dilate_flags,
    filter_continuous,
    gps_cells,
    heading_bin_deg,
    rms_masked,
    spectral_peak_hz,
)


def offset_to_curvature(offset_m, lookahead_m: float):
    """Convert a lateral offset at a forward lookahead to an implied path curvature.

    For a small-curvature arc through the origin, y(L) ~= kappa * L^2 / 2, so
    kappa ~= 2*y / L^2. Lets model path / lane center / road edge offsets be compared
    on the same 1/m axis as desired_curvature and yawRate/v.
    """
    L = float(lookahead_m)
    if L <= 0:
        raise ValueError("lookahead_m must be > 0")
    return 2.0 * np.asarray(offset_m, dtype=float) / (L * L)


def gps_course_deg(lat_deg, lon_deg):
    """Course over ground in degrees [0, 360), 0 = North, clockwise, from GPS displacement.

    Independent of the vision model and of EPAS. NaN where it cannot be computed.
    Approximate: treats finite samples as evenly spaced; intended as a coarse,
    model-independent heading reference, not a precision signal.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    out = np.full(lat.shape, np.nan)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return out
    lat0 = float(np.nanmedian(lat[ok]))
    east_m = lon * C.M_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat0))
    north_m = lat * C.M_PER_DEG_LAT
    dx = np.gradient(east_m)
    dy = np.gradient(north_m)
    course = np.degrees(np.arctan2(dx, dy)) % 360.0
    # np.gradient poisons the valid samples bordering each NaN gap, so explicitly
    # invalidate the gap samples and their immediate neighbors.
    course[dilate_flags(~ok, 1)] = np.nan
    return course


def path_curvature_from_rate(rate, v_mps, min_speed_mps: float = C.DISCRIM_MIN_SPEED_MPS):
    """Realized path curvature = angular rate / speed (1/m). NaN below the speed guard.

    Use with CAN yawRate or calibrated yawRate to get a vision-model-independent path.
    """
    rate = np.asarray(rate, dtype=float)
    v = np.asarray(v_mps, dtype=float)
    v = np.broadcast_to(v, rate.shape)
    out = np.full(rate.shape, np.nan)
    ok = np.isfinite(rate) & np.isfinite(v) & (v >= float(min_speed_mps))
    out[ok] = rate[ok] / v[ok]
    return out


def gps_path_curvature(lat_deg, lon_deg, v_mps, fs_hz: float = C.FS_HZ,
                       min_speed_mps: float = C.DISCRIM_MIN_SPEED_MPS):
    """Heading-rate curvature from GPS course (1/m). Fully model- and EPAS-independent.

    Coarse/noisy; used only as a third independent corroborator, not a primary metric.
    Because np.gradient runs on the gap-compressed valid samples, the rate can be
    slightly inflated at GPS-gap boundaries (mitigated by the dilation in gps_course_deg).
    """
    course = gps_course_deg(lat_deg, lon_deg)
    rate = np.full(course.shape, np.nan)
    ok = np.isfinite(course)
    if ok.sum() >= 3:
        unwrapped = np.unwrap(np.radians(course[ok]))
        rate[ok] = np.gradient(unwrapped) * float(fs_hz)
    return path_curvature_from_rate(rate, v_mps, min_speed_mps=min_speed_mps)


def xcorr_best(a, b, fs_hz: float, max_lag_s: float = C.DISCRIM_MAX_LAG_S):
    """Best Pearson correlation over integer lags on the largest shared finite run.

    Returns (corr, lag_s). lag_s > 0 means `a` leads `b`. (nan, nan) if undecidable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    min_len = max(12, int(fs_hz * 3))
    regions = contiguous_regions(np.isfinite(a) & np.isfinite(b), min_len=min_len)
    if not regions:
        return (math.nan, math.nan)
    s, e = max(regions, key=lambda r: r[1] - r[0])
    aa = a[s:e]
    bb = b[s:e]
    n = len(aa)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return (math.nan, math.nan)
    max_lag = int(round(max_lag_s * fs_hz))
    best_corr = -2.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = aa[-lag:], bb[:n + lag]
        elif lag > 0:
            x, y = aa[:n - lag], bb[lag:]
        else:
            x, y = aa, bb
        if len(x) < min_len or np.std(x) == 0 or np.std(y) == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best_corr:
            best_corr = c
            best_lag = lag
    if best_corr < -1.5:
        return (math.nan, math.nan)
    return (best_corr, best_lag / fs_hz)
