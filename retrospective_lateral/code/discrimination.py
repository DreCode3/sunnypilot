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
    course[~ok] = np.nan
    return course
