"""Weave band-RMS + anchor comparison metrics for the model-replay simulator.

Reuses the retrospective weave convention exactly: band-RMS of the curvature in
`C.WEAVE_BAND_HZ` at `C.FS_HZ`, computed via `filter_continuous(sig, fs, band=band)`.
All functions are nan-safe; Task 10 compares the raw numbers returned here against
`C.ANCHOR_CORR_MIN` / `C.ANCHOR_BAND_RATIO`.
"""
from __future__ import annotations
import math

import numpy as np

from model_replay_sim import config as C
from retrospective_lateral.code.signal_utils import filter_continuous

# filter_continuous needs >= max(12, fs*3) contiguous finite samples to emit a region.
_MIN_VALID = max(12, int(C.FS_HZ * 3))
# Below this the band-rms is filter/numerical residue of a (near-)constant series, not
# real weave. Curvature weave is ~1e-4..1e-2; bandpass residue of a constant is ~1e-15.
_BAND_RMS_FLOOR = 1e-12


def weave_band_rms(series, fs: float = C.FS_HZ, band=C.WEAVE_BAND_HZ) -> float:
    """RMS of the band-passed series over its finite samples.

    Nan-safe: `filter_continuous` interpolates short gaps and only filters
    contiguous finite regions. Too few valid samples → nan.
    """
    x = np.asarray(series, dtype=float)
    if x.size < _MIN_VALID or np.isfinite(x).sum() < _MIN_VALID:
        return math.nan
    filtered = filter_continuous(x, fs, band=band)
    vals = filtered[np.isfinite(filtered)]
    if vals.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(vals ** 2)))


def anchor_metrics(replayed, logged) -> dict:
    """Anchor comparison of a replayed vs logged curvature series.

    Returns {corr, band_ratio, band_rms_replayed, band_rms_logged, n_valid}:
      - corr: Pearson r over jointly-finite samples (nan if too few or a series is constant)
      - band_ratio: band_rms_replayed / band_rms_logged (nan if logged band-rms is 0/nan)
    """
    a = np.asarray(replayed, dtype=float)
    b = np.asarray(logged, dtype=float)
    both = np.isfinite(a) & np.isfinite(b)
    n_valid = int(both.sum())

    av, bv = a[both], b[both]
    if n_valid < 2 or np.std(av) == 0 or np.std(bv) == 0:
        corr = math.nan
    else:
        corr = float(np.corrcoef(av, bv)[0, 1])

    band_rms_replayed = weave_band_rms(a)
    band_rms_logged = weave_band_rms(b)
    if (not np.isfinite(band_rms_logged)) or band_rms_logged <= _BAND_RMS_FLOOR:
        band_ratio = math.nan
    elif not np.isfinite(band_rms_replayed):
        band_ratio = math.nan
    else:
        band_ratio = float(band_rms_replayed / band_rms_logged)

    return {
        "corr": corr,
        "band_ratio": band_ratio,
        "band_rms_replayed": band_rms_replayed,
        "band_rms_logged": band_rms_logged,
        "n_valid": n_valid,
    }
