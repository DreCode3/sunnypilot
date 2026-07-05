"""Tests for Task 10: the CD210 same-model fidelity ANCHOR (model_replay_sim.anchor).

Two tiers:
  * pure/fast — the gate decision (anchor_verdict) on SYNTHETIC anchor_metrics dicts and
    on synthetic replayed/logged arrays through anchor_metrics; the span selection
    (select_anchor_span) on the REAL route_b5 timeline (no model, no tinygrad).
  * (the heavy end-to-end run_cd210_anchor lives in run.py and is executed there, not in
    the unit suite — it replays vision+policy for hundreds of frames.)
"""

import math

import numpy as np
import pytest

from model_replay_sim import config as C
from model_replay_sim.anchor import (
    anchor_verdict,
    anchor_metrics_safe,
    select_anchor_span,
)


# --------------------------------------------------------------------------- #
# Pure: gate decision on synthetic metrics dicts                              #
# --------------------------------------------------------------------------- #

def test_anchor_verdict_passes_when_both_in_band():
    assert anchor_verdict({"corr": 0.99, "band_ratio": 1.0}) is True
    # exactly on the thresholds passes (>= / inclusive)
    assert anchor_verdict({"corr": C.ANCHOR_CORR_MIN, "band_ratio": C.ANCHOR_BAND_RATIO[0]}) is True
    assert anchor_verdict({"corr": C.ANCHOR_CORR_MIN, "band_ratio": C.ANCHOR_BAND_RATIO[1]}) is True


def test_anchor_verdict_fails_on_low_corr():
    assert anchor_verdict({"corr": 0.80, "band_ratio": 1.0}) is False
    assert anchor_verdict({"corr": C.ANCHOR_CORR_MIN - 1e-6, "band_ratio": 1.0}) is False


def test_anchor_verdict_fails_on_out_of_band_ratio():
    assert anchor_verdict({"corr": 0.99, "band_ratio": C.ANCHOR_BAND_RATIO[1] + 0.01}) is False
    assert anchor_verdict({"corr": 0.99, "band_ratio": C.ANCHOR_BAND_RATIO[0] - 0.01}) is False


def test_anchor_verdict_fails_on_nan():
    assert anchor_verdict({"corr": math.nan, "band_ratio": 1.0}) is False
    assert anchor_verdict({"corr": 0.99, "band_ratio": math.nan}) is False
    assert anchor_verdict({}) is False


# --------------------------------------------------------------------------- #
# Pure: gate decision exercised through real anchor_metrics on synthetic data #
# --------------------------------------------------------------------------- #

def _sine(freq_hz, n=1200, fs=C.FS_HZ, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)


def test_passing_case_corr1_ratio1_through_metrics():
    # an in-band weave reproduced exactly -> corr ~1, ratio ~1 -> PASS
    replayed = _sine(0.2)
    logged = replayed.copy()
    m = anchor_metrics_safe(replayed, logged)
    assert m["corr"] == pytest.approx(1.0, abs=1e-6)
    assert m["band_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert anchor_verdict(m) is True


def test_failing_case_low_corr_through_metrics():
    # uncorrelated noise vs the weave -> corr near 0 -> FAIL even if band amplitudes happen close
    rng = np.random.default_rng(0)
    logged = _sine(0.2)
    replayed = _sine(0.2, phase=np.pi) + 0.01 * rng.standard_normal(len(logged))  # anti-phase
    m = anchor_metrics_safe(replayed, logged)
    assert m["corr"] < C.ANCHOR_CORR_MIN
    assert anchor_verdict(m) is False


def test_failing_case_ratio_out_of_band_through_metrics():
    # same shape (corr ~1) but replayed weave amplitude doubled -> ratio ~2 -> FAIL
    logged = _sine(0.2)
    replayed = 2.0 * logged
    m = anchor_metrics_safe(replayed, logged)
    assert m["corr"] == pytest.approx(1.0, abs=1e-6)
    assert m["band_ratio"] == pytest.approx(2.0, abs=1e-3)
    assert anchor_verdict(m) is False


# --------------------------------------------------------------------------- #
# Span selection on the REAL route_b5 timeline (fast: no model)               #
# --------------------------------------------------------------------------- #

def _have_b5_timeline():
    try:
        from model_replay_sim.alignment import build_frame_timeline
        return len(build_frame_timeline("route_b5")) > 0 and (C.CACHE_ROOT / "route_b5.npz").exists()
    except Exception:
        return False


_SKIP_SPAN = not _have_b5_timeline()


@pytest.mark.skipif(_SKIP_SPAN, reason="route_b5 frames/npz not present locally")
def test_select_anchor_span_meets_warmup_and_compare_minimums():
    warmup_s, min_compare_s = 10.0, 30.0
    span = select_anchor_span("route_b5", warmup_s=warmup_s, min_compare_s=min_compare_s)

    # warm-up >= requested, comparison >= requested (in frames AND seconds). A span of N
    # frames covers (N-1) inter-frame intervals, so allow a 1-frame edge tolerance in seconds.
    assert span.warmup_frames >= int(round(warmup_s * C.FS_HZ))
    assert span.compare_frames >= int(round(min_compare_s * C.FS_HZ))
    assert span.warmup_s >= warmup_s - 2.0 / C.FS_HZ
    assert span.compare_s >= min_compare_s - 2.0 / C.FS_HZ

    # the split is consistent with the frame arrays
    n = len(span.mono_times)
    assert span.split_index == span.warmup_frames
    assert n == span.warmup_frames + span.compare_frames
    assert len(span.frame_ids) == n
    assert len(span.seg_keys) == n


@pytest.mark.skipif(_SKIP_SPAN, reason="route_b5 frames/npz not present locally")
def test_select_anchor_span_has_contiguous_frame_ids_and_sorted_mono():
    span = select_anchor_span("route_b5", warmup_s=10.0, min_compare_s=30.0)
    fids = np.asarray(span.frame_ids)
    # contiguous global frame_ids across the WHOLE span (warm-up feeds the recurrent state
    # from real data with no gap into the comparison window)
    assert np.all(np.diff(fids) == 1), "span frame_ids must be strictly consecutive (no gaps)"
    # mono_times monotonically increasing (it is timestamp_eof_s of consecutive frames)
    assert np.all(np.diff(np.asarray(span.mono_times)) > 0)


@pytest.mark.skipif(_SKIP_SPAN, reason="route_b5 frames/npz not present locally")
def test_select_anchor_span_raises_when_compare_too_long():
    # route_b5's longest eligible run is ~68s; ask for far more than it can supply.
    with pytest.raises(AssertionError, match="Cannot supply anchor span"):
        select_anchor_span("route_b5", warmup_s=10.0, min_compare_s=600.0)
