"""Tests for the cross-model weave COMPARISON driver (model_replay_sim.compare).

ALL tests are STUBBED and FAST — they monkeypatch ``compare.replay_window`` and
``compare.select_anchor_span`` with synthetic fakes so NO tinygrad / NO real model
inference runs (a heavy replay is using the CPU). The fakes inject known
``desired_curvature`` arrays (sines of chosen amplitude per bundle) so we can assert
the relative weave_band_rms, the pairwise ratios, the trust gate, and cross-scene
aggregation deterministically.
"""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from model_replay_sim import config as C
from model_replay_sim import compare as CMP


# --------------------------------------------------------------------------- #
# Synthetic fakes (no model, no tinygrad)                                      #
# --------------------------------------------------------------------------- #

# enough frames for the band-RMS filter: weave_band_rms needs >= max(12, fs*3) samples
# AND a comparison portion long enough for the 0.10-0.35 Hz band. 1200 frames @20Hz = 60s.
_N = 1200
_SPLIT = 200  # 10s warm-up @ 20Hz


@dataclass(frozen=True)
class _FakeSpan:
    route_id: str
    mono_times: np.ndarray
    frame_ids: np.ndarray
    seg_keys: tuple
    split_index: int
    warmup_frames: int
    compare_frames: int
    warmup_s: float
    compare_s: float


def _fake_span(route_id="route_b5", n=_N, split=_SPLIT, fs=C.FS_HZ):
    mono = np.arange(n) / fs
    fids = np.arange(n, dtype=np.int64)
    return _FakeSpan(
        route_id=route_id,
        mono_times=mono,
        frame_ids=fids,
        seg_keys=tuple((0, int(f)) for f in fids),
        split_index=split,
        warmup_frames=split,
        compare_frames=n - split,
        warmup_s=split / fs,
        compare_s=(n - split) / fs,
    )


def _sine(freq_hz, n, fs=C.FS_HZ, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t + phase)


# per-bundle weave amplitude — CD210 = 2x OPM7 so ratio OPM7/CD210 should be ~0.5
_AMP = {"CD210": 0.002, "OPM7": 0.001, "Nevada": 0.003}


def _make_fake_replay(amp_map=_AMP, freq_hz=0.2):
    """Return a fake replay_window that emits a 0.2 Hz sine of bundle-specific amplitude."""
    def fake_replay_window(bundle, route_id, mono_times):
        n = len(mono_times)
        amp = amp_map[bundle]
        curv = _sine(freq_hz, n, amp=amp)
        return {
            "route_id": route_id,
            "bundle": bundle,
            "mono_time": np.asarray(mono_times, float),
            "desired_curvature": curv,
            "v_ego": np.full(n, 25.0),
            "frames": [(0, i) for i in range(n)],
            "lat_action_t": 0.0,
        }
    return fake_replay_window


@pytest.fixture
def patched(monkeypatch):
    """Patch compare.select_anchor_span + compare.replay_window with synthetic fakes."""
    monkeypatch.setattr(CMP, "select_anchor_span",
                        lambda route, warmup_s=10.0, min_compare_s=30.0: _fake_span(route))
    monkeypatch.setattr(CMP, "replay_window", _make_fake_replay())
    return monkeypatch


# --------------------------------------------------------------------------- #
# compare_on_scene — shape, relative weave, ratio, trust gate                  #
# --------------------------------------------------------------------------- #

def test_compare_on_scene_shape_and_split(patched):
    res = CMP.compare_on_scene("route_b5", ["CD210", "OPM7"])
    assert res["scene_route"] == "route_b5"
    assert res["split_index"] == _SPLIT
    assert res["n_compare"] == _N - _SPLIT
    assert set(res["per_bundle"]) == {"CD210", "OPM7"}
    for b in ("CD210", "OPM7"):
        pb = res["per_bundle"][b]
        assert np.isfinite(pb["weave_band_rms"])
        assert pb["weave_band_rms"] > 0
        assert "anchor_validated" in pb
        assert "lat_action_t" in pb
        assert pb["n_finite"] == _N - _SPLIT


def test_compare_on_scene_ratio_tracks_amplitude(patched):
    # CD210 amplitude is 2x OPM7 -> OPM7/CD210 weave ratio ~ 0.5
    res = CMP.compare_on_scene("route_b5", ["CD210", "OPM7"])
    assert "OPM7/CD210" in res["ratios"]
    assert res["ratios"]["OPM7/CD210"] == pytest.approx(0.5, abs=1e-3)


def test_compare_on_scene_post_warmup_only(patched):
    # the weave is computed on curv[split:], so n_finite reflects the compare portion only
    res = CMP.compare_on_scene("route_b5", ["CD210"])
    assert res["per_bundle"]["CD210"]["n_finite"] == _N - _SPLIT


# --------------------------------------------------------------------------- #
# trust gate                                                                   #
# --------------------------------------------------------------------------- #

def test_trust_gate_false_when_any_unvalidated(patched):
    res = CMP.compare_on_scene("route_b5", ["CD210", "OPM7"])
    # OPM7 is anchor_validated=False -> trustworthy False, OPM7 listed in unvalidated
    assert res["trustworthy"] is False
    assert "OPM7" in res["unvalidated"]
    assert "CD210" not in res["unvalidated"]
    # the number is still COMPUTED (not dropped)
    assert np.isfinite(res["per_bundle"]["OPM7"]["weave_band_rms"])
    assert res["per_bundle"]["OPM7"]["anchor_validated"] is False
    assert res["per_bundle"]["CD210"]["anchor_validated"] is True


def test_trust_gate_true_when_all_validated(patched):
    res = CMP.compare_on_scene("route_b5", ["CD210"])
    assert res["trustworthy"] is True
    assert res["unvalidated"] == []


def test_trust_gate_false_when_any_unvalidated(patched):
    # OPM7 is the only remaining unvalidated bundle (CD210, Nevada both anchor-validated).
    res = CMP.compare_on_scene("route_b5", ["CD210", "OPM7"])
    assert res["trustworthy"] is False
    assert set(res["unvalidated"]) == {"OPM7"}


# --------------------------------------------------------------------------- #
# compare_models — cross-scene aggregation (median ratio)                      #
# --------------------------------------------------------------------------- #

def test_compare_models_aggregates_two_scenes(patched):
    res = CMP.compare_models(["route_b5", "route_7f"], ["CD210", "OPM7"])
    assert set(res["scenes"]) == {"route_b5", "route_7f"}
    assert len(res["per_scene"]) == 2
    # each scene's OPM7/CD210 ratio is ~0.5 -> median ratio ~0.5
    assert res["median_ratios"]["OPM7/CD210"] == pytest.approx(0.5, abs=1e-3)
    # per-scene ratios carried
    for sc in ("route_b5", "route_7f"):
        assert res["per_scene"][sc]["ratios"]["OPM7/CD210"] == pytest.approx(0.5, abs=1e-3)
    # overall trust = AND across scenes (OPM7 unvalidated -> False)
    assert res["trustworthy"] is False
    assert "OPM7" in res["unvalidated"]


def test_compare_models_trust_true_all_validated(monkeypatch):
    monkeypatch.setattr(CMP, "select_anchor_span",
                        lambda route, warmup_s=10.0, min_compare_s=30.0: _fake_span(route))
    monkeypatch.setattr(CMP, "replay_window", _make_fake_replay())
    res = CMP.compare_models(["route_b5", "route_7f"], ["CD210"])
    assert res["trustworthy"] is True
    assert res["unvalidated"] == []


def test_compare_models_median_across_differing_scenes(monkeypatch):
    # scene-dependent amplitude so the per-scene ratio differs; assert the MEDIAN is robust
    def varying_replay(bundle, route_id, mono_times):
        n = len(mono_times)
        # route_b5: OPM7/CD210 = 0.25 ; route_7f: 0.5 ; route_xx: 1.0 -> median = 0.5
        scene_factor = {"route_b5": 0.25, "route_7f": 0.5, "route_xx": 1.0}[route_id]
        base = 0.002
        amp = base if bundle == "CD210" else base * scene_factor
        return {
            "route_id": route_id, "bundle": bundle,
            "mono_time": np.asarray(mono_times, float),
            "desired_curvature": _sine(0.2, n, amp=amp),
            "v_ego": np.full(n, 25.0), "frames": [(0, i) for i in range(n)],
            "lat_action_t": 0.0,
        }
    monkeypatch.setattr(CMP, "select_anchor_span",
                        lambda route, warmup_s=10.0, min_compare_s=30.0: _fake_span(route))
    monkeypatch.setattr(CMP, "replay_window", varying_replay)
    res = CMP.compare_models(["route_b5", "route_7f", "route_xx"], ["CD210", "OPM7"])
    assert res["median_ratios"]["OPM7/CD210"] == pytest.approx(0.5, abs=1e-3)


# --------------------------------------------------------------------------- #
# run_anchor exists + run_cd210_anchor delegates (no real anchor run)         #
# --------------------------------------------------------------------------- #

def test_run_anchor_exists_and_cd210_delegates(monkeypatch):
    from model_replay_sim import anchor as A
    assert callable(A.run_anchor)
    calls = []
    monkeypatch.setattr(A, "run_anchor",
                        lambda bundle, route_id="route_b5", warmup_s=10.0, min_compare_s=30.0:
                        calls.append((bundle, route_id, warmup_s, min_compare_s)) or {"ok": True})
    out = A.run_cd210_anchor(route_id="route_b5", warmup_s=10.0, min_compare_s=30.0)
    assert out == {"ok": True}
    assert calls == [("CD210", "route_b5", 10.0, 30.0)]
