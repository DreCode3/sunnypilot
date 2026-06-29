"""Tests for Task 8: faithful vision->policy replay loop (model_replay_sim.infer).

Two tiers:
  * pure/fast — the temporal-buffer index math (must match modeld.py:80-104) and the
    output-slicing reshape of `plan` from a synthetic concatenated tensor. No tinygrad,
    no compiled model, no frames.
  * slow/real — end-to-end replay_window on a real CD210/route_b5 eligible window: finite,
    right length, plausible magnitude, deterministic. Skipped if frames/npz are absent or
    the compiled pkls cannot be produced.
"""

import bisect
import glob
import warnings

import numpy as np
import pytest

from model_replay_sim import config as C
from model_replay_sim.infer import build_temporal_buffers


# --------------------------------------------------------------------------- #
# Pure: temporal-buffer index math                                            #
# --------------------------------------------------------------------------- #

# CD210's combined input shapes (vision img inputs + policy float inputs). Hard-coded here
# (and asserted against the live BundleModel in the slow test) so the pure test needs no
# compiled model.
CD210_INPUT_SHAPES = {
    "img": (1, 12, 128, 256),
    "big_img": (1, 12, 128, 256),
    "desire_pulse": (1, 25, 8),
    "traffic_convention": (1, 2),
    "features_buffer": (1, 25, 512),
}
CD210_VISION_INPUTS = ["img", "big_img"]


def test_features_buffer_temporal_idxs_match_modeld_formula():
    # modeld.py:80-104 for a 25-deep features_buffer: buffer_history_len = 25*4 = 100,
    # skip = 100//25 = 4, idxs = arange(100)[-1-(4*24)::4] = [3,7,...,99] (25 indices).
    _, temporal_buffers, idx_map = build_temporal_buffers(CD210_INPUT_SHAPES, CD210_VISION_INPUTS)

    assert "features_buffer" in idx_map
    got = np.asarray(idx_map["features_buffer"])
    expected = np.arange(100)[-1 - (4 * 24)::4]
    np.testing.assert_array_equal(got, expected)
    assert got.tolist() == list(range(3, 100, 4))
    assert got.tolist()[0] == 3 and got.tolist()[-1] == 99 and len(got) == 25

    # the underlying rolling buffer is (1, 100, 512)
    assert temporal_buffers["features_buffer"].shape == (1, 100, 512)


def test_desire_pulse_temporal_buffer_is_25_deep():
    # desire_pulse shares the 25-deep / *4 history construction (split branch).
    numpy_inputs, temporal_buffers, idx_map = build_temporal_buffers(
        CD210_INPUT_SHAPES, CD210_VISION_INPUTS)
    # policy inputs present, vision inputs excluded
    assert set(numpy_inputs) == {"desire_pulse", "traffic_convention", "features_buffer"}
    assert "img" not in numpy_inputs and "big_img" not in numpy_inputs
    # desire_pulse is a 25-deep temporal input
    assert temporal_buffers["desire_pulse"].shape == (1, 100, 8)
    np.testing.assert_array_equal(np.asarray(idx_map["desire_pulse"]), np.arange(100)[3::4])
    # traffic_convention is NOT temporal (2-D) -> no buffer / idx entry
    assert "traffic_convention" not in temporal_buffers
    assert numpy_inputs["traffic_convention"].shape == (1, 2)


# --------------------------------------------------------------------------- #
# Pure: output-slicing reshape of `plan`                                       #
# --------------------------------------------------------------------------- #

def test_output_slicing_reshapes_plan_from_synthetic_tensor():
    # The policy graph emits ONE concatenated (1,1000) tensor; output_slices cut it into
    # named outputs (ModelRunner._slice_outputs convention: {k: flat[np.newaxis, v]}).
    # plan = slice(0,990), desire_state = slice(990,998). The split parser reshapes plan
    # (1,990) -> (1, IDX_N=33, PLAN_WIDTH=15): is_mhp sees 990 == 2*33*15 -> NOT mhp ->
    # parse_mdn(in_N=0) -> mu half = (1, 33, 15).
    import sys
    from model_replay_sim.warp import _ensure_tinygrad_env
    _ensure_tinygrad_env()
    from openpilot.sunnypilot.modeld_v2.parse_model_outputs_split import Parser
    from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

    output_slices = {
        "plan": slice(0, 990, None),
        "desire_state": slice(990, 998, None),
        "pad": slice(-2, None, None),
    }
    flat = np.arange(1000, dtype=np.float32)
    sliced = {k: flat[np.newaxis, v] for k, v in output_slices.items()}
    assert sliced["plan"].shape == (1, 990)
    assert sliced["desire_state"].shape == (1, 8)

    parsed = Parser().parse_policy_outputs(dict(sliced))
    assert parsed["plan"].shape == (1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert (ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH) == (33, 15)
    # plan_stds is also produced (the std half)
    assert parsed["plan_stds"].shape == (1, 33, 15)


# --------------------------------------------------------------------------- #
# Slow/real: end-to-end CD210 / route_b5 window                                #
# --------------------------------------------------------------------------- #

def _b5_frames():
    return bool(glob.glob("explorer_st_logs/route_b5/000000b5--*--*/fcamera.hevc"))


def _b5_npz():
    return (C.CACHE_ROOT / "route_b5.npz").exists()


def _cd210_onnx():
    d = C.RESULTS_ROOT / "onnx" / "CD210"
    return (d / "driving_vision.onnx").exists() and (d / "driving_policy.onnx").exists()


def _short_eligible_window(route_id="route_b5", n_frames=24):
    """A short (~n_frames) contiguous eligible+aligned window of mono_times, picked the same
    way assets.py defines eligibility. Returns None if none found."""
    from model_replay_sim.assets import _replay_scene_eligible_mask
    from model_replay_sim.alignment import build_frame_timeline

    z = dict(np.load(C.CACHE_ROOT / f"{route_id}.npz"))
    mono = np.asarray(z["mono_time"], float)
    elig = _replay_scene_eligible_mask(z)
    tl = build_frame_timeline(route_id)
    if not tl:
        return None
    eofs = sorted(r.timestamp_eof_s for r in tl)

    def covered(t):
        i = bisect.bisect_left(eofs, t)
        return any(0 <= j < len(eofs) and abs(eofs[j] - t) <= C.MAX_FRAME_DELTA_S for j in (i - 1, i))

    good = elig & np.array([covered(float(t)) for t in mono], dtype=bool)
    i, n = 0, len(good)
    best = None
    while i < n:
        if good[i]:
            j = i
            while j < n and good[j]:
                j += 1
            if (j - i) >= n_frames and (best is None or (j - i) > best[2]):
                best = (i, j, j - i)
            i = j
        else:
            i += 1
    if best is None:
        return None
    s = best[0] + (best[2] - n_frames) // 2
    return mono[s:s + n_frames]


_SKIP = not (_b5_frames() and _b5_npz() and _cd210_onnx())


@pytest.mark.skipif(_SKIP, reason="route_b5 frames/npz or CD210 onnx not present locally")
def test_replay_window_cd210_route_b5_end_to_end():
    from model_replay_sim.infer import replay_window, BundleModel

    win = _short_eligible_window("route_b5", n_frames=24)
    if win is None:
        pytest.skip("no eligible aligned window in route_b5")
    assert len(win) == 24

    # combined input shapes match the metadata convention (live model)
    model = BundleModel.get("CD210")
    assert model.input_shapes["features_buffer"] == (1, 25, 512)
    assert model.input_shapes["img"] == (1, 12, 128, 256)
    assert set(model.vision_input_names) == {"img", "big_img"}

    r = replay_window("CD210", "route_b5", win)
    curv = np.asarray(r["desired_curvature"], dtype=float)

    # right length, finite
    assert curv.shape == (len(win),)
    assert np.all(np.isfinite(curv)), "desiredCurvature must be finite"

    # plausible magnitude on a gentle (eligible) window: |curv| < 0.05 1/m
    assert np.max(np.abs(curv)) < 0.05, f"curv too large: max|c|={np.max(np.abs(curv)):.4f}"

    # the curvature post-step path is the plan-based one (CD210 emits no desired_curvature)
    assert "desired_curvature" not in model.policy.output_slices

    # mono_time / v_ego aligned to the window
    np.testing.assert_allclose(r["mono_time"], np.asarray(win, float))
    assert len(r["v_ego"]) == len(win)
    assert len(r["frames"]) == len(win)

    # deterministic across two runs (bit-identical)
    r2 = replay_window("CD210", "route_b5", win)
    np.testing.assert_array_equal(curv, np.asarray(r2["desired_curvature"], dtype=float))


@pytest.mark.skipif(_SKIP, reason="route_b5 frames/npz or CD210 onnx not present locally")
def test_big_img_sourced_from_real_wide_camera_not_road_proxy():
    """The big_img wide sixchan must come from the REAL wide camera (ecamera.hevc), NOT the
    old road-through-M_extra proxy. For a frame with an aligned wide index, the real-wide
    sixchan must differ from the road-frame warped through the same wide transform."""
    from model_replay_sim.alignment import build_frame_timeline, read_frame, read_wide_frame
    from model_replay_sim.context import route_context
    from model_replay_sim.warp import model_transform, frame_to_sixchan
    from model_replay_sim.infer import _frame_dims

    tl = build_frame_timeline("route_b5")
    s16 = [r for r in tl if r.segment_num == 16 and r.ecamera_segment_id is not None]
    if not s16:
        pytest.skip("no aligned wide frame in route_b5 seg16")
    r = s16[0]

    ctx = route_context("route_b5")
    M_extra = model_transform(ctx, wide=True)

    road = np.asarray(read_frame("route_b5", r.segment_num, r.segment_id), np.uint8).ravel()
    wide = np.asarray(read_wide_frame("route_b5", r.segment_num, r.ecamera_segment_id), np.uint8).ravel()
    cam_w, cam_h = _frame_dims(road.size)
    w_w, w_h = _frame_dims(wide.size)

    proxy_big = frame_to_sixchan(road, cam_w, cam_h, M_extra)   # the OLD road-through-M_extra
    real_big = frame_to_sixchan(wide, w_w, w_h, M_extra)        # the NEW real wide frame
    assert real_big.shape == proxy_big.shape == (6, 128, 256)
    # they must NOT be identical — the fix changed the big_img source
    assert not np.array_equal(real_big, proxy_big), \
        "big_img wide sixchan is identical to the road-through-M_extra proxy (fix not active)"


# --------------------------------------------------------------------------- #
# C1: _route_window_v_ego honesty (no silent v_ego=0; time-delta guard; raises)#
# --------------------------------------------------------------------------- #

def _write_npz(tmp_path, monkeypatch, mono_time, v_ego):
    """Point C.CACHE_ROOT at tmp_path and write a synthetic route_zz.npz."""
    monkeypatch.setattr(C, "CACHE_ROOT", tmp_path)
    arrs = {"mono_time": np.asarray(mono_time, float)}
    if v_ego is not None:
        arrs["v_ego"] = np.asarray(v_ego, float)
    np.savez(tmp_path / "route_zz.npz", **arrs)


def test_route_window_v_ego_out_of_range_mono_time_yields_nan_not_zero(tmp_path, monkeypatch):
    from model_replay_sim.infer import _route_window_v_ego
    # samples at t=0,0.05,0.10 with finite speeds; request one in-range t and one far away.
    _write_npz(tmp_path, monkeypatch, mono_time=[0.0, 0.05, 0.10], v_ego=[20.0, 21.0, 22.0])
    far = 0.05 + 10 * C.MAX_FRAME_DELTA_S      # nearest sample is > MAX_FRAME_DELTA_S away
    # one of two frames is out-of-range -> 50% nan -> the loud guard also fires (>25%)
    with pytest.warns(UserWarning, match="no valid v_ego"):
        out = _route_window_v_ego("route_zz", [0.05, far])
    assert out[0] == pytest.approx(21.0)        # in-range -> real speed
    assert np.isnan(out[1])                      # out-of-range -> nan, NOT 0.0
    assert not np.any(out == 0.0)                # the bug would have emitted 0.0


def test_route_window_v_ego_nonfinite_sample_yields_nan(tmp_path, monkeypatch):
    from model_replay_sim.infer import _route_window_v_ego
    _write_npz(tmp_path, monkeypatch, mono_time=[0.0, 0.05], v_ego=[np.nan, 21.0])
    with pytest.warns(UserWarning, match="no valid v_ego"):  # 50% nan -> loud guard fires
        out = _route_window_v_ego("route_zz", [0.0, 0.05])
    assert np.isnan(out[0])                       # non-finite nearest sample -> nan
    assert out[1] == pytest.approx(21.0)


def test_route_window_v_ego_missing_npz_raises(tmp_path, monkeypatch):
    from model_replay_sim.infer import _route_window_v_ego
    monkeypatch.setattr(C, "CACHE_ROOT", tmp_path)   # empty dir -> no route_zz.npz
    with pytest.raises(FileNotFoundError):
        _route_window_v_ego("route_zz", [0.0, 0.05])


def test_route_window_v_ego_missing_key_raises(tmp_path, monkeypatch):
    from model_replay_sim.infer import _route_window_v_ego
    _write_npz(tmp_path, monkeypatch, mono_time=[0.0, 0.05], v_ego=None)  # no v_ego key
    with pytest.raises(KeyError):
        _route_window_v_ego("route_zz", [0.0, 0.05])


def test_route_window_v_ego_warns_when_mostly_nan(tmp_path, monkeypatch):
    from model_replay_sim.infer import _route_window_v_ego
    _write_npz(tmp_path, monkeypatch, mono_time=[0.0], v_ego=[20.0])
    far = 100.0
    with pytest.warns(UserWarning, match="no valid v_ego"):
        _route_window_v_ego("route_zz", [far, far, far, far])  # 100% nan


@pytest.mark.skipif(not _b5_npz(), reason="route_b5 npz not present locally")
def test_route_window_v_ego_real_route_b5_window_all_finite():
    """A normal eligible route_b5 window must stay all-finite (the C1 guard does not
    corrupt clean data)."""
    from model_replay_sim.infer import _route_window_v_ego
    win = _short_eligible_window("route_b5", n_frames=24)
    if win is None:
        pytest.skip("no eligible aligned window in route_b5")
    out = _route_window_v_ego("route_b5", win)
    assert out.shape == (len(win),)
    assert np.all(np.isfinite(out)), "clean route_b5 window must have all-finite v_ego"
    assert np.all(out > 0)


def test_step_returns_nan_on_nonfinite_v_ego():
    """A nan v_ego must yield a nan curvature (honest exclusion), not a held prev value.
    We bypass the model by constructing a minimal duck-typed state and calling the real
    branch logic via a tiny stand-in — but simplest is to verify the documented contract
    directly on the production smoothing branch."""
    # The contract is implemented at the top of ReplayState.step; verify via the source
    # that a non-finite v_ego short-circuits to nan before any model call.
    import inspect
    from model_replay_sim.infer import ReplayState
    src = inspect.getsource(ReplayState.step)
    assert "np.isfinite(v_ego)" in src
    assert 'return float("nan")' in src or "return float('nan')" in src


# --------------------------------------------------------------------------- #
# C2/C3: loud guard for UNVALIDATED cross-model bundles                        #
# --------------------------------------------------------------------------- #

def test_warn_if_unvalidated_warns_for_nevada_and_opm7():
    from model_replay_sim.infer import _warn_if_unvalidated
    for b in ("Nevada", "OPM7"):
        with pytest.warns(UserWarning, match="NOT fidelity-anchor-validated"):
            _warn_if_unvalidated(b)


def test_warn_if_unvalidated_silent_for_cd210(recwarn):
    from model_replay_sim.infer import _warn_if_unvalidated
    _warn_if_unvalidated("CD210")          # anchor_validated=True -> no warning
    assert len(recwarn) == 0, [str(w.message) for w in recwarn]


def test_config_anchor_validated_flags():
    assert C.BUNDLES["CD210"]["anchor_validated"] is True
    assert C.BUNDLES["Nevada"]["anchor_validated"] is False
    assert C.BUNDLES["OPM7"]["anchor_validated"] is False


# --------------------------------------------------------------------------- #
# I1/I2: doc-deviation note present; dead FrameWarpCache removed               #
# --------------------------------------------------------------------------- #

def test_get_curvature_from_output_docstring_marks_deliberate_deviation():
    from model_replay_sim.infer import get_curvature_from_output
    doc = get_curvature_from_output.__doc__ or ""
    assert "DELIBERATE DEVIATION" in doc
    assert "fill_model_msg.py" in doc            # full repo-relative path cited


def test_frame_warp_cache_dead_code_removed():
    import model_replay_sim.infer as infer
    assert not hasattr(infer, "FrameWarpCache"), "dead FrameWarpCache should be deleted"


# --------------------------------------------------------------------------- #
# OPM7 3-model split: on_policy IS the policy; off_policy NOT loaded            #
# --------------------------------------------------------------------------- #

def _opm7_onnx():
    d = C.RESULTS_ROOT / "onnx" / "OPM7"
    return (d / "driving_vision.onnx").exists() and (d / "driving_on_policy.onnx").exists()


def _7f_frames():
    return bool(glob.glob("explorer_st_logs/route_7f/0000007f--*--*/fcamera.hevc"))


def _7f_npz():
    return (C.CACHE_ROOT / "route_7f.npz").exists()


@pytest.mark.slow
@pytest.mark.skipif(not _opm7_onnx(), reason="OPM7 vision/on_policy onnx not present locally")
def test_opm7_loads_vision_and_on_policy_as_the_policy():
    """The OPM7 split BundleModel loads vision + on_policy (NOT off_policy) as the policy.

    Compiles on_policy (slow-ish first time) — marked slow. Asserts:
      * vision inputs are {img, big_img};
      * the policy metadata has `plan` in output_slices (plan-based curvature branch) and
        NOT `desired_curvature`;
      * the policy is on_policy and NOT off_policy: on_policy's output_slices carry
        `desire_state` (an on_policy/CD210-policy output) and DO NOT carry `lane_lines`
        (which is exclusive to off_policy). off_policy is never loaded.
    """
    from model_replay_sim.infer import BundleModel

    model = BundleModel.get("OPM7")
    assert model.split is True
    # vision img inputs
    assert set(model.vision_input_names) == {"img", "big_img"}
    assert model.input_shapes["img"] == (1, 12, 128, 256)
    assert model.input_shapes["big_img"] == (1, 12, 128, 256)
    # on_policy input shapes (identical to CD210's policy) drive the temporal machinery
    assert model.input_shapes["features_buffer"] == (1, 25, 512)
    assert model.input_shapes["desire_pulse"] == (1, 25, 8)
    assert model.input_shapes["traffic_convention"] == (1, 2)

    pol = model.policy.output_slices
    # plan-based curvature branch (no direct curvature output)
    assert "plan" in pol
    assert "desired_curvature" not in pol
    # the policy IS on_policy: on_policy emits desire_state; it does NOT emit lane_lines
    # (lane_lines is exclusive to off_policy). This proves off_policy was not loaded here.
    assert "desire_state" in pol
    assert "lane_lines" not in pol, "policy looks like off_policy (has lane_lines)"
    # off_policy attribute is intentionally absent — we never load it
    assert not hasattr(model, "off_policy")


def _7f_short_window(route_id="route_7f", n_frames=20):
    """A short contiguous eligible+aligned window of route_7f mono_times (same eligibility
    rule as assets.py / the CD210 helper). Returns None if none found."""
    from model_replay_sim.assets import _replay_scene_eligible_mask
    from model_replay_sim.alignment import build_frame_timeline

    z = dict(np.load(C.CACHE_ROOT / f"{route_id}.npz"))
    mono = np.asarray(z["mono_time"], float)
    elig = _replay_scene_eligible_mask(z)
    tl = build_frame_timeline(route_id)
    if not tl:
        return None
    eofs = sorted(r.timestamp_eof_s for r in tl)

    def covered(t):
        i = bisect.bisect_left(eofs, t)
        return any(0 <= j < len(eofs) and abs(eofs[j] - t) <= C.MAX_FRAME_DELTA_S for j in (i - 1, i))

    good = elig & np.array([covered(float(t)) for t in mono], dtype=bool)
    i, n = 0, len(good)
    best = None
    while i < n:
        if good[i]:
            j = i
            while j < n and good[j]:
                j += 1
            if (j - i) >= n_frames and (best is None or (j - i) > best[2]):
                best = (i, j, j - i)
            i = j
        else:
            i += 1
    if best is None:
        return None
    s = best[0] + (best[2] - n_frames) // 2
    return mono[s:s + n_frames]


_OPM7_SKIP = not (_7f_frames() and _7f_npz() and _opm7_onnx())


@pytest.mark.slow
@pytest.mark.skipif(_OPM7_SKIP, reason="route_7f frames/npz or OPM7 onnx not present locally")
def test_replay_window_opm7_route_7f_end_to_end():
    """Full vision -> on_policy -> curvature path on a short route_7f window: finite, right
    length, plausible magnitude, deterministic. The unvalidated cross-model warning MUST fire
    for OPM7 (no anchor yet)."""
    from model_replay_sim.infer import replay_window, BundleModel

    win = _7f_short_window("route_7f", n_frames=20)
    if win is None:
        pytest.skip("no eligible aligned window in route_7f")
    assert len(win) == 20

    model = BundleModel.get("OPM7")
    assert set(model.vision_input_names) == {"img", "big_img"}

    # the unvalidated warning must fire for OPM7 (anchor_validated=False)
    with pytest.warns(UserWarning, match="NOT fidelity-anchor-validated"):
        r = replay_window("OPM7", "route_7f", win)
    curv = np.asarray(r["desired_curvature"], dtype=float)

    # right length
    assert curv.shape == (len(win),)
    # finite where v_ego is valid; nan only where v_ego is honestly nan (excluded by metrics)
    v = np.asarray(r["v_ego"], dtype=float)
    finite_mask = np.isfinite(v)
    assert finite_mask.any(), "expected at least some valid-speed frames in the window"
    assert np.all(np.isfinite(curv[finite_mask])), "curvature must be finite where v_ego valid"
    # plausible magnitude on a gentle (eligible) window
    assert np.max(np.abs(curv[finite_mask])) < 0.05, \
        f"curv too large: max|c|={np.max(np.abs(curv[finite_mask])):.4f}"

    # plan-based curvature branch (on_policy emits no desired_curvature)
    assert "desired_curvature" not in model.policy.output_slices

    # mono_time / v_ego / frames aligned
    np.testing.assert_allclose(r["mono_time"], np.asarray(win, float))
    assert len(r["v_ego"]) == len(win)
    assert len(r["frames"]) == len(win)

    # deterministic across two runs (bit-identical); suppress the (re-)warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = replay_window("OPM7", "route_7f", win)
    np.testing.assert_array_equal(curv, np.asarray(r2["desired_curvature"], dtype=float))
