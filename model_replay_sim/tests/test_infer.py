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
