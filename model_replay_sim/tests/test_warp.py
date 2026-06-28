import glob

import numpy as np
import pytest

from model_replay_sim.warp import (
    apply_camera_offset,
    contig_nv12_to_visionbuf,
    nv12_layouts,
)

# mici/os04c10 (Comma 4) — this project's device.
MICI_W, MICI_H = 1344, 760


# --------------------------------------------------------------------------- #
# Pure math (no tinygrad, no frames)                                          #
# --------------------------------------------------------------------------- #

def test_shear_zero_offset_is_identity_on_transform():
    # offset 0 -> shear is identity -> transform unchanged.
    M = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 4.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    intr = np.array([[910.0, 0.0, 256.0], [0.0, 910.0, 47.6], [0.0, 0.0, 1.0]])
    out = apply_camera_offset(M, intr, height=1.4, offset_param=0.0)
    assert out.dtype == np.float32
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, M, atol=1e-6)


def test_shear_matches_camera_offset_helper_formula():
    # Reproduce the exact formula from camera_offset_helper.apply_camera_offset and
    # confirm our function matches it (shear @ M).
    M = np.eye(3, dtype=np.float32)
    cy = 47.6
    intr = np.array([[910.0, 0.0, 256.0], [0.0, 910.0, cy], [0.0, 0.0, 1.0]])
    height, offset = 1.4, 0.06
    shear = np.eye(3, dtype=np.float32)
    shear[0, 1] = offset / height
    shear[0, 2] = -offset / height * cy
    expected = (shear @ M).astype(np.float32)
    out = apply_camera_offset(M, intr, height, offset)
    np.testing.assert_allclose(out, expected, atol=1e-6)
    # specific entries land where the formula says
    assert abs(out[0, 1] - offset / height) < 1e-6
    assert abs(out[0, 2] - (-offset / height * cy)) < 1e-6


def test_apply_camera_offset_returns_float32_3x3():
    M = np.eye(3, dtype=np.float64)  # accept float64 input
    intr = np.eye(3)
    out = apply_camera_offset(M, intr, height=1.22, offset_param=0.1)
    assert out.shape == (3, 3) and out.dtype == np.float32


# --------------------------------------------------------------------------- #
# Stride reconciliation (uses get_nv12_info -> tinygrad env, but no GPU work)  #
# --------------------------------------------------------------------------- #

def test_nv12_layouts_mici_known_values():
    L = nv12_layouts(MICI_W, MICI_H)
    assert L["framereader_contiguous_size"] == MICI_W * MICI_H * 3 // 2 == 1_532_160
    assert L["stride"] == 1408 and L["stride_pad"] == 64
    assert L["visionbuf_yuv_size"] == 2_428_928
    assert L["uv_offset"] == 1_081_344
    # the two layouts genuinely differ -> reconciliation is required, not optional
    assert L["visionbuf_yuv_size"] != L["framereader_contiguous_size"]


def test_contig_to_visionbuf_places_real_pixels_and_pads_rest():
    L = nv12_layouts(MICI_W, MICI_H)
    stride, uv_offset = L["stride"], L["uv_offset"]
    rng = np.random.default_rng(7)
    nv12 = rng.integers(0, 256, size=MICI_W * MICI_H * 3 // 2, dtype=np.uint8)
    out = contig_nv12_to_visionbuf(nv12, MICI_W, MICI_H)
    assert out.size == L["visionbuf_yuv_size"] and out.dtype == np.uint8

    # Y rows: real pixels at [:, :cam_w], padded columns zero.
    y_out = out[:MICI_H * stride].reshape(MICI_H, stride)
    y_src = nv12[:MICI_W * MICI_H].reshape(MICI_H, MICI_W)
    np.testing.assert_array_equal(y_out[:, :MICI_W], y_src)
    assert (y_out[:, MICI_W:] == 0).all()

    # UV rows at uv_offset.
    uv_out = out[uv_offset:uv_offset + (MICI_H // 2) * stride].reshape(MICI_H // 2, stride)
    uv_src = nv12[MICI_W * MICI_H:].reshape(MICI_H // 2, MICI_W)
    np.testing.assert_array_equal(uv_out[:, :MICI_W], uv_src)
    assert (uv_out[:, MICI_W:] == 0).all()


def test_contig_to_visionbuf_rejects_wrong_size():
    with pytest.raises(ValueError, match="contiguous nv12 size"):
        contig_nv12_to_visionbuf(np.zeros(10, dtype=np.uint8), MICI_W, MICI_H)


# --------------------------------------------------------------------------- #
# Real route_b5 frame -> model input tensor (skip if no frames locally)        #
# --------------------------------------------------------------------------- #

def _b5_seg16():
    return bool(glob.glob("explorer_st_logs/route_b5/000000b5--*--16/fcamera.hevc"))


@pytest.mark.skipif(not _b5_seg16(), reason="no route_b5 seg-16 fcamera frames locally")
def test_real_frame_to_model_input_shape_dtype_determinism():
    from model_replay_sim.alignment import build_frame_timeline, read_frame
    from model_replay_sim.context import route_context
    from model_replay_sim.warp import frame_to_model_input, frame_to_sixchan, model_transform

    ctx = route_context("route_b5")
    assert (ctx.device_type, ctx.road_sensor) == ("mici", "os04c10")

    tl = build_frame_timeline("route_b5")
    seg16 = [r for r in tl if r.segment_num == 16]
    assert seg16, "route_b5 seg16 should have frames"
    sid = seg16[0].segment_id
    nv12 = read_frame("route_b5", 16, sid)
    cam_w, cam_h = MICI_W, MICI_H
    assert nv12.shape == (cam_w * cam_h * 3 // 2,)

    M = model_transform(ctx, wide=False)
    assert M.shape == (3, 3) and M.dtype == np.float32

    # First frame (no history): both halves = current frame.
    out = frame_to_model_input(nv12, cam_w, cam_h, M, prev_sixchan=None)
    assert out.shape == (1, 12, 128, 256)   # EXACT metadata img shape
    assert out.dtype == np.uint8            # onnx vision input is UINT8
    # warp-only path is uint8, never normalized -> full byte range present
    assert out.min() >= 0 and out.max() <= 255 and out.max() > out.min()

    # Determinism: identical frame + transform -> identical tensor.
    out2 = frame_to_model_input(nv12, cam_w, cam_h, M, prev_sixchan=None)
    np.testing.assert_array_equal(out, out2)

    # prev/current wiring: with prev=None both halves equal; the sixchan of the current
    # frame is exactly the second half (channels 6:12).
    cur6 = frame_to_sixchan(nv12, cam_w, cam_h, M)
    np.testing.assert_array_equal(out[0, 6:12], cur6)
    np.testing.assert_array_equal(out[0, 0:6], cur6)  # prev=None duplicates current

    # With a distinct previous sixchan, the first half changes, second half stays current.
    nv12_prev = read_frame("route_b5", 16, seg16[1].segment_id) if len(seg16) > 1 else nv12
    prev6 = frame_to_sixchan(nv12_prev, cam_w, cam_h, M)
    out3 = frame_to_model_input(nv12, cam_w, cam_h, M, prev_sixchan=prev6)
    np.testing.assert_array_equal(out3[0, 0:6], prev6)
    np.testing.assert_array_equal(out3[0, 6:12], cur6)


@pytest.mark.skipif(not _b5_seg16(), reason="no route_b5 seg-16 fcamera frames locally")
def test_wide_transform_uses_ecam_and_bigmodel():
    from model_replay_sim.context import route_context
    from model_replay_sim.warp import model_transform

    ctx = route_context("route_b5")
    M_road = model_transform(ctx, wide=False)
    M_wide = model_transform(ctx, wide=True)
    assert M_road.shape == (3, 3) and M_wide.shape == (3, 3)
    # road (fcam, medmodel) and wide (ecam, sbigmodel) warps differ.
    assert not np.allclose(M_road, M_wide)
