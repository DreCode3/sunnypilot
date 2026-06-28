"""Task 7: warp PIXEL pipeline — decoded camera frame -> model input tensor.

Turns a decoded nv12 camera frame into the CD210/Nevada/OPM7 **vision** model input
tensor ``img``/``big_img`` of shape ``(1, 12, 128, 256)`` dtype ``uint8`` (12 channels =
6 current-frame sixchan + 6 previous-frame sixchan; per-frame 6 = 4 sub-sampled Y planes
+ U + V, each 128x256 = ``MEDMODEL_INPUT_SIZE``/2).

FIDELITY: reuses the PRODUCTION compiled-tinygrad warp+pack verbatim
(``selfdrive/modeld/compile_warp.make_frame_prepare`` -> ``frames_to_tensor``), the same
code the device runs. We do NOT reimplement the warp in numpy. The model's normalization
lives INSIDE the onnx graph (the onnx ``img`` input is UINT8), so this pixel pipeline is
uint8 end-to-end with no normalization — matching production.

THE STRIDE RECONCILIATION (the crux of this task):
  ``FrameReader(...,pix_fmt="nv12").get(idx)`` returns a flat *contiguous* nv12 buffer of
  size ``cam_w*cam_h*3//2`` (Y plane at row-stride ``cam_w``, then interleaved UVUV at
  row-stride ``cam_w``). The production ``make_frame_prepare`` / ``get_nv12_info`` instead
  assume the on-device *VisionBuf* layout, which has stride padding
  (``stride = align(cam_w,128) > cam_w``) and a UV plane offset of ``stride*y_height``.
  Feeding the contiguous buffer straight in reads garbage.

  We reconcile by REPACKING the contiguous frame into the strided VisionBuf layout
  (:func:`contig_nv12_to_visionbuf`) before handing it to ``make_frame_prepare``. This is
  provably faithful: ``warp_perspective_tinygrad`` only ever samples source index
  ``y*(w_src+stride_pad)+x`` for ``0<=x<cam_w`` and ``0<=y<cam_h`` (after a round+clip to
  ``[0,w_src-1]``/``[0,cam_h-1]``), so the padded columns/rows are never read — only the
  real pixels matter, and we place them at exactly the strided positions the indexing
  expects. The UV deinterleave (``uv[:, 0::2]``=U, ``uv[:, 1::2]``=V) is identical between
  layouts; only the row stride differs.

  For mici/os04c10 (the Comma 4, this project's device): cam_w=1344, cam_h=760,
  contiguous size 1_532_160, VisionBuf yuv_size 2_428_928, stride 1408 (pad 64),
  uv_offset 1_081_344.

TRANSFORM: :func:`model_transform` builds the forward 3x3 the production code feeds to the
warp as its ``M_inv`` argument: ``get_warp_matrix(rpyCalib, intrinsics, bigmodel)`` then the
sunnypilot camera-offset shear (``apply_camera_offset``). ``warp_perspective_tinygrad``
treats this matrix as dst(model-pixel)->src(camera-pixel), so NO inversion is applied —
matching ``sunnypilot/modeld_v2/{modeld,warp}.py`` which pass ``get_warp_matrix(...)``
straight through as the warp's ``M_inv`` arg.

Analysis-only. Imports tinygrad — run under ``model_replay_sim.env`` semantics (DEBUG=0
DEV=CPU IMAGE=0 THREADS=0; ``tinygrad_repo`` + repo root on sys.path) set BEFORE importing
tinygrad. :func:`_ensure_tinygrad_env` arranges this for in-process use.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from model_replay_sim import config as C


def _ensure_tinygrad_env() -> None:
    """Set the tinygrad env + sys.path the way ``env.replay_env`` does, BEFORE the first
    tinygrad import in this process. Idempotent; safe to call repeatedly."""
    for k, v in C.TINYGRAD_ENV.items():
        os.environ.setdefault(k, v)
    for p in (str(C.TINYGRAD_PATH), str(C.REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Pure transform math (no tinygrad / no frame needed)
# ---------------------------------------------------------------------------

def apply_camera_offset(model_transform: np.ndarray, intrinsics: np.ndarray,
                        height: float, offset_param: float) -> np.ndarray:
    """The sunnypilot camera-offset shear, reproduced from
    ``sunnypilot/modeld_v2/camera_offset_helper.CameraOffsetHelper.apply_camera_offset``.
    Pure: shears the warp matrix to model a lateral camera mounting offset.
    ``shear = I; shear[0,1] = offset/height; shear[0,2] = -offset/height * cy`` then
    ``shear @ model_transform`` (float32). ``cy = intrinsics[1,2]``."""
    cy = float(intrinsics[1, 2])
    shear = np.eye(3, dtype=np.float32)
    shear[0, 1] = offset_param / height
    shear[0, 2] = -offset_param / height * cy
    return (shear @ model_transform).astype(np.float32)


def model_transform(ctx, wide: bool = False, camera_offset: float | None = None) -> np.ndarray:
    """The forward 3x3 model warp matrix for ``ctx`` (a Task-4 ``RouteContext``).

    = ``get_warp_matrix(rpyCalib, intrinsics, bigmodel_frame=wide)`` then the camera-offset
    shear. Mirrors ``sunnypilot/modeld_v2/modeld.py`` lines 294-298: the ROAD ("img") path
    uses ``bigmodel_frame=False`` + fcam intrinsics (or ecam if main-wide); the WIDE
    ("big_img") path uses ``bigmodel_frame=True`` + ecam intrinsics. This project's device
    (mici, road=fcam) is NOT main-wide, so ``wide=False`` -> fcam, ``wide=True`` -> ecam.

    ``camera_offset`` defaults to ``ctx.camera_offset``. (Production EMAs the live param via
    ``0.9*old+0.1*new``; for a static-scene replay we use the steady value directly — pass
    an explicit value to model the EMA if needed.)

    Returns a 3x3 float32 ndarray.
    """
    _ensure_tinygrad_env()
    from openpilot.common.transformations.model import get_warp_matrix
    from openpilot.common.transformations.camera import DEVICE_CAMERAS

    if camera_offset is None:
        camera_offset = float(getattr(ctx, "camera_offset", 0.0))

    dc = DEVICE_CAMERAS[(str(ctx.device_type), str(ctx.road_sensor))]
    intrinsics = dc.ecam.intrinsics if wide else dc.fcam.intrinsics

    rpy = np.asarray(ctx.rpy_calib, dtype=np.float32)
    M = get_warp_matrix(rpy, intrinsics, bigmodel_frame=wide).astype(np.float32)
    M = apply_camera_offset(M, intrinsics, float(ctx.height), float(camera_offset))
    return M.astype(np.float32)


# ---------------------------------------------------------------------------
# Stride reconciliation: contiguous FrameReader nv12 -> strided VisionBuf layout
# ---------------------------------------------------------------------------

def nv12_layouts(cam_w: int, cam_h: int) -> dict:
    """Both nv12 layouts for ``cam_w x cam_h`` — what FrameReader gives vs what
    ``make_frame_prepare`` expects — so callers/tests can assert the reconciliation."""
    _ensure_tinygrad_env()
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    stride, y_height, uv_height, yuv_size = get_nv12_info(cam_w, cam_h)
    return {
        "cam_w": cam_w, "cam_h": cam_h,
        "framereader_contiguous_size": cam_w * cam_h * 3 // 2,
        "stride": stride, "y_height": y_height, "uv_height": uv_height,
        "visionbuf_yuv_size": yuv_size,
        "stride_pad": stride - cam_w,
        "uv_offset": stride * y_height,
    }


def contig_nv12_to_visionbuf(nv12_flat: np.ndarray, cam_w: int, cam_h: int) -> np.ndarray:
    """Repack a *contiguous* FrameReader nv12 frame into the strided device VisionBuf
    layout that ``make_frame_prepare``/``get_nv12_info`` index. See module docstring for
    the faithfulness argument (padded region is never sampled by the warp).

    Returns a contiguous uint8 array of size ``get_nv12_info(cam_w,cam_h)[3]``.
    """
    nv12_flat = np.ascontiguousarray(nv12_flat, dtype=np.uint8).ravel()
    expected = cam_w * cam_h * 3 // 2
    if nv12_flat.size != expected:
        raise ValueError(f"contiguous nv12 size {nv12_flat.size} != cam_w*cam_h*3//2={expected} "
                         f"for {cam_w}x{cam_h}")
    L = nv12_layouts(cam_w, cam_h)
    stride, y_height, yuv_size, uv_offset = L["stride"], L["y_height"], L["visionbuf_yuv_size"], L["uv_offset"]

    out = np.zeros(yuv_size, dtype=np.uint8)
    # Y plane: cam_h rows of cam_w real pixels into a stride-padded region.
    y_src = nv12_flat[:cam_w * cam_h].reshape(cam_h, cam_w)
    out[:cam_h * stride].reshape(cam_h, stride)[:, :cam_w] = y_src
    # UV plane (interleaved UVUV, cam_h//2 rows of cam_w bytes) at uv_offset, same stride.
    uv_src = nv12_flat[cam_w * cam_h:].reshape(cam_h // 2, cam_w)
    out[uv_offset:uv_offset + (cam_h // 2) * stride].reshape(cam_h // 2, stride)[:, :cam_w] = uv_src
    return out


# ---------------------------------------------------------------------------
# Frame -> sixchan -> model input tensor
# ---------------------------------------------------------------------------

# Cache the per-(cam_w,cam_h) production frame_prepare closure (it precomputes strides).
_FRAME_PREPARE_CACHE: dict[tuple[int, int], object] = {}


def _get_frame_prepare(cam_w: int, cam_h: int):
    key = (cam_w, cam_h)
    fp = _FRAME_PREPARE_CACHE.get(key)
    if fp is None:
        _ensure_tinygrad_env()
        from openpilot.selfdrive.modeld.compile_warp import make_frame_prepare, MEDMODEL_INPUT_SIZE
        model_w, model_h = MEDMODEL_INPUT_SIZE
        fp = make_frame_prepare(cam_w, cam_h, model_w, model_h)
        _FRAME_PREPARE_CACHE[key] = fp
    return fp


def frame_to_sixchan(nv12_flat: np.ndarray, cam_w: int, cam_h: int,
                     transform: np.ndarray) -> np.ndarray:
    """Warp + sixchan-pack ONE frame -> ``(6, 128, 256)`` uint8 ndarray, using the
    production ``make_frame_prepare`` (compiled tinygrad warp -> ``frames_to_tensor``).

    ``transform`` is the forward 3x3 :func:`model_transform` (fed to the warp as its
    ``M_inv`` arg, exactly as production does — no inversion).
    """
    _ensure_tinygrad_env()
    from tinygrad.tensor import Tensor

    M = np.ascontiguousarray(transform, dtype=np.float32).reshape(3, 3)
    strided = contig_nv12_to_visionbuf(nv12_flat, cam_w, cam_h)
    yuv_size = strided.size

    blob = Tensor.from_blob(strided.ctypes.data, (yuv_size,), dtype="uint8").realize()
    # M_inv MUST be a default-device (CPU) tensor here, NOT device='NPY'. NPY is a
    # JIT-input marker in production; outside the TinyJit it yields a device-less kernel
    # ("needs a renderer"). A CPU tensor gives the identical computation.
    M_t = Tensor(M)
    out = _get_frame_prepare(cam_w, cam_h)(blob, M_t)
    arr = out.numpy()
    # keep the strided buffer alive until the blob has been consumed
    del strided, blob
    return np.ascontiguousarray(arr, dtype=np.uint8)


def frame_to_model_input(nv12_flat: np.ndarray, cam_w: int, cam_h: int,
                         transform: np.ndarray,
                         prev_sixchan: np.ndarray | None = None) -> np.ndarray:
    """Build the vision model input tensor ``(1, 12, 128, 256)`` uint8 for ONE frame.

    Channels 0:6 = the previous frame's sixchan, 6:12 = the current frame's sixchan —
    matching production ``make_update_img_input``: the rolling buffer is rotated by 6 and
    the model input is ``cat(buffer[:6], buffer[-6:])`` = (oldest 6, newest 6). For a
    standalone pair that is (prev, current).

    ``prev_sixchan``: the previous frame's ``(6,128,256)`` uint8 sixchan (e.g. from a prior
    :func:`frame_to_sixchan`). If ``None`` (first frame / no history) the current frame is
    duplicated into both halves — the same warm-up behavior as starting from a zeroed buffer
    that has been filled with the first frame.

    Returns a contiguous uint8 ndarray ``(1,12,128,256)``. Use ``frame_to_sixchan`` on this
    frame to obtain the sixchan to pass as ``prev_sixchan`` for the NEXT frame.
    """
    cur = frame_to_sixchan(nv12_flat, cam_w, cam_h, transform)  # (6,128,256)
    if prev_sixchan is None:
        prev = cur
    else:
        prev = np.ascontiguousarray(prev_sixchan, dtype=np.uint8)
        if prev.shape != cur.shape:
            raise ValueError(f"prev_sixchan shape {prev.shape} != current {cur.shape}")
    pair = np.concatenate([prev, cur], axis=0)            # (12,128,256)
    return np.ascontiguousarray(pair[None], dtype=np.uint8)  # (1,12,128,256)
