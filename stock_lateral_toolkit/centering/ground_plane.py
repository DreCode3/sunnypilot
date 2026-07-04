"""Flat-road ground-plane projection for M1 video ground truth.

Frames (common/transformations/README.md):
  calibrated frame: [x fwd, y RIGHT, z down] — model laneLines y lives here.
  road frame (get_view_frame_from_road_frame): [x fwd, y LEFT, z up].
Homography: KE = K @ get_view_frame_from_road_frame(rpyCalib, height) (3x4); road-plane
points [x, y, 0, 1] project through H = KE[:, [0, 1, 3]]. Feeding liveCalibration's
rpyCalib is exact here because device_from_road = R(rpy) @ diag([1,-1,-1]) ==
device_from_calib @ calib_from_road, and stock cal_roll == 0.

Everything in this module is CAMERA-relative (the same origin the model's laneLines
use). Vehicle-centerline conversion (the tape-measured lever arm) happens ONLY in
m1_offsets.py. Uncertainty budget: pre-registration doc §2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpilot.common.transformations.camera import DEVICE_CAMERAS, get_view_frame_from_road_frame


def fcam_intrinsics(device_type: str = "mici", sensor: str = "os04c10") -> np.ndarray:
    return DEVICE_CAMERAS[(device_type, sensor)].fcam.intrinsics


def road_homography(rpy_calib, height_m: float, intrinsics: np.ndarray) -> np.ndarray:
    """3x3 homography: road-plane [x_fwd, y_left, 1] -> pixel homogeneous [u, v, 1]."""
    roll, pitch, yaw = (float(v) for v in rpy_calib)
    ke = np.asarray(intrinsics, float) @ get_view_frame_from_road_frame(roll, pitch, yaw, float(height_m))
    return ke[:, [0, 1, 3]]


def pixel_from_road(H: np.ndarray, x_fwd_m: float, y_left_m: float) -> tuple[float, float]:
    p = H @ np.array([x_fwd_m, y_left_m, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def road_from_pixel(H: np.ndarray, u_px: float, v_px: float) -> tuple[float, float]:
    q = np.linalg.solve(H, np.array([u_px, v_px, 1.0]))
    if abs(q[2]) < 1e-12 or (q[0] / q[2]) <= 0.0:
        raise ValueError(f"pixel ({u_px:.0f},{v_px:.0f}) maps above the horizon / behind the camera")
    return float(q[0] / q[2]), float(q[1] / q[2])


def y_cal_from_y_road(y_left_m: float) -> float:
    """road frame y (+LEFT) -> calibrated frame y (+RIGHT)."""
    return -float(y_left_m)


def sigma_frame_m() -> float:
    """Pre-registered per-frame 1-sigma (M0 §2): quadrature of the declared terms."""
    from stock_lateral_toolkit.centering import config as CC
    return float(np.sqrt(CC.SIGMA_PIXEL_M ** 2 + CC.SIGMA_ROLL_M ** 2
                         + CC.SIGMA_YAW_M ** 2 + CC.SIGMA_MISC_M ** 2))
