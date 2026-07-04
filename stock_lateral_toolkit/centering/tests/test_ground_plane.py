import numpy as np
import pytest


FX = FY = 1141.5   # DEVICE_CAMERAS[("mici","os04c10")].fcam focal (2688//2 x 1520//2)
CX, CY = 672.0, 380.0
H = 1.22


def test_identity_rpy_known_projection():
    """With rpy=(0,0,0), a road point (x=10 m, y=0) must land at u=cx,
    v = cy + fy*h/x (pinhole flat-road geometry) — this pins the whole
    convention stack (road frame y-LEFT, view frame permutation, intrinsics)."""
    from stock_lateral_toolkit.centering import ground_plane as G
    K = G.fcam_intrinsics()
    assert K[0, 0] == pytest.approx(FX) and K[1, 2] == pytest.approx(CY)
    Hm = G.road_homography([0.0, 0.0, 0.0], H, K)
    u, v = G.pixel_from_road(Hm, 10.0, 0.0)
    assert u == pytest.approx(CX, abs=0.5)
    assert v == pytest.approx(CY + FY * H / 10.0, abs=0.5)
    # a point 1.8 m to the LEFT (road y positive-left) must be LEFT in the image (u < cx)
    u_left, _ = G.pixel_from_road(Hm, 10.0, 1.8)
    assert u_left < CX - 100


def test_round_trip_with_real_calibration():
    from stock_lateral_toolkit.centering import ground_plane as G
    K = G.fcam_intrinsics()
    Hm = G.road_homography([0.0, 0.0429, -0.0498], H, K)   # stock05 median rpy
    for x in (6.0, 10.0, 18.0):
        for y in (-3.0, -1.7, 0.0, 1.7, 3.0):
            u, v = G.pixel_from_road(Hm, x, y)
            xr, yr = G.road_from_pixel(Hm, u, v)
            assert xr == pytest.approx(x, abs=1e-6)
            assert yr == pytest.approx(y, abs=1e-6)


def test_above_horizon_raises():
    from stock_lateral_toolkit.centering import ground_plane as G
    Hm = G.road_homography([0.0, 0.0, 0.0], H, G.fcam_intrinsics())
    with pytest.raises(ValueError):
        G.road_from_pixel(Hm, 672.0, 100.0)   # well above the horizon row


def test_sign_helper():
    from stock_lateral_toolkit.centering import ground_plane as G
    assert G.y_cal_from_y_road(1.8) == -1.8   # road +LEFT -> calibrated +RIGHT
