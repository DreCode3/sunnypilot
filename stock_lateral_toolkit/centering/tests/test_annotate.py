import numpy as np
import pytest


RPY = [0.0, 0.0429, -0.0498]
H = 1.22


def _paint_image(y_left=1.75, y_right=-1.72):
    """Synthetic 760x1344 luma: dark road with two bright 0.125 m paint stripes
    at road-frame lateral y_left / y_right, painted by forward projection."""
    from stock_lateral_toolkit.centering import ground_plane as G
    img = np.full((760, 1344), 60.0)
    Hm = G.road_homography(RPY, H, G.fcam_intrinsics())
    for x in np.arange(5.0, 25.0, 0.02):
        for yc in (y_left, y_right):
            for y in np.arange(yc - 0.0625, yc + 0.0625, 0.01):
                u, v = G.pixel_from_road(Hm, float(x), float(y))
                ui, vi = int(round(u)), int(round(v))
                if 0 <= vi < 760 and 0 <= ui < 1344:
                    img[vi, ui] = 220.0
    return img, Hm


def test_propose_line_recovers_painted_stripes():
    from stock_lateral_toolkit.centering import annotate as A
    img, Hm = _paint_image()
    for x_m in (8.0, 12.0, 16.0):
        left = A.propose_line(img, Hm, x_m, "left")
        right = A.propose_line(img, Hm, x_m, "right")
        assert left["auto_ok"] and right["auto_ok"]
        assert left["y_road"] == pytest.approx(1.75, abs=0.03)
        assert right["y_road"] == pytest.approx(-1.72, abs=0.03)


def test_propose_line_flags_blank_road():
    from stock_lateral_toolkit.centering import annotate as A
    from stock_lateral_toolkit.centering import ground_plane as G
    rng = np.random.default_rng(0)
    img = np.full((760, 1344), 60.0) + rng.normal(0, 2.0, (760, 1344))
    Hm = G.road_homography(RPY, H, G.fcam_intrinsics())
    p = A.propose_line(img, Hm, 12.0, "left")
    assert not p["auto_ok"]          # nothing mark-like -> low contrast


def test_luma_plane_shape():
    from stock_lateral_toolkit.centering import annotate as A
    flat = np.arange(1344 * 760 * 3 // 2, dtype=np.uint8)
    y = A.luma_plane(flat)
    assert y.shape == (760, 1344)
    assert y[0, 5] == 5.0
