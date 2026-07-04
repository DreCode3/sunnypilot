import numpy as np
import pytest


def _rows(y_left=1.7, y_right=-1.9):
    """Proposals for one frame at 3 distances, both sides accepted. Road-frame y
    (+LEFT): left line +1.7, right line -1.9 => midpoint road -0.1 => canonical
    offset (car LEFT of center, calibrated y of midpoint) = +0.1."""
    rows = []
    for x in (8.0, 12.0, 16.0):
        rows.append(dict(frame_idx=0, x_m=x, side="left", y_road=y_left, contrast=9.0,
                         auto_ok=True, verdict="", corrected_u_px="", u_px=0.0, v_px=0.0))
        rows.append(dict(frame_idx=0, x_m=x, side="right", y_road=y_right, contrast=9.0,
                         auto_ok=True, verdict="", corrected_u_px="", u_px=0.0, v_px=0.0))
    return rows


def test_frame_offset_sign_convention():
    from stock_lateral_toolkit.centering import m1_offsets as M
    off = M.frame_offset_cam(_rows())
    assert off == pytest.approx(+0.10, abs=1e-9)   # car 0.10 m LEFT of center


def test_trust_rule():
    from stock_lateral_toolkit.centering import m1_offsets as M
    reviewed = [dict(verdict="accept", u_px=100.0, corrected_u_px="")] * 17 \
             + [dict(verdict="correct", u_px=100.0, corrected_u_px="103.0")] * 1 \
             + [dict(verdict="reject", u_px=100.0, corrected_u_px="")] * 2
    ok, frac = M.trust_check(reviewed)
    assert ok and frac == pytest.approx(0.90)
    reviewed_bad = [dict(verdict="reject", u_px=0.0, corrected_u_px="")] * 5 \
                 + [dict(verdict="accept", u_px=0.0, corrected_u_px="")] * 5
    ok2, frac2 = M.trust_check(reviewed_bad)
    assert not ok2 and frac2 == pytest.approx(0.50)


def test_frame_offset_requires_both_sides():
    from stock_lateral_toolkit.centering import m1_offsets as M
    rows = [r for r in _rows() if r["side"] == "left"]
    assert M.frame_offset_cam(rows) is None
