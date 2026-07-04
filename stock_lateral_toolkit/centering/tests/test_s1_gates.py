import numpy as np
import pytest


def _synth(slope=1.0, weave_jitter=0.0):
    offsets = [round(-0.10 + 0.02 * i, 3) for i in range(11)]
    rows = []
    for off in offsets:
        rows.append(dict(offset=off, d_center_m=slope * off,
                         band_ratio=1.0 + weave_jitter * abs(off) / 0.10,
                         corr_vs_zero=0.999, low_band_ratio=1.0))
    controls = {(-0.005): 1.005, (0.005): 0.996}   # -> NB = max(0.03, 2*0.005) = 0.03
    return rows, controls


def test_gates_pass_on_clean_unit_response():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth()
    g = S.evaluate_gates(rows, controls, determinism_max_delta=0.0, p_cam=0.08)
    assert g["monotonic"] and g["slope_ok"] and g["weave_ok"] and g["curve_ok"]
    assert g["slope"] == pytest.approx(1.0, abs=0.01)
    assert g["delta_star_m"] == pytest.approx(0.08, abs=0.011)  # snapped to the grid
    assert g["all_pass"] is True


def test_weave_gate_fails_outside_noise_band():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth(weave_jitter=0.2)      # band_ratio up to 1.2 > hard cap
    g = S.evaluate_gates(rows, controls, 0.0, p_cam=0.08)
    assert not g["weave_ok"] and g["all_pass"] is False


def test_slope_gate_fails_on_flat_response():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth(slope=0.1)
    g = S.evaluate_gates(rows, controls, 0.0, p_cam=0.08)
    assert not g["slope_ok"] and g["all_pass"] is False


def test_delta_star_out_of_range_flags():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth(slope=0.6)             # delta* = 0.09/0.6 = 0.15 > 0.10
    g = S.evaluate_gates(rows, controls, 0.0, p_cam=0.09)
    assert g["delta_star_in_range"] is False and g["all_pass"] is False
