"""S0/M2a tests: CameraOffset counterfactual plumbing + per-frame output capture.

The shear test compares the SIM's transform against the PRODUCTION
CameraOffsetHelper math (imported from sunnypilot/modeld_v2) — the fidelity
claim of the whole S1 sweep rests on this equivalence.
"""
import inspect
import types

import numpy as np
import pytest


def _fake_ctx(camera_offset=0.0):
    # rpy/height from route_stock05 (medians); device = comma four
    return types.SimpleNamespace(
        rpy_calib=[0.0, 0.0429, -0.0498],
        height=1.22,
        device_type="mici",
        road_sensor="os04c10",
        camera_offset=camera_offset,
    )


def test_window_transforms_shear_matches_production():
    from model_replay_sim.infer import window_transforms
    from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
    from openpilot.common.transformations.camera import DEVICE_CAMERAS

    ctx = _fake_ctx()
    dc = DEVICE_CAMERAS[("mici", "os04c10")]

    M0_main, M0_extra, used0 = window_transforms(ctx, camera_offset=0.0)
    Mx_main, Mx_extra, usedx = window_transforms(ctx, camera_offset=0.04)

    assert used0 == 0.0 and usedx == 0.04
    exp_main = CameraOffsetHelper.apply_camera_offset(M0_main, dc.fcam.intrinsics, 1.22, 0.04)
    exp_extra = CameraOffsetHelper.apply_camera_offset(M0_extra, dc.ecam.intrinsics, 1.22, 0.04)
    np.testing.assert_allclose(Mx_main, exp_main, atol=1e-6)
    np.testing.assert_allclose(Mx_extra, exp_extra, atol=1e-6)
    # offset must actually change the matrix
    assert not np.allclose(M0_main, Mx_main)


def test_window_transforms_default_uses_ctx_offset():
    from model_replay_sim.infer import window_transforms
    ctx = _fake_ctx(camera_offset=0.07)
    M_none, _, used = window_transforms(ctx, camera_offset=None)
    M_explicit, _, _ = window_transforms(ctx, camera_offset=0.07)
    assert used == pytest.approx(0.07)
    np.testing.assert_array_equal(M_none, M_explicit)


def test_replay_window_signature_has_new_params():
    from model_replay_sim.infer import replay_window
    params = inspect.signature(replay_window).parameters
    assert "camera_offset" in params and params["camera_offset"].default is None
    assert "capture_outputs" in params and params["capture_outputs"].default == ()
