def _base():
    return dict(proceed_to_r=True, p_cam=0.10, dmid=0.01, dwidth=0.02, logged=0.11,
                crown_significant=False, crown_component=0.0)


def test_trained_preference():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**_base())
    assert v["class"] == "TRAINED_PATH_PREFERENCE"
    assert v["camera_offset_is_the_lever"] is True


def test_definition_only():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "p_cam": 0.02, "dmid": 0.09})
    assert v["class"] == "MODEL_FRAME_DEFINITION_ONLY"
    assert v["camera_offset_is_the_lever"] is False


def test_mixed():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "p_cam": 0.08, "dmid": 0.06})
    assert v["class"] == "MIXED_TRANSLATION_PREFERENCE"
    assert v["camera_offset_is_the_lever"] is True


def test_crown_dominant_blocks_camera_offset():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "crown_significant": True, "crown_component": 0.08})
    assert "CROWN" in v["class"]
    assert v["camera_offset_is_the_lever"] is False   # crown fraction 0.08/0.11 > 0.5


def test_method_gate_blocks_everything():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "proceed_to_r": False})
    assert v["class"] == "NO_VERDICT_METHOD_DISAGREEMENT"


def test_no_deficit():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "p_cam": 0.01, "dmid": 0.01, "logged": 0.02})
    assert v["class"] == "NO_DEFICIT_MEASURABLE"
