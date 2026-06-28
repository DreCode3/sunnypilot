from model_replay_sim import config as C
def test_merged_config_shape():
    assert C.REPO_ROOT.name == "sunnypilot"
    assert C.RESULTS_ROOT == C.REPO_ROOT/"retrospective_lateral"/"results"/"model_replay"
    assert C.WEAVE_BAND_HZ == (0.10, 0.35)
    assert C.GENTLE_CURV_ABS_MAX_1PM == 0.005
    assert C.ANCHOR_CORR_MIN == 0.95
    assert C.ANCHOR_BAND_RATIO == (0.85, 1.15)
    assert C.BUNDLES["CD210"]["full_sha"] == "55f66e2246359c6593605399a0199d94d13ad90d"
    assert C.BUNDLES["CD210"]["repo"] == "commaai/openpilot" and C.BUNDLES["CD210"]["split"] is False
    assert C.BUNDLES["Nevada"]["full_sha"] == "3193eac5e385aa010694a8ac192ff38ffe000193"
    assert C.BUNDLES["OPM7"]["full_sha"] == "052692b25d63c5ddda276b5c2271383b6aff129f" and C.BUNDLES["OPM7"]["split"] is True
