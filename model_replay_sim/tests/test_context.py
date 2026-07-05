import glob, pytest
from model_replay_sim.context import bundle_lat_smooth_seconds, route_context, lateral_delay_input

def test_opm7_lat_smooth_is_zero_from_pinned_commit():
    assert bundle_lat_smooth_seconds("OPM7") == 0.0      # FIX 2: sourced from 052692b2..., not a literal

def test_cd210_nevada_lat_smooth_are_floats_from_their_commits():
    for b in ("CD210", "Nevada"):
        v = bundle_lat_smooth_seconds(b)
        assert isinstance(v, float)

def test_lateral_delay_input_uses_base_plus_bundle_const_no_literal():
    # base selection: LagdToggle -> LagdValueCache else liveDelay.lateralDelay; then + bundle LAT_SMOOTH
    class Ctx:  # minimal stand-in
        lagd_toggle = True; lagd_value_cache = 0.30; live_lateral_delay = 0.20
    assert abs(lateral_delay_input(Ctx, "OPM7") - (0.30 + 0.0)) < 1e-9      # toggle on -> cache; OPM7 +0.0
    class Ctx2:
        lagd_toggle = False; lagd_value_cache = 0.30; live_lateral_delay = 0.20
    assert abs(lateral_delay_input(Ctx2, "OPM7") - (0.20 + 0.0)) < 1e-9     # toggle off -> liveDelay

def _b5():
    return "route_b5" if glob.glob("explorer_st_logs/route_b5/000000b5--*--*/rlog.zst") else None

@pytest.mark.skipif(_b5() is None, reason="no route_b5 rlog")
def test_route_context_reads_real_route():
    ctx = route_context("route_b5")
    assert len(ctx.rpy_calib) == 3 and all(isinstance(x, float) for x in ctx.rpy_calib)
    assert ctx.active_bundle_internal_name == "C210M"          # route_b5 ran CD210
    assert isinstance(ctx.live_lateral_delay, float)
