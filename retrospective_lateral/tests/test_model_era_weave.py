from retrospective_lateral.code import model_era_weave as E


def _route(dis_rms, eng_rms, bins=(25.0, 30.0)):
    prof = {}
    for lo in bins:
        prof[("model_y20", lo, "eng")] = eng_rms
        prof[("model_y20", lo, "dis")] = dis_rms
    return prof


def test_aggregate_by_model_compares_open_loop_weave_across_models():
    per_route = {
        "route_1": _route(6e-4, 3.5e-4), "route_2": _route(5.5e-4, 3.4e-4),   # CD210 (loud)
        "route_3": _route(3.0e-4, 3.0e-4), "route_4": _route(2.8e-4, 2.9e-4),  # OPM7 (quiet)
    }
    labels = {"route_1": "C210M", "route_2": "C210M", "route_3": "OPM7", "route_4": "OPM7"}
    rows = E.aggregate_by_model(per_route, labels, signals=("model_y20",))
    by_model = {r["model"]: r for r in rows}
    assert set(by_model) == {"CD210", "OPM7"}
    assert by_model["CD210"]["routes"] == 2 and by_model["OPM7"]["routes"] == 2
    # the loud model has higher open-loop (disengaged) weave than the quiet one
    assert by_model["CD210"]["disengaged_weave_rms_1e4"] > by_model["OPM7"]["disengaged_weave_rms_1e4"]


def test_aggregate_ignores_unlabeled_and_empty():
    per_route = {"route_1": _route(6e-4, 3.5e-4), "route_x": _route(1e-4, 1e-4), "route_e": {}}
    labels = {"route_1": "C210M", "route_x": "unknown"}  # route_x model not in MODEL_GROUPS
    rows = E.aggregate_by_model(per_route, labels, signals=("model_y20",))
    assert {r["model"] for r in rows} == {"CD210"}
