import types

import numpy as np
import pytest


def _fake_timeline(n, t0=1000.0, dt=0.05):
    rows = []
    for i in range(n):
        rows.append(types.SimpleNamespace(frame_id=100 + i, timestamp_eof_s=t0 + i * dt,
                                          segment_num=i // 1200, segment_id=i % 1200))
    return tuple(rows)


def test_select_scene_windows_carves_disjoint_windows(monkeypatch):
    import model_replay_sim.anchor as anchor_mod
    from stock_lateral_toolkit.centering import windows as W

    tl = _fake_timeline(3000)
    good = np.ones(3000, dtype=bool)
    monkeypatch.setattr(anchor_mod, "_eligible_aligned_timeline_mask", lambda route: (good, tl))

    ws = W.select_scene_windows("route_fake", n_windows=2, warmup_frames=200,
                                compare_frames=1200, min_compare_frames=600)
    assert len(ws) == 2
    assert len(ws[0]["mono_times"]) == 1400 and ws[0]["split_index"] == 200
    assert len(ws[1]["mono_times"]) == 1400
    # disjoint and ordered
    assert ws[0]["frame_ids"][-1] < ws[1]["frame_ids"][0]


def test_select_scene_windows_short_run_fallback(monkeypatch):
    import model_replay_sim.anchor as anchor_mod
    from stock_lateral_toolkit.centering import windows as W

    tl = _fake_timeline(2300)  # 1400 + 900: second window uses the fallback size
    good = np.ones(2300, dtype=bool)
    monkeypatch.setattr(anchor_mod, "_eligible_aligned_timeline_mask", lambda route: (good, tl))

    ws = W.select_scene_windows("route_fake", 2, 200, 1200, min_compare_frames=600)
    assert len(ws) == 2
    assert len(ws[1]["mono_times"]) == 900          # warmup 200 + compare 700 remainder
    assert ws[1]["n_compare"] == 700


def test_select_scene_windows_no_run_raises(monkeypatch):
    import model_replay_sim.anchor as anchor_mod
    from stock_lateral_toolkit.centering import windows as W

    tl = _fake_timeline(100)
    monkeypatch.setattr(anchor_mod, "_eligible_aligned_timeline_mask",
                        lambda route: (np.ones(100, dtype=bool), tl))
    with pytest.raises(RuntimeError):
        W.select_scene_windows("route_fake", 1, 200, 1200, min_compare_frames=600)
