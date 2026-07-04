import numpy as np


def _synth_cache(n=8000, dt=0.05):
    """Straight-line eastbound drive at 20 m/s with all gates open."""
    t = 1000.0 + np.arange(n) * dt
    lon0, lat0 = -84.7, 33.9
    lon = lon0 + (20.0 * np.arange(n) * dt) / (111320.0 * np.cos(np.radians(lat0)))
    return {
        "mono_time": t,
        "v_ego": np.full(n, 20.0),
        "lat": np.full(n, lat0) , "lon": lon,
        "lat_active": np.ones(n), "steering_pressed": np.zeros(n),
        "blinker": np.zeros(n), "lane_change_state": np.zeros(n),
        "lane_prob_left": np.full(n, 0.9), "lane_prob_right": np.full(n, 0.9),
    }


def test_sample_respects_separation_and_is_deterministic():
    from stock_lateral_toolkit.centering import frame_sampler as FS
    z = _synth_cache()
    in_win = np.zeros(len(z["mono_time"]), dtype=bool)
    in_win[2000:4000] = True
    a = FS.sample(z, in_win)
    b = FS.sample(z, in_win)
    assert a == b                                    # seeded => deterministic
    assert len(a) >= 40
    t = np.sort(z["mono_time"][a])
    assert np.min(np.diff(t)) >= 5.0 - 1e-9          # MIN_FRAME_SEPARATION_S
    # in-window preference: the 100 s window holds at most 20 slots at 5 s separation;
    # the seeded round-robin fill empirically achieves 15-17 (random-packing density <1);
    # assert a strong-preference floor rather than an unattainable perfect packing
    n_in = int(np.sum(in_win[a]))
    assert n_in >= 15


def test_eligibility_blocks_curves():
    from stock_lateral_toolkit.centering import frame_sampler as FS
    z = _synth_cache()
    # Bend the GPS track into a constant-curvature arc for the middle third.
    # (A raw position-domain wiggle like `lat += cumsum(sin(theta))` has a
    # curvature -- the 2nd derivative -- that swings back through zero partway
    # across the segment, so it doesn't stay reliably above STRAIGHT_CURV_MAX
    # everywhere. Integrating a linearly-growing heading instead yields a
    # genuine constant-curvature arc, so the whole segment interior sits
    # unambiguously above threshold.)
    n = len(z["lon"])
    seg = slice(n // 3, 2 * n // 3)
    length = len(range(*seg.indices(n)))
    ds = float(z["v_ego"][0]) * (z["mono_time"][1] - z["mono_time"][0])  # meters/sample
    kappa = 0.005  # 1/m curvature, 10x STRAIGHT_CURV_MAX -> unambiguous curve
    heading = kappa * np.arange(length) * ds
    y_local = np.cumsum(ds * np.sin(heading))
    z["lat"] = z["lat"].copy()
    z["lat"][seg] += y_local / 110540.0
    ok, _head = FS.eligibility(z)
    assert ok[: n // 4].mean() > 0.8                 # straight part eligible
    assert ok[n // 3 + 200: 2 * n // 3 - 200].mean() < 0.5   # curved part mostly blocked
