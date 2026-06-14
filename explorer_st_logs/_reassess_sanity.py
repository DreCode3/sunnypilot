import numpy as np, glob

# Compare my extraction vs provided cache to make sure I'm reading the same data
for route in ['route_b1', 'route_b2', 'route_b8']:
    mine = np.load(f'explorer_st_logs/_reassess_mine_{route}.npz')
    cand = glob.glob(f'explorer_st_logs/_cache_reassess/{route}.npz')
    print(f"=== {route} ===")
    print(f"  mine keys: {list(mine.keys())}")
    if cand:
        cache = np.load(cand[0])
        print(f"  cache keys: {list(cache.keys())}")
        for k in ['spd', 'steer', 'cmd', 'pos', 'yaw']:
            if k in mine and k in cache:
                m = mine[k]; c = cache[k]
                # they may differ in length / grid alignment; compare distributions
                mv = m[np.isfinite(m)]; cv = c[np.isfinite(c)]
                print(f"  {k:6s}: mine n={len(mv):6d} med={np.median(mv):+.5f} std={np.std(mv):.5f} | "
                      f"cache n={len(cv):6d} med={np.median(cv):+.5f} std={np.std(cv):.5f}")
    # basic engaged-straight summary from mine
    spd = mine['spd']; steer = mine['steer']; press = mine['press']
    latact = mine['latact']; yaw = mine['yaw']; cmd = mine['cmd']; pos = mine['pos']
    eng = latact >= 0.5
    straight = np.abs(yaw * spd) < 0.6
    band = (spd >= 40*0.44704) & (spd <= 80*0.44704)
    sel = eng & straight & band & np.isfinite(steer) & np.isfinite(pos)
    print(f"  engaged-straight-band(40-80mph): n={sel.sum()}  "
          f"median spd={np.median(spd[sel])/0.44704:.1f}mph  "
          f"mean steer={np.mean(steer[sel]):+.3f}deg  mean pos={np.mean(pos[sel]):+.4f}m")
