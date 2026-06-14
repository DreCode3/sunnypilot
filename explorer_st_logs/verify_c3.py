import sys, glob, json
sys.path.insert(0, '.')
sys.path.insert(0, 'opendbc_repo')
import numpy as np
from openpilot.tools.lib.logreader import LogReader

MPS_TO_MPH = 2.23694

def extract_route(route):
    segs = sorted(glob.glob(f'explorer_st_logs/{route}/000000*--*/'),
                  key=lambda p: int(p.rstrip('/').split('--')[-1]))
    # per-channel raw (time, value) lists
    t_lane, v_off = [], []        # model lane offset pos = -(left_y+right_y)/2
    t_la, v_la = [], []           # latActive (engaged)
    t_v, v_v = [], []             # vEgo
    t_yr, v_yr = [], []           # yawRate
    for seg in segs:
        f = seg + 'rlog.zst'
        try:
            lr = LogReader(f)
        except Exception as e:
            print(f"  skip {seg}: {e}", file=sys.stderr)
            continue
        for msg in lr:
            w = msg.which()
            t = msg.logMonoTime * 1e-9
            if w == 'modelV2':
                m = msg.modelV2
                lls = m.laneLines
                if len(lls) >= 3 and len(lls[1].y) > 0 and len(lls[2].y) > 0:
                    left_y = lls[1].y[0]
                    right_y = lls[2].y[0]
                    t_lane.append(t); v_off.append(-(left_y + right_y) / 2.0)
            elif w == 'carControl':
                t_la.append(t); v_la.append(1.0 if msg.carControl.latActive else 0.0)
            elif w == 'carState':
                cs = msg.carState
                t_v.append(t); v_v.append(cs.vEgo)
            elif w == 'liveLocationKalman':
                av = msg.liveLocationKalman.angularVelocityCalibrated.value
                if len(av) >= 3:
                    t_yr.append(t); v_yr.append(av[2])
    return (np.array(t_lane), np.array(v_off),
            np.array(t_la), np.array(v_la),
            np.array(t_v), np.array(v_v),
            np.array(t_yr), np.array(v_yr))

def build_grid(data):
    t_lane, v_off, t_la, v_la, t_v, v_v, t_yr, v_yr = data
    # sort each channel by time
    def srt(t, v):
        o = np.argsort(t); return t[o], v[o]
    t_lane, v_off = srt(t_lane, v_off)
    t_la, v_la = srt(t_la, v_la)
    t_v, v_v = srt(t_v, v_v)
    t_yr, v_yr = srt(t_yr, v_yr)
    # grid = model timestamps (the channel of interest); interp others onto it
    grid = t_lane
    if len(grid) == 0:
        return None
    la = np.interp(grid, t_la, v_la) if len(t_la) else np.zeros_like(grid)
    v = np.interp(grid, t_v, v_v) if len(t_v) else np.zeros_like(grid)
    yr = np.interp(grid, t_yr, v_yr) if len(t_yr) else np.zeros_like(grid)
    return grid, v_off, la, v, yr

def analyze(route):
    data = extract_route(route)
    g = build_grid(data)
    if g is None:
        print(f"{route}: NO model data"); return None
    grid, off, la, v, yr = g
    # engaged: latActive interpolated -> require >0.5 (essentially ==1)
    engaged = la > 0.5
    # straight: |yawRate * vEgo| < 0.6  (lateral accel proxy)
    straight = np.abs(yr * v) < 0.6
    mph = v * MPS_TO_MPH
    band = (mph >= 40) & (mph <= 80)
    mask = engaged & straight & band & np.isfinite(off)
    return {'route': route, 'off': off[mask], 'mph': mph[mask],
            'n_total': len(grid), 'n_engaged_straight_band': int(mask.sum())}

res = {}
for r in ['route_b1', 'route_b2', 'route_b8']:
    a = analyze(r)
    res[r] = a
    if a:
        print(f"{r}: n_total_model={a['n_total']}, n_engaged_straight_band(40-80)={a['n_engaged_straight_band']}, "
              f"mean_off={np.mean(a['off']):.4f}m, median_off={np.median(a['off']):.4f}m")

# WEAK = b1+b2 pooled
weak_off = np.concatenate([res['route_b1']['off'], res['route_b2']['off']])
gold_off = res['route_b8']['off']
print(f"\nWEAK (b1+b2) pooled: n={len(weak_off)}, mean={np.mean(weak_off):.4f}m, median={np.median(weak_off):.4f}m")
print(f"GOLD (b8):           n={len(gold_off)}, mean={np.mean(gold_off):.4f}m, median={np.median(gold_off):.4f}m")

# Per 5-mph speed bin
print("\n=== Per 5-mph speed bin (mean offset, n) ===")
bins = np.arange(40, 85, 5)
def binstats(off, mph):
    out = {}
    for lo in bins:
        hi = lo + 5
        m = (mph >= lo) & (mph < hi)
        if m.sum() > 0:
            out[lo] = (np.mean(off[m]), int(m.sum()))
    return out
weak_mph = np.concatenate([res['route_b1']['mph'], res['route_b2']['mph']])
gold_mph = res['route_b8']['mph']
ws = binstats(weak_off, weak_mph)
gs = binstats(gold_off, gold_mph)
print(f"{'bin':>6} | {'WEAK mean (n)':>22} | {'GOLD mean (n)':>22}")
for lo in bins:
    w = f"{ws[lo][0]:+.4f} (n={ws[lo][1]})" if lo in ws else "-"
    gg = f"{gs[lo][0]:+.4f} (n={gs[lo][1]})" if lo in gs else "-"
    print(f"{lo:>3}-{lo+5:<2} | {w:>22} | {gg:>22}")

print("\n=== Per route ===")
for r in ['route_b1', 'route_b2', 'route_b8']:
    a = res[r]
    print(f"{r}: mean={np.mean(a['off']):+.4f}m, median={np.median(a['off']):+.4f}m, n={len(a['off'])}")
