import sys, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
import numpy as np
from openpilot.tools.lib.logreader import LogReader

def extract_route(route):
    segs = sorted(glob.glob(f'explorer_st_logs/{route}/000000*--*/'),
                  key=lambda p: int(p.rstrip('/').split('--')[-1]))
    # per-channel time/value lists
    cs_t, cs_v, cs_ang, cs_press = [], [], [], []
    cc_t, cc_lat = [], []
    ll_t, ll_yaw = [], []
    for seg in segs:
        f = seg + 'rlog.zst'
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                w = msg.which()
                t = msg.logMonoTime * 1e-9
                if w == 'carState':
                    cs_t.append(t)
                    cs_v.append(msg.carState.vEgo)
                    cs_ang.append(msg.carState.steeringAngleDeg)
                    cs_press.append(1 if msg.carState.steeringPressed else 0)
                elif w == 'carControl':
                    cc_t.append(t)
                    cc_lat.append(1 if msg.carControl.latActive else 0)
                elif w == 'liveLocationKalman':
                    yaw = msg.liveLocationKalman.angularVelocityCalibrated.value[2]
                    ll_t.append(t)
                    ll_yaw.append(yaw)
        except Exception as e:
            print(f"  WARN {f}: {e}", file=sys.stderr)
    return (np.array(cs_t), np.array(cs_v), np.array(cs_ang), np.array(cs_press),
            np.array(cc_t), np.array(cc_lat), np.array(ll_t), np.array(ll_yaw))

if __name__ == '__main__':
    os.makedirs('explorer_st_logs/_c4', exist_ok=True)
    for route in ['route_b1', 'route_b2', 'route_b8']:
        print(f"Extracting {route}...", flush=True)
        cs_t, cs_v, cs_ang, cs_press, cc_t, cc_lat, ll_t, ll_yaw = extract_route(route)
        print(f"  carState n={len(cs_t)}  carControl n={len(cc_t)}  llk n={len(ll_t)}")
        if len(cs_t):
            dt = np.diff(np.sort(cs_t))
            dt = dt[(dt > 0) & (dt < 1)]
            print(f"  carState median dt={np.median(dt)*1000:.1f}ms (={1/np.median(dt):.1f}Hz)")
        np.savez(f'explorer_st_logs/_c4/{route}.npz',
                 cs_t=cs_t, cs_v=cs_v, cs_ang=cs_ang, cs_press=cs_press,
                 cc_t=cc_t, cc_lat=cc_lat, ll_t=ll_t, ll_yaw=ll_yaw)
    print("done")
