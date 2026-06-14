import sys, glob, os
sys.path.insert(0, '.')
sys.path.insert(0, 'opendbc_repo')
import numpy as np
from openpilot.tools.lib.logreader import LogReader

def extract_route(route):
    segs = sorted(glob.glob(f'explorer_st_logs/{route}/000000*--*/'),
                  key=lambda p: int(p.rstrip('/').split('--')[-1]))
    # collect per-channel (time, value) lists
    cs_t, cs_v, cs_steer, cs_press = [], [], [], []   # carState: vEgo, steeringAngleDeg, steeringPressed
    cc_t, cc_latact, cc_cmd = [], [], []              # carControl: latActive, actuators.curvature
    ll_t, ll_lat, ll_lon, ll_yaw = [], [], [], []     # liveLocationKalman
    mv_t, mv_pos = [], []                             # modelV2 lane offset pos
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
                    cs_steer.append(msg.carState.steeringAngleDeg)
                    cs_press.append(1.0 if msg.carState.steeringPressed else 0.0)
                elif w == 'carControl':
                    cc_t.append(t)
                    cc_latact.append(1.0 if msg.carControl.latActive else 0.0)
                    cc_cmd.append(msg.carControl.actuators.curvature)
                elif w == 'liveLocationKalman':
                    llk = msg.liveLocationKalman
                    ll_t.append(t)
                    pg = llk.positionGeodetic.value
                    ll_lat.append(pg[0]); ll_lon.append(pg[1])
                    ll_yaw.append(llk.angularVelocityCalibrated.value[2])
                elif w == 'modelV2':
                    mv = msg.modelV2
                    try:
                        ly = mv.laneLines[1].y[0]
                        ry = mv.laneLines[2].y[0]
                        mv_t.append(t)
                        mv_pos.append(-(ly + ry) / 2.0)
                    except Exception:
                        pass
        except Exception as e:
            print(f"  WARN seg {seg}: {e}", file=sys.stderr)
    cs_t = np.array(cs_t); order = np.argsort(cs_t)
    cs_t = cs_t[order]
    cs_v = np.array(cs_v)[order]
    cs_steer = np.array(cs_steer)[order]
    cs_press = np.array(cs_press)[order]
    cc_t = np.array(cc_t); o2 = np.argsort(cc_t); cc_t = cc_t[o2]
    cc_latact = np.array(cc_latact)[o2]; cc_cmd = np.array(cc_cmd)[o2]
    ll_t = np.array(ll_t); o3 = np.argsort(ll_t); ll_t = ll_t[o3]
    ll_lat = np.array(ll_lat)[o3]; ll_lon = np.array(ll_lon)[o3]; ll_yaw = np.array(ll_yaw)[o3]
    mv_t = np.array(mv_t); o4 = np.argsort(mv_t); mv_t = mv_t[o4]; mv_pos = np.array(mv_pos)[o4]

    # common 50Hz grid over carState span
    t0, t1 = cs_t[0], cs_t[-1]
    grid = np.arange(t0, t1, 0.02)
    def rs(src_t, src_v):
        if len(src_t) < 2:
            return np.full_like(grid, np.nan)
        return np.interp(grid, src_t, src_v)
    out = dict(
        t=grid,
        spd=rs(cs_t, cs_v),
        steer=rs(cs_t, cs_steer),
        press=rs(cs_t, cs_press),
        lat=rs(ll_t, ll_lat),
        lon=rs(ll_t, ll_lon),
        yaw=rs(ll_t, ll_yaw),
        latact=rs(cc_t, cc_latact),
        cmd=rs(cc_t, cc_cmd),
        pos=rs(mv_t, mv_pos),
    )
    print(f"{route}: grid {len(grid)} samples, {(t1-t0):.0f}s; "
          f"cs={len(cs_t)} cc={len(cc_t)} llk={len(ll_t)} mv={len(mv_t)}")
    return out

if __name__ == '__main__':
    for route in ['route_b1', 'route_b2', 'route_b8']:
        d = extract_route(route)
        np.savez(f'explorer_st_logs/_reassess_mine_{route}.npz', **d)
        print(f"  saved _reassess_mine_{route}.npz")
