import sys, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
import numpy as np
from openpilot.tools.lib.logreader import LogReader

def extract_route(route):
    segs = sorted(glob.glob(f'explorer_st_logs/{route}/000000*--*/'),
                  key=lambda p: int(p.rstrip('/').split('--')[-1]))
    # collect per-channel (time, value)
    cs_t, cs_v, cs_yr_t = [], [], []  # carState vEgo, steeringAngle
    veg_t, veg_v = [], []
    eng_t, eng_v = [], []             # carControl latActive
    mdl_t, mdl_off = [], []           # modelV2 lane offset
    yr_t, yr_v = [], []               # liveLocationKalman yawRate
    for seg in segs:
        path = seg + 'rlog.zst'
        if not os.path.exists(path):
            continue
        try:
            for msg in LogReader(path):
                w = msg.which()
                t = msg.logMonoTime * 1e-9
                if w == 'carState':
                    veg_t.append(t); veg_v.append(msg.carState.vEgo)
                elif w == 'carControl':
                    eng_t.append(t); eng_v.append(1.0 if msg.carControl.latActive else 0.0)
                elif w == 'modelV2':
                    ll = msg.modelV2.laneLines
                    if len(ll) >= 3 and len(ll[1].y) > 0 and len(ll[2].y) > 0:
                        off = -(ll[1].y[0] + ll[2].y[0]) / 2.0
                        mdl_t.append(t); mdl_off.append(off)
                elif w == 'liveLocationKalman':
                    av = msg.liveLocationKalman.angularVelocityCalibrated.value
                    if len(av) >= 3:
                        yr_t.append(t); yr_v.append(av[2])
        except Exception as e:
            print('  seg err', seg, e, file=sys.stderr)
    out = dict(
        veg_t=np.array(veg_t), veg_v=np.array(veg_v),
        eng_t=np.array(eng_t), eng_v=np.array(eng_v),
        mdl_t=np.array(mdl_t), mdl_off=np.array(mdl_off),
        yr_t=np.array(yr_t), yr_v=np.array(yr_v),
    )
    return out

if __name__ == '__main__':
    os.makedirs('explorer_st_logs/_c5_raw', exist_ok=True)
    for r in ['route_b1', 'route_b2', 'route_b8']:
        print('extracting', r, flush=True)
        d = extract_route(r)
        for k, v in d.items():
            print('  ', k, v.shape)
        np.savez('explorer_st_logs/_c5_raw/' + r + '.npz', **d)
    print('DONE')
