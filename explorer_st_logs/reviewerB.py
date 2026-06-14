#!/usr/bin/env python3
"""REVIEWER B independent re-analysis + confound attack.
Loads each route's raw timeseries ONCE, caches to .npz, then runs all analyses.
Differs from v2: (a) my own band-RMS via Welch-style PSD AND naive std on engaged straights,
(b) rigorous GPS block-overlap & speed/bearing match, (c) POS noise-floor (HF >2Hz) vs steer,
(d) sharp-curve takeover characterization (apply_curvature vs model desiredCurvature)."""
import sys, glob, os, math
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

FS = 20.0
ROUTES = ['route_b1', 'route_b2', 'route_b8']
CACHE = 'explorer_st_logs/_rb_cache'
os.makedirs(CACHE, exist_ok=True)


def extract(rid):
    cf = f'{CACHE}/{rid}.npz'
    if os.path.exists(cf):
        return rid
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'),
                  key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    tl=[];lat=[];lon=[];yaw=[]
    tc=[];veg=[];sa=[];prs=[]
    te=[];eng=[];cmd=[]
    tm=[];pos=[];desc=[]
    for sd in segs:
        f = sd + 'rlog.zst'
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                w = msg.which()
                if w == 'liveLocationKalman':
                    g = msg.liveLocationKalman
                    if not g.positionGeodetic.valid or not g.angularVelocityCalibrated.valid:
                        continue
                    v = g.positionGeodetic.value; av = g.angularVelocityCalibrated.value
                    tl.append(msg.logMonoTime*1e-9); lat.append(float(v[0])); lon.append(float(v[1])); yaw.append(float(av[2]))
                elif w == 'carState':
                    cs = msg.carState
                    tc.append(msg.logMonoTime*1e-9); veg.append(float(cs.vEgo)); sa.append(float(cs.steeringAngleDeg)); prs.append(1.0 if cs.steeringPressed else 0.0)
                elif w == 'carControl':
                    te.append(msg.logMonoTime*1e-9); eng.append(1.0 if msg.carControl.latActive else 0.0)
                    try: cmd.append(float(msg.carControl.actuators.curvature))
                    except Exception: cmd.append(0.0)
                elif w == 'modelV2':
                    mdl = msg.modelV2
                    ll = mdl.laneLines
                    if len(ll) > 2 and len(ll[1].y) > 0 and len(ll[2].y) > 0:
                        tm.append(msg.logMonoTime*1e-9); pos.append(-(float(ll[1].y[0]) + float(ll[2].y[0]))/2)
                        # desired curvature from model: action.desiredCurvature if present
                        dc = float('nan')
                        try:
                            dc = float(mdl.action.desiredCurvature)
                        except Exception:
                            dc = float('nan')
                        desc.append(dc)
        except Exception:
            continue
    if len(tl) < 400 or len(tc) < 400 or len(te) < 400:
        np.savez(cf, empty=True)
        return rid
    tl=np.array(tl);o=np.argsort(tl);tl,lat,lon,yaw=tl[o],np.array(lat)[o],np.array(lon)[o],np.array(yaw)[o]
    tc=np.array(tc);oc=np.argsort(tc);tc,veg,sa,prs=tc[oc],np.array(veg)[oc],np.array(sa)[oc],np.array(prs)[oc]
    te=np.array(te);oe=np.argsort(te);te,eng,cmd=te[oe],np.array(eng)[oe],np.array(cmd)[oe]
    tu = np.arange(tl[0], tl[-1], 1/FS)
    latu=np.interp(tu,tl,lat);lonu=np.interp(tu,tl,lon);yawu=np.interp(tu,tl,yaw)
    vegu=np.interp(tu,tc,veg);sau=np.interp(tu,tc,sa);prsu=np.interp(tu,tc,prs)
    engu=np.interp(tu,te,eng);cmdu=np.interp(tu,te,cmd)
    if len(tm) > 100:
        tm=np.array(tm);om=np.argsort(tm);tm=tm[om];posv=np.array(pos)[om];descv=np.array(desc)[om]
        posu=np.interp(tu,tm,posv);descu=np.interp(tu,tm,descv)
    else:
        posu=np.full_like(tu, np.nan);descu=np.full_like(tu, np.nan)
    np.savez(cf, tu=tu, lat=latu, lon=lonu, yaw=yawu, veg=vegu, sa=sau, prs=prsu,
             eng=engu, cmd=cmdu, pos=posu, desc=descu, empty=False)
    return rid


if __name__ == '__main__':
    import concurrent.futures as cf2
    with cf2.ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(extract, ROUTES))
    for r in ROUTES:
        d = np.load(f'{CACHE}/{r}.npz')
        if d.get('empty', np.array(True)):
            print(f'{r}: EMPTY')
        else:
            print(f'{r}: {len(d["tu"])} samples ({len(d["tu"])/FS/60:.1f} min)')
