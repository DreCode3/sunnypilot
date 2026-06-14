#!/usr/bin/env python3
"""ONE-TIME extractor: read each route's rlogs ONCE, resample all driver-relevant channels to 50Hz,
cache to explorer_st_logs/_cache_reassess/{rid}.npz so downstream multi-lens analysis never re-reads logs.

Channels (50Hz uniform grid over each route's wall-clock span):
  t        relative seconds
  lat,lon  GPS (liveLocationKalman.positionGeodetic)
  yaw      angularVelocityCalibrated.z (yaw rate, rad/s)
  spd      vEgo (m/s)
  steer    carState.steeringAngleDeg (WHEEL angle, deg) -- model-INDEPENDENT
  pos      model lane offset = -(laneLines[1].y0+laneLines[2].y0)/2  (CD210 both groups -> comparable)
  modpath  modelV2.position.y[0] (model planned lateral, m)  -- centering intent
  cmd      carControl.actuators.curvature (commanded curvature, 1/m)
  curv     carControl.actuators.curvature? no -> desiredCurvature from controlsState if present
  latact   carControl.latActive (1/0)   -- the CORRECT engaged flag under MADS
  press    carState.steeringPressed (1/0)

NOTE: 50Hz preserves felt bands up to 25Hz Nyquist (human steering feel <5Hz). carState/carControl
native ~100Hz downsample to 50Hz (no meaningful 25-50Hz EPS content). location/model native 20Hz upsample.
We also store native-rate steer samples (t_steer, steer_native) so a lens can verify resampling didn't alias.
"""
import sys, glob, os, math
import numpy as np
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

FS = 50.0
import os as _os
ROUTES = _os.environ.get('REASSESS_ROUTES', 'route_b1,route_b2,route_b8').split(',')  # override via env
OUTDIR = 'explorer_st_logs/_cache_reassess'


def load_route(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'),
                  key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    tl=[];lat=[];lon=[];yaw=[]
    tc=[];veg=[];sa=[];prs=[]
    te=[];eng=[];cmd=[]
    tm=[];pos=[];mpath=[]
    nseg=0
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
                    tc.append(msg.logMonoTime*1e-9); veg.append(float(cs.vEgo))
                    sa.append(float(cs.steeringAngleDeg)); prs.append(1.0 if cs.steeringPressed else 0.0)
                elif w == 'carControl':
                    cc = msg.carControl
                    te.append(msg.logMonoTime*1e-9); eng.append(1.0 if cc.latActive else 0.0)
                    try: cmd.append(float(cc.actuators.curvature))
                    except Exception: cmd.append(0.0)
                elif w == 'modelV2':
                    m = msg.modelV2
                    ll = m.laneLines
                    if len(ll) > 2 and len(ll[1].y) > 0 and len(ll[2].y) > 0:
                        tm.append(msg.logMonoTime*1e-9)
                        pos.append(-(float(ll[1].y[0]) + float(ll[2].y[0]))/2)
                        try:
                            mpath.append(float(m.position.y[0]))
                        except Exception:
                            mpath.append(np.nan)
            nseg += 1
        except Exception as e:
            sys.stderr.write(f'{rid} seg {sd} err {e}\n')
            continue
    if len(tl) < 400 or len(tc) < 400 or len(te) < 400:
        return rid, None
    def sortpair(t, *arrs):
        t = np.array(t); o = np.argsort(t)
        return (t[o],) + tuple(np.array(a)[o] for a in arrs)
    tl, lat, lon, yaw = sortpair(tl, lat, lon, yaw)
    tc, veg, sa, prs = sortpair(tc, veg, sa, prs)
    te, eng, cmd = sortpair(te, eng, cmd)
    have_model = len(tm) > 100
    if have_model:
        tm, pos, mpath = sortpair(tm, pos, mpath)
    t0 = tl[0]; t1 = tl[-1]
    tu = np.arange(t0, t1, 1/FS)
    out = dict(
        t=(tu - t0).astype(np.float32),
        lat=np.interp(tu, tl, lat).astype(np.float64),
        lon=np.interp(tu, tl, lon).astype(np.float64),
        yaw=np.interp(tu, tl, yaw).astype(np.float32),
        spd=np.interp(tu, tc, veg).astype(np.float32),
        steer=np.interp(tu, tc, sa).astype(np.float32),
        press=(np.interp(tu, tc, prs) > 0.5).astype(np.int8),
        latact=(np.interp(tu, te, eng) > 0.5).astype(np.int8),
        cmd=np.interp(tu, te, cmd).astype(np.float32),
        nseg=np.int32(nseg),
    )
    if have_model:
        out['pos'] = np.interp(tu, tm, pos).astype(np.float32)
        out['modpath'] = np.interp(tu, tm, mpath).astype(np.float32)
    else:
        out['pos'] = np.full(len(tu), np.nan, np.float32)
        out['modpath'] = np.full(len(tu), np.nan, np.float32)
    # native-rate steer for anti-alias verification
    out['t_steer'] = (tc - t0).astype(np.float32)
    out['steer_native'] = sa.astype(np.float32)
    return rid, out


os.makedirs(OUTDIR, exist_ok=True)
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    for rid, d in ex.map(load_route, ROUTES):
        if d is None:
            print(f'{rid}: NO DATA'); continue
        np.savez_compressed(f'{OUTDIR}/{rid}.npz', **d)
        dur = d['t'][-1]
        eng_frac = float(np.mean(d['latact']))
        print(f'{rid}: {int(d["nseg"])} seg, {dur/60:.1f} min, {len(d["t"])} samp@50Hz, '
              f'latActive {eng_frac*100:.0f}%, spd med {np.median(d["spd"][d["latact"]>0])*2.237:.1f} mph, '
              f'model {"Y" if not np.all(np.isnan(d["pos"])) else "N"}')
print('cache ->', OUTDIR)
