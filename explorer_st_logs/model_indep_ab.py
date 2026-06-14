#!/usr/bin/env python3
"""Model-INDEPENDENT same-road lateral A/B: golden(strong-PI OPM7) vs weak(weak-PI OPM7) vs today(CD210).
Signals from CAN/locationd, NOT the driving model's perception:
  aLat = yawRate(liveLocationKalman.angularVelocityCalibrated.z) * vEgo(carState)  -> actual lateral motion
  override = carState.steeringPressed                                              -> driver intervention
Per 4s window: aLat hunt-band 0.5-1.5Hz RMS, override fraction, GPS cell, speed, |aLat| (curve/straight class).
Compare groups on SHARED ~55m cells, robust (median + sign). Curve cells separated (the complaint zone)."""
import sys, glob, os, math
import numpy as np
from collections import defaultdict
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

CELL = 0.0005; FS = 20.0; W = 4.0
GROUPS = {'GOLDEN': ['route_7f', 'route_95', 'route_99', 'route_98'],
          'WEAK':   ['route_a0', 'route_9b', 'route_9d'],
          'TODAY':  ['route_b1', 'route_b2']}


def band_rms(x, lo=0.5, hi=1.5):
    x = x - np.mean(x); n = len(x)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1 / FS); p = np.abs(X) ** 2
    return math.sqrt(2 * np.sum(p[(f >= lo) & (f < hi)]) / (n * n))


def load(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    tl = []; lat = []; lon = []; yaw = []; tc = []; veg = []; prs = []
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
                    tl.append(msg.logMonoTime * 1e-9); lat.append(float(v[0])); lon.append(float(v[1])); yaw.append(float(av[2]))
                elif w == 'carState':
                    cs = msg.carState
                    tc.append(msg.logMonoTime * 1e-9); veg.append(float(cs.vEgo)); prs.append(1.0 if cs.steeringPressed else 0.0)
        except Exception:
            continue
    if len(tl) < 200 or len(tc) < 200:
        return rid, []
    tl = np.array(tl); o = np.argsort(tl); tl, lat, lon, yaw = tl[o], np.array(lat)[o], np.array(lon)[o], np.array(yaw)[o]
    tc = np.array(tc); oc = np.argsort(tc); tc, veg, prs = tc[oc], np.array(veg)[oc], np.array(prs)[oc]
    tu = np.arange(tl[0], tl[-1], 1 / FS)
    latu = np.interp(tu, tl, lat); lonu = np.interp(tu, tl, lon); yawu = np.interp(tu, tl, yaw)
    vegu = np.interp(tu, tc, veg); prsu = (np.interp(tu, tc, prs) > 0.5).astype(float)
    aLat = yawu * vegu
    wins = []; n = int(W * FS)
    for i in range(0, len(tu) - n, n):
        sl = slice(i, i + n); la = latu[sl]; lo = lonu[sl]
        if not (np.all(la > 33) and np.all(la < 35.5) and np.all(lo > -85.5) and np.all(lo < -83.5)):
            continue
        wins.append(dict(cell=(round(np.median(la) / CELL), round(np.median(lo) / CELL)),
                         hunt=band_rms(aLat[sl]), ovr=float(np.mean(prsu[sl])),
                         spd=float(np.median(vegu[sl]) * 2.237), absalat=float(np.median(np.abs(aLat[sl])))))
    return rid, wins


allr = [r for v in GROUPS.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=10) as ex:
    res = dict(ex.map(load, allr))
for r in allr:
    print(f'  {r}: {len(res.get(r, []))} windows')


def gcells(names):
    cd = defaultdict(list)
    for nm in names:
        for w in res.get(nm, []):
            cd[w['cell']].append(w)
    return {c: dict(hunt=float(np.mean([w['hunt'] for w in ws])), ovr=float(np.mean([w['ovr'] for w in ws])),
                    spd=float(np.median([w['spd'] for w in ws])), absalat=float(np.median([w['absalat'] for w in ws])),
                    n=len(ws)) for c, ws in cd.items()}


G = {k: gcells(v) for k, v in GROUPS.items()}
print(f'\ncells: GOLDEN {len(G["GOLDEN"])}  WEAK {len(G["WEAK"])}  TODAY {len(G["TODAY"])}')


def cmp(a, b, curve='all'):
    common = [c for c in G[a] if c in G[b]]
    if curve == 'curve':
        common = [c for c in common if max(G[a][c]['absalat'], G[b][c]['absalat']) > 1.0]
    elif curve == 'straight':
        common = [c for c in common if G[a][c]['absalat'] < 0.6 and G[b][c]['absalat'] < 0.6]
    if len(common) < 5:
        print(f'    {a} vs {b}: only {len(common)} shared {curve} cells'); return
    hr = [G[b][c]['hunt'] / G[a][c]['hunt'] for c in common if G[a][c]['hunt'] > 1e-3]
    bw = sum(1 for c in common if G[b][c]['hunt'] > G[a][c]['hunt'])
    oa = 100 * np.mean([G[a][c]['ovr'] for c in common]); ob = 100 * np.mean([G[b][c]['ovr'] for c in common])
    ha = np.median([G[a][c]['hunt'] for c in common]); hb = np.median([G[b][c]['hunt'] for c in common])
    print(f'    {a}->{b} ({len(common)}c): aLat-hunt med {ha:.3f}->{hb:.3f} m/s2 ({100*(hb-ha)/ha:+.0f}%), {b} worse {bw}/{len(common)}, ratio {np.median(hr):.2f} | override {oa:.1f}%->{ob:.1f}%')


for cv in ['all', 'curve', 'straight']:
    print(f'\n--- {cv} cells ---')
    cmp('GOLDEN', 'WEAK', cv); cmp('GOLDEN', 'TODAY', cv); cmp('WEAK', 'TODAY', cv)

import json
json.dump({k: {f'{c[0]}_{c[1]}': v for c, v in G[k].items()} for k in G},
          open('explorer_st_logs/model_indep_ab.json', 'w'))
print('\nsaved -> explorer_st_logs/model_indep_ab.json')
