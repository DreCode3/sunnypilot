#!/usr/bin/env python3
"""ENGAGED-GATED model-independent same-road lateral A/B (verification re-run of model_indep_ab.py).
Fixes the three QA-anticipated holes:
  1. ENGAGEMENT GATING: only windows where openpilot was actually steering (carControl.latActive >= 80% of window).
     -> override = steeringPressed WHILE engaged (real intervention, not manual/tuning driving).
  2. SLOW BAND: report 0.1-0.5Hz slow-weave RMS alongside 0.5-1.5Hz hunt (felt long-drive wander may be slow).
  3. HEADING MATCH: cells matched across groups must share travel direction (<60 deg) -> divided-highway
     carriageways / opposite curve directions don't get pooled.
Robust median + sign-count + two-sided sign-test p (normal approx). aLat = yawRate*vEgo (model-independent)."""
import sys, glob, os, math
import numpy as np
from collections import defaultdict
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

CELL = 0.0005; FS = 20.0; W = 4.0; ENG_MIN = 0.8
GROUPS = {'GOLDEN': ['route_7f', 'route_95', 'route_99', 'route_98'],
          'WEAK':   ['route_a0', 'route_9b', 'route_9d'],
          'TODAY':  ['route_b1', 'route_b2']}


def band_rms(x, lo, hi):
    x = x - np.mean(x); n = len(x)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1 / FS); p = np.abs(X) ** 2
    return math.sqrt(2 * np.sum(p[(f >= lo) & (f < hi)]) / (n * n))


def bearing(la0, lo0, la1, lo1):
    dlat = la1 - la0; dlon = (lo1 - lo0) * math.cos(math.radians((la0 + la1) / 2))
    return math.degrees(math.atan2(dlon, dlat)) % 360


def adiff(a, b):  # circular degrees diff 0..180
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def load(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    tl = []; lat = []; lon = []; yaw = []; tc = []; veg = []; prs = []; te = []; eng = []
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
                elif w == 'carControl':
                    te.append(msg.logMonoTime * 1e-9); eng.append(1.0 if msg.carControl.latActive else 0.0)
        except Exception:
            continue
    if len(tl) < 200 or len(tc) < 200 or len(te) < 200:
        return rid, []
    tl = np.array(tl); o = np.argsort(tl); tl, lat, lon, yaw = tl[o], np.array(lat)[o], np.array(lon)[o], np.array(yaw)[o]
    tc = np.array(tc); oc = np.argsort(tc); tc, veg, prs = tc[oc], np.array(veg)[oc], np.array(prs)[oc]
    te = np.array(te); oe = np.argsort(te); te, eng = te[oe], np.array(eng)[oe]
    tu = np.arange(tl[0], tl[-1], 1 / FS)
    latu = np.interp(tu, tl, lat); lonu = np.interp(tu, tl, lon); yawu = np.interp(tu, tl, yaw)
    vegu = np.interp(tu, tc, veg); prsu = np.interp(tu, tc, prs); engu = np.interp(tu, te, eng)
    aLat = yawu * vegu
    wins = []; n = int(W * FS)
    for i in range(0, len(tu) - n, n):
        sl = slice(i, i + n)
        if np.mean(engu[sl]) < ENG_MIN:        # require OP actively steering most of the window
            continue
        la = latu[sl]; lo = lonu[sl]
        if not (np.all(la > 33) and np.all(la < 35.5) and np.all(lo > -85.5) and np.all(lo < -83.5)):
            continue
        wins.append(dict(cell=(round(np.median(la) / CELL), round(np.median(lo) / CELL)),
                         hunt=band_rms(aLat[sl], 0.5, 1.5), slow=band_rms(aLat[sl], 0.1, 0.5),
                         ovr=float(np.mean(prsu[sl] > 0.5)),
                         brg=bearing(la[0], lo[0], la[-1], lo[-1]),
                         spd=float(np.median(vegu[sl]) * 2.237), absalat=float(np.median(np.abs(aLat[sl])))))
    return rid, wins


allr = [r for v in GROUPS.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=9) as ex:
    res = dict(ex.map(load, allr))
for r in allr:
    print(f'  {r}: {len(res.get(r, []))} engaged windows')


def gcells(names):
    cd = defaultdict(list)
    for nm in names:
        for w in res.get(nm, []):
            cd[w['cell']].append(w)
    out = {}
    for c, ws in cd.items():
        if len(ws) < 2:
            continue
        # circular mean bearing
        bx = np.mean([math.cos(math.radians(w['brg'])) for w in ws]); by = np.mean([math.sin(math.radians(w['brg'])) for w in ws])
        out[c] = dict(hunt=float(np.median([w['hunt'] for w in ws])), slow=float(np.median([w['slow'] for w in ws])),
                      ovr=float(np.mean([w['ovr'] for w in ws])), spd=float(np.median([w['spd'] for w in ws])),
                      absalat=float(np.median([w['absalat'] for w in ws])),
                      brg=math.degrees(math.atan2(by, bx)) % 360, n=len(ws))
    return out


G = {k: gcells(v) for k, v in GROUPS.items()}
print(f'\nengaged cells: GOLDEN {len(G["GOLDEN"])}  WEAK {len(G["WEAK"])}  TODAY {len(G["TODAY"])}')


def signp(k, n):  # two-sided sign-test p, normal approx with continuity correction
    if n == 0:
        return 1.0
    z = (abs(k - n / 2) - 0.5) / math.sqrt(n / 4)
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


def cmp(a, b, metric='hunt', curve='all', hdg=True, spd_tol=6.0):
    common = []
    for c in G[a]:
        if c not in G[b]:
            continue
        if hdg and adiff(G[a][c]['brg'], G[b][c]['brg']) > 60:   # same travel direction only
            continue
        if abs(G[a][c]['spd'] - G[b][c]['spd']) > spd_tol:       # speed-matched
            continue
        if curve == 'curve' and max(G[a][c]['absalat'], G[b][c]['absalat']) <= 1.0:
            continue
        if curve == 'straight' and not (G[a][c]['absalat'] < 0.6 and G[b][c]['absalat'] < 0.6):
            continue
        common.append(c)
    if len(common) < 5:
        print(f'    {a}->{b} {metric}/{curve}: only {len(common)} shared cells'); return
    bw = sum(1 for c in common if G[b][c][metric] > G[a][c][metric])
    ratio = [G[b][c][metric] / G[a][c][metric] for c in common if G[a][c][metric] > 1e-4]
    ma = np.median([G[a][c][metric] for c in common]); mb = np.median([G[b][c][metric] for c in common])
    oa = 100 * np.mean([G[a][c]['ovr'] for c in common]); ob = 100 * np.mean([G[b][c]['ovr'] for c in common])
    sa = np.median([G[a][c]['spd'] for c in common]); sb = np.median([G[b][c]['spd'] for c in common])
    print(f'    {a}->{b} ({len(common)}c, spd {sa:.0f}/{sb:.0f}mph): {metric} med {ma:.3f}->{mb:.3f} ({100*(mb-ma)/ma:+.0f}%), '
          f'{b} worse {bw}/{len(common)} (sign p={signp(bw, len(common)):.2f}), ratio {np.median(ratio):.2f} | ovr {oa:.1f}%->{ob:.1f}%')


for metric in ['hunt', 'slow']:
    for cv in ['all', 'curve', 'straight']:
        print(f'\n--- {metric} band | {cv} cells (engaged, same-dir, speed-matched) ---')
        cmp('GOLDEN', 'WEAK', metric, cv); cmp('GOLDEN', 'TODAY', metric, cv); cmp('WEAK', 'TODAY', metric, cv)

import json
json.dump({k: {f'{c[0]}_{c[1]}': v for c, v in G[k].items()} for k in G},
          open('explorer_st_logs/model_indep_gated.json', 'w'))
print('\nsaved -> explorer_st_logs/model_indep_gated.json')
