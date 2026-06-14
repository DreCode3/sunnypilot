#!/usr/bin/env python3
"""b8 TEST: did restoring golden PI authority on CD210 suppress the slow weave?
  CD210_WEAK = b1,b2  (CD210 + weak PI: lc_kp 0.0001, int_cap 0.3, decay 0.98)  -- baseline
  CD210_GOLD = b8     (CD210 + golden PI: lc_kp 0.0005, int_cap interp->1.0, decay 0.995)  -- the test
  OPM7_GOLD  = 7f,95,99,98 (OPM7 + golden PI)  -- absolute reference (the "awesome" no-weave level)
Single variable WEAK->GOLD = PI authority on the SAME CD210 model. PASS = slow band 0.1-0.5Hz DROPS,
hunt band 0.5-1.5Hz flat (control). Same methodology as slow_weave_model.py (engaged, 16s, same-dir, robust)."""
import sys, glob, os, math
import numpy as np
from collections import defaultdict
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

BLOCK = 0.0025; FS = 20.0; WIN = 16.0; HOP = 8.0; ENG_MIN = 0.9
GROUPS = {'CD210_WEAK': ['route_b1', 'route_b2'],
          'CD210_GOLD': ['route_b8'],
          'OPM7_GOLD':  ['route_7f', 'route_95', 'route_99', 'route_98']}


def band_rms(x, lo, hi):
    n = len(x); t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1 / FS); p = np.abs(X) ** 2
    return math.sqrt(2 * np.sum(p[(f >= lo) & (f < hi)]) / (n * n))


def bearing(la0, lo0, la1, lo1):
    dlat = la1 - la0; dlon = (lo1 - lo0) * math.cos(math.radians((la0 + la1) / 2))
    return math.degrees(math.atan2(dlon, dlat)) % 360


def adiff(a, b):
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
    if len(tl) < 400 or len(tc) < 400 or len(te) < 400:
        return rid, []
    tl = np.array(tl); o = np.argsort(tl); tl, lat, lon, yaw = tl[o], np.array(lat)[o], np.array(lon)[o], np.array(yaw)[o]
    tc = np.array(tc); oc = np.argsort(tc); tc, veg, prs = tc[oc], np.array(veg)[oc], np.array(prs)[oc]
    te = np.array(te); oe = np.argsort(te); te, eng = te[oe], np.array(eng)[oe]
    tu = np.arange(tl[0], tl[-1], 1 / FS)
    latu = np.interp(tu, tl, lat); lonu = np.interp(tu, tl, lon); yawu = np.interp(tu, tl, yaw)
    vegu = np.interp(tu, tc, veg); prsu = np.interp(tu, tc, prs); engu = np.interp(tu, te, eng)
    aLat = yawu * vegu
    wins = []; n = int(WIN * FS); hop = int(HOP * FS)
    for i in range(0, len(tu) - n, hop):
        sl = slice(i, i + n)
        if np.mean(engu[sl]) < ENG_MIN:
            continue
        la = latu[sl]; lo = lonu[sl]
        if not (np.all(la > 33) and np.all(la < 35.5) and np.all(lo > -85.5) and np.all(lo < -83.5)):
            continue
        wins.append(dict(blk=(round(np.median(la) / BLOCK), round(np.median(lo) / BLOCK)),
                         slow=band_rms(aLat[sl], 0.1, 0.5), hunt=band_rms(aLat[sl], 0.5, 1.5),
                         ovr=float(np.mean(prsu[sl] > 0.5)), brg=bearing(la[0], lo[0], la[-1], lo[-1]),
                         spd=float(np.median(vegu[sl]) * 2.237), absalat=float(np.median(np.abs(aLat[sl])))))
    return rid, wins


allr = [r for v in GROUPS.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=9) as ex:
    res = dict(ex.map(load, allr))
WINS = {g: [w for r in rs for w in res.get(r, [])] for g, rs in GROUPS.items()}
for r in allr:
    print(f'  {r}: {len(res.get(r, []))} eng 16s-win')
print('engaged windows: ' + '  '.join(f'{g} {len(WINS[g])}' for g in GROUPS))


def gblocks(group):
    bd = defaultdict(list)
    for w in WINS[group]:
        bd[w['blk']].append(w)
    out = {}
    for b, ws in bd.items():
        bx = np.mean([math.cos(math.radians(w['brg'])) for w in ws]); by = np.mean([math.sin(math.radians(w['brg'])) for w in ws])
        out[b] = dict(slow=float(np.median([w['slow'] for w in ws])), hunt=float(np.median([w['hunt'] for w in ws])),
                      ovr=float(np.mean([w['ovr'] for w in ws])), spd=float(np.median([w['spd'] for w in ws])),
                      absalat=float(np.median([w['absalat'] for w in ws])),
                      brg=math.degrees(math.atan2(by, bx)) % 360, n=len(ws))
    return out


B = {g: gblocks(g) for g in GROUPS}
print('coarse blocks: ' + '  '.join(f'{g} {len(B[g])}' for g in GROUPS))


def signp(k, n):
    if n == 0:
        return 1.0
    z = (abs(k - n / 2) - 0.5) / math.sqrt(n / 4)
    return max(0.0, min(1.0, math.erfc(z / math.sqrt(2))))


def mannwhitney(a, b):
    a = np.asarray(a); b = np.asarray(b); n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return None
    allv = np.concatenate([a, b]); order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    R1 = np.sum(ranks[:n1]); U1 = R1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2; sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mu) / sd
    return z, max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2))))


def block_pair(a, b, metric='slow', curve='all', spd_tol=8.0):
    common = []
    for k in B[a]:
        if k not in B[b]:
            continue
        if adiff(B[a][k]['brg'], B[b][k]['brg']) > 60:
            continue
        if abs(B[a][k]['spd'] - B[b][k]['spd']) > spd_tol:
            continue
        if curve == 'curve' and max(B[a][k]['absalat'], B[b][k]['absalat']) <= 1.0:
            continue
        if curve == 'straight' and not (B[a][k]['absalat'] < 0.6 and B[b][k]['absalat'] < 0.6):
            continue
        common.append(k)
    if len(common) < 4:
        print(f'    [pair] {a}->{b} {metric}/{curve}: only {len(common)} shared blocks'); return
    bw = sum(1 for k in common if B[b][k][metric] > B[a][k][metric])
    ratio = [B[b][k][metric] / B[a][k][metric] for k in common if B[a][k][metric] > 1e-4]
    ma = np.median([B[a][k][metric] for k in common]); mb = np.median([B[b][k][metric] for k in common])
    oa = 100 * np.mean([B[a][k]['ovr'] for k in common]); ob = 100 * np.mean([B[b][k]['ovr'] for k in common])
    sa = np.median([B[a][k]['spd'] for k in common]); sb = np.median([B[b][k]['spd'] for k in common])
    print(f'    [pair] {a}->{b} ({len(common)}blk spd{sa:.0f}/{sb:.0f}): {metric} med {ma:.3f}->{mb:.3f} '
          f'({100*(mb-ma)/ma:+.0f}%) {b}-worse {bw}/{len(common)} signp={signp(bw,len(common)):.2f} ratio={np.median(ratio):.2f} | ovr {oa:.1f}->{ob:.1f}%')


def dist_test(a, b, metric='slow', curve='all', lo_mph=40, hi_mph=80):
    shared = {k for k in (set(B[a]) & set(B[b])) if adiff(B[a][k]['brg'], B[b][k]['brg']) <= 60}
    def pool(g):
        out = []
        for w in WINS[g]:
            if w['blk'] not in shared:
                continue
            if adiff(w['brg'], B[a][w['blk']]['brg']) > 60:
                continue
            if not (lo_mph <= w['spd'] <= hi_mph):
                continue
            if curve == 'curve' and w['absalat'] <= 1.0:
                continue
            if curve == 'straight' and w['absalat'] >= 0.6:
                continue
            out.append(w[metric])
        return out
    pa, pb = pool(a), pool(b)
    if len(pa) < 5 or len(pb) < 5:
        print(f'    [dist] {a}->{b} {metric}/{curve}: n {len(pa)}/{len(pb)} too few ({len(shared)} shared-dir blks)'); return
    mw = mannwhitney(pa, pb)
    print(f'    [dist] {a}->{b} {metric}/{curve}: n {len(pa)}/{len(pb)}  med {np.median(pa):.3f}->{np.median(pb):.3f} '
          f'({100*(np.median(pb)-np.median(pa))/np.median(pa):+.0f}%)  MW z={mw[0]:+.2f} p={mw[1]:.3f}')


print('\n############ THE TEST: CD210_WEAK -> CD210_GOLD (PI authority on same CD210) ############')
for metric in ['slow', 'hunt']:
    print(f'\n========== {metric.upper()} band ==========')
    for cv in ['all', 'straight', 'curve']:
        print(f'  --- {cv} ---')
        block_pair('CD210_WEAK', 'CD210_GOLD', metric, cv)
        dist_test('CD210_WEAK', 'CD210_GOLD', metric, cv)

print('\n############ REFERENCE: where does CD210_GOLD land vs OPM7_GOLD (the no-weave target)? ############')
for cv in ['all', 'straight']:
    print(f'  --- slow / {cv} ---')
    block_pair('OPM7_GOLD', 'CD210_GOLD', 'slow', cv)
    dist_test('OPM7_GOLD', 'CD210_GOLD', 'slow', cv)

print('\n############ absolute slow-band medians (engaged, straight, 40-80mph) ############')
for g in GROUPS:
    vals = [w['slow'] for w in WINS[g] if w['absalat'] < 0.6 and 40 <= w['spd'] <= 80]
    if vals:
        print(f'  {g}: median slow {np.median(vals):.3f}  (n={len(vals)})  | median hunt '
              f'{np.median([w["hunt"] for w in WINS[g] if w["absalat"]<0.6 and 40<=w["spd"]<=80]):.3f}')
