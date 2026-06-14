#!/usr/bin/env python3
"""Straight-line center-keeping hunt from DENSE modelV2 (20Hz) — CD210 (b1) vs OPM7.
CX1 logs straights sparsely (~1Hz) so can't measure this; modelV2 is always 20Hz.
On straight road (rolling-mean |desiredCurvature| < 5e-4, speed>13 m/s), measure:
  A_std  = lateral-position wander amplitude (A=-(L+R)/2; on straights aLat~0 so NO lean confound)
  A_cps  = lateral wander reversals/s
  desC_cps = model desiredCurvature crossings/s (the model jitter the driver feels as 'can't settle')
"""
import sys, glob
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from openpilot.tools.lib.logreader import LogReader
from explorer_st_logs.phase1_validation import crossings_per_sec

CD210 = ['route_b1']
OPM7 = ['route_aa', 'route_ac', 'route_ad', 'route_af', 'route_b0']


def load_mv2(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    t = []; dc = []; A = []; spd = []
    for sd in segs:
        f = sd + 'rlog.zst'
        try:
            for msg in LogReader(f):
                if msg.which() != 'modelV2':
                    continue
                m = msg.modelV2
                ll = m.laneLines
                if len(ll) < 4 or len(ll[1].y) < 1:
                    continue
                try:
                    v = m.velocity.x[0]
                except Exception:
                    v = float('nan')
                t.append(msg.logMonoTime * 1e-9)
                dc.append(float(m.action.desiredCurvature))
                A.append(-(ll[1].y[0] + ll[2].y[0]) / 2.0)
                spd.append(v)
        except Exception:
            continue
    if len(t) < 200:
        return None
    t = np.array(t); o = np.argsort(t)
    return dict(t=t[o], dc=np.array(dc)[o], A=np.array(A)[o], spd=np.array(spd)[o])


def runs(mask, minlen):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= minlen:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def analyze(rid):
    d = load_mv2(rid)
    if d is None:
        return None
    t, dc, A, spd = d['t'], d['dc'], d['A'], d['spd']
    dcroll = np.convolve(np.abs(dc), np.ones(40) / 40, mode='same')  # ~2s -> straight road
    sm = (dcroll < 5e-4) & (spd > 13)
    out = []
    for a, b in runs(sm, 120):  # >=6s at 20Hz
        ts = t[a:b]
        if ts[-1] - ts[0] < 6:
            continue
        dt = float(np.median(np.diff(ts))) or 0.05
        out.append(dict(dur=float(ts[-1] - ts[0]), spd=float(np.median(spd[a:b]) * 2.237),
                        A_std=float(np.std(A[a:b])),
                        A_cps=crossings_per_sec(A[a:b], dt, threshold=0.05),
                        desC_cps=crossings_per_sec(dc[a:b], dt, threshold=4e-5)))
    return dict(rid=rid, runs=out)


def med(xs):
    xs = [x for x in xs if x is not None and not np.isnan(x)]
    return float(np.median(xs)) if xs else float('nan')


def main():
    routes = CD210 + OPM7
    res = {}
    with cf.ProcessPoolExecutor(max_workers=min(len(routes), 16)) as ex:
        for rid, r in zip(routes, ex.map(analyze, routes)):
            res[rid] = r
            print(f'  {rid}: {len(r["runs"]) if r else 0} straight runs', flush=True)

    def grp(rids):
        return [r for rid in rids if res.get(rid) for r in res[rid]['runs']]

    cd = grp(CD210); op = grp(OPM7)
    print('\n' + '=' * 92)
    print('STRAIGHT-LINE CENTER-KEEPING (dense modelV2) — CD210 (b1) vs OPM7')
    print('=' * 92)
    print(f'  {"group":<8}{"runs":>5}{"tot_s":>7}{"spd":>6}{"A_std(m)":>10}{"A_cps":>8}{"desC_cps":>10}')
    for nm, g in [('CD210', cd), ('OPM7', op)]:
        if not g:
            print(f'  {nm}: no runs'); continue
        tot = sum(r['dur'] for r in g)
        print(f'  {nm:<8}{len(g):>5}{tot:>7.0f}{med([r["spd"] for r in g]):>6.0f}{med([r["A_std"] for r in g]):>10.4f}'
              f'{med([r["A_cps"] for r in g]):>8.2f}{med([r["desC_cps"] for r in g]):>10.2f}')
    print('\n  -- speed-matched 45-60 mph --')
    for nm, g in [('CD210', cd), ('OPM7', op)]:
        gs = [r for r in g if 45 <= r['spd'] < 60]
        if not gs:
            print(f'  {nm}: none 45-60'); continue
        print(f'  {nm:<8}{len(gs):>5}{sum(r["dur"] for r in gs):>7.0f}{med([r["spd"] for r in gs]):>6.0f}'
              f'{med([r["A_std"] for r in gs]):>10.4f}{med([r["A_cps"] for r in gs]):>8.2f}{med([r["desC_cps"] for r in gs]):>10.2f}')
    print('\n  A_std=lateral wander amplitude (higher=more hunting); A_cps=wander reversals/s; desC_cps=model jitter/s')


if __name__ == '__main__':
    main()
