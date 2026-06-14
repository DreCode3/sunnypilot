#!/usr/bin/env python3
"""Straight-line lane-keeping hunt: CD210 (b1) vs OPM7 baseline.
On contiguous straight runs (|cmd|<5e-4, v>15, no override), measure lateral wander (lOff std),
wheel hunting (cmd crossings/s) and model hunting (des crossings/s). Tests the felt 'can't hold center'."""
import sys, glob
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.analyze_curves_v2 import load_cx1, IDX
from explorer_st_logs.phase1_validation import route_prefix_for, crossings_per_sec

CD210 = ['route_b1']
OPM7 = ['route_aa', 'route_ac', 'route_ad', 'route_af', 'route_b0']  # live-lOff, same config, OPM7


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
    p = route_prefix_for(rid)
    if not p:
        return None
    arr, t = load_cx1(p)
    if arr is None or len(arr) < 300:
        return None
    cmd = arr[:, IDX['cmd']]; des = arr[:, IDX['des']]; v = arr[:, IDX['v']]
    ovr = arr[:, IDX['ovr']]; lOff = arr[:, IDX['lOff']]
    k = 40
    desroll = np.convolve(np.abs(des), np.ones(k) / k, mode='same')  # ~2s avg -> straight ROAD (allows hunting wiggle)
    sm = (desroll < 5e-4) & (v > 15) & (ovr < 0.5)
    out = []
    for a, b in runs(sm, 120):  # >= ~6s contiguous straight road
        ts = t[a:b]
        if ts[-1] - ts[0] < 6:
            continue
        n = int((ts[-1] - ts[0]) * 20) + 1
        tu = ts[0] + np.arange(n) / 20.0
        cmdu = np.interp(tu, ts, cmd[a:b]); desu = np.interp(tu, ts, des[a:b]); lou = np.interp(tu, ts, lOff[a:b])
        out.append(dict(dur=float(ts[-1] - ts[0]), spd=float(np.median(v[a:b]) * 2.237),
                        lOff_std=float(np.std(lou)),
                        lOff_cps=crossings_per_sec(lou, 0.05, threshold=0.03),  # lateral wander reversals (m/s)
                        cmd_cps=crossings_per_sec(cmdu, 0.05),
                        des_cps=crossings_per_sec(desu, 0.05)))
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
        runs = []
        for rid in rids:
            if res.get(rid):
                runs += res[rid]['runs']
        return runs

    cd = grp(CD210); op = grp(OPM7)
    print('\n' + '=' * 90)
    print('STRAIGHT-LINE LANE-KEEPING — CD210 (b1) vs OPM7 baseline (contiguous straights >=8s, v>15)')
    print('=' * 90)
    print(f'  {"group":<8}{"runs":>5}{"spd_mph":>9}{"lOff_std(m)":>13}{"lOff_cps":>10}{"cmd_cps":>9}{"des_cps":>9}')
    for nm, g in [('CD210', cd), ('OPM7', op)]:
        if not g:
            print(f'  {nm}: no runs'); continue
        print(f'  {nm:<8}{len(g):>5}{med([r["spd"] for r in g]):>9.0f}{med([r["lOff_std"] for r in g]):>13.4f}'
              f'{med([r["lOff_cps"] for r in g]):>10.2f}{med([r["cmd_cps"] for r in g]):>9.2f}{med([r["des_cps"] for r in g]):>9.2f}')
    # speed-matched (45-60mph, where CD210 has data)
    print('\n  -- speed-matched 45-60 mph straights --')
    for nm, g in [('CD210', cd), ('OPM7', op)]:
        gs = [r for r in g if 45 <= r['spd'] < 60]
        if not gs:
            print(f'  {nm}: none in 45-60'); continue
        print(f'  {nm:<8}{len(gs):>5}{med([r["spd"] for r in gs]):>9.0f}{med([r["lOff_std"] for r in gs]):>13.4f}'
              f'{med([r["lOff_cps"] for r in gs]):>10.2f}{med([r["cmd_cps"] for r in gs]):>9.2f}{med([r["des_cps"] for r in gs]):>9.2f}')
    print('\n  lOff_std=lateral wander amplitude; lOff_cps=wander reversals/s; cmd_cps=wheel hunt; des_cps=model hunt')


if __name__ == '__main__':
    main()
