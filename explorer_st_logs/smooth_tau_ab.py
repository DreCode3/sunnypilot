#!/usr/bin/env python3
"""smooth_tau single-variable A/B (both CD210):
  BASE route_b1 = smooth_tau (0.12, 0.04)
  TEST route_b2 = smooth_tau (0.25, 0.12)   <- iter3, just deployed
Hypothesis: stronger command EMA low-passes CD210's high-freq jitter ->
  STRAIGHTS: A_std/A_cps DOWN (less felt lateral wander = the center-hunt target),
             desC_cps ~SAME (model output unchanged = sanity check),
  CURVES:    alat_band ~same/down, possible cmd_cps down; watch A_apex (cutting) + apex lag.
Loads dense modelV2 once per route; parallel over the 2 routes (M4 Max)."""
import sys, glob, math, json
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.analyze_curves_v2 import load_cx1, IDX, detect_curve_events
from explorer_st_logs.phase1_validation import per_event_metrics, route_prefix_for, crossings_per_sec
from explorer_st_logs.straight_hunting_mv2 import load_mv2, runs

BASE = 'route_b1'   # CD210 + smooth_tau 0.12/0.04
TEST = 'route_b2'   # CD210 + smooth_tau 0.25/0.12


def analyze(rid):
    d = load_mv2(rid)  # dense 20Hz modelV2 (loaded once)
    straights = []
    if d is not None:
        t, dc, A, spd = d['t'], d['dc'], d['A'], d['spd']
        dcroll = np.convolve(np.abs(dc), np.ones(40) / 40, mode='same')
        sm = (dcroll < 5e-4) & (spd > 13)
        for a, b in runs(sm, 120):
            ts = t[a:b]
            if ts[-1] - ts[0] < 6:
                continue
            dt = float(np.median(np.diff(ts))) or 0.05
            straights.append(dict(spd=float(np.median(spd[a:b]) * 2.237), dur=float(ts[-1] - ts[0]),
                                  A_std=float(np.std(A[a:b])),
                                  A_cps=crossings_per_sec(A[a:b], dt, threshold=0.05),
                                  desC_cps=crossings_per_sec(dc[a:b], dt, threshold=4e-5)))
    curves = []
    p = route_prefix_for(rid)
    if p:
        try:
            arr, t = load_cx1(p)
        except Exception:
            arr = None
        if arr is not None and len(arr) >= 300:
            aLat = arr[:, IDX['aLat']]
            for e in detect_curve_events(arr, t, require_dir_match_pct=0.0):
                m = per_event_metrics(arr, t, e)
                if not m:
                    continue
                A = float('nan')
                if d is not None and len(d['t']) > 1:
                    A = float(np.interp(e['t_apex'], d['t'], d['A']))
                curves.append(dict(dir=e['direction'], mclass=e['magnitude_class'], speed=m['mean_speed_mph'],
                                   alat_band=m['alat_band_0_5_3'], jerk=m['jerk_rms'], cmd_cps=m['cmd_cps'],
                                   A_apex=A, aLat_apex=float(aLat[e['apex_idx']]), had_override=bool(e['had_override'])))
    drive_min = float((t[-1] - t[0]) / 60.0) if p and arr is not None else float('nan')
    return dict(rid=rid, straights=straights, curves=curves, drive_min=drive_min)


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else float('nan')


def main():
    with cf.ProcessPoolExecutor(max_workers=2) as ex:
        res = {r['rid']: r for r in ex.map(analyze, [BASE, TEST])}
    b, tst = res[BASE], res[TEST]
    sb, st = b['straights'], tst['straights']
    cb, ct = b['curves'], tst['curves']

    print('=' * 94)
    print(f'smooth_tau A/B  —  BASE {BASE} (0.12/0.04, {b["drive_min"]:.0f}min)  vs  TEST {TEST} (0.25/0.12, {tst["drive_min"]:.0f}min)   [both CD210]')
    print('=' * 94)

    print('\n--- STRAIGHTS (target): A_std/A_cps DOWN = less hunt | desC_cps ~SAME = model unchanged ---')
    print(f'  {"grp":<6}{"runs":>5}{"spd":>6}{"A_std(m)":>10}{"A_cps":>8}{"desC_cps":>10}')
    for nm, g in [('BASE', sb), ('TEST', st)]:
        if not g:
            print(f'  {nm}: no straight runs'); continue
        print(f'  {nm:<6}{len(g):>5}{med([r["spd"] for r in g]):>6.0f}{med([r["A_std"] for r in g]):>10.4f}'
              f'{med([r["A_cps"] for r in g]):>8.2f}{med([r["desC_cps"] for r in g]):>10.2f}')
    print('  -- speed-matched bins --')
    for lo, hi in [(25, 40), (40, 50), (50, 65)]:
        line = f'    {lo}-{hi}mph '
        for nm, g in [('BASE', sb), ('TEST', st)]:
            gs = [r for r in g if lo <= r['spd'] < hi]
            line += f'| {nm} N={len(gs)} A_std={med([r["A_std"] for r in gs]):.4f} A_cps={med([r["A_cps"] for r in gs]):.2f} desC={med([r["desC_cps"] for r in gs]):.2f} '
        print(line)

    print('\n--- CURVES by magnitude: alat_band(osc), cmd_cps, jerk, A_apex(+=right/inside) ---')
    print(f'  {"grp":<6}{"mag":<9}{"N":>4}{"spd":>5}{"alat_band":>10}{"cmd_cps":>8}{"jerk":>7}{"A_apex":>8}')
    for nm, g in [('BASE', cb), ('TEST', ct)]:
        for mg in ('gentle', 'moderate', 'sharp'):
            sub = [r for r in g if r['mclass'] == mg]
            if not sub:
                continue
            print(f'  {nm:<6}{mg:<9}{len(sub):>4}{med([r["speed"] for r in sub]):>5.0f}{med([r["alat_band"] for r in sub]):>10.1f}'
                  f'{med([r["cmd_cps"] for r in sub]):>8.2f}{med([r["jerk"] for r in sub]):>7.2f}{med([r["A_apex"] for r in sub]):>8.2f}')

    print('\n--- overrides ---')
    for nm, g in [('BASE', cb), ('TEST', ct)]:
        print(f'  {nm}: {sum(1 for r in g if r["had_override"])} override events / {len(g)} curve events')

    json.dump({'S': {'b1': sb, 'b2': st}, 'C': {'b1': cb, 'b2': ct}},
              open('explorer_st_logs/smooth_tau_ab.json', 'w'), default=str)
    print('\nsaved -> explorer_st_logs/smooth_tau_ab.json')


if __name__ == '__main__':
    main()
