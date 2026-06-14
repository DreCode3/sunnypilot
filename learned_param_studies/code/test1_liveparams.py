#!/usr/bin/env python3
"""TEST 1 — learned VEHICLE/STEERING params (paramsd liveParameters) vs lateral performance.
Params: steerRatio, angleOffsetAvg, stiffness.  Lateral battery: weave_path (PRIMARY, model-indep path curvature),
weave_steer, centering_abs, hunt_steer.
Views per pair (QA-corrected 2026-06-13):
  ACROSS-drive (raw, confounded), CONFIG-STRATIFIED (CD210/weak only -> removes PI+model confound),
  PARTIAL across-drive controlling SPEED+CONFIG (the real confounds -- the old 'order' time-control was noise),
  WITHIN-drive PRIMARY = detrended + SPEED-CONTROLLED (speed is the dominant weave confound, within r~-0.65).
Significance comes from EXACT/permutation p (not the anti-conservative bootstrap CI); the whole battery is
BH-FDR corrected (q). Read the WITHIN-drive(speed-ctrl) + STRATIFIED as the de-confounded signal.
  .venv311/bin/python learned_param_studies/code/test1_liveparams.py --tables learned_param_studies/results"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import shared

PARAMS = ['steerRatio', 'angleOffsetAvg', 'stiffness']
LAT = ['weave_path', 'weave_steer', 'centering_abs', 'hunt_steer']


def run(tdir):
    seg = pd.read_csv(os.path.join(tdir, 'segment_table.csv'))
    drv = pd.read_csv(os.path.join(tdir, 'drive_table.csv'))
    drv['pi_bin'] = (drv['pi_set'] == 'golden').astype(int)
    strat = drv[(drv['driving_model'] == 'CD210') & (drv['pi_set'] == 'weak')]   # homogeneous subset
    print(f"drives total={len(drv)}  CD210/weak-stratified={len(strat)}  segments={len(seg)}")
    print("interpret: NEG r between a param and weave/centering = higher param -> LESS weave (param helps).")
    print("PRIMARY = within-drive (detrended, speed-controlled) + BH-FDR q over the battery. CIs are descriptive only.\n")
    out = []
    for p in PARAMS:
        for L in LAT:
            ad = shared.spearman_ci(drv[p], drv[L])
            st = shared.spearman_ci(strat[p], strat[L]) if len(strat) >= 5 else dict(r=np.nan, p=np.nan, n=len(strat))
            pa = shared.partial_spearman(drv, p, L, ['spd_mph', 'pi_bin'])
            wd = shared.within_drive_spearman(seg, p, L, ctrl='spd_mph', detrend=True)
            wd_raw = shared.within_drive_spearman(seg, p, L, ctrl=None, detrend=True)
            out.append(dict(param=p, lateral=L, across=ad, strat=st, partial=pa, within=wd, within_raw=wd_raw))
    qs = shared.bh_fdr([o['within']['p'] for o in out])   # FDR over the PRIMARY (within, speed-ctrl) battery
    cur = None
    for o, q in zip(out, qs):
        if o['param'] != cur:
            cur = o['param']; print(f"\n========== {cur} ==========")
        print(f"  {o['lateral']:<14} across:{shared.fmt(o['across']):<40} strat(CD210wk):{shared.fmt(o['strat'])}")
        print(f"  {'':<14} partial(spd,cfg): r={o['partial']['r']:+.2f} p={o['partial']['p']:.3f} n={o['partial']['n']:<4}"
              f"  within-raw:{shared.fmt(o['within_raw'])}")
        caveat = '  [within-drive p is a FLOOR: nonlinear within-drive drift -> gate ~0.08-0.13 FPR; see across]' \
            if o['param'] in ('angleOffsetAvg', 'steerRatio', 'stiffness') else ''
        print(f"  {'':<14} PRIMARY within(spd-ctrl,detrend): {shared.fmt(o['within'], q=q)}{caveat}")
    rows = [dict(param=o['param'], lateral=o['lateral'], across_r=o['across']['r'], across_p=o['across']['p'],
                 strat_r=o['strat'].get('r'), partial_r=o['partial']['r'], partial_p=o['partial']['p'],
                 within_r=o['within']['r'], within_p=o['within']['p'], within_q=q,
                 within_raw_r=o['within_raw']['r'], within_raw_p=o['within_raw']['p'],
                 within_ndrives=o['within'].get('n_drives'))
            for o, q in zip(out, qs)]
    pd.DataFrame(rows).to_csv(os.path.join(tdir, 'test1_liveparams_results.csv'), index=False)
    nrob = int(np.sum((qs < 0.10) & np.isfinite(qs)))
    print(f"\nwrote {tdir}/test1_liveparams_results.csv   |  BH-FDR survivors (q<0.10): {nrob}/{len(out)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tables', required=True); run(ap.parse_args().tables)
