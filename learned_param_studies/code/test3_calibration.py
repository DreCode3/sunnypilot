#!/usr/bin/env python3
"""TEST 3 — learned CAMERA CALIBRATION (liveCalibration rpyCalib) vs lateral performance.
Params: cal_yaw, cal_pitch, cal_roll (deg). PRIMARY lateral target = centering_signed (a camera-YAW error biases
where the model thinks the lane center is); secondary = centering_abs, weave_path, weave_steer.
QA-corrected 2026-06-13:
  * cal_yaw and cal_roll are PERFECTLY rank-collinear (Spearman=1.000: calibrationd hard-codes roll=0 in the
    observation, so published roll is a deterministic function of yaw) -> they are STATISTICALLY INDISTINGUISHABLE;
    a result for cal_yaw IS the same result for cal_roll. cal_roll is reported but tagged.
  * within-drive is detrended + SPEED-CONTROLLED. cal_pitch's old within-drive '+0.29' was a SPEED CONFOUND
    (cal_pitch~speed~-0.4, speed~weave~-0.65; within-drive pitch only varies ~0.01deg = calib noise) -> it
    collapses under speed control. The across-drive partial now controls SPEED+CONFIG+MODEL (cal_pitch tracks
    model-era), not the old noise 'order'.
  * significance from exact/permutation p; battery BH-FDR corrected (q). Cleanest = within-drive(speed-ctrl) + stratified.
  .venv311/bin/python learned_param_studies/code/test3_calibration.py --tables learned_param_studies/results"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import shared

PARAMS = ['cal_yaw', 'cal_pitch', 'cal_roll']
LAT = ['centering_signed', 'centering_abs', 'weave_path', 'weave_steer']


def run(tdir):
    seg = pd.read_csv(os.path.join(tdir, 'segment_table.csv'))
    drv = pd.read_csv(os.path.join(tdir, 'drive_table.csv'))
    drv['pi_bin'] = (drv['pi_set'] == 'golden').astype(int)
    drv['model_bin'] = drv['driving_model'].astype(str).str.contains('CD210').astype(int)
    strat = drv[(drv['driving_model'] == 'CD210') & (drv['pi_set'] == 'weak')]
    # confirm + warn on the cal_yaw==cal_roll rank-collinearity in THIS dataset
    collin = drv[['cal_yaw', 'cal_roll']].dropna()
    rho = collin['cal_yaw'].corr(collin['cal_roll'], method='spearman') if len(collin) >= 3 else np.nan
    print(f"drives={len(drv)}  CD210/weak-stratified={len(strat)}  segments={len(seg)}")
    print(f"cal_yaw~cal_roll Spearman={rho:+.3f}  ({'COLLINEAR: cal_roll == cal_yaw, do not interpret separately' if np.isfinite(rho) and rho > 0.999 else 'not perfectly collinear here'})")
    print("interpret: a camera-yaw miscalibration should shift centering_signed. PRIMARY = within-drive (speed-ctrl,")
    print("           detrend) + BH-FDR q. Partial controls SPEED+CONFIG+MODEL. CIs descriptive only.\n")
    out = []
    for p in PARAMS:
        for L in LAT:
            ad = shared.spearman_ci(drv[p], drv[L])
            st = shared.spearman_ci(strat[p], strat[L]) if len(strat) >= 5 else dict(r=np.nan, p=np.nan, n=len(strat))
            pa = shared.partial_spearman(drv, p, L, ['spd_mph', 'pi_bin', 'model_bin'])
            wd = shared.within_drive_spearman(seg, p, L, ctrl='spd_mph', detrend=True)
            wd_raw = shared.within_drive_spearman(seg, p, L, ctrl=None, detrend=True)
            out.append(dict(param=p, lateral=L, across=ad, strat=st, partial=pa, within=wd, within_raw=wd_raw))
    qs = shared.bh_fdr([o['within']['p'] for o in out])
    cur = None
    for o, q in zip(out, qs):
        if o['param'] != cur:
            cur = o['param']
            tag = '  [== cal_yaw, collinear]' if cur == 'cal_roll' and np.isfinite(rho) and rho > 0.999 else ''
            print(f"\n========== {cur}{tag} ==========")
        print(f"  {o['lateral']:<17} across:{shared.fmt(o['across']):<40} strat(CD210wk):{shared.fmt(o['strat'])}")
        print(f"  {'':<17} partial(spd,cfg,model): r={o['partial']['r']:+.2f} p={o['partial']['p']:.3f} n={o['partial']['n']:<4}"
              f"  within-raw:{shared.fmt(o['within_raw'])}")
        print(f"  {'':<17} PRIMARY within(spd-ctrl,detrend): {shared.fmt(o['within'], q=q)}")
    rows = [dict(param=o['param'], lateral=o['lateral'], across_r=o['across']['r'], across_p=o['across']['p'],
                 strat_r=o['strat'].get('r'), partial_r=o['partial']['r'], partial_p=o['partial']['p'],
                 within_r=o['within']['r'], within_p=o['within']['p'], within_q=q,
                 within_raw_r=o['within_raw']['r'], within_raw_p=o['within_raw']['p'])
            for o, q in zip(out, qs)]
    pd.DataFrame(rows).to_csv(os.path.join(tdir, 'test3_calibration_results.csv'), index=False)
    nrob = int(np.sum((qs < 0.10) & np.isfinite(qs)))
    print(f"\nwrote {tdir}/test3_calibration_results.csv   |  BH-FDR survivors (q<0.10): {nrob}/{len(out)}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tables', required=True); run(ap.parse_args().tables)
