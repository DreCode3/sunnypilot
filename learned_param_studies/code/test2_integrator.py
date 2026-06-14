#!/usr/bin/env python3
"""TEST 2 — the cumulative PI lane-centering INTEGRATOR (our warm-started LaneBiasIntegral) vs lateral performance.
Params: int_abs (mean |integral|), int_railed_frac (fraction at the 0.30 cap). Lateral: weave_path, centering_abs,
centering_signed, weave_steer. The integrator VARIES WITHIN a drive (it accumulates), so WITHIN-DRIVE is PRIMARY.
QA-corrected 2026-06-13: within-drive is detrended + SPEED-CONTROLLED; significance from permutation p; BH-FDR over
the battery. CAUSAL CAVEAT printed below -- a within-drive int<->centering association is mechanically REVERSE-CAUSAL
(the integrator winds up in RESPONSE to an offset; raw 1Hz telemetry shows |int| LAGS offset by +1-2s), so a
surviving int~centering correlation is the controller doing its job, NOT evidence the integrator drives weave.
NOTE: only drives with the 'LC:' telemetry have an integrator signal (excludes OPM7/older + mostly-manual drives).
  .venv311/bin/python learned_param_studies/code/test2_integrator.py --tables learned_param_studies/results"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import shared

PARAMS = ['int_abs', 'int_railed_frac']
LAT = ['weave_path', 'centering_abs', 'centering_signed', 'weave_steer']


def run(tdir):
    seg = pd.read_csv(os.path.join(tdir, 'segment_table.csv'))
    drv = pd.read_csv(os.path.join(tdir, 'drive_table.csv'))
    seg_i = seg.dropna(subset=['int_abs'])
    print(f"drives with integrator telemetry: {seg_i['drive_id'].nunique()}  segments={len(seg_i)}")
    print("interpret: WITHIN-DRIVE (speed-ctrl, detrend) is PRIMARY. A surviving int~centering link is REVERSE-CAUSAL")
    print("           (integrator responds to offset, lags it +1-2s) -- it works, it does not 'cause' weave.")
    print("           Only a POS int_abs~weave that survives FDR would implicate wind-up in weave.\n")
    out = []
    for p in PARAMS:
        for L in LAT:
            wd = shared.within_drive_spearman(seg_i, p, L, ctrl='spd_mph', detrend=True)
            wd_raw = shared.within_drive_spearman(seg_i, p, L, ctrl=None, detrend=True)
            ad = shared.spearman_ci(drv[p], drv[L])
            wkw = shared.within_drive_spearman(seg_i[seg_i['pi_set'] == 'weak'], p, L, ctrl='spd_mph', detrend=True)
            gdw = shared.within_drive_spearman(seg_i[seg_i['pi_set'] == 'golden'], p, L, ctrl='spd_mph', detrend=True)
            out.append(dict(param=p, lateral=L, within=wd, within_raw=wd_raw, across=ad, weak=wkw, gold=gdw))
    qs = shared.bh_fdr([o['within']['p'] for o in out])
    cur = None
    for o, q in zip(out, qs):
        if o['param'] != cur:
            cur = o['param']; print(f"\n========== {cur} ==========")
        print(f"  {o['lateral']:<16} PRIMARY within(spd-ctrl):{shared.fmt(o['within'], q=q)}")
        print(f"  {'':<16} within-raw:{shared.fmt(o['within_raw']):<40} across:{shared.fmt(o['across'])}")
        print(f"  {'':<16} within-WEAK:{shared.fmt(o['weak']):<40} within-GOLD:{shared.fmt(o['gold'])}")
    rows = [dict(param=o['param'], lateral=o['lateral'], within_r=o['within']['r'], within_p=o['within']['p'], within_q=q,
                 within_raw_r=o['within_raw']['r'], within_raw_p=o['within_raw']['p'], across_r=o['across']['r'],
                 within_weak_r=o['weak']['r'], within_gold_r=o['gold']['r'])
            for o, q in zip(out, qs)]
    pd.DataFrame(rows).to_csv(os.path.join(tdir, 'test2_integrator_results.csv'), index=False)
    nrob = int(np.sum((qs < 0.10) & np.isfinite(qs)))
    print(f"\nwrote {tdir}/test2_integrator_results.csv   |  BH-FDR survivors (q<0.10): {nrob}/{len(out)} "
          f"(reverse-causal where lateral is centering_*)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tables', required=True); run(ap.parse_args().tables)
