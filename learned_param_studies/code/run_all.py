#!/usr/bin/env python3
"""Run all three learned-param tests, then apply ONE GLOBAL BH-FDR across the COMBINED primary battery
(every within-drive, speed-controlled correlation in the suite). The honest multiple-comparison family is the
whole suite, not a single test -- per-test FDR (8-12 hypotheses) is too lenient. Prints the globally-robust set,
tags the integrator->centering rows as mechanically reverse-causal, and states the net lever verdict.
  .venv311/bin/python learned_param_studies/code/run_all.py --tables learned_param_studies/results"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import shared, test1_liveparams, test2_integrator, test3_calibration

FILES = [('liveparams', 'test1_liveparams_results.csv'),
         ('integrator', 'test2_integrator_results.csv'),
         ('calibration', 'test3_calibration_results.csv')]


def run(tdir):
    for m in (test1_liveparams, test2_integrator, test3_calibration):
        print("\n" + "#" * 72)
        m.run(tdir)
    frames = []
    for test, fn in FILES:
        d = pd.read_csv(os.path.join(tdir, fn)); d['test'] = test
        frames.append(d[['test', 'param', 'lateral', 'within_r', 'within_p']])
    allr = pd.concat(frames, ignore_index=True)
    # De-duplicate RANK-COLLINEAR params (e.g. cal_roll == cal_yaw, Spearman=1.000: calibrationd hard-codes
    # roll=0 so published roll is a deterministic function of yaw) from the FDR family, so the effective family
    # size m is honest (not inflated by carrying the same hypothesis twice). Keep the first-seen of any pair.
    drv = pd.read_csv(os.path.join(tdir, 'drive_table.csv'))
    params = list(dict.fromkeys(allr['param']))
    drop, dup_of = set(), {}
    for i, a in enumerate(params):
        for b in params[i + 1:]:
            if b in drop or a not in drv.columns or b not in drv.columns:
                continue
            s = drv[[a, b]].dropna()
            if len(s) >= 3 and abs(s[a].corr(s[b], method='spearman')) > 0.999:
                drop.add(b); dup_of[b] = a
    allr['in_family'] = ~allr['param'].isin(drop)
    allr['q_global'] = np.nan
    allr.loc[allr['in_family'], 'q_global'] = shared.bh_fdr(allr.loc[allr['in_family'], 'within_p'].values)
    allr['note'] = np.where(allr['test'].eq('integrator') & allr['lateral'].astype(str).str.startswith('centering'),
                            'REVERSE-CAUSAL (controller responds to offset; not a lever)', '')
    allr.loc[~allr['in_family'], 'note'] = allr.loc[~allr['in_family'], 'param'].map(
        lambda p: f'collinear dup of {dup_of.get(p, "?")}; excluded from FDR family')
    if drop:
        print(f"\nde-duplicated rank-collinear params from FDR family: "
              f"{', '.join(f'{b}==' + dup_of[b] for b in sorted(drop))}  -> effective m={int(allr['in_family'].sum())}")
    allr = allr.sort_values('within_p').reset_index(drop=True)
    print("\n" + "=" * 72)
    print(f"GLOBAL BH-FDR across the full PRIMARY battery (within-drive, speed-controlled): n_tests={len(allr)}")
    print("=" * 72)
    with pd.option_context('display.width', 200, 'display.max_columns', 50, 'display.max_colwidth', 60):
        show = allr.copy()
        show['within_r'] = show['within_r'].round(2); show['within_p'] = show['within_p'].round(3)
        show['q_global'] = show['q_global'].round(3)
        print(show.head(12).to_string(index=False))
    m = int(allr['in_family'].sum())
    rob = allr[(allr['q_global'] < 0.10) & np.isfinite(allr['q_global'])]
    lever = rob[rob['note'] == '']                              # robust AND not reverse-causal
    lever = lever[~lever['lateral'].astype(str).str.startswith('centering')]   # a weave/hunt lever, not centering
    print(f"\nglobally robust (q<0.10): {len(rob)}/{m} (effective family)")
    print(f"  of those, reverse-causal integrator->centering rows: {(rob['note'] != '').sum()}")
    print(f"  CANDIDATE LEVERS (robust, not reverse-causal, target=weave/hunt): {len(lever)}")
    if len(lever):
        for _, r in lever.iterrows():
            print(f"    -> {r['test']}/{r['param']} ~ {r['lateral']}: r={r['within_r']:+.2f} q={r['q_global']:.3f}")
    else:
        print("    -> 0 candidate WEAVE/HUNT levers (centering levers, if any, are reported in test3).")
        print("       No learned parameter is a validated weave/hunt lever in this dataset.")
    allr.to_csv(os.path.join(tdir, 'GLOBAL_fdr_summary.csv'), index=False)
    print(f"\nwrote {tdir}/GLOBAL_fdr_summary.csv")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tables', required=True); run(ap.parse_args().tables)
