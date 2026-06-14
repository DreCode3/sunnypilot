#!/usr/bin/env python3
"""FINAL verified trend test: apply the ONLY calibrated method (blockperm on tau=days-since-reset, epoch_fe=False,
FPR ~0.04-0.05) to the REAL data, with conservative reset detection. Reports the realized epoch structure, the
calibrated p for FULL + weak-only (and the calendar predictor for contrast, FLAGGED as having no calibrated test),
and re-confirms calibration AT the realized epoch sizes (so we know the test is valid for THIS structure).
  .venv311/bin/python longitudinal_weave/final_trend.py --workers 12"""
import argparse, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'learned_param_studies', 'code'))
import analyze as AN
import calib_candidates as CC


def run_bp(sub, ycol, tcol, controls, L=4, nperm=8000, seed=1):
    cols = [ycol, tcol] + controls + ['epoch']
    s = sub[cols].dropna()
    if len(s) < 8:
        return None
    ry, rt, ep, n = CC.prep(s, ycol, tcol, controls, epoch_fe=False)
    if np.std(rt) < 1e-9:
        return None
    p = CC.test_blockperm(ry, rt, ep, L, nperm, np.random.default_rng(seed))
    return dict(n=n, n_epoch=len(np.unique(ep)), slope=CC._slope(ry, rt), p=float(p))


def calib_at(sizes, tcol, reps, workers, L=4, nboot=400):
    """re-confirm blockperm FPR + power at the REALIZED epoch sizes."""
    def rate(bt, rho=0.3):
        ss = np.random.SeedSequence(abs(hash((tcol, bt, tuple(sizes)))) & 0xffffffff).spawn(reps)
        args = [('blockperm', sizes, rho, -0.06, bt, L, nboot, int(s.generate_state(1)[0]), tcol, False) for s in ss]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            ps = [p for p in ex.map(CC._one, args) if np.isfinite(p)]
        return float(np.mean([p < 0.05 for p in ps])) if ps else np.nan, len(ps)
    fpr, nf = rate(0.0); p2, _ = rate(0.02); p4, _ = rate(0.04)
    print(f"  calibration @ realized sizes (tcol={tcol}): FPR={fpr:.3f} (N={nf}) | power b=0.02:{p2:.2f} b=0.04:{p4:.2f}")
    return fpr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--reps', type=int, default=400); a = ap.parse_args()
    workers = a.workers or (os.cpu_count() or 4)
    df = pd.read_csv('longitudinal_weave/results/drive_longitudinal.csv')
    df = df[df['n_elig'] >= 3].copy(); df['pi_bin'] = (df['pi_set'] == 'golden').astype(int)
    d = AN.detect_resets(df)
    t = pd.to_datetime(d['wall_date'], unit='s'); d['cal_days'] = (t - t.min()).dt.total_seconds() / 86400
    sizes = [int(n) for n in d.groupby('epoch').size().values]
    print(f"drives={len(d)}  epochs={len(sizes)}  sizes={sorted(sizes, reverse=True)}  resets@={d.loc[d['is_reset'],'date'].tolist()}\n")

    print("=== PRIMARY (calibrated): weave_path ~ tau(days_since_reset), speed+config controlled, blockperm ===")
    full = run_bp(d, 'weave_path', 'days_since_reset', ['spd_mph', 'pi_bin'])
    print(f"  FULL : n={full['n']} epochs={full['n_epoch']} slope={full['slope']:+.4f}/day  p={full['p']:.4f}")
    wk = d[d['pi_set'] == 'weak']
    weak = run_bp(wk, 'weave_path', 'days_since_reset', ['spd_mph'])
    if weak: print(f"  WEAK : n={weak['n']} epochs={weak['n_epoch']} slope={weak['slope']:+.4f}/day  p={weak['p']:.4f}")

    print("\n--- calibration re-confirmed AT the realized epoch structure (is the tau test valid here?) ---")
    calib_at(sizes, 'tau', a.reps, workers)

    print("\n=== CONTRAST: calendar-time predictor (NO calibrated test exists — report FPR to prove it) ===")
    fullc = run_bp(d, 'weave_path', 'cal_days', ['spd_mph', 'pi_bin'])
    print(f"  FULL weave~calendar: slope={fullc['slope']:+.4f}/day p_raw={fullc['p']:.4f}  <-- UNTRUSTWORTHY (see FPR below)")
    calib_at(sizes, 'gday', a.reps, workers)


if __name__ == '__main__':
    main()
