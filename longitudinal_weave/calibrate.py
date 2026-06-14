#!/usr/bin/env python3
"""Monte-Carlo the FALSE-POSITIVE RATE of analyze.trend_test's circular-shift-within-epoch permutation, under H0
(weave is autocorrelated within epoch + speed-dependent, but has NO trend with tau). A calibrated test must reject
at ~5%. Also a POWER check (inject a known tau slope, measure detection). Reuses the EXACT estimator from analyze.py
(no re-implementation -> no drift). PARALLEL across cores (reps fan out; inner permutation is serial to avoid nested
pools). Mirrors learned_param_studies/qa_calibration.py discipline.
  .venv311/bin/python longitudinal_weave/calibrate.py --csv longitudinal_weave/results/drive_longitudinal.csv \
       --reps 600 --nperm 300 [--workers N]"""
import argparse, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'learned_param_studies', 'code'))
import analyze as AN
import shared


def ar1(n, rho, rng):
    e = rng.standard_normal(n); x = np.empty(n); x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + np.sqrt(1 - rho * rho) * e[i]
    return x


def synth(epoch_sizes, rho, beta_speed, beta_tau, rng):
    """Build a drive table under a given truth. weave = base + beta_speed*(speed) + beta_tau*tau + AR1(rho) noise
    + a per-epoch random level. beta_tau=0 => H0 (FPR); beta_tau>0 => power. speed is random + a mild calendar
    drift (so speed is a genuine time-correlated confound to be controlled, as in the real data)."""
    rows = []
    gday = 0.0
    for e, n in enumerate(epoch_sizes):
        lvl = rng.normal(0, 0.5)
        tau = np.sort(rng.uniform(0, 20, n))                       # days-since-reset, monotone within epoch
        spd = 58 - 0.03 * gday + rng.normal(0, 6, n)               # speed drifts mildly over calendar time
        noise = ar1(n, rho, rng)
        weave = 1.6 + lvl + beta_speed * (spd - 58) + beta_tau * tau + 0.5 * noise
        pi = int(e % 2)                                            # alternating config to exercise the covariate
        for i in range(n):
            rows.append(dict(weave=weave[i], tau=tau[i], speed=spd[i], pi_bin=pi, epoch=e))
            gday += 0.7
    return pd.DataFrame(rows)


def trend_p_serial(df, nperm, seed):
    r0, ry, rt, epoch, n = AN.trend_r(df, 'weave', 'tau', ['speed', 'pi_bin'])
    if not np.isfinite(r0):
        return np.nan
    rng = np.random.default_rng(seed); c = 1
    for _ in range(nperm):
        if abs(shared._fast_pearson(AN._circshift_within_epoch(ry, epoch, rng), rt)) >= abs(r0) - 1e-12:
            c += 1
    return c / (nperm + 1)


def _rep(args):
    epoch_sizes, rho, beta_speed, beta_tau, nperm, seed = args
    rng = np.random.default_rng(seed)
    df = synth(epoch_sizes, rho, beta_speed, beta_tau, rng)
    return trend_p_serial(df, nperm, int(rng.integers(1 << 30)))


def battery(epoch_sizes, reps, nperm, workers):
    workers = workers or (os.cpu_count() or 4)
    def rate(rho, bs, bt, label):
        ss = np.random.SeedSequence(hash((rho, bs, bt)) & 0xffffffff).spawn(reps)
        args = [(epoch_sizes, rho, bs, bt, nperm, int(s.generate_state(1)[0])) for s in ss]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            ps = [p for p in ex.map(_rep, args) if np.isfinite(p)]
        rej = float(np.mean([p < 0.05 for p in ps])) if ps else np.nan
        print(f"  {label:<46}: rate(p<0.05)={rej:.3f} (N={len(ps)})")
        return rej
    print(f"epoch sizes (real): {epoch_sizes}  sum={sum(epoch_sizes)}  reps={reps} nperm={nperm}")
    print("\n--- FPR under H0 (beta_tau=0): target ~0.05, >~0.08 = anti-conservative=FAIL ---")
    for rho in (0.0, 0.3, 0.6):
        rate(rho, -0.06, 0.0, f"H0 speed-confounded, AR({rho})")
    print("\n--- POWER (can n detect a real tau slope?) at AR(0.3) ---")
    for bt in (0.01, 0.02, 0.04):
        rate(0.3, -0.06, bt, f"beta_tau={bt}/day (~{bt*15:.2f} weave over 15d)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--reps', type=int, default=600)
    ap.add_argument('--nperm', type=int, default=300); ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--sizes', default='', help='comma-sep epoch sizes to override (else derived from data)')
    a = ap.parse_args()
    if a.sizes:
        sizes = [int(x) for x in a.sizes.split(',')]
    else:
        df = pd.read_csv(a.csv); df = df[df['n_elig'] >= 3].copy()
        df['pi_bin'] = (df['pi_set'] == 'golden').astype(int)
        d = AN.detect_resets(df)
        sizes = [int(n) for n in d.groupby('epoch').size().values if n >= 2]
    battery(sizes, a.reps, a.nperm, a.workers)


if __name__ == '__main__':
    main()
