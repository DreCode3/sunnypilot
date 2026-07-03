#!/usr/bin/env python3
"""LENS-2 design verification: calibrate a significance test for "weave rises with TIME (within a regime),
controlling speed+config" that holds ~5% FPR under realistic conditions (autocorrelated weave, speed-confound
that drifts with calendar, ~22-68 drives, few/uncertain resets).

Candidates evaluated (all operate on the speed[+config]-residualized weave-vs-TIME slope, calendar order):
  (M) moving-block bootstrap of the residualized weave~time slope (block len tuned to autocorr)
  (S) stationary (geometric-block) bootstrap of the same slope (Politis-Romano)
  (C) cluster-robust OLS by epoch (HC by epoch cluster) -> t vs normal/t
  (A) AR(1)-prewhitened OLS (Cochrane-Orcutt-ish) of residualized weave on time
  (P) corrected block-PERMUTATION of the time variable in contiguous blocks (block len = autocorr)

Synthetic truth mirrors calibrate.synth (speed drifts with calendar; AR(1) weave noise; per-epoch level;
alternating config). beta_tau=0 => FPR; beta_tau>0 => power. HARD COST BOUND: reps<=400 fast.

  .venv311/bin/python stock_lateral_toolkit/calib_candidates.py --reps 400 --sizes 8,12,6 [--workers N]
"""
import argparse, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))


# ---------- residualize helper ----------
def _resid(y, Xcols):
    C = np.column_stack([np.ones(len(y))] + list(Xcols))
    Q, _ = np.linalg.qr(C)
    return y - Q @ (Q.T @ y)


def _slope(y, t):
    """OLS slope of y on t (both already residualized of nuisance + each other not needed: use raw OLS)."""
    tc = t - t.mean()
    return float(np.dot(tc, y - y.mean()) / np.dot(tc, tc))


def _demean_epoch(v, epoch):
    out = v.copy()
    for e in np.unique(epoch):
        m = epoch == e
        out[m] = v[m] - v[m].mean()
    return out


def prep(df, ycol, tcol, controls, epoch_fe=False):
    """Return calendar-ordered residualized weave (on controls), residualized time (on controls), epoch.
    epoch_fe=True ALSO removes per-epoch means from y and t (within-epoch demean) -> isolates the within-epoch
    slope (the sawtooth) and treats per-epoch level as a nuisance fixed effect. This is the correct framing for
    the time-since-reset hypothesis and removes the per-epoch-level confound that wrecks tiny-epoch structures."""
    cols = [ycol, tcol] + controls + ['epoch']
    d = df[cols].dropna().reset_index(drop=True)
    y = d[ycol].values.astype(float)
    t = d[tcol].values.astype(float)
    epoch = d['epoch'].values
    X = [d[c].values.astype(float) for c in controls]
    ry = _resid(y, X)          # weave with speed+config removed
    rt = _resid(t, X)          # time with speed+config removed (FWL: slope of ry~rt == partial slope)
    if epoch_fe:
        ry = _demean_epoch(ry, epoch)
        rt = _demean_epoch(rt, epoch)
    return ry, rt, epoch, len(d)


# ============================================================
# (M) moving-block bootstrap of the residualized slope
# ============================================================
def _mbb_indices(n, L, rng):
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=nb) if n > L else np.zeros(nb, int)
    idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
    return idx


def test_mbb(ry, rt, epoch, L, nboot, rng, stationary=False, epoch_aware=False):
    obs = _slope(ry, rt)
    n = len(ry)
    # epoch-aware: blocks never straddle reset boundaries (each epoch resampled internally; tiny epochs = whole)
    ep_idx = [np.where(epoch == e)[0] for e in np.unique(epoch)] if epoch_aware else None
    boots = np.empty(nboot)
    for b in range(nboot):
        if epoch_aware:
            parts = []
            for ei in ep_idx:
                m = len(ei)
                if m <= L:
                    parts.append(ei)                       # whole epoch as one block
                else:
                    nb = int(np.ceil(m / L))
                    starts = rng.integers(0, m - L + 1, size=nb)
                    parts.append(ei[np.concatenate([np.arange(s, s + L) for s in starts])[:m]])
            idx = np.concatenate(parts)
        elif stationary:
            # geometric block lengths, mean L (Politis-Romano)
            idx = []
            while len(idx) < n:
                s = rng.integers(0, n)
                bl = rng.geometric(1.0 / L)
                idx.extend([(s + k) % n for k in range(bl)])
            idx = np.array(idx[:n])
        else:
            idx = _mbb_indices(n, L, rng)
        boots[b] = _slope(ry[idx], rt[idx])
    # two-sided p via bootstrap SE + normal (recenter on obs; bootstrap estimates sampling var of slope)
    se = boots.std(ddof=1)
    if se == 0:
        return np.nan
    z = obs / se
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(abs(z)))


# ============================================================
# (C) cluster-robust OLS by epoch
# ============================================================
def test_cluster(ry, rt, epoch):
    n = len(ry)
    G = len(np.unique(epoch))
    if G < 2:
        return np.nan
    # slope of ry on rt (intercept ~0 since residualized, keep it anyway)
    X = np.column_stack([np.ones(n), rt])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ ry)
    resid = ry - X @ beta
    # cluster-robust meat
    meat = np.zeros((2, 2))
    for g in np.unique(epoch):
        m = epoch == g
        Xg = X[m]; ug = resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    # small-sample correction (Stata-style)
    c = (G / (G - 1)) * ((n - 1) / (n - 2))
    V = c * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(V[1, 1])
    if se == 0 or not np.isfinite(se):
        return np.nan
    t = beta[1] / se
    from scipy.stats import t as tdist
    return 2 * (1 - tdist.cdf(abs(t), df=G - 1))   # df = G-1 (cluster count)


# ============================================================
# (A) AR(1)-prewhitened OLS
# ============================================================
def test_ar1(ry, rt, epoch):
    n = len(ry)
    # estimate rho from OLS residuals of ry~rt (calendar order, pooled; ignore epoch breaks -> conservative-ish)
    X = np.column_stack([np.ones(n), rt])
    beta = np.linalg.lstsq(X, ry, rcond=None)[0]
    e = ry - X @ beta
    rho = np.clip(np.dot(e[:-1], e[1:]) / np.dot(e[:-1], e[:-1]), -0.95, 0.95) if np.dot(e[:-1], e[:-1]) > 0 else 0.0
    # prewhiten within each epoch (don't difference across reset boundaries)
    yw, Xw = [], []
    for g in np.unique(epoch):
        m = np.where(epoch == g)[0]
        if len(m) == 1:
            yw.append(ry[m]); Xw.append(X[m]); continue
        yy = ry[m]; XX = X[m]
        yw.append(yy[1:] - rho * yy[:-1])
        Xw.append(XX[1:] - rho * XX[:-1])
    yw = np.concatenate(yw); Xw = np.concatenate(Xw)
    if len(yw) < 4:
        return np.nan
    bw = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    ew = yw - Xw @ bw
    dof = len(yw) - 2
    s2 = np.dot(ew, ew) / dof
    XtXinv = np.linalg.inv(Xw.T @ Xw)
    se = np.sqrt(s2 * XtXinv[1, 1])
    if se == 0 or not np.isfinite(se):
        return np.nan
    t = bw[1] / se
    from scipy.stats import t as tdist
    return 2 * (1 - tdist.cdf(abs(t), df=dof))


# ============================================================
# (P) contiguous-block permutation of TIME (Spearman-style on residuals)
# ============================================================
def test_blockperm(ry, rt, epoch, L, nperm, rng):
    obs = abs(_slope(ry, rt))
    n = len(ry)
    nb = int(np.ceil(n / L))
    # build contiguous blocks of indices in calendar order
    bounds = [min(i * L, n) for i in range(nb + 1)]
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(nb) if bounds[i + 1] > bounds[i]]
    cnt = 1
    for _ in range(nperm):
        order = rng.permutation(len(blocks))
        idx = np.concatenate([blocks[k] for k in order])
        # permute ry block-wise against fixed rt
        if abs(_slope(ry[idx], rt)) >= obs - 1e-12:
            cnt += 1
    return cnt / (nperm + 1)


# ============================================================
# synthetic harness (mirror calibrate.synth)
# ============================================================
def synth(epoch_sizes, rho, beta_speed, beta_tau, rng):
    rows = []; gday = 0.0
    for e, n in enumerate(epoch_sizes):
        lvl = rng.normal(0, 0.5)
        tau = np.sort(rng.uniform(0, 20, n))
        spd = 58 - 0.03 * gday + rng.normal(0, 6, n)
        # AR1 noise
        ee = rng.standard_normal(n); x = np.empty(n); x[0] = ee[0]
        for i in range(1, n):
            x[i] = rho * x[i - 1] + np.sqrt(1 - rho * rho) * ee[i]
        weave = 1.6 + lvl + beta_speed * (spd - 58) + beta_tau * tau + 0.5 * x
        pi = int(e % 2)
        for i in range(n):
            rows.append(dict(weave=weave[i], tau=tau[i], speed=spd[i], pi_bin=pi, epoch=e, gday=gday))
            gday += 0.7
    return pd.DataFrame(rows)


def _one(args):
    method, epoch_sizes, rho, bs, bt, L, nboot, seed, tcol, epoch_fe = args
    rng = np.random.default_rng(seed)
    df = synth(epoch_sizes, rho, bs, bt, rng)
    # calendar order = row order (gday increasing). control speed + config; TIME var = tcol
    # tcol='tau' tests time-since-reset (the driver hypothesis); 'gday' tests calendar trend.
    ry, rt, epoch, n = prep(df, 'weave', tcol, ['speed', 'pi_bin'], epoch_fe=epoch_fe)
    if np.std(rt) < 1e-9:
        return np.nan
    if method == 'mbb':
        return test_mbb(ry, rt, epoch, L, nboot, rng, stationary=False)
    if method == 'mbb_ea':
        return test_mbb(ry, rt, epoch, L, nboot, rng, epoch_aware=True)
    if method == 'sbb':
        return test_mbb(ry, rt, epoch, L, nboot, rng, stationary=True)
    if method == 'cluster':
        return test_cluster(ry, rt, epoch)
    if method == 'ar1':
        return test_ar1(ry, rt, epoch)
    if method == 'blockperm':
        return test_blockperm(ry, rt, epoch, L, nboot, rng)
    return np.nan


def battery(epoch_sizes, reps, workers, L, nboot, methods, tcol='tau', epoch_fe=False):
    workers = workers or (os.cpu_count() or 4)

    def rate(method, rho, bs, bt):
        ss = np.random.SeedSequence(hash((method, rho, bs, bt, tcol, epoch_fe)) & 0xffffffff).spawn(reps)
        args = [(method, epoch_sizes, rho, bs, bt, L, nboot, int(s.generate_state(1)[0]), tcol, epoch_fe) for s in ss]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            ps = [p for p in ex.map(_one, args) if np.isfinite(p)]
        return float(np.mean([p < 0.05 for p in ps])) if ps else np.nan, len(ps)

    print(f"epoch sizes: {epoch_sizes} sum={sum(epoch_sizes)} reps={reps} L={L} nboot={nboot} tcol={tcol} epoch_fe={epoch_fe}")
    for method in methods:
        print(f"\n==== METHOD = {method} ====")
        print("  -- FPR (beta_tau=0) target ~0.05 --")
        for rho in (0.0, 0.3, 0.6):
            r, nn = rate(method, rho, -0.06, 0.0)
            print(f"    H0 AR({rho}): FPR={r:.3f} (N={nn})")
        print("  -- POWER at AR(0.3) --")
        for bt in (0.02, 0.04):
            r, nn = rate(method, 0.3, -0.06, bt)
            print(f"    beta_tau={bt}: power={r:.3f} (N={nn})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=400)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--L', type=int, default=4, help='block length for MBB/SBB/blockperm')
    ap.add_argument('--nboot', type=int, default=300)
    ap.add_argument('--sizes', default='')
    ap.add_argument('--methods', default='mbb,sbb,cluster,ar1,blockperm')
    ap.add_argument('--tcol', default='tau', help='tau (time-since-reset) or gday (calendar)')
    ap.add_argument('--epoch_fe', action='store_true', help='within-epoch demean (remove per-epoch level nuisance)')
    a = ap.parse_args()
    if not a.sizes:
        sys.exit("--sizes is required (comma-separated per-epoch drive counts), e.g. --sizes 8,12,6. "
                 "(The old fork-era fallback derived sizes from longitudinal_weave/analyze.py, which was retired.)")
    sizes = [int(x) for x in a.sizes.split(',')]
    battery(sizes, a.reps, a.workers, a.L, a.nboot, a.methods.split(','), a.tcol, a.epoch_fe)


if __name__ == '__main__':
    main()
