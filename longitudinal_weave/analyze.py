#!/usr/bin/env python3
"""LONGITUDINAL trend analysis: does WEAVE rise with TIME-SINCE-(learned-param)-RESET and drop at resets,
REGARDLESS of config? Inputs drive_longitudinal.csv (one row per drive, chronological by route_counter).

DESIGN (the hypothesis the driver posed):
  H1: within a reset-epoch, weave GROWS as the learned params drift from their fresh/reset state; a reset
      (calibration recal OR paramsd re-init) RESETS the clock -> a sawtooth in weave vs calendar time.
  Discriminator: time-since-reset must predict weave BEYOND a monotonic calendar trend (a pure calendar/seasonal
      or road trend would NOT reset at param-resets). So we test tau_since_reset CONTROLLING calendar time.

CONFOUNDS handled: SPEED (dominant, within r~-0.65) always controlled; CONFIG (pi_set golden/weak, + manual era)
  as covariate so the effect is "regardless of customizations"; ROUTE/road acknowledged (model-indep yawRate/vEgo
  metric) + a within-config robustness pass.

STATS (calibrated, parallel): rank-partial association of weave with tau, residualizing speed+config. Significance
  by a WITHIN-EPOCH CIRCULAR-SHIFT permutation (cyclically shift weave within each epoch: preserves within-epoch
  serial autocorrelation, breaks only the trend phase) -> calibrated for autocorrelated/trended series, unlike a
  plain shuffle. Parallelized across all cores. FPR is Monte-Carlo verified separately (calibrate.py).

  .venv311/bin/python longitudinal_weave/analyze.py --csv longitudinal_weave/results/drive_longitudinal.csv \
       --out longitudinal_weave/results [--workers N]
RESET DETECTION is data-driven (see detect_resets) and its thresholds are reported + sensitivity-swept; finalize
after inspecting the actual sr/aoa/calPerc trajectory."""
import argparse, os, sys
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'learned_param_studies', 'code'))
import shared   # bh_fdr, _fast_pearson, _rankdata


# ---------------- reset detection ----------------
def detect_resets(df, sr_jump=0.7, aoa_jump=1.2, calperc_drop=10.0, calperc_floor=90.0):
    """Mark a drive as a RESET point (start of a new epoch) when the learned state discontinuously returns toward
    a fresh value vs the PREVIOUS drive IN CALENDAR ORDER's END state:
      * paramsd: |sr_start - prev_sr_end| > sr_jump  OR  |aoa_start - prev_aoa_end| > aoa_jump
        (a jump = paramsd re-initialized: reflash / CarParams change / our manual steerRatio edit -> re-learning)
      * calibration: calperc dropped (calperc_min < calperc_floor) OR calperc_start << prev calperc_end-calperc_drop
        (NOTE: in the current local set calPerc is 100 throughout -> this signal fires nothing here)
      * reflash: the device route_counter DECREASED vs the previous calendar drive (counter went backwards in
        time = the realdata counter was reset = a reflash/factory-clear -> learned params cleared).
    CRITICAL: ordered by WALL_DATE (the device route_counter is NOT monotonic in time -- it reset >=once).
    Returns df with is_reset(bool), epoch(int), drives_since_reset(int), days_since_reset(float). First drive
    is always an epoch start. Transparent + parameterized so sensitivity can be swept."""
    d = df.sort_values('wall_date').reset_index(drop=True).copy()
    is_reset = [True]
    for i in range(1, len(d)):
        p, c = d.iloc[i - 1], d.iloc[i]
        sr_r = np.isfinite(c['sr_start']) and np.isfinite(p['sr_end']) and abs(c['sr_start'] - p['sr_end']) > sr_jump
        aoa_r = np.isfinite(c['aoa_start']) and np.isfinite(p['aoa_end']) and abs(c['aoa_start'] - p['aoa_end']) > aoa_jump
        cal_r = (np.isfinite(c['calperc_min']) and c['calperc_min'] < calperc_floor) or \
                (np.isfinite(c['calperc_start']) and np.isfinite(p['calperc_end']) and
                 c['calperc_start'] < p['calperc_end'] - calperc_drop)
        ctr_r = np.isfinite(c['route_counter']) and np.isfinite(p['route_counter']) and c['route_counter'] < p['route_counter']
        is_reset.append(bool(sr_r or aoa_r or cal_r or ctr_r))
    d['is_reset'] = is_reset
    d['epoch'] = np.cumsum(is_reset)
    t = pd.to_datetime(d['wall_date'], unit='s', errors='coerce')
    d['drives_since_reset'] = np.nan; d['days_since_reset'] = np.nan
    for e, g in d.groupby('epoch'):
        idx = g.index; t0 = t.loc[idx[0]]
        d.loc[idx, 'drives_since_reset'] = np.arange(len(idx))
        d.loc[idx, 'days_since_reset'] = [((t.loc[i] - t0).total_seconds() / 86400.0)
                                          if (pd.notna(t.loc[i]) and pd.notna(t0)) else np.nan for i in idx]
    return d


# ---------------- core trend statistic ----------------
def _resid_on(R, cols):
    C = np.column_stack([np.ones(len(R))] + [R[c].values for c in cols]) if cols else np.ones((len(R), 1))
    Q, _ = np.linalg.qr(C)
    return lambda v: v - Q @ (Q.T @ v)


def trend_r(df, ycol, taucol, controls, eps_col='epoch'):
    """Rank-partial correlation of weave(y) with tau, residualizing controls (speed + config dummies). Pooled
    across drives. Returns (r, resid_y, tau_rank, epoch, n)."""
    cols = [ycol, taucol] + controls + [eps_col]
    d = df[cols].dropna().reset_index(drop=True)
    if len(d) < 8:
        return np.nan, None, None, None, len(d)
    R = d[[ycol, taucol] + controls].rank()
    resid = _resid_on(R, controls)
    ry = resid(R[ycol].values); rt = resid(R[taucol].values)
    if np.std(ry) == 0 or np.std(rt) == 0:
        return np.nan, None, None, None, len(d)
    return shared._fast_pearson(ry, rt), ry, rt, d[eps_col].values, len(d)


def _circshift_within_epoch(ry, epoch, rng):
    out = ry.copy()
    for e in np.unique(epoch):
        idx = np.where(epoch == e)[0]
        if len(idx) > 1:
            out[idx] = np.roll(ry[idx], int(rng.integers(1, len(idx))))   # circular shift != 0
    return out


def trend_test(df, ycol='weave_path', taucol='days_since_reset', controls=('spd_mph',),
               config=('pi_bin',), nperm=10000, workers=0, seed=1):
    """Circular-shift-within-epoch permutation test for a weave-vs-tau trend, speed+config controlled. Parallel."""
    controls = list(controls) + [c for c in config if c in df.columns]
    r0, ry, rt, epoch, n = trend_r(df, ycol, taucol, controls)
    if not np.isfinite(r0):
        return dict(r=np.nan, p=np.nan, n=n, slope=np.nan)
    workers = workers or (os.cpu_count() or 4)
    per = max(1, nperm // workers)
    seedseq = np.random.SeedSequence(seed).spawn(workers)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_run_star, [(ry, rt, epoch, ss, per, r0) for ss in seedseq]))
    cnt = 1 + sum(c for c, _ in res); tot = 1 + sum(k for _, k in res)
    # slope in weave-units per day via simple OLS on raw (not rank) for interpretability
    dd = df[[ycol, taucol] + controls].dropna()
    X = np.column_stack([np.ones(len(dd)), dd[taucol].values] + [dd[c].values for c in controls])
    beta = np.linalg.lstsq(X, dd[ycol].values, rcond=None)[0]
    return dict(r=float(r0), p=float(cnt / tot), n=int(n), slope=float(beta[1]))


def _run_star(a):
    ry, rt, epoch, ss, k, r0 = a
    rng = np.random.default_rng(ss); c = 0
    for _ in range(k):
        r = shared._fast_pearson(_circshift_within_epoch(ry, epoch, rng), rt)
        if abs(r) >= abs(r0) - 1e-12:
            c += 1
    return c, k


# ---------------- driver ----------------
def run(csv, out, workers):
    df = pd.read_csv(csv)
    df = df[df['n_elig'] >= 3].copy()                          # need >=3 eligible engaged-straight segments for a weave value
    df['pi_bin'] = (df['pi_set'] == 'golden').astype(int)
    d = detect_resets(df)
    d.to_csv(os.path.join(out, 'drive_epochs.csv'), index=False)
    n_ep = d['epoch'].nunique(); n_reset = int(d['is_reset'].sum())
    print(f"drives with weave: {len(d)} | epochs: {n_ep} | reset points: {n_reset}")
    print(d.groupby('epoch').agg(n=('drive_id', 'size'), start=('date', 'first'),
                                 span_days=('days_since_reset', 'max')).to_string())
    res = {}
    print("\n=== PRIMARY: weave_path ~ days_since_reset (speed+config controlled) ===")
    res['primary'] = trend_test(d, 'weave_path', 'days_since_reset', ('spd_mph',), ('pi_bin',), workers=workers)
    print(f"  r={res['primary']['r']:+.3f} p={res['primary']['p']:.4f} slope={res['primary']['slope']:+.4f}/day n={res['primary']['n']}")
    print("\n=== DISCRIMINATOR: does tau_since_reset survive controlling CALENDAR time? (sawtooth vs monotone) ===")
    d['cal_days'] = (pd.to_datetime(d['wall_date'], unit='s') - pd.to_datetime(d['wall_date'], unit='s').min()).dt.total_seconds() / 86400
    res['vs_calendar'] = trend_test(d, 'weave_path', 'days_since_reset', ('spd_mph', 'cal_days'), ('pi_bin',), workers=workers)
    res['calendar_only'] = trend_test(d, 'weave_path', 'cal_days', ('spd_mph',), ('pi_bin',), workers=workers)
    print(f"  tau|calendar: r={res['vs_calendar']['r']:+.3f} p={res['vs_calendar']['p']:.4f}   "
          f"calendar alone: r={res['calendar_only']['r']:+.3f} p={res['calendar_only']['p']:.4f}")
    print("\n=== ROBUSTNESS: within-config + drive-count tau ===")
    for nm, sub in [('weak-only', d[d['pi_set'] == 'weak']), ('golden-only', d[d['pi_set'] == 'golden'])]:
        if sub['epoch'].nunique() >= 1 and len(sub) >= 8:
            r = trend_test(sub, 'weave_path', 'days_since_reset', ('spd_mph',), (), workers=workers)
            print(f"  {nm:<12}: r={r['r']:+.3f} p={r['p']:.4f} n={r['n']}"); res[nm] = r
    res['drivecount_tau'] = trend_test(d, 'weave_path', 'drives_since_reset', ('spd_mph',), ('pi_bin',), workers=workers)
    print(f"  tau=drives_since_reset: r={res['drivecount_tau']['r']:+.3f} p={res['drivecount_tau']['p']:.4f}")
    pd.DataFrame(res).T.to_csv(os.path.join(out, 'trend_results.csv'))
    print(f"\nwrote {out}/trend_results.csv + drive_epochs.csv")
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); run(a.csv, a.out, a.workers)
