#!/usr/bin/env python3
"""Shared logic for the learned-parameter correlation studies: lateral-performance metric battery, per-segment
table builder, and correlation tools that respect the hard confounds (time, config, route are entangled across
~10 drives). Key tools: across-drive Spearman with bootstrap CI; PARTIAL correlation (control covariates);
WITHIN-DRIVE correlation (demean per drive -> removes all between-drive confounds incl route & config); config
stratification. Lateral metrics use the validated mask-after-filter band-RMS on engaged, straight, no-override,
no-lead, lane-quality-gated, speed-banded samples."""
import os, math
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d
from scipy import stats

FS = 50.0
WEAVE_BAND = (0.10, 0.35)
HUNT_BAND = (0.50, 1.50)
ROAD_LP_HZ = 0.035
ROAD_CURV_ABS_MAX = 0.0015
SPEED_BAND_MPH = (30.0, 80.0)
SEG_S = 45.0          # segment length (>=30s for the 0.10 Hz edge); within-drive unit
MIN_SEG_ELIG_S = 25.0
LEAD_HEADWAY_S = 1.6


def _sos(lo, hi, btype):
    return butter(3, [lo, hi], btype='band', fs=FS, output='sos') if btype == 'band' else butter(2, lo, btype='low', fs=FS, output='sos')

def _fillguard(x, maxgap_s=0.5):
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 10:
        return None
    idx = np.arange(len(x)); out = np.interp(idx, idx[ok], x[ok])
    near = np.interp(idx, idx[ok], idx[ok]); out[np.abs(idx - near) > maxgap_s * FS] = np.nan
    return out

def bandpass(x, lo, hi):
    xf = _fillguard(x)
    if xf is None:
        return np.full(len(x), np.nan)
    g = np.isfinite(xf)
    if g.sum() < 60:
        return np.full(len(x), np.nan)
    y = np.full(len(x), np.nan); y[g] = sosfiltfilt(_sos(lo, hi, 'band'), np.where(g, xf, 0.0))[g]
    return y

def lowpass(x, hz):
    xf = _fillguard(x)
    if xf is None:
        return np.full(len(x), np.nan)
    g = np.isfinite(xf); y = np.full(len(x), np.nan)
    if g.sum() < 30:
        return y
    y[g] = sosfiltfilt(_sos(hz, None, 'low'), np.where(g, xf, 0.0))[g]
    return y

def rms_masked(filt, m):
    v = filt[m]; v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan


def load(cache, did):
    f = os.path.join(cache, f'{did}.npz')
    return {k: np.load(f)[k] for k in np.load(f).files} if os.path.exists(f) else None


def eligibility(P):
    v = P['v'].astype(float); yaw = P['yaw'].astype(float)
    curv = np.divide(yaw, v, out=np.full_like(v, np.nan), where=(v > 5) & np.isfinite(yaw))
    road = lowpass(curv, ROAD_LP_HZ)
    er = lambda m, r: uniform_filter1d(m.astype(float), 2 * r + 1, mode='nearest') >= 1.0
    di = lambda m, r: uniform_filter1d(m.astype(float), 2 * r + 1, mode='nearest') > 0.0
    eng = er(P['latact'] > 0.5, int(2 * FS))
    nopress = ~di(P['press'] > 0.5, int(FS))
    noblink = ~di(P['blinker'] > 0.5, int(FS))
    nolc = P['lcs'] <= 0.001
    spd = v * 2.23694
    sg = (spd >= SPEED_BAND_MPH[0]) & (spd <= SPEED_BAND_MPH[1])
    cg = np.abs(road) <= ROAD_CURV_ABS_MAX
    valid = (P['canvalid'] > 0.5) & np.isfinite(curv)
    nolead = np.ones(len(v), bool)
    if 'lead_p' in P:
        thw = np.divide(P['lead_d'].astype(float), v, out=np.full_like(v, 1e9), where=v > 1)
        nolead = ~di((P['lead_p'].astype(float) > 0.5) & (thw < LEAD_HEADWAY_S), int(2 * FS))
    return eng & nopress & noblink & nolc & sg & cg & valid & nolead, curv, road, spd


def _lc_int_grid(P):
    """interpolate the sparse 1 Hz |integrator| telemetry onto the 50 Hz grid (slowly-varying controller state)."""
    n = len(P['t'])
    if 'lc_t' not in P or len(P['lc_t']) < 3:
        return np.full(n, np.nan)
    t = P['t'].astype(float)
    return np.interp(t, P['lc_t'].astype(float), np.abs(P['lc_int'].astype(float)), left=np.nan, right=np.nan)


def segment_table(P, meta):
    """per ~45s engaged-straight segment: lateral metrics + learned-param medians."""
    elig, curv, road, spd = eligibility(P)
    n = len(P['t']); w = int(SEG_S * FS)
    bf_curv = bandpass(curv, *WEAVE_BAND); bf_steer = bandpass(P['steer'].astype(float), *WEAVE_BAND)
    hunt = bandpass(P['steer'].astype(float), *HUNT_BAND)
    lc_int = _lc_int_grid(P)
    rows = []
    for i in range(0, n - w, w):
        sl = slice(i, i + w); m = elig[sl]
        if m.sum() / FS < MIN_SEG_ELIG_S:
            continue
        def md(k):
            x = P[k][sl][m] if k in P else np.array([np.nan]); x = x[np.isfinite(x)]
            return float(np.median(x)) if len(x) else np.nan
        rows.append(dict(
            drive_id=meta['drive_id'], pi_set=meta['pi_set'], driving_model=meta['driving_model'],
            wall_date=meta['wall_date'], order=meta['order'], seg_t=float(P['t'][i]),
            # --- LATERAL PERFORMANCE battery ---
            weave_path=rms_masked(bf_curv, _idx(sl, m, n)) * 1e4,
            weave_steer=rms_masked(bf_steer, _idx(sl, m, n)),
            hunt_steer=rms_masked(hunt, _idx(sl, m, n)),
            centering_abs=float(np.nanmean(np.abs(P['pos'][sl][m]))) if np.isfinite(P['pos'][sl][m]).any() else np.nan,
            centering_signed=float(np.nanmean(P['pos'][sl][m])) if np.isfinite(P['pos'][sl][m]).any() else np.nan,
            spd_mph=float(np.nanmedian(spd[sl][m])), elig_s=float(m.sum() / FS),
            # --- LEARNED / CUMULATIVE params (segment median) ---
            steerRatio=md('sr'), angleOffsetAvg=md('aoa'), angleOffset=md('ao'), stiffness=md('stf'),
            cal_yaw=md('cal_yaw'), cal_pitch=md('cal_pitch'), cal_roll=md('cal_roll'),
            int_abs=float(np.nanmean(lc_int[sl][m])) if np.isfinite(lc_int[sl][m]).any() else np.nan,
            int_railed_frac=float(np.nanmean(lc_int[sl][m] >= 0.299)) if np.isfinite(lc_int[sl][m]).any() else np.nan,
        ))
    return rows


def _idx(sl, m, n):
    full = np.zeros(n, bool); full[sl] = m
    return full


# ---------- significance helpers (calibrated for small-n / autocorrelated within-drive data) ----------
# WHY these exist (verified 2026-06-13, 4-lens QA): the previous tools were anti-conservative ->
#   * spearman_ci's percentile bootstrap CI excluded 0 at ~12% FPR at n=5 (nominal 5%) -> false "SIG".
#   * within_drive_spearman's cluster-bootstrap CI ran at 9-13% FPR (~2x too liberal).
# FIX: significance ALWAYS comes from an EXACT/permutation test (calibrated), never from a CI excluding 0.
# CIs are kept for description only. All families are corrected with BH-FDR (bh_fdr) at the test level.
def _rankdata(a):
    return stats.rankdata(a)


def _fast_pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def spearman_perm_p(x, y, nperm=10000, seed=7, exact_max_n=7):
    """Two-sided permutation p-value for Spearman r. EXACT (enumerate all n! permutations) when
    n<=exact_max_n (n=5 -> 120 perms), else Monte-Carlo. Use this in place of the asymptotic/bootstrap
    p at small n. Operates on ranks (permuting ranks == permuting values for Spearman)."""
    d = pd.DataFrame({'x': x, 'y': y}).dropna()
    n = len(d)
    if n < 4 or d['x'].nunique() < 3 or d['y'].nunique() < 3:
        return np.nan
    rx = _rankdata(d['x'].values); ry = _rankdata(d['y'].values)
    r0 = abs(_fast_pearson(rx, ry))
    if n <= exact_max_n:
        from itertools import permutations
        rs = [abs(_fast_pearson(rx, ry[list(p)])) for p in permutations(range(n))]
        rs = np.array(rs)
        return float(np.mean(rs >= r0 - 1e-12))
    rg = np.random.default_rng(seed); cnt = 1
    for _ in range(nperm):
        if abs(_fast_pearson(rx, rg.permutation(ry))) >= r0 - 1e-12:
            cnt += 1
    return float(cnt / (nperm + 1))


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values (NaN-safe). Apply over the whole correlation battery of a test."""
    p = np.asarray(pvals, float); q = np.full(len(p), np.nan)
    ok = np.isfinite(p); m = int(ok.sum())
    if m == 0:
        return q
    idx = np.where(ok)[0]; order = idx[np.argsort(p[idx])]
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k); q[i] = prev
    return q


# ---------- correlation tools ----------
def spearman_ci(x, y, nboot=5000, seed=11):
    """Spearman r + EXACT/permutation two-sided p (the significance gate, d['p']) plus a bootstrap CI
    (DESCRIPTIVE ONLY -- percentile bootstrap is anti-conservative at small n, so never gate on CI-excludes-0)."""
    d = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(d) < 5:
        return dict(r=np.nan, p=np.nan, p_asym=np.nan, lo=np.nan, hi=np.nan, n=len(d))
    r, p_asym = stats.spearmanr(d['x'], d['y'])
    p_perm = spearman_perm_p(d['x'].values, d['y'].values, seed=seed)
    rg = np.random.default_rng(seed); bs = []
    for _ in range(nboot):
        s = d.sample(len(d), replace=True, random_state=rg.integers(1 << 30))
        if s['x'].nunique() > 2 and s['y'].nunique() > 2:
            bs.append(stats.spearmanr(s['x'], s['y'])[0])
    lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return dict(r=float(r), p=float(p_perm), p_asym=float(p_asym), lo=float(lo), hi=float(hi), n=int(len(d)))


def partial_spearman(df, xcol, ycol, controls, nperm=10000, seed=9):
    """Rank-partial Spearman of x,y given controls. Significance via a FREEDMAN-LANE permutation: permute the
    y rank-residual and RE-RESIDUALIZE on the control basis each permutation (naive residual-permutation is
    anti-conservative at small n / many controls -- ~9% FPR at n=10,2-ctrl, ~13% at 3-ctrl). NEVER feed a
    near-constant or noise control (e.g. the clustered wall-clock 'order' -- 12/13 drives shared a 17s
    timestamp); partialling on noise inflates r and burns a dof. NOTE: at the suite's n=10 this view is still
    weak -- read it as descriptive; the headline rests on the within-drive primary, not the partial."""
    cols = [xcol, ycol] + list(controls)
    d = df[cols].dropna()
    if len(d) < len(controls) + 5:
        return dict(r=np.nan, p=np.nan, n=len(d), n_ctrl=len(controls))
    R = d.rank()
    C = np.column_stack([np.ones(len(d))] + [R[c].values for c in controls])
    Q, _ = np.linalg.qr(C)
    def resid_of(vals):
        return vals - Q @ (Q.T @ vals)
    rx, ry = resid_of(R[xcol].values), resid_of(R[ycol].values)
    # near-singular guard: if x or y is (almost) fully explained by the controls, the residual is float noise.
    if np.std(rx) <= 1e-9 * max(1.0, np.ptp(R[xcol].values)) or np.std(ry) <= 1e-9 * max(1.0, np.ptp(R[ycol].values)):
        return dict(r=np.nan, p=np.nan, n=int(len(d)), n_ctrl=len(controls))
    r = _fast_pearson(rx, ry)
    rg = np.random.default_rng(seed); r0 = abs(r); cnt = 1
    for _ in range(nperm):
        yp = rg.permutation(ry); yd = yp - Q @ (Q.T @ yp)     # Freedman-Lane re-residualization
        if abs(_fast_pearson(rx, yd)) >= r0 - 1e-12:
            cnt += 1
    return dict(r=float(r), p=float(cnt / (nperm + 1)), n=int(len(d)), n_ctrl=len(controls))


def within_drive_spearman(df, xcol, ycol, ctrl='spd_mph', detrend=True, nperm=4000, seed=13, nboot=2000):
    """Within-drive association of x and y, de-confounded. Per drive:
      * DETREND (default ON): remove a per-drive LINEAR trend (vs segment index) from x and y, so a shared
        slow drift over a drive is NOT read as correlation (this is what spuriously inflated cal_pitch~weave).
      * ctrl (default 'spd_mph'): residualize x and y on the covariate WITHIN each drive -- SPEED is the
        dominant weave confound (within-drive speed~weave ~ -0.65). Pass ctrl=None to disable.
        CAVEAT: this treats speed as a CONFOUND, not a MEDIATOR. If a param affected weave PARTLY via speed
        (param -> driver slows -> less weave), speed-control would partially block that real effect (under-claim
        risk). The tests deliberately also report the ctrl=None / within_raw view as the hedge against this.
    Significance = WITHIN-DRIVE FREEDMAN-LANE PERMUTATION. NOTE (QA round 1, 2026-06-14): naively permuting
    the OLS residuals is anti-conservative (~9% FPR) because residuals are linearly constrained (orthogonal to
    the [1,trend,ctrl] basis, sum to zero) so permuting them breaks exchangeability and fattens the null tails.
    Freedman-Lane fixes it: per drive permute the REDUCED-model y-residual and RE-RESIDUALIZE it on the basis
    each permutation (the re-projection restores the lost-dof, recalibrating to ~5%). x-residual stays fixed.
    The cluster-bootstrap CI is DESCRIPTIVE ONLY (it ran ~2x too liberal -- never gate on it).
    CALIBRATION (verified, qa_calibration.py): ~0.04-0.07 FPR for params with ~linear within-drive drift. A small
    FAMILY with strong NONLINEAR within-drive drift sits on a mild ~0.08-0.13 anti-conservative floor a linear
    detrend can't remove (angleOffsetAvg most, quadGain~0.35; also steerRatio~weave_steer and stiffness~hunt_steer;
    a quadratic detrend does NOT fix it and steals dof on 4-13-seg drives). Treat those params' within-drive p as a
    FLOOR and corroborate with the across view. IMMATERIAL to the verdict by DISJOINTNESS: every floor pair has a
    real within-drive p>=0.2 (far from significance) and every signal-bearing pair has a well-calibrated null
    (FPR ~0.02-0.08), so no pair has BOTH inflation and a near-significant result; a 2x-inflation stress model still
    yields 0 weave/hunt levers."""
    need = ['drive_id', xcol, ycol] + ([ctrl] if ctrl else [])
    d = df[need].dropna().copy()
    per_drive = []   # (Q, xd, yres): Q=orthonormal basis of [1,trend,ctrl]; xd,yres = residuals on Q
    for _, g in d.groupby('drive_id'):
        if len(g) < 4:
            continue
        n = len(g); cols = [np.ones(n)]
        if detrend:
            t = np.arange(n, dtype=float); cols.append(t - t.mean())
        if ctrl:
            cv = g[ctrl].values.astype(float); cols.append(cv - cv.mean())
        Z = np.column_stack(cols)
        Q, _ = np.linalg.qr(Z)                       # orthonormal basis; resid(v) = v - Q (Qᵀ v)
        def resid(v):
            return v - Q @ (Q.T @ v)
        xv = g[xcol].values.astype(float); yv = g[ycol].values.astype(float)
        xd = resid(xv); yres = resid(yv)
        # scale-relative degeneracy guard: a (near-)constant column leaves a QR residual of ~1e-16 float noise,
        # so exact ==0 would NOT fire and we'd rank pure noise. Skip when residual std is negligible vs the raw range.
        if np.std(xd) <= 1e-9 * max(1.0, np.ptp(xv)) or np.std(yres) <= 1e-9 * max(1.0, np.ptp(yv)):
            continue
        per_drive.append((Q, xd, yres))
    if len(per_drive) < 2:
        return dict(r=np.nan, p=np.nan, lo=np.nan, hi=np.nan, n_drives=len(per_drive), n=0)
    def pooled(perm_rng=None):
        xs, ys = [], []
        for Q, xd, yres in per_drive:
            if perm_rng is not None:                 # Freedman-Lane: permute y-residual, RE-residualize on Q
                yp = perm_rng.permutation(yres); yd = yp - Q @ (Q.T @ yp)
            else:
                yd = yres
            xs.append(xd); ys.append(yd)
        X = np.concatenate(xs); Y = np.concatenate(ys)
        if np.std(X) == 0 or np.std(Y) == 0:
            return np.nan
        return stats.spearmanr(X, Y)[0]
    r = pooled(); npts = int(sum(len(p[1]) for p in per_drive))
    rg = np.random.default_rng(seed); r0 = abs(r); cnt = 1
    for _ in range(nperm):
        rr = pooled(perm_rng=rg)
        if np.isfinite(rr) and abs(rr) >= r0 - 1e-12:
            cnt += 1
    p_perm = cnt / (nperm + 1)
    rgb = np.random.default_rng(seed + 1); bs = []
    for _ in range(nboot):
        idx = rgb.integers(0, len(per_drive), len(per_drive))
        xs = np.concatenate([per_drive[i][1] for i in idx]); ys = np.concatenate([per_drive[i][2] for i in idx])
        if np.std(xs) > 0 and np.std(ys) > 0:
            bs.append(stats.spearmanr(xs, ys)[0])
    lo, hi = (np.nanpercentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
    return dict(r=float(r) if np.isfinite(r) else np.nan, p=float(p_perm), lo=float(lo), hi=float(hi),
                n_drives=len(per_drive), n=int(npts))


def fmt(d, q=None):
    """Format a result. Significance ALWAYS from the permutation/exact p (d['p']) -- NEVER from a CI excluding 0.
    Pass q (BH-FDR q-value over the test's battery) to show the multiple-comparison-corrected verdict."""
    r = d.get('r', float('nan')); p = d.get('p', float('nan'))
    tag = ''
    if np.isfinite(p):
        if q is not None and np.isfinite(q):
            tag = ' robust(q<.10)' if q < 0.10 else (' p<.05*' if p < 0.05 else ' ns')
        else:
            tag = ' p<.05*' if p < 0.05 else ' ns'
    ci = f" CI[{d['lo']:+.2f},{d['hi']:+.2f}]" if np.isfinite(d.get('lo', np.nan)) else ''
    qs = f" q={q:.2f}" if (q is not None and np.isfinite(q)) else ''
    return f"r={r:+.2f} p={p:.3f}{qs}{ci}{tag} (n={d.get('n')},drv={d.get('n_drives', '-')})"
