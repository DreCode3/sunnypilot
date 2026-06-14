#!/usr/bin/env python3
"""Controlled-test weave analysis (ANALYSIS_PLAN.md) — revision incorporating the adversarial review.

KEY DESIGN (review fixes folded in):
  * SPACE-ANCHORED matching: every eligible 50Hz sample is snapped to a fixed global GPS grid (GPS_CELL_M);
    the per-(pass x cell) band-RMS is computed directly over that cell's eligible samples — same physical
    stretch always lands in the same stratum across passes (M1).
  * MASK-AFTER-FILTER: band-pass the CONTINUOUS pass signal once (tiny-gap-guarded), then RMS over the
    eligibility-masked samples only — never interpolate across gated-out road/override (M3).
  * PASS is the unit: per-(pass,cell) -> median over passes within a stratum; cluster bootstrap over passes;
    WITHIN-STRATUM exchangeable label permutation with (1+c)/(1+n) p-estimator (M4).
  * MATCHED %-denominator over the same shared strata (M6); CIRCULAR heading, drop undefined-heading (M7);
    AUDIT HALTS on config/build/integrator/interleave/speed/calibration failure (M5); lead gate (M9).
  * PRIMARY = P2 path curvature; P1 steering is a speed-RESIDUALIZED corroborator (1/v^2 confound) (S1);
    permutation is the binding gate, CI descriptive+direction-explicit (S5); §9 battery enforced on P1&P2 (S4);
    power/variance printed to finalize n (S2).

Run (after extract.py): <python w/ numpy scipy pandas> analyze.py --cache <dir> --out <dir>
"""
import argparse, json, math, os, sys, itertools
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, os.path.dirname(__file__))
import config as K
M_PER_DEG_LAT = 111320.0


# ---------------- signal helpers ----------------
def _sos(lo, hi, fs, btype):
    return butter(3, [lo, hi], btype='band', fs=fs, output='sos') if btype == 'band' \
        else butter(2, lo, btype=btype, fs=fs, output='sos')

def fill_guarded(x, max_gap_s=K.MAX_INTERP_GAP_S, fs=K.FS):
    """interpolate ONLY across gaps <= max_gap_s; longer gaps stay NaN (no fabricated data)."""
    x = np.asarray(x, float); ok = np.isfinite(x)
    if ok.sum() < 10:
        return None
    idx = np.arange(len(x)); out = np.interp(idx, idx[ok], x[ok])
    # re-NaN samples whose nearest real sample is farther than max_gap
    near = np.interp(idx, idx[ok], idx[ok])
    out[np.abs(idx - near) > max_gap_s * fs] = np.nan
    return out

def bandpass_continuous(x, lo, hi, fs=K.FS):
    """filter the continuous series ONCE; returns filtered array with NaN where input had long gaps."""
    xf = fill_guarded(x)
    if xf is None:
        return np.full(len(x), np.nan)
    good = np.isfinite(xf)
    if good.sum() < 60:
        return np.full(len(x), np.nan)
    y = np.full(len(x), np.nan)
    xs = np.where(good, xf, 0.0)
    y[good] = sosfiltfilt(_sos(lo, hi, fs, 'band'), xs)[good]
    return y

def lowpass_continuous(x, hz, fs=K.FS):
    xf = fill_guarded(x)
    if xf is None:
        return np.full(len(x), np.nan)
    good = np.isfinite(xf); y = np.full(len(x), np.nan)
    if good.sum() < 30:
        return y
    y[good] = sosfiltfilt(_sos(hz, None, fs, 'low'), np.where(good, xf, 0.0))[good]
    return y

def rms_masked(filtered, mask):
    v = filtered[mask]; v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else np.nan

def circ_mean_deg(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if len(a) == 0:
        return np.nan
    return float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(a))), np.mean(np.cos(np.radians(a))))) % 360)

def welch_peak(x, lo, hi, fs=K.FS):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < fs * 12:
        return np.nan
    f, p = welch(x - x.mean(), fs=fs, nperseg=int(min(2048, len(x))))
    b = (f >= lo) & (f <= hi)
    return float(f[b][np.argmax(p[b])]) if (b.any() and np.nanmax(p[b]) > 0) else np.nan

def agg_stat(v, stat):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan
    if stat == 'trim20':
        v = np.sort(v); k = int(len(v) * 0.2)
        return float(np.mean(v[k:len(v) - k])) if len(v) - 2 * k > 0 else float(np.median(v))
    return float(np.median(v))


# ---------------- load ----------------
def load(cache):
    summ = pd.read_csv(os.path.join(cache, 'pass_summary.csv'))
    passes = {r['pass_id']: {k: np.load(os.path.join(cache, f"{r['pass_id']}.npz"))[k]
                             for k in np.load(os.path.join(cache, f"{r['pass_id']}.npz")).files}
              for _, r in summ.iterrows() if os.path.exists(os.path.join(cache, f"{r['pass_id']}.npz"))}
    return summ, passes


# ---------------- eligibility (+ continuous derived signals) ----------------
def eligibility(P, S):
    v = P['v'].astype(float)
    ys = S['yaw_source']
    yaw = P['yaw_cal'].astype(float) if ys == 'calibrated' else P['yaw_cs'].astype(float)
    # gate whole-sample on a single consistent yaw source (no mid-series mixing) — drop where chosen source NaN
    actual_curv = np.divide(yaw, v, out=np.full_like(v, np.nan), where=(v > 5.0) & np.isfinite(yaw))
    road_lp = lowpass_continuous(actual_curv, K.ROAD_LP_HZ)

    def er(mask, r):
        return uniform_filter1d(mask.astype(float), 2 * r + 1, mode='nearest') >= 1.0

    def di(mask, r):
        return uniform_filter1d(mask.astype(float), 2 * r + 1, mode='nearest') > 0.0

    eng = er(P['latact'] > 0.5, int(K.ENGAGE_ERODE_S * K.FS))
    nopress = ~di(P['press'] > 0.5, int(K.OVERRIDE_BUFFER_S * K.FS))
    noblink = ~di(P['blinker'] > 0.5, int(K.BLINKER_BUFFER_S * K.FS))
    nolc = P['lcs'] <= 0.001
    spd = v * 2.23694
    spd_gate = (spd >= S['speed_band'][0]) & (spd <= S['speed_band'][1])
    curv_gate = np.abs(road_lp) <= S['road_curv_abs_max']
    lane_ok = (P['lane_p'] > K.LANE_PROB_MIN) & (P['lane_w'] >= K.LANE_WIDTH_RANGE[0]) & (P['lane_w'] <= K.LANE_WIDTH_RANGE[1])
    valid = (P['canvalid'] > 0.5) & np.isfinite(actual_curv)
    # lead gate (M9) if the channel exists
    nolead = np.ones(len(v), bool)
    if 'lead_d_rel' in P and 'lead_prob' in P:
        thw = np.divide(P['lead_d_rel'].astype(float), v, out=np.full_like(v, 1e9), where=v > 1.0)
        lead = (P['lead_prob'].astype(float) > 0.5) & (thw < K.LEAD_HEADWAY_S)
        nolead = ~di(lead, int(K.LEAD_BUFFER_S * K.FS))
    elig = eng & nopress & noblink & nolc & spd_gate & curv_gate & lane_ok & valid & nolead
    return elig, actual_curv, road_lp, spd


# ---------------- per-(pass x cell) metrics ----------------
def cell_metrics(pid, P, meta, S):
    elig, curv, road_lp, spd = eligibility(P, S)
    lo, hi = S['band']; v = P['v'].astype(float)
    bf_steer = bandpass_continuous(P['steer'].astype(float), lo, hi)
    bf_curv = bandpass_continuous(curv, lo, hi)
    bf_alat = bf_curv * (v ** 2)
    bf_cmd = bandpass_continuous(P['cmd_curv'].astype(float), lo, hi)
    bf_des = bandpass_continuous(P['des_curv'].astype(float), lo, hi)
    bf_modely = bandpass_continuous(P['model_y20'].astype(float), lo, hi)
    bf_rate = bandpass_continuous(P['steer_rate'].astype(float), lo, hi)
    hunt_steer = bandpass_continuous(P['steer'].astype(float), *K.HUNT_BAND)
    # global GPS grid
    lat = P['lat']; lon = P['lon']; la0 = np.nanmedian(lat)
    gx = np.floor(lon * M_PER_DEG_LAT * math.cos(math.radians(la0)) / S['gps_cell_m'])
    gy = np.floor(lat * M_PER_DEG_LAT / S['gps_cell_m'])
    cellid = gx * 1e7 + gy
    rows = []
    eidx = np.where(elig & np.isfinite(cellid))[0]
    if len(eidx) == 0:
        return rows
    for cell in np.unique(cellid[eidx]):
        m = elig & (cellid == cell) & np.isfinite(cellid)
        if m.sum() / K.FS < K.MIN_CELL_ELIGIBLE_S:
            continue
        span = slice(np.where(m)[0][0], np.where(m)[0][-1] + 1)
        pf_path = welch_peak(np.where(elig[span], bf_curv[span], np.nan), lo, hi)
        latacc_rms = rms_masked(bf_alat, m)
        disp_cm = (100 * 2 * math.sqrt(2) * latacc_rms / ((2 * math.pi * pf_path) ** 2)) if (np.isfinite(pf_path) and pf_path > 0 and np.isfinite(latacc_rms)) else np.nan
        # episode: rolling 8s RMS of band-passed steer over eligible cell samples
        ew = int(K.EPISODE_WIN_S * K.FS); s2 = np.where(m, np.nan_to_num(bf_steer) ** 2, np.nan)
        roll = np.sqrt(uniform_filter1d(np.nan_to_num(s2), ew, mode='nearest'))
        epi = float(np.mean(roll[m] > K.EPISODE_STEER_RMS_DEG)) if m.any() else np.nan
        p2 = rms_masked(bf_curv, m) * 1e4
        rows.append(dict(
            pass_id=pid, driving_model=meta['driving_model'], pi_set=meta['pi_set'],
            config=f"{meta['driving_model']}/{meta['pi_set']}", direction=str(meta.get('direction', '')),
            cell=float(cell), elig_s=float(m.sum() / K.FS),
            spd_mph=float(np.nanmedian(spd[m])), v_mps=float(np.nanmedian(v[m])),
            road_curv=float(np.nanmedian(road_lp[m])), heading=circ_mean_deg(P['heading'][m]),
            cal_yaw_deg=float(np.degrees(np.nanmedian(P['cal_yaw'][m]))) if np.isfinite(P['cal_yaw'][m]).any() else np.nan,
            # PRIMARY + corroborators
            P2_curv_rms_1e4=p2, P1_steer_rms_deg=rms_masked(bf_steer, m),
            S1_disp_pp_cm=disp_cm, S2_latacc_rms=latacc_rms, S3_episode_frac=epi,
            S4_steer_per_path=(rms_masked(bf_steer, m) / p2) if (p2 and np.isfinite(p2) and p2 > 0) else np.nan,
            S5_peak_hz=pf_path, S6_rate_rms=rms_masked(bf_rate, m),
            S7_offset_mean=float(np.nanmean(P['lane_c_y20'][m])),
            hunt_steer_rms=rms_masked(hunt_steer, m),
            # signal chain (weave-band RMS at each stage)
            chain_modely_m=rms_masked(bf_modely, m), chain_des_1e4=rms_masked(bf_des, m) * 1e4,
            chain_cmd_1e4=rms_masked(bf_cmd, m) * 1e4, chain_actual_1e4=p2, chain_steer_deg=rms_masked(bf_steer, m),
        ))
    return rows


def add_strata(df, S):
    out = df.copy()
    out['speed_bin'] = np.floor(out['spd_mph'] / S['speed_bin_mph'])
    out['curv_bin'] = np.floor(out['road_curv'] / K.CURV_BIN)
    out['hoct'] = np.where(out['heading'].notna(), np.floor(((out['heading'] + 22.5) % 360) / 45), np.nan)
    out = out[out['hoct'].notna()].copy()                       # drop undefined-heading cells (M7)
    dircol = out['direction'] if S.get('direction') == 'per' else ''
    out['stratum'] = (out['cell'].astype(str) + '|' + out['hoct'].astype(str) + '|' +
                      out['speed_bin'].astype(str) + '|' + out['curv_bin'].astype(str) +
                      ('|' + out['direction'].astype(str) if S.get('direction') == 'per' else ''))
    return out


# ---------------- matched comparison (pass-level) ----------------
def _pass_cell_value(g, metric, stat):
    """one scalar per pass in a (stratum,config) group = median over that pass's cells."""
    return g.groupby('pass_id')[metric].median()

def matched_agg(df, metric, ca, cb, stat):
    diffs, weights, signs, base_vals, base_w = [], [], [], [], []
    for _, g in df.groupby('stratum'):
        pa = _pass_cell_value(g[g['config'] == ca], metric, stat)
        pb = _pass_cell_value(g[g['config'] == cb], metric, stat)
        pa = pa[np.isfinite(pa)]; pb = pb[np.isfinite(pb)]
        if len(pa) == 0 or len(pb) == 0:
            continue
        va = agg_stat(pa.values, stat); vb = agg_stat(pb.values, stat)
        w = min(len(pa), len(pb))                      # weight by # distinct PASSES (S3)
        diffs.append(vb - va); weights.append(w); signs.append(np.sign(vb - va))
        base_vals.append(va); base_w.append(w)
    if not diffs:
        return np.nan, np.nan, [], 0
    diffs = np.array(diffs); weights = np.array(weights, float)
    agg = float(np.average(diffs, weights=weights))
    base = float(np.average(base_vals, weights=base_w))   # matched denominator (M6)
    return agg, base, signs, len(diffs)


def compare(df, metric, ca, cb, S, nboot=K.N_BOOT, nperm=4000):
    sub = df[df['config'].isin([ca, cb])].copy()
    obs, base, signs, nstr = matched_agg(sub, metric, ca, cb, S['stat'])
    pa = list(sub[sub['config'] == ca]['pass_id'].unique()); pb = list(sub[sub['config'] == cb]['pass_id'].unique())
    r = dict(metric=metric, comparison=f'{cb}_minus_{ca}', a=ca, b=cb, n_passes_a=len(pa), n_passes_b=len(pb),
             n_strata=nstr, effect=obs, base=base,
             effect_pct=(100 * obs / base) if (base and np.isfinite(base) and base != 0) else np.nan,
             sign_consistency_neg=float(np.mean(np.asarray(signs) < 0)) if signs else np.nan,
             sign_consistency_pos=float(np.mean(np.asarray(signs) > 0)) if signs else np.nan)
    rg = np.random.default_rng(K.RNG_SEED)
    if len(pa) >= 2 and len(pb) >= 2 and nstr >= 1:
        by = {p: sub[sub['pass_id'] == p] for p in pa + pb}
        boots = []
        for _ in range(nboot):
            sa = rg.choice(pa, len(pa), True); sb = rg.choice(pb, len(pb), True)
            # relabel duplicated passes uniquely so a doubled pass actually reweights
            parts = []
            for j, p in enumerate(list(sa) + list(sb)):
                d = by[p].copy(); d['pass_id'] = f'{p}#{j}'; parts.append(d)
            a, _, _, _ = matched_agg(pd.concat(parts, ignore_index=True), metric, ca, cb, S['stat'])
            if np.isfinite(a):
                boots.append(a)
        if boots:
            r['ci_lo'], r['ci_hi'] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        # WITHIN-STRATUM exchangeable permutation (M4): permute pass->config labels inside each stratum
        passes_cfg = sub.drop_duplicates('pass_id').set_index('pass_id')['config'].to_dict()
        perms = []
        for _ in range(nperm):
            relab = {}
            for _, g in sub.groupby('stratum'):
                ps = list(g['pass_id'].unique()); cur = [passes_cfg[p] for p in ps]
                rg.shuffle(cur)
                relab.update(dict(zip(ps, cur)))
            pdf = sub.copy(); pdf['config'] = pdf['pass_id'].map(relab)
            a, _, _, _ = matched_agg(pdf, metric, ca, cb, S['stat'])
            if np.isfinite(a):
                perms.append(a)
        if perms:
            r['perm_p'] = float((1 + np.sum(np.abs(perms) >= abs(obs) - 1e-12)) / (1 + len(perms)))  # (1+c)/(1+n)
    return r


# ---------------- power (S2) ----------------
def power_report(df, metric):
    out = {}
    pp = df.groupby(['config', 'pass_id'])[metric].median().reset_index()
    cvs = []
    for cfg, g in pp.groupby('config'):
        vals = g[metric].dropna().values
        out[cfg] = dict(n_pass=int(len(vals)), mean=float(np.mean(vals)) if len(vals) else np.nan,
                        sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan)
        if len(vals) > 1 and np.mean(vals) != 0:
            cvs.append(np.std(vals, ddof=1) / np.mean(vals))
    cv = float(np.nanmean(cvs)) if cvs else np.nan
    n80 = None
    if cv and cv > 0:
        d = abs(K.WIN_EFFECT_PCT / 100) / cv
        n80 = math.ceil(2 * (2.8 / d) ** 2) if d > 0 else None
    out['_pooled_cv'] = cv; out['_n_per_arm_for_80pct_power_at_threshold'] = n80
    return out


# ---------------- decision (PRE-REGISTERED §11) ----------------
def _sig_win(r, sign):
    if r is None or not np.isfinite(r.get('effect_pct', np.nan)):
        return False
    eff = r['effect_pct']
    thr = (eff <= K.WIN_EFFECT_PCT) if sign < 0 else (eff >= -K.WIN_EFFECT_PCT)
    ci_dir = (r.get('ci_hi', 1) < 0) if sign < 0 else (r.get('ci_lo', -1) > 0)  # direction-explicit (S5)
    pval = r.get('perm_p', 1.0) < K.WIN_PERM_P                                   # permutation is binding
    sc = (r.get('sign_consistency_neg', 0) if sign < 0 else r.get('sign_consistency_pos', 0)) >= K.WIN_SIGN_CONSISTENCY
    return thr and ci_dir and pval and sc

def decide(primary, corrob, battery_pass, n_passes_a, n_passes_b, n_strata_primary):
    if not (n_passes_a >= K.MIN_PASSES_FOR_CLAIM and n_passes_b >= K.MIN_PASSES_FOR_CLAIM and n_strata_primary >= K.MIN_MATCHED_STRATA):
        return 'INSUFFICIENT_DATA', f'need >= {K.MIN_PASSES_FOR_CLAIM} passes/config and >= {K.MIN_MATCHED_STRATA} strata (perm floor + power; §11/§12)'
    for sign, label in [(-1, 'WIN_REDUCES_WEAVE'), (+1, 'WORSENS_WEAVE')]:
        cor_ok = any(_sig_win(c, sign) for c in corrob if c is not None)   # corroborator must itself be significant (S6)
        if _sig_win(primary, sign) and cor_ok and battery_pass:
            return label, 'primary (P2) significant + a significant corroborator + survived robustness battery'
    return 'NO_DIFFERENCE_OR_INCONCLUSIVE', 'did not meet the locked criteria (effect/CI-direction/perm/sign-consistency/battery)'


# ---------------- execution audit (HALTS) ----------------
def audit(summ, passes, df):
    a = {}
    a['passes'] = int(len(summ))
    if 'pi_set_recovered' in summ:
        a['n_per_config'] = {f'{m}/{p}': int(n) for (m, p), n in summ.groupby(['driving_model', 'pi_set_recovered']).size().items()}
        a['config_recovered_matches_declared'] = bool(
            (summ['pi_set_recovered'].astype(str).str.lower() == summ['pi_set_declared'].astype(str).str.lower()).all()) if 'pi_set_declared' in summ else None
        a['no_unknown_config'] = bool((summ['pi_set_recovered'].astype(str) != 'unknown').all())
    a['builds'] = sorted(set(summ['build_commit'].dropna().astype(str))) if 'build_commit' in summ else []
    a['single_build'] = len(a['builds']) <= 1
    a['integrator_reset_ok'] = bool((summ['recovered_int_start_abs'].fillna(1).astype(float) < 0.05).all()) if 'recovered_int_start_abs' in summ else None
    # interleaving: configs should not be perfectly blocked in drive order (manifest row order)
    seq = list(summ['pass_id'].index) and list(summ.get('pi_set_recovered', summ.get('pi_set_declared', pd.Series())))
    changes = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1]) if len(seq) > 1 else 0
    a['interleaved'] = bool(len(seq) > 3 and changes >= len(seq) // 2)
    # camera-calibration stability across configs (M5/should-add)
    if len(df) and 'cal_yaw_deg' in df:
        by = df.groupby('config')['cal_yaw_deg'].median()
        a['cal_yaw_by_config'] = {k: round(float(v), 3) for k, v in by.items()}
        a['cal_yaw_stable'] = bool(np.nanmax(by.values) - np.nanmin(by.values) <= K.CAL_YAW_MAX_SPREAD_DEG) if len(by) > 1 else None
    if 'spd_med_mph' in summ:
        a['speed_med_by_config'] = {f'{m}/{p}': round(float(s), 1) for (m, p), s in summ.groupby(['driving_model', 'pi_set_recovered'])['spd_med_mph'].median().items()}
    a['lead_gate_active'] = bool(len(passes) and 'lead_prob' in next(iter(passes.values())))
    # overall hard-stop conditions
    hard = [a.get('config_recovered_matches_declared'), a.get('no_unknown_config'), a.get('single_build'),
            a.get('integrator_reset_ok'), a.get('cal_yaw_stable')]
    a['AUDIT_OK'] = all(c is not False for c in hard)   # None (unknown) does not hard-fail; False does
    a['_warnings'] = [k for k, v in dict(config=a.get('config_recovered_matches_declared'), build=a.get('single_build'),
                      integrator=a.get('integrator_reset_ok'), calibration=a.get('cal_yaw_stable'),
                      interleaved=a.get('interleaved'), lead_gate=a.get('lead_gate_active')).items() if v is False or v is None]
    return a


def settings(**o):
    S = dict(band=K.WEAVE_BAND, road_curv_abs_max=K.ROAD_CURV_ABS_MAX, gps_cell_m=K.GPS_CELL_M,
             speed_bin_mph=K.SPEED_BIN_MPH, yaw_source='calibrated', stat='median',
             speed_band=K.SPEED_BAND_MPH, direction='both')
    S.update(o)
    return S

def build_cells(summ, passes, S):
    rows = []
    for _, r in summ.iterrows():
        if r['pass_id'] not in passes:
            continue
        meta = dict(driving_model=str(r.get('driving_model', '')),
                    pi_set=str(r.get('pi_set_recovered', r.get('pi_set_declared', ''))),
                    direction=str(r.get('direction', '')))
        rows += cell_metrics(r['pass_id'], passes[r['pass_id']], meta, S)
    return add_strata(pd.DataFrame(rows), S)


def battery(summ, passes, ca, cb, metric):
    """run §9 robustness; return fraction of variants keeping same-sign + significant."""
    results = []
    for knob, vals in K.ROBUSTNESS.items():
        for v in vals:
            Sx = settings(**{knob: v})
            dfx = build_cells(summ, passes, Sx)
            if cb not in set(dfx['config']) or ca not in set(dfx['config']):
                continue
            r = compare(dfx, metric, ca, cb, Sx, nboot=1200, nperm=1200)
            results.append(r)
    sig = [r for r in results if np.isfinite(r.get('effect_pct', np.nan))]
    if not sig:
        return 0.0, results
    base_sign = np.sign(np.median([r['effect_pct'] for r in sig]))
    survive = [r for r in sig if np.sign(r['effect_pct']) == base_sign and r.get('perm_p', 1) < K.WIN_PERM_P]
    return (len(survive) / len(sig)), results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    summ, passes = load(a.cache)
    S = settings(); df = build_cells(summ, passes, S)
    ex = audit(summ, passes, df)
    json.dump(ex, open(os.path.join(a.out, 'execution_audit.json'), 'w'), indent=2, default=str)
    print('EXECUTION AUDIT:', json.dumps(ex, indent=2, default=str))
    df.to_csv(os.path.join(a.out, 'cell_metrics.csv'), index=False)
    configs = sorted(df['config'].unique()); print('configs:', configs, '| cells:', len(df))

    pairs = []
    for m in set(df['driving_model']):
        if f'{m}/weak' in configs and f'{m}/golden' in configs:
            pairs.append((f'{m}/weak', f'{m}/golden'))
    for pi in set(df['pi_set']):
        if f'CD210/{pi}' in configs and f'OPM7/{pi}' in configs:
            pairs.append((f'CD210/{pi}', f'OPM7/{pi}'))

    comp_rows, decisions, power = [], {}, {}
    metrics = [K.PRIMARY_METRIC, 'P1_steer_rms_deg', 'S1_disp_pp_cm', 'S3_episode_frac', 'S4_steer_per_path']
    for ca, cb in pairs:
        for metric in metrics:
            comp_rows.append(compare(df, metric, ca, cb, S))
        cm = pd.DataFrame([c for c in comp_rows if c['comparison'] == f'{cb}_minus_{ca}'])
        prim = cm[cm['metric'] == K.PRIMARY_METRIC].to_dict('records')
        cors = [cm[cm['metric'] == x].to_dict('records') for x in ['P1_steer_rms_deg', 'S1_disp_pp_cm', 'S3_episode_frac']]
        cors = [c[0] for c in cors if c]
        if prim and (prim[0]['n_passes_a'] >= K.MIN_PASSES_FOR_CLAIM and prim[0]['n_passes_b'] >= K.MIN_PASSES_FOR_CLAIM
                     and prim[0]['n_strata'] >= K.MIN_MATCHED_STRATA and ex['AUDIT_OK']):
            bpass, _ = battery(summ, passes, ca, cb, K.PRIMARY_METRIC)
        else:
            bpass = False
        verdict = ('AUDIT_FAILED', 'execution audit failed: ' + ','.join(ex['_warnings'])) if not ex['AUDIT_OK'] else \
            decide(prim[0] if prim else None, cors, bpass, prim[0]['n_passes_a'] if prim else 0,
                   prim[0]['n_passes_b'] if prim else 0, prim[0]['n_strata'] if prim else 0)
        decisions[f'{cb}_minus_{ca}'] = dict(verdict=verdict[0], why=verdict[1], battery_survival=bpass,
                                             primary=prim[0] if prim else None)
        power[f'{cb}_minus_{ca}'] = power_report(df[df['config'].isin([ca, cb])], K.PRIMARY_METRIC)

    pd.DataFrame(comp_rows).to_csv(os.path.join(a.out, 'comparisons.csv'), index=False)
    df.groupby('config')[['chain_modely_m', 'chain_des_1e4', 'chain_cmd_1e4', 'chain_actual_1e4', 'chain_steer_deg', 'hunt_steer_rms']].median().reset_index().to_csv(os.path.join(a.out, 'signal_chain.csv'), index=False)
    json.dump(decisions, open(os.path.join(a.out, 'decision.json'), 'w'), indent=2, default=str)
    json.dump(power, open(os.path.join(a.out, 'power.json'), 'w'), indent=2, default=str)

    print('\n=== POWER (between-pass variance -> n for 80% power at the threshold) ===')
    for k, pr in power.items():
        print(f"  {k}: pooled CV={pr.get('_pooled_cv')}  n/arm for 80% @ {K.WIN_EFFECT_PCT}% = {pr.get('_n_per_arm_for_80pct_power_at_threshold')}")
    print('\n=== DECISIONS ===')
    for k, d in decisions.items():
        p = d.get('primary') or {}
        print(f"  {k}: {d['verdict']} | P2 eff={p.get('effect_pct')} p={p.get('perm_p')} strata={p.get('n_strata')} passes={p.get('n_passes_a')}/{p.get('n_passes_b')} battery={d['battery_survival']}")
    print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
