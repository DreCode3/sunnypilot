#!/usr/bin/env python3
"""Extract per-PASS signal caches for the controlled-test analysis (ANALYSIS_PLAN.md §2, §13).
Reads a manifest CSV of passes -> {folder, driving_model, declared pi_set, direction} and writes one
.npz per pass with: continuous channels resampled to 50 Hz (the full signal chain + both yaw sources +
heading), sparse 'LC:' controller telemetry (off/int/P/I/curv for integrator-reset/saturation & config
proof), and metadata (build commit, carParams, recovered lc_kp/lc_ki/cap).

Env: run with the repo's openpilot python, FROM REPO ROOT:
    .venv311/bin/python controlled_test_analysis/code/extract.py --manifest <passes.csv> --out <cache_dir>

Manifest columns (header required): pass_id,folder,driving_model,pi_set_declared,direction,intended_speed_mph
  folder = a directory containing segment subdirs each with rlog.zst (e.g. explorer_st_logs/route_b8)
"""
import argparse, glob, json, math, os, re, sys
import numpy as np
import csv as _csv
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

FS = 50.0
LC = re.compile(r'off=(-?[\d.]+) ll=(-?[\d.]+) pos=(-?[\d.]+) scl=([\d.]+) conf=([\d.]+) wid=([\d.]+) '
                r'int=(-?[\d.]+) P=(-?[\d.eE+-]+) I=(-?[\d.eE+-]+) curv=(-?[\d.]+) spd=(-?[\d.]+)')
LANE_CHANGE = {"off": 0, "preLaneChange": 1, "laneChangeStarting": 2, "laneChangeFinishing": 3}


def interp_xy(xs, ys, xq):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    if len(x) < 2 or xq < x[0] or xq > x[-1]:
        return np.nan
    return float(np.interp(xq, x, y))


def lane_center_y(model, xq):
    ll = model.laneLines; lp = model.laneLineProbs
    if len(ll) < 3 or len(lp) < 3:
        return np.nan, np.nan, np.nan
    ly = interp_xy(ll[1].x, ll[1].y, xq); ry = interp_xy(ll[2].x, ll[2].y, xq)
    if not (np.isfinite(ly) and np.isfinite(ry)):
        return np.nan, np.nan, float(min(lp[1], lp[2]))
    return 0.5 * (ly + ry), abs(ry - ly), float(min(lp[1], lp[2]))


def extract_pass(folder):
    segs = sorted(glob.glob(f'{folder}/000000*--*/') or glob.glob(f'{folder}/*/'),
                  key=lambda d: int(re.findall(r'(\d+)/?$', d)[-1]) if re.findall(r'(\d+)/?$', d) else 0)
    C = {k: [] for k in ['t_cs', 'v', 'steer', 'steer_rate', 'yaw_cs', 'press', 'lblink', 'rblink', 'canvalid',
                         't_cc', 'latact', 'cmd_curv',
                         't_ct', 'des_curv', 'ctl_curv',
                         't_m', 'model_y20', 'lane_c_y20', 'lane_w', 'lane_p', 'lcs',
                         't_l', 'lat', 'lon', 'yaw_cal',
                         't_lead', 'lead_prob', 'lead_d',
                         't_lc', 'lc_off', 'lc_int', 'lc_P', 'lc_I', 'lc_curv',
                         't_cal', 'cal_yaw']}
    meta = {'build_commit': None, 'carParams': {}}
    for sd in segs:
        f = sd + 'rlog.zst'
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                w = msg.which(); t = msg.logMonoTime * 1e-9
                if w == 'carState':
                    cs = msg.carState
                    C['t_cs'].append(t); C['v'].append(cs.vEgo); C['steer'].append(cs.steeringAngleDeg)
                    C['steer_rate'].append(cs.steeringRateDeg); C['yaw_cs'].append(cs.yawRate)
                    C['press'].append(1.0 if cs.steeringPressed else 0.0)
                    C['lblink'].append(1.0 if cs.leftBlinker else 0.0); C['rblink'].append(1.0 if cs.rightBlinker else 0.0)
                    C['canvalid'].append(1.0 if cs.canValid else 0.0)
                elif w == 'carControl':
                    cc = msg.carControl
                    C['t_cc'].append(t); C['latact'].append(1.0 if cc.latActive else 0.0)
                    try: C['cmd_curv'].append(float(cc.actuators.curvature))
                    except Exception: C['cmd_curv'].append(np.nan)
                elif w == 'controlsState':
                    st = msg.controlsState
                    C['t_ct'].append(t)
                    try: C['des_curv'].append(float(st.desiredCurvature))
                    except Exception: C['des_curv'].append(np.nan)
                    try: C['ctl_curv'].append(float(st.curvature))
                    except Exception: C['ctl_curv'].append(np.nan)
                elif w == 'modelV2':
                    m = msg.modelV2
                    C['t_m'].append(t)
                    C['model_y20'].append(interp_xy(m.position.x, m.position.y, 20.0))
                    cy, cw, cp = lane_center_y(m, 20.0)
                    C['lane_c_y20'].append(cy); C['lane_w'].append(cw); C['lane_p'].append(cp)
                    C['lcs'].append(LANE_CHANGE.get(str(m.meta.laneChangeState), -1))
                    # lead vehicle (M9): prob + relative distance, for the lead-follow gate
                    lp, ld = 0.0, 1e9
                    try:
                        if len(m.leadsV3) > 0:
                            lp = float(m.leadsV3[0].prob); ld = float(m.leadsV3[0].x[0])
                    except Exception:
                        pass
                    C['t_lead'].append(t); C['lead_prob'].append(lp); C['lead_d'].append(ld)
                elif w == 'liveLocationKalman':
                    g = msg.liveLocationKalman
                    C['t_l'].append(t)
                    if g.positionGeodetic.valid and len(g.positionGeodetic.value) >= 2:
                        C['lat'].append(float(g.positionGeodetic.value[0])); C['lon'].append(float(g.positionGeodetic.value[1]))
                    else:
                        C['lat'].append(np.nan); C['lon'].append(np.nan)
                    if g.angularVelocityCalibrated.valid and len(g.angularVelocityCalibrated.value) >= 3:
                        C['yaw_cal'].append(float(g.angularVelocityCalibrated.value[2]))
                    else:
                        C['yaw_cal'].append(np.nan)
                elif w == 'liveCalibration':
                    rpy = list(msg.liveCalibration.rpyCalib)
                    C['t_cal'].append(t); C['cal_yaw'].append(rpy[2] if len(rpy) > 2 else np.nan)
                elif w == 'carParams' and not meta['carParams']:
                    cp = msg.carParams
                    meta['carParams'] = dict(carFingerprint=str(cp.carFingerprint), wheelbase=float(cp.wheelbase),
                                             steerRatio=float(cp.steerRatio), steerActuatorDelay=float(cp.steerActuatorDelay))
                elif w == 'logMessage':
                    s = msg.logMessage
                    if 'LC:' in s:
                        try: txt = json.loads(s).get('msg', '')
                        except Exception: txt = s
                        m = LC.search(txt)
                        if m:
                            off, ll, pos, scl, conf, wid, integ, P, I, curv, spd = map(float, m.groups())
                            C['t_lc'].append(t); C['lc_off'].append(off); C['lc_int'].append(integ)
                            C['lc_P'].append(P); C['lc_I'].append(I); C['lc_curv'].append(curv)
                        if meta['build_commit'] is None and ('"commit"' in s or '"branch"' in s):
                            try: meta['build_commit'] = json.loads(s).get('ctx', {}).get('commit')
                            except Exception: pass
        except Exception as e:
            sys.stderr.write(f'  seg {sd} err {e}\n')
    return C, meta


def resample(C):
    def arr(k): return np.asarray(C[k], float)
    tcs = arr('t_cs')
    if len(tcs) < 200:
        return None
    t0 = tcs.min(); t1 = tcs.max(); tu = np.arange(t0, t1, 1 / FS)

    MAX_GAP = 0.5  # s — do NOT fabricate data across longer source gaps (M8)

    def R(tk, yk):
        t = arr(tk); y = arr(yk)
        if len(t) < 2:
            return np.full_like(tu, np.nan)
        o = np.argsort(t); t = t[o]; y = y[o]
        out = np.interp(tu, t, y, left=np.nan, right=np.nan)
        nearest = t[np.clip(np.searchsorted(t, tu), 0, len(t) - 1)]
        prev = t[np.clip(np.searchsorted(t, tu) - 1, 0, len(t) - 1)]
        gap = np.minimum(np.abs(tu - nearest), np.abs(tu - prev))
        out[gap > MAX_GAP] = np.nan
        return out

    def Rb(tk, yk):  # nearest for flags
        t = arr(tk); y = arr(yk)
        if len(t) < 2:
            return np.zeros_like(tu)
        o = np.argsort(t); t = t[o]; y = y[o]
        idx = np.clip(np.searchsorted(t, tu), 0, len(t) - 1)
        return (y[idx] > 0.5).astype(float)

    lat = R('t_l', 'lat'); lon = R('t_l', 'lon')
    # heading from GPS gradient, smoothed over ~0.5s (circular-safe: smooth dx,dy then atan2)
    hd = np.full_like(tu, np.nan)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() > 50:
        la0 = np.nanmedian(lat); k = max(3, int(0.5 * FS))
        dx = np.gradient(np.nan_to_num(lon)) * 111320.0 * math.cos(math.radians(la0)); dy = np.gradient(np.nan_to_num(lat)) * 110540.0
        kern = np.ones(k) / k
        dxs = np.convolve(dx, kern, 'same'); dys = np.convolve(dy, kern, 'same')
        hd = (np.degrees(np.arctan2(dxs, dys))) % 360
        hd[~ok] = np.nan
    out = dict(
        t=(tu - t0).astype(np.float32), lat=lat, lon=lon, heading=hd.astype(np.float32),
        v=R('t_cs', 'v').astype(np.float32), steer=R('t_cs', 'steer').astype(np.float32),
        steer_rate=R('t_cs', 'steer_rate').astype(np.float32),
        yaw_cs=R('t_cs', 'yaw_cs').astype(np.float32), yaw_cal=R('t_l', 'yaw_cal').astype(np.float32),
        latact=Rb('t_cc', 'latact'), press=Rb('t_cs', 'press'),
        blinker=((Rb('t_cs', 'lblink') + Rb('t_cs', 'rblink')) > 0).astype(float),
        canvalid=Rb('t_cs', 'canvalid'),
        cmd_curv=R('t_cc', 'cmd_curv').astype(np.float32), des_curv=R('t_ct', 'des_curv').astype(np.float32),
        ctl_curv=R('t_ct', 'ctl_curv').astype(np.float32),
        model_y20=R('t_m', 'model_y20').astype(np.float32), lane_c_y20=R('t_m', 'lane_c_y20').astype(np.float32),
        lane_w=R('t_m', 'lane_w').astype(np.float32), lane_p=R('t_m', 'lane_p').astype(np.float32),
        lcs=R('t_m', 'lcs').astype(np.float32), cal_yaw=R('t_cal', 'cal_yaw').astype(np.float32),
        lead_prob=R('t_lead', 'lead_prob').astype(np.float32), lead_d_rel=R('t_lead', 'lead_d').astype(np.float32),
    )
    # sparse LC telemetry (kept at native ~1 Hz)
    out['lc_t'] = (arr('t_lc') - t0).astype(np.float32) if len(C['t_lc']) else np.zeros(0, np.float32)
    for k in ['lc_off', 'lc_int', 'lc_P', 'lc_I', 'lc_curv']:
        out[k] = arr(k).astype(np.float32)
    return out


def recover_config(out):
    """Recover lc_kp = P/off, lc_ki = I/int, and max|int| from LC telemetry (proves config, §1)."""
    off = out.get('lc_off'); P = out.get('lc_P'); I = out.get('lc_I'); integ = out.get('lc_int')
    res = dict(lc_kp=None, lc_ki=None, max_abs_int=None, n_lc=int(len(off)) if off is not None else 0)
    if off is None or len(off) < 5:
        return res
    ok = np.abs(off) > 0.02
    if ok.sum() >= 5:
        res['lc_kp'] = float(np.median(P[ok] / off[ok]))
    oki = np.abs(integ) > 0.02
    if oki.sum() >= 5:
        res['lc_ki'] = float(np.median(I[oki] / integ[oki]))
    res['max_abs_int'] = float(np.max(np.abs(integ)))
    # integrator-reset check (S9): |int| at the FIRST engaged LC sample (not merely the first logged one)
    res['int_start_abs'] = float(np.abs(integ[0])) if len(integ) else None
    try:
        eng_idx = np.where(out['latact'] > 0.5)[0]
        if len(eng_idx) and len(out.get('lc_t', [])):
            t_eng = float(out['t'][eng_idx[0]])
            after = np.where(out['lc_t'] >= t_eng)[0]
            if len(after):
                res['int_start_abs'] = float(np.abs(integ[after[0]]))
    except Exception:
        pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(_csv.DictReader(open(a.manifest)))
    summary = []
    for r in rows:
        pid = r['pass_id']; outp = os.path.join(a.out, f'{pid}.npz')
        if os.path.exists(outp) and not a.force:
            print(f'{pid}: exists, skip'); continue
        print(f'{pid}: extracting {r["folder"]}')
        C, meta = extract_pass(r['folder'])
        out = resample(C)
        if out is None:
            print(f'  {pid}: insufficient data'); continue
        cfg = recover_config(out)
        np.savez_compressed(outp, **out)
        meta_row = dict(pass_id=pid, **{k: r.get(k, '') for k in
                        ['driving_model', 'pi_set_declared', 'direction', 'intended_speed_mph']},
                        build_commit=meta['build_commit'], **{f'cp_{k}': v for k, v in meta['carParams'].items()},
                        **{f'recovered_{k}': v for k, v in cfg.items()},
                        pi_set_recovered=('golden' if (cfg['lc_kp'] or 0) >= 0.0003 else ('weak' if cfg['lc_kp'] else 'unknown')),
                        dur_min=float(out['t'][-1] / 60), eng_pct=float(np.mean(out['latact']) * 100),
                        spd_med_mph=float(np.nanmedian(out['v'][out['latact'] > 0.5]) * 2.23694) if (out['latact'] > 0.5).any() else np.nan)
        summary.append(meta_row)
        print(f"  -> {pid}: {meta_row['dur_min']:.1f}min eng {meta_row['eng_pct']:.0f}% "
              f"lc_kp={cfg['lc_kp']} pi_recovered={meta_row['pi_set_recovered']} declared={r.get('pi_set_declared')}")
    if summary:
        keys = sorted({k for row in summary for k in row})
        with open(os.path.join(a.out, 'pass_summary.csv'), 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for row in summary:
                w.writerow(row)
        print(f'wrote {a.out}/pass_summary.csv ({len(summary)} passes)')


if __name__ == '__main__':
    main()
