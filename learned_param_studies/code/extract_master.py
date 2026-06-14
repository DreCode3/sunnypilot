#!/usr/bin/env python3
"""Master extractor for the LEARNED-PARAMETER correlation studies. For every accumulated drive, pull — resampled
to a 50 Hz grid (slow params nearest/interp, gap-guarded so dropouts don't fabricate data):
  LATERAL signals: v, steeringAngleDeg, yawRate(calibrated), latActive, steeringPressed, blinker, lane-change,
                   model lane offset pos=-(left+right)/2, lat/lon, lead prob/dist (for gating).
  LEARNED / CUMULATIVE params:
    (1) liveParameters: steerRatio, angleOffsetAverageDeg, angleOffsetDeg, stiffnessFactor
    (2) PI integrator: 1 Hz 'LC:' telemetry off/int/P/I/curv  (+ recovered lc_kp = config)
    (3) liveCalibration: rpyCalib roll/pitch/yaw (deg)
  META: wall-clock date (from clocks), driving_model (manifest), n segs.
Run FROM REPO ROOT:  .venv311/bin/python learned_param_studies/code/extract_master.py --manifest <m.csv> --out <dir>
Manifest cols: drive_id,folder,driving_model,note
"""
import argparse, glob, json, math, os, re, sys, csv as _csv
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

FS = 50.0; MAX_GAP = 0.5
LC = re.compile(r'off=(-?[\d.]+) ll=(-?[\d.]+) pos=(-?[\d.]+) scl=([\d.]+) conf=([\d.]+) wid=([\d.]+) '
                r'int=(-?[\d.]+) P=(-?[\d.eE+-]+) I=(-?[\d.eE+-]+) curv=(-?[\d.]+) spd=(-?[\d.]+)')
LANE = {"off": 0, "preLaneChange": 1, "laneChangeStarting": 2, "laneChangeFinishing": 3}


def ix(xs, ys, xq):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    if len(x) < 2 or xq < x[0] or xq > x[-1]:
        return np.nan
    return float(np.interp(xq, x, y))


def _seg_files(folder):
    """rlog file paths in segment order, handling BOTH on-disk layouts:
    new  route_XX/000000XX--hash--N/rlog.zst   |   old  route_XX/rlog_N(.zst) flat."""
    new = sorted(glob.glob(f'{folder}/000000*--*/'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    if new:
        return [sd + 'rlog.zst' for sd in new]
    def _n(p):
        s = os.path.basename(p).split('rlog_')[-1].split('.')[0]
        return int(s) if s.isdigit() else -1
    return sorted(glob.glob(f'{folder}/rlog_*'), key=_n)


def extract(folder):
    files = _seg_files(folder)
    A = {k: [] for k in ['t_cs', 'v', 'steer', 'press', 'lblink', 'rblink', 'canvalid',
                         't_cc', 'latact', 't_m', 'pos', 'lcs', 'lead_p', 'lead_d',
                         't_l', 'lat', 'lon', 'yaw',
                         't_lp', 'sr', 'aoa', 'ao', 'stf',
                         't_cal', 'roll', 'pitch', 'yaw_c', 'calp',
                         't_lc', 'lc_off', 'lc_int', 'lc_P', 'lc_I', 'lc_curv']}
    wall = None; nseg = 0
    for f in files:
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                w = msg.which(); t = msg.logMonoTime * 1e-9
                if w == 'carState':
                    cs = msg.carState
                    A['t_cs'].append(t); A['v'].append(cs.vEgo); A['steer'].append(cs.steeringAngleDeg)
                    A['press'].append(1.0 if cs.steeringPressed else 0.0)
                    A['lblink'].append(1.0 if cs.leftBlinker else 0.0); A['rblink'].append(1.0 if cs.rightBlinker else 0.0)
                    A['canvalid'].append(1.0 if cs.canValid else 0.0)
                elif w == 'carControl':
                    A['t_cc'].append(t); A['latact'].append(1.0 if msg.carControl.latActive else 0.0)
                elif w == 'modelV2':
                    m = msg.modelV2; ll = m.laneLines
                    A['t_m'].append(t)
                    if len(ll) > 2 and len(ll[1].y) and len(ll[2].y):
                        A['pos'].append(-(float(ll[1].y[0]) + float(ll[2].y[0])) / 2)
                    else:
                        A['pos'].append(np.nan)
                    A['lcs'].append(LANE.get(str(m.meta.laneChangeState), -1))
                    lp, ld = 0.0, 1e9
                    try:
                        if len(m.leadsV3):
                            lp = float(m.leadsV3[0].prob); ld = float(m.leadsV3[0].x[0])
                    except Exception:
                        pass
                    A['lead_p'].append(lp); A['lead_d'].append(ld)
                elif w == 'liveLocationKalman':
                    g = msg.liveLocationKalman; A['t_l'].append(t)
                    if g.positionGeodetic.valid and len(g.positionGeodetic.value) >= 2:
                        A['lat'].append(float(g.positionGeodetic.value[0])); A['lon'].append(float(g.positionGeodetic.value[1]))
                    else:
                        A['lat'].append(np.nan); A['lon'].append(np.nan)
                    A['yaw'].append(float(g.angularVelocityCalibrated.value[2]) if (g.angularVelocityCalibrated.valid and len(g.angularVelocityCalibrated.value) >= 3) else np.nan)
                elif w == 'liveParameters':
                    lp = msg.liveParameters
                    if lp.valid:
                        A['t_lp'].append(t); A['sr'].append(float(lp.steerRatio)); A['aoa'].append(float(lp.angleOffsetAverageDeg))
                        A['ao'].append(float(lp.angleOffsetDeg)); A['stf'].append(float(lp.stiffnessFactor))
                elif w == 'liveCalibration':
                    lc = msg.liveCalibration; rpy = list(lc.rpyCalib); A['t_cal'].append(t)
                    A['roll'].append(math.degrees(rpy[0]) if len(rpy) > 0 else np.nan)
                    A['pitch'].append(math.degrees(rpy[1]) if len(rpy) > 1 else np.nan)
                    A['yaw_c'].append(math.degrees(rpy[2]) if len(rpy) > 2 else np.nan)
                    try: A['calp'].append(float(lc.calPerc))      # calibration % -> drops to ~0 & re-climbs on a RESET
                    except Exception: A['calp'].append(np.nan)
                elif w == 'clocks':
                    # Use the MAX wall time over the drive (end-of-drive, AFTER NTP/GPS sync). The FIRST
                    # clocks reading is a pre-sync boot/RTC value -> previously 12/13 drives collapsed to a
                    # single stale 17s window, which made the date-order time-control pure noise. (QA fix.)
                    try:
                        wt = float(msg.clocks.wallTimeNanos) * 1e-9
                        if wall is None or wt > wall:
                            wall = wt
                    except Exception: pass
                elif w == 'logMessage':
                    s = msg.logMessage
                    if 'LC:' in s:
                        try: txt = json.loads(s).get('msg', '')
                        except Exception: txt = s
                        mm = LC.search(txt)
                        if mm:
                            o, ll_, ps, sc, cf, wd, iv, P, I, cv, sp = map(float, mm.groups())
                            A['t_lc'].append(t); A['lc_off'].append(o); A['lc_int'].append(iv)
                            A['lc_P'].append(P); A['lc_I'].append(I); A['lc_curv'].append(cv)
            nseg += 1
        except Exception as e:
            sys.stderr.write(f'  {sd} err {e}\n')
    return A, wall, nseg


def resample(A):
    def arr(k): return np.asarray(A[k], float)
    tcs = arr('t_cs')
    if len(tcs) < 200:
        return None
    t0 = tcs.min(); tu = np.arange(t0, tcs.max(), 1 / FS)

    def R(tk, yk):
        t = arr(tk); y = arr(yk)
        if len(t) < 2:
            return np.full_like(tu, np.nan)
        o = np.argsort(t); t = t[o]; y = y[o]
        out = np.interp(tu, t, y, left=np.nan, right=np.nan)
        nn = t[np.clip(np.searchsorted(t, tu), 0, len(t) - 1)]; pv = t[np.clip(np.searchsorted(t, tu) - 1, 0, len(t) - 1)]
        out[np.minimum(np.abs(tu - nn), np.abs(tu - pv)) > MAX_GAP] = np.nan
        return out

    def Rb(tk, yk):
        t = arr(tk); y = arr(yk)
        if len(t) < 2:
            return np.zeros_like(tu)
        o = np.argsort(t); return (np.interp(tu, t[o], y[o]) > 0.5).astype(float)

    out = dict(t=(tu - t0).astype(np.float32), lat=R('t_l', 'lat'), lon=R('t_l', 'lon'),
               v=R('t_cs', 'v').astype(np.float32), steer=R('t_cs', 'steer').astype(np.float32),
               yaw=R('t_l', 'yaw').astype(np.float32), pos=R('t_m', 'pos').astype(np.float32),
               latact=Rb('t_cc', 'latact'), press=Rb('t_cs', 'press'),
               blinker=((Rb('t_cs', 'lblink') + Rb('t_cs', 'rblink')) > 0).astype(float),
               canvalid=Rb('t_cs', 'canvalid'), lcs=R('t_m', 'lcs').astype(np.float32),
               lead_p=R('t_m', 'lead_p').astype(np.float32), lead_d=R('t_m', 'lead_d').astype(np.float32),
               sr=R('t_lp', 'sr').astype(np.float32), aoa=R('t_lp', 'aoa').astype(np.float32),
               ao=R('t_lp', 'ao').astype(np.float32), stf=R('t_lp', 'stf').astype(np.float32),
               cal_roll=R('t_cal', 'roll').astype(np.float32), cal_pitch=R('t_cal', 'pitch').astype(np.float32),
               cal_yaw=R('t_cal', 'yaw_c').astype(np.float32), cal_perc=R('t_cal', 'calp').astype(np.float32))
    out['lc_t'] = (arr('t_lc') - t0).astype(np.float32) if len(A['t_lc']) else np.zeros(0, np.float32)
    for k in ['lc_off', 'lc_int', 'lc_P', 'lc_I', 'lc_curv']:
        out[k] = arr(k).astype(np.float32)
    return out


def recover_lc_kp(out):
    off = out.get('lc_off'); P = out.get('lc_P')
    if off is None or len(off) < 5:
        return None
    ok = np.abs(off) > 0.02
    return float(np.median(P[ok] / off[ok])) if ok.sum() >= 5 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(_csv.DictReader(open(a.manifest))); summ = []
    for r in rows:
        did = r['drive_id']; outp = os.path.join(a.out, f'{did}.npz')
        if os.path.exists(outp) and not a.force:
            print(f'{did}: skip'); continue
        print(f'{did}: {r["folder"]}')
        A, wall, nseg = extract(r['folder']); out = resample(A)
        if out is None:
            print(f'  {did}: insufficient'); continue
        kp = recover_lc_kp(out)
        np.savez_compressed(outp, **out)
        row = dict(drive_id=did, driving_model=r.get('driving_model', ''), note=r.get('note', ''),
                   wall_date=wall, nseg=nseg, lc_kp=kp,
                   pi_set=('golden' if (kp or 0) >= 0.0003 else ('weak' if kp else 'unknown/manual')),
                   sr_med=float(np.nanmedian(out['sr'])), aoa_med=float(np.nanmedian(out['aoa'])),
                   stf_med=float(np.nanmedian(out['stf'])), cal_yaw_med=float(np.nanmedian(out['cal_yaw'])),
                   cal_pitch_med=float(np.nanmedian(out['cal_pitch'])), cal_roll_med=float(np.nanmedian(out['cal_roll'])),
                   eng_pct=float(np.mean(out['latact']) * 100), dur_min=float(out['t'][-1] / 60))
        summ.append(row)
        print(f"  -> {did}: {row['dur_min']:.0f}min eng{row['eng_pct']:.0f}% pi={row['pi_set']} "
              f"steerRatio={row['sr_med']:.2f} angleOffAvg={row['aoa_med']:.2f} calYaw={row['cal_yaw_med']:.2f}")
    if summ:
        keys = sorted({k for s in summ for k in s})
        with open(os.path.join(a.out, 'drive_summary.csv'), 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=keys); w.writeheader()
            for s in summ:
                w.writerow(s)
        print(f'wrote {a.out}/drive_summary.csv ({len(summ)} drives)')


if __name__ == '__main__':
    main()
