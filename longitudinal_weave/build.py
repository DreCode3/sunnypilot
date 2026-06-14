#!/usr/bin/env python3
"""LONGITUDINAL weave extractor (PARALLEL): ONE row per historical drive ->
  * proven model-indep WEAVE (engaged-straight 0.1-0.35Hz band-RMS of yawRate/vEgo = weave_path; + weave_steer)
    as the per-drive median over eligible segments, plus median speed (the dominant confound) + centering.
  * LEARNED-param TRAJECTORY: steerRatio/angleOffsetAverage START & END, stiffness, rpyCalib, and calPerc
    START/MIN/END (calibration % -> collapses & re-climbs on a RESET = cleanest reset signal).
  * CONFIG: recovered lc_kp -> pi_set; engaged fraction.  reliable DATE (max-clocks) + route_counter (hex->int).
Reuses learned_param_studies extract_master (BOTH on-disk formats + calPerc, 3-round QA'd) and shared.segment_table.
PARALLEL across all CPU cores (ProcessPoolExecutor). Resumable: per-route npz cache + skips routes already in CSV.
  .venv311/bin/python longitudinal_weave/build.py --out longitudinal_weave/results [--workers N] [--limit N] [--only a,b]"""
import argparse, glob, os, sys, warnings
import numpy as np, pandas as pd
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'learned_param_studies', 'code'))
import extract_master as EM
import shared
warnings.filterwarnings('ignore')


def first_last(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(x[0]), float(x[-1])) if len(x) else (np.nan, np.nan)


def hexc(rid):
    s = rid.split('_')[-1]
    try: return int(s, 16)
    except ValueError: return -1


def drive_row(rid, folder, cache_dir):
    cache = os.path.join(cache_dir, f'{rid}.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True); P = {k: z[k] for k in z.files}
        wall = float(P.pop('wall_date')); nseg = int(P.pop('nseg'))
    else:
        A, wall, nseg = EM.extract(folder); P = EM.resample(A)
        if P is None:
            return None
        np.savez_compressed(cache, wall_date=(wall if wall else np.nan), nseg=nseg, **P)
        wall = wall if wall else np.nan
    kp = EM.recover_lc_kp(P)
    meta = dict(drive_id=rid, pi_set=('golden' if (kp or 0) >= 0.0003 else ('weak' if kp else 'unknown/manual')),
                driving_model='', wall_date=wall, order=hexc(rid))
    sdf = pd.DataFrame(shared.segment_table(P, meta))
    med = lambda c: float(sdf[c].median()) if len(sdf) and c in sdf else np.nan
    sr0, sr1 = first_last(P['sr']); aoa0, aoa1 = first_last(P['aoa'])
    cp = np.asarray(P.get('cal_perc', []), float); cp = cp[np.isfinite(cp)]
    eng = float(np.nanmean(np.asarray(P['latact'], float))) if len(P['latact']) else np.nan
    return dict(drive_id=rid, route_counter=hexc(rid), wall_date=wall,
                date=(datetime.fromtimestamp(wall, timezone.utc).strftime('%Y-%m-%d %H:%M') if np.isfinite(wall) else ''),
                n_segs=nseg, n_elig=len(sdf), engaged_frac=eng,
                weave_path=med('weave_path'), weave_steer=med('weave_steer'), hunt_steer=med('hunt_steer'),
                centering_abs=med('centering_abs'), centering_signed=med('centering_signed'), spd_mph=med('spd_mph'),
                sr_start=sr0, sr_end=sr1, aoa_start=aoa0, aoa_end=aoa1,
                stf=float(np.nanmedian(P['stf'])), cal_yaw=float(np.nanmedian(P['cal_yaw'])),
                cal_pitch=float(np.nanmedian(P['cal_pitch'])), cal_roll=float(np.nanmedian(P['cal_roll'])),
                calperc_start=(float(cp[0]) if len(cp) else np.nan), calperc_min=(float(cp.min()) if len(cp) else np.nan),
                calperc_end=(float(cp[-1]) if len(cp) else np.nan), lc_kp=kp, pi_set=meta['pi_set'])


def _worker(args):
    rid, folder, cache_dir = args
    try:
        r = drive_row(rid, folder, cache_dir)
        return r if r is not None else {'drive_id': rid, '_error': 'insufficient'}
    except Exception as e:
        return {'drive_id': rid, '_error': repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--routes', default='explorer_st_logs/route_*')
    ap.add_argument('--out', required=True); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', default=''); ap.add_argument('--workers', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); cache = os.path.join(a.out, 'cache'); os.makedirs(cache, exist_ok=True)
    outcsv = os.path.join(a.out, 'drive_longitudinal.csv')
    done = set(pd.read_csv(outcsv)['drive_id']) if os.path.exists(outcsv) else set()
    dirs = [d for d in sorted(glob.glob(a.routes)) if os.path.isdir(d)]
    if a.only:
        want = set(a.only.split(',')); dirs = [d for d in dirs if os.path.basename(d) in want]
    dirs = sorted(dirs, key=lambda d: hexc(os.path.basename(d)))
    if a.limit: dirs = dirs[:a.limit]
    tasks = [(os.path.basename(d), d, cache) for d in dirs if os.path.basename(d) not in done]
    workers = a.workers or (os.cpu_count() or 4)
    print(f"routes total={len(dirs)} | to-do={len(tasks)} | workers={workers}")
    rows, errs, n = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, t): t[0] for t in tasks}
        for f in as_completed(futs):
            r = f.result(); n += 1
            if r is None or '_error' in r:
                errs += 1; print(f"[{n}/{len(tasks)}] {r.get('drive_id','?')}: ERR {r.get('_error')}"); continue
            rows.append(r)
            print(f"[{n}/{len(tasks)}] {r['drive_id']}: {r['date']} weave={r['weave_path']:.2f} spd={r['spd_mph']:.0f} "
                  f"elig={r['n_elig']} sr={r['sr_start']:.2f}->{r['sr_end']:.2f} calP={r['calperc_start']:.0f}/{r['calperc_min']:.0f} {r['pi_set']}")
            if len(rows) % 15 == 0:                                    # periodic safe write (resumable; idempotent)
                base = pd.read_csv(outcsv) if os.path.exists(outcsv) else pd.DataFrame()
                pd.concat([base, pd.DataFrame(rows)], ignore_index=True).drop_duplicates('drive_id', keep='last').to_csv(outcsv, index=False)
    if rows:
        base = pd.read_csv(outcsv) if os.path.exists(outcsv) else pd.DataFrame()
        pd.concat([base, pd.DataFrame(rows)], ignore_index=True).drop_duplicates('drive_id', keep='last').to_csv(outcsv, index=False)
    final = pd.read_csv(outcsv) if os.path.exists(outcsv) else pd.DataFrame()
    print(f"\nwrote {outcsv}  | total drives={len(final)}  errors this run={errs}")


if __name__ == '__main__':
    main()
