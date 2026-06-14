#!/usr/bin/env python3
"""Build the per-segment lateral×param table for all drives (run after extract_master.py).
   .venv311/bin/python learned_param_studies/code/build_table.py --cache <dir> --out <dir>"""
import argparse, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import shared


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cache', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    summ = pd.read_csv(os.path.join(a.cache, 'drive_summary.csv'))
    # date-order: wall_date if present else file order.
    summ = summ.sort_values('wall_date', na_position='last').reset_index(drop=True)
    summ['order'] = np.arange(len(summ))
    # VALIDATE the time axis before anyone uses it as a control. Real drives span days/weeks; if wall_date
    # collapses to a tiny window it is a pre-sync clock artifact (the original bug: 12/13 drives in a 17s
    # window) and 'order' is NOISE -> it MUST NOT be fed to partial_spearman. (QA fix: tests now use
    # speed+config controls, never 'order'.)
    wd = summ['wall_date'].dropna()
    span_h = (wd.max() - wd.min()) / 3600.0 if len(wd) >= 2 else 0.0
    # CLUSTER-based validity (QA round 1 fix): a single real-timestamp outlier was inflating max-min span and
    # raw nunique (ms ties) -> the old flag falsely passed on data where 12/13 drives sit in a 17s pre-sync
    # window. Count timestamp clusters: a >60s gap between sorted drives starts a new cluster. Real drives are
    # days/weeks apart -> many clusters; the pre-sync artifact collapses to ~1-2 clusters regardless of span.
    n_clusters = 0
    if len(wd) >= 1:
        ws = np.sort(wd.values); n_clusters = 1 + int((np.diff(ws) > 60.0).sum())
    time_valid = bool(span_h >= 1.0 and n_clusters >= max(3, int(0.5 * len(summ))))
    summ['time_valid'] = time_valid
    print(f"time axis: wall_date span={span_h:.2f}h, clusters(>60s apart)={n_clusters}/{len(summ)} -> "
          f"time_valid={time_valid}" + ("" if time_valid else "  [!] wall_date collapses to too few clusters "
          "(pre-sync clock artifact); 'order' is NOT a usable time-control -- do NOT partial on it. Re-extract "
          "with the fixed clocks-max logic to repair."))
    rows = []
    for _, r in summ.iterrows():
        P = shared.load(a.cache, r['drive_id'])
        if P is None:
            print(f"  {r['drive_id']}: no cache"); continue
        meta = dict(drive_id=r['drive_id'], pi_set=str(r.get('pi_set', '')), driving_model=str(r.get('driving_model', '')),
                    wall_date=float(r['wall_date']) if pd.notna(r.get('wall_date')) else np.nan, order=int(r['order']))
        segs = shared.segment_table(P, meta)
        rows += segs
        print(f"  {r['drive_id']} ({meta['pi_set']}/{meta['driving_model']}): {len(segs)} eng-straight segments")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, 'segment_table.csv'), index=False)
    # per-drive aggregate (median)
    agg_cols = ['weave_path', 'weave_steer', 'hunt_steer', 'centering_abs', 'centering_signed', 'spd_mph',
                'steerRatio', 'angleOffsetAvg', 'angleOffset', 'stiffness', 'cal_yaw', 'cal_pitch', 'cal_roll',
                'int_abs', 'int_railed_frac']
    drv = df.groupby(['drive_id', 'pi_set', 'driving_model', 'order'], as_index=False)[agg_cols].median()
    drv = drv.merge(summ[['drive_id', 'wall_date', 'time_valid', 'note']], on='drive_id', how='left').sort_values('order')
    drv['n_segs'] = df.groupby('drive_id').size().reindex(drv['drive_id']).values
    drv.to_csv(os.path.join(a.out, 'drive_table.csv'), index=False)
    print(f"\nwrote segment_table.csv ({len(df)} segs) + drive_table.csv ({len(drv)} drives)")
    print("\n=== per-drive (date-ordered) ===")
    show = ['drive_id', 'pi_set', 'driving_model', 'n_segs', 'spd_mph', 'weave_path', 'centering_abs',
            'steerRatio', 'angleOffsetAvg', 'cal_yaw', 'cal_pitch', 'int_abs']
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(drv[show].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
