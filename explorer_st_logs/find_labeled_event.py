#!/usr/bin/env python3
"""Find the curve event nearest to a target GPS coordinate, dump all signals."""

import sys
import os
import glob
import argparse
import math
import numpy as np

sys.path.insert(0, '/Users/dregilley/Documents/GitHub/sunnypilot')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_curves_v2 import load_cx1, detect_curve_events, IDX


def find_rlogs(route_prefix):
    files = sorted(glob.glob(os.path.join(route_prefix, 'rlog_*.zst')))
    if files:
        return files
    seg_dirs = sorted(glob.glob(route_prefix + '--*'),
                      key=lambda d: int(d.rsplit('--', 1)[-1]))
    out = []
    for sd in seg_dirs:
        rlog = os.path.join(sd, 'rlog.zst')
        if os.path.exists(rlog):
            out.append(rlog)
    return out


def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters between two lat/lon points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def pull_gps_and_signals(route_prefix):
    """Returns dict with gps timestamps + lat/lon and signal arrays."""
    from openpilot.tools.lib.logreader import LogReader

    files = find_rlogs(route_prefix)
    if not files:
        return None

    gps_t = []
    gps_lat = []
    gps_lon = []
    # Try liveLocationKalman first, fall back to gpsLocationExternal
    found_llk = 0
    found_gps = 0
    for fp in files:
        try:
            for msg in LogReader(fp):
                w = msg.which()
                t = msg.logMonoTime / 1e9
                if w == 'liveLocationKalman':
                    llk = msg.liveLocationKalman
                    if hasattr(llk, 'positionGeodetic') and llk.positionGeodetic.valid:
                        val = llk.positionGeodetic.value
                        gps_t.append(t)
                        gps_lat.append(float(val[0]))
                        gps_lon.append(float(val[1]))
                        found_llk += 1
                elif w == 'gpsLocationExternal' and found_llk == 0:
                    g = msg.gpsLocationExternal
                    gps_t.append(t)
                    gps_lat.append(float(g.latitude))
                    gps_lon.append(float(g.longitude))
                    found_gps += 1
        except Exception:
            pass

    print(f'  GPS samples: {len(gps_t)} (liveLocationKalman={found_llk}, gps={found_gps})')
    return {
        't': np.array(gps_t),
        'lat': np.array(gps_lat),
        'lon': np.array(gps_lon),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('route_prefix')
    ap.add_argument('--lat', type=float, required=True)
    ap.add_argument('--lon', type=float, required=True)
    ap.add_argument('--window', type=float, default=5.0,
                    help='Show signals in ±window_sec around event apex')
    args = ap.parse_args()

    print(f'[1/3] Loading CX1 from {args.route_prefix}...', flush=True)
    arr, t_cx1 = load_cx1(args.route_prefix)
    print(f'  {len(arr)} CX1 rows, span {t_cx1[-1] - t_cx1[0]:.1f}s')

    print(f'[2/3] Pulling GPS from rlog...', flush=True)
    gps = pull_gps_and_signals(args.route_prefix)
    if gps is None or len(gps['t']) == 0:
        print('  No GPS data.')
        return

    print(f'[3/3] Finding nearest pass to ({args.lat}, {args.lon})...', flush=True)
    distances = np.array([haversine_m(args.lat, args.lon, la, lo)
                          for la, lo in zip(gps['lat'], gps['lon'])])
    closest_idx = int(np.argmin(distances))
    t_closest = gps['t'][closest_idx]
    print(f'  Closest pass: t={t_closest:.1f}s, dist={distances[closest_idx]:.1f}m, '
          f'GPS=({gps["lat"][closest_idx]:.5f}, {gps["lon"][closest_idx]:.5f})')

    # All passes within 20m
    near_mask = distances < 20.0
    n_near = int(near_mask.sum())
    print(f'  {n_near} GPS samples within 20m of target')
    if n_near > 0:
        nears = np.where(near_mask)[0]
        print(f'  Time range of passes: {gps["t"][nears[0]]:.1f}s to {gps["t"][nears[-1]]:.1f}s')

    # Find curve events; pick the one with apex closest to t_closest
    print()
    print(f'  Detecting curve events...', flush=True)
    events = detect_curve_events(arr, t_cx1)
    print(f'  {len(events)} events. Finding nearest by apex time...')
    apex_dists = [abs(e['t_apex'] - t_closest) for e in events]
    if not apex_dists:
        print('No events found.')
        return
    nearest_ev_i = int(np.argmin(apex_dists))
    ev = events[nearest_ev_i]
    print(f'  → Event {nearest_ev_i}: class={ev["magnitude_class"]}, '
          f'apex at t={ev["t_apex"]:.1f}s, '
          f'v_mph={ev["mean_v_mph"]:.1f}, dur={ev["duration_sec"]:.1f}s, '
          f'peak_cmd={ev["peak_cmd"]:.5f}')
    print(f'  apex-to-target time delta: {ev["t_apex"] - t_closest:+.1f}s')

    # Pull signals for ±window_sec around apex
    t_lo = ev['t_apex'] - args.window
    t_hi = ev['t_apex'] + args.window
    mask = (t_cx1 >= t_lo) & (t_cx1 <= t_hi)
    if mask.sum() < 5:
        print('Not enough samples in window.')
        return

    sub = arr[mask]
    t_rel = t_cx1[mask] - ev['t_apex']

    fields_to_print = [
        ('v_mph',   lambda r: r[IDX['v']] * 2.237),
        ('cmd',     lambda r: r[IDX['cmd']]),
        ('des',     lambda r: r[IDX['des']]),
        ('pred',    lambda r: r[IDX['pred']]),
        ('ema',     lambda r: r[IDX['ema']]),
        ('preRL',   lambda r: r[IDX['preRL']]),
        ('rl',      lambda r: r[IDX['rl']]),
        ('meas',    lambda r: r[IDX['meas']]),
        ('aLat',    lambda r: r[IDX['aLat']]),
        ('ang',     lambda r: r[IDX['ang']]),
        ('dAng',    lambda r: r[IDX['dAng']]),
        ('lOff',    lambda r: r[IDX['lOff']]),
        ('lInt',    lambda r: r[IDX['lInt']]),
        ('ovr',     lambda r: r[IDX['ovr']]),
        ('tq',      lambda r: r[IDX['tq']]),
        ('p4Tau',   lambda r: r[IDX['p4Tau']]),
    ]

    print()
    print('=' * 130)
    print(f'WAVEFORM — event {nearest_ev_i} @ t_apex={ev["t_apex"]:.1f}s ({ev["magnitude_class"]}, '
          f'{ev["mean_v_mph"]:.1f} mph, {ev["duration_sec"]:.1f}s)')
    print('=' * 130)
    print(f'  {"t_rel":>7} ' + ' '.join(f'{n:>10}' for n, _ in fields_to_print))
    print('  ' + '-' * 130)
    for i, t in enumerate(t_rel):
        vals = [fn(sub[i]) for _, fn in fields_to_print]
        print(f'  {t:>+7.2f} ' + ' '.join(f'{v:>+10.5f}' for v in vals))

    # Summary metrics
    print()
    cmd_arr = sub[:, IDX['cmd']]
    des_arr = sub[:, IDX['des']]
    aLat_arr = sub[:, IDX['aLat']]
    ang_arr = sub[:, IDX['ang']]
    ovr_arr = sub[:, IDX['ovr']]
    dt_avg = float(np.mean(np.diff(t_rel)))
    print(f'  SUMMARY of event window (±{args.window}s, {len(sub)} samples, avg dt={dt_avg*1000:.1f}ms):')
    print(f'    peak |cmd|:        {np.max(np.abs(cmd_arr)):.5f}')
    print(f'    peak |des|:        {np.max(np.abs(des_arr)):.5f}')
    print(f'    peak |aLat|:       {np.max(np.abs(aLat_arr)):.3f} m/s²')
    print(f'    aLat RMS:          {np.sqrt(np.mean(aLat_arr**2)):.3f} m/s²')
    if len(aLat_arr) > 2:
        jerk = np.diff(aLat_arr) / np.diff(t_rel + ev['t_apex'])
        print(f'    jerk RMS:          {np.sqrt(np.mean(jerk**2)):.3f} m/s³')
        print(f'    jerk peak:         {np.max(np.abs(jerk)):.3f} m/s³')
    if len(ang_arr) > 2:
        ang_rate = np.diff(ang_arr) / np.diff(t_rel + ev['t_apex'])
        print(f'    steering rate peak: {np.max(np.abs(ang_rate)):.2f} deg/s')
        print(f'    steering rate RMS:  {np.sqrt(np.mean(ang_rate**2)):.2f} deg/s')
    # Direction reversals in cmd
    cmd_rate = np.diff(cmd_arr) / np.diff(t_rel + ev['t_apex'])
    cmd_sign_changes = int(np.sum(np.diff(np.sign(cmd_rate[np.abs(cmd_rate) > 4e-5])) != 0))
    duration = float(t_rel[-1] - t_rel[0])
    print(f'    cmd direction reversals: {cmd_sign_changes} in {duration:.1f}s = {cmd_sign_changes/duration:.2f} cps')
    print(f'    overrides:         {int((ovr_arr > 0.5).sum())} samples')


if __name__ == '__main__':
    main()
