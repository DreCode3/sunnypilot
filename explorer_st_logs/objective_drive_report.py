#!/usr/bin/env python3
"""Objective curve-performance report across recent drives — NO subjective inputs.

Uses the validated alat_band_0.5-3 Hz metric (phase1.per_event_metrics) and attaches
GPS (via find_labeled_event.pull_gps_and_signals) so the objectively-worst curves can
be cross-checked against subjective reports AFTER this runs.

Context: iter3 was NEVER deployed (verified 2026-06-07); all routes here ran the
current/pre-iter3 config (smooth_tau 0.12/0.04). So NEW vs BASELINE should be similar
(same config) — differences are road/speed, not tuning.
"""
import sys, json
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.analyze_curves_v2 import load_cx1, detect_curve_events, IDX  # noqa
from explorer_st_logs.phase1_validation import per_event_metrics, route_prefix_for
from explorer_st_logs.find_labeled_event import pull_gps_and_signals

NEW = ['route_ab', 'route_ac', 'route_ad', 'route_af', 'route_b0']  # current-config drives (ae empty)
BASELINE = ['route_aa', 'route_9b']                                  # pre-iter3 / validated baselines
SHARP = ('moderate', 'sharp')


def gps_at(gps, t):
    if not gps or len(gps['t']) < 2:
        return (None, None)
    return (round(float(np.interp(t, gps['t'], gps['lat'])), 6),
            round(float(np.interp(t, gps['t'], gps['lon'])), 6))


def analyze(route_id, want_gps):
    prefix = route_prefix_for(route_id)
    if not prefix:
        return []
    try:
        arr, t = load_cx1(prefix)
    except Exception as e:
        print(f'  {route_id}: load error {e}', flush=True); return []
    if arr is None or len(arr) < 100:
        return []
    events = detect_curve_events(arr, t)
    gps = None
    if want_gps:
        try:
            gps = pull_gps_and_signals(prefix)
        except Exception as e:
            print(f'  {route_id}: gps error {e}', flush=True); gps = None
    rows = []
    for ev in events:
        m = per_event_metrics(arr, t, ev)
        if not m:
            continue
        lat, lon = gps_at(gps, ev['t_apex']) if gps else (None, None)
        m.update(route=route_id, lat=lat, lon=lon, peak_cmd=float(ev['peak_cmd']),
                 direction=ev['direction'], had_override=bool(ev.get('had_override', False)),
                 t_apex=float(ev['t_apex']))
        rows.append(m)
    return rows


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float('nan')


print('Loading (re-reads CX1 + GPS; a few min)...', flush=True)
data = {}
for r in NEW:
    data[r] = analyze(r, want_gps=True); print(f'  {r}: {len(data[r])} events', flush=True)
for r in BASELINE:
    data[r] = analyze(r, want_gps=False); print(f'  {r}: {len(data[r])} events', flush=True)

print('\n' + '=' * 104)
print('PER-ROUTE — moderate+sharp curves (CURRENT config; iter3 NOT active on any of these)')
print('=' * 104)
print(f'{"route":<10}{"N_ms":>6}{"N_all":>6}{"alat_band":>12}{"jerk_rms":>10}{"cmd_cps":>9}{"alat_cps":>9}{"spd_mph":>9}')
for r in NEW + BASELINE:
    rows = data[r]; ms = [x for x in rows if x['magnitude_class'] in SHARP]
    print(f'{r:<10}{len(ms):>6}{len(rows):>6}{med([x["alat_band_0_5_3"] for x in ms]):>12.1f}'
          f'{med([x["jerk_rms"] for x in ms]):>10.3f}{med([x["cmd_cps"] for x in ms]):>9.2f}'
          f'{med([x["alat_cps"] for x in ms]):>9.2f}{med([x["mean_speed_mph"] for x in ms]):>9.1f}')

print('\n' + '=' * 104)
print('SPEED-BINNED alat_band_0.5-3 (moderate+sharp): NEW (ab-b0) vs BASELINE (aa+9b) at matched speed')
print('=' * 104)
new_ms = [x for r in NEW for x in data[r] if x['magnitude_class'] in SHARP]
base_ms = [x for r in BASELINE for x in data[r] if x['magnitude_class'] in SHARP]
print(f'{"speed bin":<14}{"NEW":<28}{"BASELINE":<28}{"delta":>10}')
for lo, hi in [(20, 35), (35, 45), (45, 55), (55, 65)]:
    n = [x['alat_band_0_5_3'] for x in new_ms if lo <= x['mean_speed_mph'] < hi]
    b = [x['alat_band_0_5_3'] for x in base_ms if lo <= x['mean_speed_mph'] < hi]
    ns = f'N={len(n)} med={med(n):.1f}' if n else 'no events'
    bs = f'N={len(b)} med={med(b):.1f}' if b else 'no events'
    d = f'{med(n) - med(b):+.1f}' if n and b else '—'
    print(f'{lo}-{hi} mph    {ns:<28}{bs:<28}{d:>10}')

print('\n' + '=' * 104)
print('OVERRIDE RATE (from curves_v2 JSON: n_overrides / drive-min)')
print('=' * 104)
for r in NEW + BASELINE:
    try:
        j = json.load(open(f'explorer_st_logs/{r}_curves_v2.json'))
        nov = len(j.get('overrides', [])); mins = j.get('t_span_sec', 0) / 60
        rate = nov / mins if mins > 0 else float('nan')
        print(f'{r:<10} {nov:>3} overrides / {mins:>6.1f} min = {rate:.3f}/min')
    except Exception as e:
        print(f'{r}: {e}')

print('\n' + '=' * 104)
print('OBJECTIVE WORST CURVES — NEW drives, moderate+sharp, ranked by alat_band_0.5-3 — with GPS')
print('=' * 104)
ranked = sorted(new_ms, key=lambda x: -x['alat_band_0_5_3'])
print(f'{"#":<3}{"route":<9}{"alat_band":>10}{"alat_cps":>9}{"jerk":>8}{"cmd_cps":>8}{"spd":>6}{"dir":>6}  GPS (lat,lon)')
for i, x in enumerate(ranked[:20], 1):
    gps = f'{x["lat"]:.5f}, {x["lon"]:.5f}' if x['lat'] is not None else 'no-gps'
    print(f'{i:<3}{x["route"]:<9}{x["alat_band_0_5_3"]:>10.1f}{x["alat_cps"]:>9.2f}{x["jerk_rms"]:>8.2f}'
          f'{x["cmd_cps"]:>8.2f}{x["mean_speed_mph"]:>6.0f}{x["direction"]:>6}  {gps}')
json.dump(ranked, open('explorer_st_logs/objective_worst_curves.json', 'w'), indent=1, default=str)
print(f'\nSaved {len(ranked)} ranked worst curves -> explorer_st_logs/objective_worst_curves.json')
print('(All on CURRENT config — iter3 was not active. Ready for subjective cross-check.)')
