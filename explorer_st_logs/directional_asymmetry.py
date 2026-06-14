#!/usr/bin/env python3
"""Test the user's directional-asymmetry hypothesis (current config, iter3 NOT active):
  H1: LEFT curves oscillate / ping-pong more than RIGHT  (metrics: alat_band, alat_cps, hunting crossings_per_sec)
  H2: RIGHT curves oversteer more than LEFT              (metrics: overshoot_pct, peak_bias_pct, override rate)
Speed-matched (controls for speed confound). Pools all same-config routes for N.
Also isolates the two user-named curves and shows them by heading/direction.
"""
import sys, json
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.analyze_curves_v2 import load_cx1, detect_curve_events, IDX  # noqa
from explorer_st_logs.phase1_validation import per_event_metrics, route_prefix_for
from explorer_st_logs.find_labeled_event import pull_gps_and_signals, haversine_m

ROUTES = ['route_ab', 'route_ac', 'route_ad', 'route_af', 'route_b0', 'route_aa', 'route_9b']
SHARP = ('moderate', 'sharp')


def load_route(rid):
    prefix = route_prefix_for(rid)
    if not prefix:
        return []
    try:
        arr, t = load_cx1(prefix)
    except Exception as e:
        print(f'  {rid}: load error {e}', flush=True); return []
    if arr is None or len(arr) < 100:
        return []
    events = detect_curve_events(arr, t)
    try:
        cj = {round(e['metadata']['t_apex'], 2): e
              for e in json.load(open(f'explorer_st_logs/{rid}_curves_v2.json'))['events']}
    except Exception:
        cj = {}
    try:
        gps = pull_gps_and_signals(prefix)
    except Exception:
        gps = None
    rows = []
    for ev in events:
        m = per_event_metrics(arr, t, ev)
        if not m:
            continue
        ce = cj.get(round(ev['t_apex'], 2), {})
        cm = ce.get('metrics', {}); hu = ce.get('hunting', {})
        lat = lon = None
        if gps and len(gps['t']) > 1:
            lat = float(np.interp(ev['t_apex'], gps['t'], gps['lat']))
            lon = float(np.interp(ev['t_apex'], gps['t'], gps['lon']))
        rows.append(dict(route=rid, direction=ev['direction'], mclass=ev['magnitude_class'],
                         speed=m['mean_speed_mph'], alat_band=m['alat_band_0_5_3'], alat_cps=m['alat_cps'],
                         cmd_cps=m['cmd_cps'], jerk=m['jerk_rms'],
                         overshoot_pct=cm.get('overshoot_pct'), peak_bias_pct=cm.get('peak_bias_pct'),
                         crossings=hu.get('crossings_per_sec'), had_override=bool(ev.get('had_override', False)),
                         lat=lat, lon=lon, peak_cmd=float(ev['peak_cmd'])))
    return rows


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float('nan')


ALL = []
for r in ROUTES:
    rows = load_route(r); ALL += rows; print(f'{r}: {len(rows)} events', flush=True)

ms = [x for x in ALL if x['mclass'] in SHARP]
L = [x for x in ms if x['direction'] == 'left']
R = [x for x in ms if x['direction'] == 'right']

print('\n' + '=' * 92)
print('LEFT vs RIGHT — moderate+sharp, all same-config routes pooled (N_L=%d, N_R=%d)' % (len(L), len(R)))
print('=' * 92)
print(f'{"metric":<26}{"LEFT":>16}{"RIGHT":>16}   hypothesis')
rows_spec = [
    ('alat_band (oscillation)', 'alat_band', 'H1: L>R'),
    ('alat_cps', 'alat_cps', 'H1: L>R'),
    ('hunting crossings/s', 'crossings', 'H1: L>R'),
    ('jerk_rms', 'jerk', 'H1: L>R'),
    ('overshoot_pct', 'overshoot_pct', 'H2: R>L'),
    ('peak_bias_pct', 'peak_bias_pct', '—'),
    ('mean speed mph', 'speed', 'confound check'),
]
for name, key, hyp in rows_spec:
    print(f'{name:<26}{med([x[key] for x in L]):>16.3f}{med([x[key] for x in R]):>16.3f}   {hyp}')
lo_ov = sum(1 for x in L if x['had_override']) / len(L) if L else float('nan')
ro_ov = sum(1 for x in R if x['had_override']) / len(R) if R else float('nan')
print(f'{"override fraction":<26}{lo_ov:>16.2f}{ro_ov:>16.2f}   H2: R>L')

print('\n' + '=' * 92)
print('SPEED-MATCHED LEFT vs RIGHT (controls for speed confound)')
print('=' * 92)
print(f'{"bin":<10}{"L: N  alat_band  ovshoot  cross":<40}{"R: N  alat_band  ovshoot  cross":<40}')
for lo, hi in [(20, 35), (35, 45), (45, 55)]:
    Lb = [x for x in L if lo <= x['speed'] < hi]; Rb = [x for x in R if lo <= x['speed'] < hi]
    ls = f'N={len(Lb)} alat={med([x["alat_band"] for x in Lb]):.1f} ov={med([x["overshoot_pct"] for x in Lb]):.3f} cr={med([x["crossings"] for x in Lb]):.2f}'
    rs = f'N={len(Rb)} alat={med([x["alat_band"] for x in Rb]):.1f} ov={med([x["overshoot_pct"] for x in Rb]):.3f} cr={med([x["crossings"] for x in Rb]):.2f}'
    print(f'{lo}-{hi}mph  {ls:<40}{rs:<40}')


def near(x, lat, lon, m=150):
    return x['lat'] is not None and haversine_m(lat, lon, x['lat'], x['lon']) < m


for label, clat, clon in [('33.90032,-84.59893  (user: EB/left worse oscillation; WB/right better)', 33.90032, -84.59893),
                          ('33.89518,-84.58368  (user: right oversteers, crosses centerline, override)', 33.89518, -84.58368)]:
    print('\n' + '=' * 92)
    print('NAMED CURVE ' + label)
    print('=' * 92)
    hits = sorted([x for x in ALL if near(x, clat, clon)], key=lambda x: -(x['alat_band'] or 0))
    if not hits:
        print('  (no events within 150 m — curve may not have triggered detection on these drives)')
    print(f'  {"route":<9}{"dir":>6}{"spd":>5}{"alat_band":>11}{"overshoot":>11}{"peak_bias":>11}{"cross":>7}{"ovr":>5}')
    for x in hits:
        print(f'  {x["route"]:<9}{x["direction"]:>6}{x["speed"]:>5.0f}{(x["alat_band"] or float("nan")):>11.1f}'
              f'{(x["overshoot_pct"] if x["overshoot_pct"] is not None else float("nan")):>11.3f}'
              f'{(x["peak_bias_pct"] if x["peak_bias_pct"] is not None else float("nan")):>11.3f}'
              f'{(x["crossings"] if x["crossings"] is not None else float("nan")):>7.2f}{str(x["had_override"]):>5}')

json.dump(ALL, open('explorer_st_logs/all_events_directional.json', 'w'), indent=1, default=str)
print(f'\nSaved {len(ALL)} events -> explorer_st_logs/all_events_directional.json')
