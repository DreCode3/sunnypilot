#!/usr/bin/env python3
"""CD210 (route_b1) vs OPM7 baseline — full A/B across ALL curve magnitudes + straights.
Same carcontroller config on both; only the driving model changed -> isolates the MODEL's effect.
Metrics: apex-cutting A (camera lane offset; root-cause), des_cps (MODEL desiredCurvature crossings =
the 'softer steering' signal), cmd_cps (post-controller), alat_band (oscillation), jerk (comfort),
override, + straight-line steadiness. Parallel over routes (M4 Max)."""
import sys, glob, json, math
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.analyze_curves_v2 import load_cx1, IDX, detect_curve_events
from explorer_st_logs.phase1_validation import per_event_metrics, route_prefix_for, crossings_per_sec
from explorer_st_logs.discriminator_run import load_modelv2

CD210 = ['route_b1']
OPM7  = ['route_aa', 'route_ab', 'route_ac', 'route_ad', 'route_af', 'route_b0']  # recent, same config, OPM7
MAG = ('gentle', 'moderate', 'sharp')


def signal_cps(arr, t, ev, idx, w=3.0, fs=20.0, thresh=4e-5):
    lo, hi = ev['t_apex'] - w, ev['t_apex'] + w
    m = (t >= lo) & (t <= hi)
    if m.sum() < 5:
        return float('nan')
    ti = t[m]; si = arr[m, idx]
    if ti[-1] - ti[0] <= 0:
        return float('nan')
    n = int((ti[-1] - ti[0]) * fs) + 1
    tu = ti[0] + np.arange(n) / fs
    return crossings_per_sec(np.interp(tu, ti, si), 1.0 / fs, threshold=thresh)


def analyze(rid):
    prefix = route_prefix_for(rid)
    if not prefix:
        return None
    try:
        arr, t = load_cx1(prefix)
    except Exception:
        return None
    if arr is None or len(arr) < 300:
        return None
    mv = load_modelv2(prefix)
    cmd = arr[:, IDX['cmd']]; des = arr[:, IDX['des']]; v = arr[:, IDX['v']]
    ovr = arr[:, IDX['ovr']]; aLat = arr[:, IDX['aLat']]
    st = (np.abs(cmd) < 5e-4) & (v > 15) & (ovr < 0.5)
    straight = {'n': int(st.sum()),
                'des_std': float(np.std(des[st])) if st.sum() > 50 else float('nan'),
                'des_cps_global': float(crossings_per_sec(des, np.median(np.diff(t)) or 0.05)),
                'spd_med': float(np.median(v[st]) * 2.237) if st.sum() > 50 else float('nan')}
    drive_min = float((t[-1] - t[0]) / 60.0)
    events = detect_curve_events(arr, t, require_dir_match_pct=0.0)
    rows = []
    for e in events:
        m = per_event_metrics(arr, t, e)
        if not m:
            continue
        A = Ac = float('nan')
        if mv is not None and len(mv['t']) > 1:
            A = float(np.interp(e['t_apex'], mv['t'], mv['A']))
            Ac = float(np.interp(e['t_apex'], mv['t'], mv['conf']))
        rows.append(dict(route=rid, dir=e['direction'], mclass=e['magnitude_class'],
                         speed=m['mean_speed_mph'], alat_band=m['alat_band_0_5_3'], jerk=m['jerk_rms'],
                         cmd_cps=m['cmd_cps'], des_cps=signal_cps(arr, t, e, IDX['des']),
                         A_apex=A, A_conf=Ac, aLat_apex=float(aLat[e['apex_idx']]),
                         had_override=bool(e['had_override'])))
    return dict(rid=rid, rows=rows, straight=straight, drive_min=drive_min)


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else float('nan')


def confok(r):
    return not math.isnan(r.get('A_conf', float('nan'))) and r['A_conf'] > 0.6


def slope_A_aLat(rows):
    x = np.array([r['aLat_apex'] for r in rows if confok(r)])
    y = np.array([r['A_apex'] for r in rows if confok(r)])
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 8:
        return None
    g, b = np.polyfit(x[m], y[m], 1)
    return float(g), int(m.sum())


def main():
    routes = CD210 + OPM7
    res = {}
    with cf.ProcessPoolExecutor(max_workers=min(len(routes), 16)) as ex:
        for rid, r in zip(routes, ex.map(analyze, routes)):
            res[rid] = r
            print(f'  {rid}: {len(r["rows"]) if r else 0} events, {r["drive_min"]:.1f} min' if r else f'  {rid}: skip', flush=True)

    def group(rids):
        rows = []; dmin = 0.0; straights = []
        for rid in rids:
            if res.get(rid):
                rows += res[rid]['rows']; dmin += res[rid]['drive_min']; straights.append(res[rid]['straight'])
        return rows, dmin, straights

    cd_rows, cd_min, cd_st = group(CD210)
    op_rows, op_min, op_st = group(OPM7)

    print('\n' + '=' * 100)
    print('CD210 (route_b1) vs OPM7 baseline  — same carcontroller; model-only A/B')
    print('=' * 100)
    print(f'  events: CD210 {len(cd_rows)} ({cd_min:.0f} min) | OPM7 {len(op_rows)} ({op_min:.0f} min)')
    print(f'  speed (mph) median[IQR]: CD210 {med([r["speed"] for r in cd_rows]):.0f} | OPM7 {med([r["speed"] for r in op_rows]):.0f}  (match check)')

    print('\n--- BY MAGNITUDE: A_apex(+=right/inside-cut), alat_band(osc), jerk, cmd_cps, des_cps(model hunting) ---')
    print(f'  {"grp":<6}{"mag":<9}{"N":>4}{"A_apex":>9}{"alat_band":>10}{"jerk":>7}{"cmd_cps":>8}{"des_cps":>8}')
    for gname, rows in [('CD210', cd_rows), ('OPM7', op_rows)]:
        for mg in MAG:
            sub = [r for r in rows if r['mclass'] == mg]
            subA = [r for r in sub if confok(r)]
            if not sub:
                continue
            print(f'  {gname:<6}{mg:<9}{len(sub):>4}{med([r["A_apex"] for r in subA]):>9.3f}'
                  f'{med([r["alat_band"] for r in sub]):>10.1f}{med([r["jerk"] for r in sub]):>7.2f}'
                  f'{med([r["cmd_cps"] for r in sub]):>8.2f}{med([r["des_cps"] for r in sub]):>8.2f}')

    print('\n--- SPEED-MATCHED (moderate+sharp), by direction: A_apex | alat_band | des_cps ---')
    for d in ('left', 'right'):
        print(f'  direction={d}')
        for lo, hi in [(20, 35), (35, 45), (45, 55)]:
            row = f'    {lo}-{hi}mph '
            for gname, rows in [('CD210', cd_rows), ('OPM7', op_rows)]:
                sub = [r for r in rows if r['dir'] == d and r['mclass'] in ('moderate', 'sharp') and lo <= r['speed'] < hi]
                subA = [r for r in sub if confok(r)]
                row += f'| {gname} N={len(sub)} A={med([r["A_apex"] for r in subA]):+.2f} osc={med([r["alat_band"] for r in sub]):.0f} desC={med([r["des_cps"] for r in sub]):.2f} '
            print(row)

    print('\n--- APEX-CUTTING SCALING (A ~ aLat slope; lean common-mode cancels in the CD210-vs-OPM7 delta) ---')
    for gname, rows in [('CD210', cd_rows), ('OPM7', op_rows)]:
        s = slope_A_aLat([r for r in rows if r['mclass'] in MAG])
        print(f'  {gname}: A~aLat slope = {s[0]:+.3f} (n={s[1]})' if s else f'  {gname}: insufficient')

    print('\n--- HUNTING on STRAIGHTS (model des steadiness) ---')
    for gname, sts in [('CD210', cd_st), ('OPM7', op_st)]:
        print(f'  {gname}: des_std={med([s["des_std"] for s in sts]):.5f}  des_cps_global={med([s["des_cps_global"] for s in sts]):.2f}  straight_spd={med([s["spd_med"] for s in sts]):.0f}mph')

    print('\n--- OVERRIDE rate (curve-event overrides / drive-min) ---')
    for gname, rows, dmin in [('CD210', cd_rows, cd_min), ('OPM7', op_rows, op_min)]:
        nov = sum(1 for r in rows if r['had_override'])
        print(f'  {gname}: {nov} override-events / {dmin:.0f} min = {nov/max(dmin,1):.3f}/min  (right: {sum(1 for r in rows if r["had_override"] and r["dir"]=="right")}, left: {sum(1 for r in rows if r["had_override"] and r["dir"]=="left")})')

    json.dump({'CD210': cd_rows, 'OPM7': op_rows}, open('explorer_st_logs/cd210_compare.json', 'w'), default=str)
    print('\nsaved -> explorer_st_logs/cd210_compare.json')


if __name__ == '__main__':
    main()
