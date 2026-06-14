#!/usr/bin/env python3
"""Discriminator analysis (corrected v3 design) — Explorer ST right-curve over-turn.

Outputs (see discriminator_design.md):
  H0  per-route yaw_offset (meas on cmd-based straights) — measurement bias.
  H2  plant-gain regression: dCE = Δψ_GPS − Δψ_cmd  vs  Δψ_cmd  (slope=gain; intercept=additive plant bias).
  L/R symmetry of dCE + the lOff drift symptom.
  Road-independence: distinct right bends (GPS-clustered) showing rightward drift.
  Banking: regress lOff drift on road-bank measured on the APPROACH straight (per-route zeroed).
H1 (model/perception) is NOT positively identified here — only by-elimination (plant faithful + drift).
Usage: discriminator_run.py [route_xx route_yy ...]   (default: all CX1 routes)
"""
import sys, os, glob, json, math
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from explorer_st_logs.analyze_curves_v2 import load_cx1, IDX, detect_curve_events
from openpilot.tools.lib.logreader import LogReader

LAG = 0.30          # global cmd->execution lag (s); integral (endpoint) form is delay-robust anyway
SHARP = ('moderate', 'sharp')


def cx1_routes():
    out = []
    for d in sorted(glob.glob('explorer_st_logs/route_*/')):
        rid = d.rstrip('/').split('/')[-1]
        segs = glob.glob(d + '000000*--*/')
        if not segs:
            continue
        out.append((rid, sorted(segs)[0].rstrip('/').rsplit('--', 1)[0]))
    return out


def load_gps(prefix):
    segs = sorted(glob.glob(prefix + '--*'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    t = []; lat = []; lon = []; vN = []; vE = []; roll = []
    for sd in segs:
        f = os.path.join(sd, 'rlog.zst')
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                if msg.which() != 'liveLocationKalman':
                    continue
                llk = msg.liveLocationKalman
                if not (llk.positionGeodetic.valid and llk.velocityNED.valid):
                    continue
                p = llk.positionGeodetic.value; v = llk.velocityNED.value; o = llk.orientationNED.value
                t.append(msg.logMonoTime * 1e-9); lat.append(p[0]); lon.append(p[1])
                vN.append(v[0]); vE.append(v[1]); roll.append(o[0])
        except Exception:
            continue  # corrupt segment — skip
    if len(t) < 100:
        return None
    t = np.array(t); o = np.argsort(t)
    g = dict(t=t[o], lat=np.array(lat)[o], lon=np.array(lon)[o],
             vN=np.array(vN)[o], vE=np.array(vE)[o], roll=np.array(roll)[o])
    g['psi'] = np.unwrap(np.arctan2(g['vE'], g['vN']))   # +=clockwise(right) heading
    g['spd'] = np.hypot(g['vN'], g['vE'])
    return g


def load_modelv2(prefix):
    """Metric A — camera lane-line lateral offset, available on ALL routes (modelV2 always runs).
    Convention VERIFIED on data: laneLines[1].y (left) <0, laneLines[2].y (right) >0 (positive-RIGHT build).
    car-offset-RIGHT = -(left_y[0] + right_y[0])/2  (midpoint>0 => car LEFT of center, so negate => +=RIGHT)."""
    segs = sorted(glob.glob(prefix + '--*'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    t = []; A = []; conf = []
    for sd in segs:
        f = os.path.join(sd, 'rlog.zst')
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                if msg.which() != 'modelV2':
                    continue
                m = msg.modelV2
                ll = m.laneLines; lp = m.laneLineProbs
                if len(ll) < 4 or len(lp) < 4 or len(ll[1].y) < 1:
                    continue
                A.append(-(ll[1].y[0] + ll[2].y[0]) / 2.0)   # +=car RIGHT of center
                conf.append(min(lp[1], lp[2]))
                t.append(msg.logMonoTime * 1e-9)
        except Exception:
            continue
    if len(t) < 50:
        return None
    t = np.array(t); o = np.argsort(t)
    return dict(t=t[o], A=np.array(A)[o], conf=np.array(conf)[o])


def hav(a, b, c, d):
    R = 6371000.0
    p1, p2 = math.radians(a), math.radians(c); dp = math.radians(c - a); dl = math.radians(d - b)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def analyze_route(rid, prefix):
    try:
        arr, tc = load_cx1(prefix)
    except Exception:
        return None
    if arr is None or len(arr) < 500:
        return None
    gps = load_gps(prefix)
    if gps is None:
        return None
    mv = load_modelv2(prefix)
    cmd = arr[:, IDX['cmd']]; des = arr[:, IDX['des']]; meas = arr[:, IDX['meas']]
    v = arr[:, IDX['v']]; lOff = arr[:, IDX['lOff']]; ovr = arr[:, IDX['ovr']]; aLat = arr[:, IDX['aLat']]
    # H0: yaw_offset on cmd-based straights
    st = (np.abs(cmd) < 5e-4) & (v > 15) & (ovr < 0.5)
    yaw_off = float(np.median(meas[st])) if st.sum() > 30 else float('nan')
    # per-route roll zero on GPS straights
    dpsi_dt = np.gradient(gps['psi'], gps['t'])
    gst = (np.abs(dpsi_dt) < 0.02) & (gps['spd'] > 15)
    roll_zero = float(np.median(gps['roll'][gst])) if gst.sum() > 30 else float(np.median(gps['roll']))
    events = detect_curve_events(arr, tc, require_dir_match_pct=0.0)  # C-A: strip meas-based gate
    rows = []
    for e in events:
        if e['magnitude_class'] not in SHARP:
            continue
        si, ei = e['start_idx'], e['end_idx']
        # M5: pre-override window — cut at first override onset within the event
        ov_on = np.where(ovr[si:ei + 1] > 0.5)[0]
        end_i = ei
        cut = len(ov_on) > 0
        if cut:
            end_i = si + int(ov_on[0]) - 1
        if end_i <= si + 2:
            continue  # too short after cut
        w = slice(si, end_i + 1)
        dt = np.gradient(tc[si:end_i + 1])
        dpsi_cmd = float(np.sum(cmd[w] * v[w] * dt))
        dpsi_des = float(np.sum(des[w] * v[w] * dt))
        dpsi_meas = float(np.sum((meas[w] - yaw_off) * v[w] * dt)) if not math.isnan(yaw_off) else float('nan')
        t0, t1 = tc[si] + LAG, tc[end_i] + LAG
        if t1 <= gps['t'][0] or t0 >= gps['t'][-1]:
            continue
        dpsi_gps = float(np.interp(t1, gps['t'], gps['psi']) - np.interp(t0, gps['t'], gps['psi']))
        lOff_apex = float(lOff[e['apex_idx']]); lOff_start = float(lOff[si])
        ts = e['t_start']; ta = e['t_apex']
        am = (gps['t'] >= ts - 3.0) & (gps['t'] <= ts - 0.5)
        bank = (float(np.median(gps['roll'][am])) - roll_zero) if am.sum() > 3 else float('nan')
        # Metric A (camera lane-line offset, +=car RIGHT of center) + banking inputs
        A_apex = float('nan'); A_conf = float('nan')
        if mv is not None and len(mv['t']) > 1:
            A_apex = float(np.interp(ta, mv['t'], mv['A']))
            A_conf = float(np.interp(ta, mv['t'], mv['conf']))
        aLat_apex = float(aLat[e['apex_idx']])
        roll_apex = float(np.interp(ta + LAG, gps['t'], gps['roll']))
        rows.append(dict(
            route=rid, dir=e['direction'], mclass=e['magnitude_class'], v=float(e['mean_v_mph']),
            dpsi_cmd=dpsi_cmd, dpsi_des=dpsi_des, dpsi_gps=dpsi_gps, dpsi_meas=dpsi_meas,
            dCE=dpsi_gps - dpsi_cmd, lOff_apex=lOff_apex, lOff_drift=lOff_apex - lOff_start,
            A_apex=A_apex, A_conf=A_conf, aLat_apex=aLat_apex, roll_apex=roll_apex,
            bank=bank, had_override=bool(e['had_override']), override_cut=cut,
            lat=float(np.interp(ta, gps['t'], gps['lat'])), lon=float(np.interp(ta, gps['t'], gps['lon'])),
            peak_cmd=float(e['peak_cmd'])))
    return dict(rid=rid, yaw_off=yaw_off, roll_zero=roll_zero, rows=rows)


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(xs)) if xs else float('nan')


def fit(x, y):
    x = np.array(x); y = np.array(y); m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 5:
        return None
    g, b = np.polyfit(x, y, 1)
    resid = y - (g * x + b); n = len(x)
    sxx = np.sum((x - x.mean()) ** 2)
    se_g = float(np.sqrt(np.sum(resid ** 2) / (n - 2) / sxx)) if sxx > 0 else float('nan')
    se_b = float(np.sqrt(np.sum(resid ** 2) / (n - 2) * (1 / n + x.mean() ** 2 / sxx))) if sxx > 0 else float('nan')
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float('nan')
    return dict(slope=float(g), se_slope=se_g, intercept=float(b), se_intercept=se_b, r2=r2, n=n)


def cluster(rows, radius_m=80):
    centers = []  # (lat, lon)
    for r in rows:
        assigned = False
        for ci, (la, lo) in enumerate(centers):
            if hav(la, lo, r['lat'], r['lon']) < radius_m:
                r['bend'] = ci; assigned = True; break
        if not assigned:
            r['bend'] = len(centers); centers.append((r['lat'], r['lon']))
    return len(centers)


def _worker(rid, pref):
    try:
        return analyze_route(rid, pref)
    except Exception:
        return None


def main():
    routes = sys.argv[1:] or [r for r, _ in cx1_routes()]
    rmap = dict(cx1_routes())
    tasks = [(rid, rmap[rid]) for rid in routes if rid in rmap]
    allrows = []; routeinfo = []
    nproc = min(len(tasks), max(1, (os.cpu_count() or 4)))
    print(f'[parallel] analyzing {len(tasks)} routes across {nproc} worker processes', flush=True)
    with cf.ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(_worker, rid, pref): rid for rid, pref in tasks}
        for fut in cf.as_completed(futs):
            rid = futs[fut]
            res = fut.result()
            if not res:
                print(f'  {rid}: skipped (no cx1/gps)', flush=True); continue
            routeinfo.append((rid, res['yaw_off'], res['roll_zero'], len(res['rows'])))
            allrows.extend(res['rows'])
            print(f'  {rid}: {len(res["rows"])} m+s events, yaw_off={res["yaw_off"]:+.5f}', flush=True)

    # GPS sign sanity (verify-don't-assume): dpsi_gps should track dpsi_cmd
    ratio = med([r['dpsi_gps'] / r['dpsi_cmd'] for r in allrows if abs(r['dpsi_cmd']) > 0.02])
    print(f'\n[SIGN CHECK] median(dpsi_gps/dpsi_cmd) = {ratio:+.2f}  (should be >0; ~1 if GPS tracks cmd)')

    print('\n' + '=' * 96)
    print('H0 — yaw_offset (meas on straights, +=right). Expect ~+0.00035 if constant sensor bias')
    print('=' * 96)
    yoffs = [y for _, y, _, _ in routeinfo if not math.isnan(y)]
    print(f'  per-route median range: {min(yoffs):+.5f} .. {max(yoffs):+.5f} ; overall median {med(yoffs):+.5f} (n={len(yoffs)} routes)')

    from collections import defaultdict
    rl = defaultdict(list)
    for r in allrows:
        rl[r['route']].append(r['lOff_apex'])
    dead = {rt for rt, vs in rl.items() if sum(1 for x in vs if x == 0.0) / len(vs) > 0.5}
    live = [r for r in allrows if r['route'] not in dead]
    print(f'\n[lOff DEAD routes (excluded from lOff-only stats): {len(dead)} of {len(rl)}] {sorted(dead)}')

    def conf_ok(r):
        return not math.isnan(r.get('A_conf', float('nan'))) and r['A_conf'] > 0.6
    L = [r for r in allrows if r['dir'] == 'left']; R = [r for r in allrows if r['dir'] == 'right']
    LA = [r for r in L if conf_ok(r)]; RA = [r for r in R if conf_ok(r)]

    print('\n' + '=' * 96)
    print(f'L/R SYMMETRY — moderate+sharp (N_L={len(L)}, N_R={len(R)})')
    print('=' * 96)
    print(f'  {"A_apex (+=RIGHT) [METRIC A, all routes, conf>0.6]":<46} L={med([r["A_apex"] for r in LA]):+.4f}  R={med([r["A_apex"] for r in RA]):+.4f}  (N_L={len(LA)},N_R={len(RA)})')
    print(f'  {"dCE (Δψ_GPS−Δψ_cmd) [GPS, all routes]":<46} L={med([r["dCE"] for r in L]):+.4f}  R={med([r["dCE"] for r in R]):+.4f}')
    print(f'  {"lOff_apex (+=left) [live routes only]":<46} L={med([r["lOff_apex"] for r in L if r["route"] not in dead]):+.4f}  R={med([r["lOff_apex"] for r in R if r["route"] not in dead]):+.4f}')

    print('\n' + '=' * 96)
    print('H2 — PLANT-GAIN regression  dCE = g·Δψ_cmd + b  (slope=gain; intercept=additive plant bias; H1 INVISIBLE here)')
    print('=' * 96)
    for nm, sub in [('RIGHT', R), ('LEFT', L), ('ALL', allrows)]:
        fres = fit([r['dpsi_cmd'] for r in sub], [r['dCE'] for r in sub])
        if fres:
            print(f'  {nm:<6} n={fres["n"]:3d}  slope={fres["slope"]:+.3f}±{fres["se_slope"]:.3f}  intercept={fres["intercept"]:+.5f}±{fres["se_intercept"]:.5f}  R²={fres["r2"]:.2f}')

    cluster(allrows)
    RAb = defaultdict(list)
    for r in RA:
        RAb[r['bend']].append(r['A_apex'])
    print('\n' + '=' * 96)
    print('#1 METRIC A — rightward drift across distinct RIGHT bends (A_apex>+0.10m=car right; conf>0.6; ALL routes)')
    print('=' * 96)
    driftA = sum(1 for vs in RAb.values() if med(vs) > 0.10)
    print(f'  RIGHT bends (conf-ok A): {len(RAb)}; with rightward drift (median A>+0.10m): {driftA} ({100*driftA/max(len(RAb),1):.0f}%)')
    print(f'  median A_apex over conf-ok right events: {med([r["A_apex"] for r in RA]):+.3f} m (n={len(RA)})')

    print('\n' + '=' * 96)
    print('#2 A vs lOff (live routes) — do clean camera offset & controller lOff agree? (both +=right)')
    print('=' * 96)
    p2 = [(r['A_apex'], -r['lOff_apex']) for r in live if conf_ok(r) and r['lOff_apex'] != 0.0]
    if len(p2) > 5:
        a2 = np.array([p[0] for p in p2]); l2 = np.array([p[1] for p in p2])
        print(f'  corr(A_apex, -lOff_apex) = {float(np.corrcoef(a2, l2)[0,1]):+.2f}  (n={len(p2)}; +1=agree)')

    print('\n' + '=' * 96)
    print('#3 A vs GPS dCE — does camera drift track the GPS execution metric? (RIGHT, conf>0.6)')
    print('=' * 96)
    p3 = [(r['A_apex'], r['dCE']) for r in RA if not math.isnan(r['dCE'])]
    if len(p3) > 5:
        a3 = np.array([p[0] for p in p3]); c3 = np.array([p[1] for p in p3])
        print(f'  corr(A_apex, dCE) = {float(np.corrcoef(a3, c3)[0,1]):+.2f}  (n={len(p3)})')

    print('\n' + '=' * 96)
    print('#4 BANKING (H3) — isolate apex road-bank (pooled roll~aLat), then regress A-drift on road-bank (RIGHT)')
    print('=' * 96)
    bk = [(r['aLat_apex'], r['roll_apex']) for r in allrows if not math.isnan(r['roll_apex'])]
    lean = fit([p[0] for p in bk], [p[1] for p in bk]) if len(bk) > 10 else None
    if lean:
        k = lean['slope']
        print(f'  pooled body-lean: roll = {k:+.4f}·aLat {lean["intercept"]:+.4f}  (R²={lean["r2"]:.2f}, n={lean["n"]})')
        for r in RA:
            r['road_bank'] = r['roll_apex'] - k * r['aLat_apex']
        fbk = fit([r['road_bank'] for r in RA], [r['A_apex'] for r in RA])
        print(f'  A_apex vs isolated road_bank (RIGHT): {fbk}')
        print('  (slope≠0 sig ⇒ banking drives drift = H3; ≈0 ⇒ not banking)')

    print('\n' + '=' * 96)
    print('#5 PASS-TO-PASS consistency — per multi-pass RIGHT bend, within-bend A_apex spread (consistent line?)')
    print('=' * 96)
    spreads = [float(np.std(vs)) for vs in RAb.values() if len(vs) >= 2]
    if spreads:
        print(f'  {len(spreads)} multi-pass right bends; median within-bend A_apex std = {med(spreads):.3f} m (small=consistent)')

    json.dump(dict(routeinfo=routeinfo, rows=allrows), open('explorer_st_logs/discriminator_results.json', 'w'), default=str)
    print(f'\nSaved {len(allrows)} events -> explorer_st_logs/discriminator_results.json')


if __name__ == '__main__':
    main()
