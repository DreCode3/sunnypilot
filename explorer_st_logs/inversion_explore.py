#!/usr/bin/env python3
"""FOUNDATIONAL CHECK for the control-law inversion — purely arithmetic on logged values, NO modeling/conjecture.
Tests whether the inversion is even valid before building any predictor.

From LC lines (carcontroller.py:360-372): curv = apply_curvature = model_desired + pi_p + pi_i, with
P=pi_p=lc_kp*off, I=pi_i=lc_ki*int. So model_desired = curv - P - I (the smoothed MODEL command).

Questions (all answerable by arithmetic, no assumptions):
 1. DECOMPOSITION: on straights, how big is the MODEL command (curv-P-I) vs the PI trim (P+I)? If the model
    dominates, "disturbance c* = P+I" is FALSE and the simple inversion is invalid.
 2. INTEGRAL DEMAND: for GOLD where the integral is UNSATURATED and settled (off~0), what integral value is
    actually needed to center? How often would that needed value exceed WEAK's 0.30 cap (=> weak shortfall)?
 3. AUTHORITY SHORTFALL: the I-term curvature weak physically cannot deliver = lc_ki*(needed_int - 0.30) where
    needed_int>0.30. This is provable (logged), independent of any plant model.
Straight gate: |curv|<0.005 (the integral-accumulation gate). Steady gate (for #2): compare consecutive LC samples."""
import sys, glob, os, json, re
import numpy as np
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
PAT = re.compile(r'off=(-?[\d.]+) ll=(-?[\d.]+) pos=(-?[\d.]+) scl=([\d.]+) conf=([\d.]+) wid=([\d.]+) '
                 r'int=(-?[\d.]+) P=(-?[\d.eE+-]+) I=(-?[\d.eE+-]+) curv=(-?[\d.]+) spd=(-?[\d.]+)')


def scan(rid):
    rows = []
    for sd in sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/')):
        f = sd + 'rlog.zst'
        if not os.path.exists(f): continue
        try:
            for msg in LogReader(f):
                if msg.which() != 'logMessage': continue
                s = msg.logMessage
                if 'LC:' not in s: continue
                try: txt = json.loads(s).get('msg', '')
                except Exception: txt = s
                m = PAT.search(txt)
                if m: rows.append(tuple(map(float, m.groups())))
        except Exception: continue
    return rid, np.array(rows) if rows else np.empty((0, 11))


allr = [r for v in GROUPS.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    res = dict(ex.map(scan, allr))

for label, rids in GROUPS.items():
    A = np.vstack([res[r] for r in rids if len(res[r])])
    off, ll, pos, scl, conf, wid, integ, P, I, curv, spd = [A[:, i] for i in range(11)]
    straight = np.abs(curv) < 0.005
    o, lc, ii, pp, iv, cv = off[straight], ll[straight], integ[straight], P[straight], I[straight], curv[straight]
    model_cmd = cv - pp - iv          # the smoothed MODEL command
    pi_trim = pp + iv                 # the PI contribution
    print(f'\n===== {label}: {straight.sum()} straight LC samples =====')
    print(f'  |MODEL cmd|=|curv-P-I| median {np.median(np.abs(model_cmd)):.6f}  vs  |PI trim|=|P+I| median {np.median(np.abs(pi_trim)):.6f}'
          f'   ratio model/PI = {np.median(np.abs(model_cmd))/np.median(np.abs(pi_trim)):.1f}x')
    print(f'  => PI is {100*np.median(np.abs(pi_trim))/np.median(np.abs(cv)+1e-9):.0f}% of the total command (rest is model)')
    # fraction where PI and model agree in sign (PI reinforcing) vs oppose
    print(f'  PI vs model same-sign: {100*np.mean(np.sign(pi_trim)==np.sign(model_cmd)):.0f}% (if PI opposes model, it is trimming model bias)')

# INTEGRAL DEMAND (GOLD): where integral is settled & unsaturated, what int centers the car?
A = res['route_b8']
off, ll, pos, scl, conf, wid, integ, P, I, curv, spd = [A[:, i] for i in range(11)]
straight = np.abs(curv) < 0.005
# settled: |off| small (well-centered) AND not at cap -> int value reflects the trim needed to hold center
cap_g = np.interp(spd, [20., 30.], [0.30, 1.0])
unsat = np.abs(integ) < 0.95*cap_g
centered = np.abs(off) < 0.05
m = straight & unsat & centered
print(f'\n===== GOLD INTEGRAL DEMAND (straight, unsaturated, |off|<0.05 = integrator did its job) =====')
print(f'  n={m.sum()}  needed |int| to center: median {np.median(np.abs(integ[m])):.3f}  '
      f'p75 {np.percentile(np.abs(integ[m]),75):.3f}  p90 {np.percentile(np.abs(integ[m]),90):.3f}  max {np.max(np.abs(integ[m])):.3f}')
print(f'  frac of these where needed |int| > 0.30 (WEAK cap) => WEAK would be SHORTED: {100*np.mean(np.abs(integ[m])>0.30):.0f}%')
print(f'  frac > 0.50: {100*np.mean(np.abs(integ[m])>0.50):.0f}%   (weak short by lc_ki*(need-0.30) curvature it cannot deliver)')
# authority shortfall in curvature units where need>0.30
need = np.abs(integ[m]); short = need[need>0.30]
if len(short):
    print(f'  median curvature SHORTFALL weak cannot deliver = lc_ki*(need-0.30) = {0.0002*np.median(short-0.30):.6f} 1/m '
          f'(over the {len(short)} shorted samples)')
print('\nINTERPRETATION: if model dominates the command (#1), the disturbance is NOT P+I and a closed-form')
print('off-prediction needs the model+plant offset-response (NOT in logs) => that step would be CONJECTURE.')
print('What IS provable w/o a model: the integral-authority shortfall (#2,#3) + the location-paired offset (separate).')
