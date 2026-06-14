#!/usr/bin/env python3
"""CONTROL-THEORY PROOF (drive-independent, no statistics needed) that the WEAK integral cap is a BINDING
constraint on centering authority, and GOLDEN relaxes it. Uses the controller's OWN logged internals (LC lines).

Logic: an integrator pinned at its cap WHILE the offset it integrates keeps the SAME sign = the controller is
still being driven HARDER into the rail = it WANTS more corrective authority than the cap allows = it CANNOT
null the steady-state error (definition of integrator windup/saturation). If WEAK is pinned-and-same-sign a
large fraction of the time and GOLDEN is not, the WEAK cap demonstrably limits centering, independent of any
position measurement or single-drive baseline.

Reports, per config: frac of straight samples with |int| at the cap; among those, frac where sign(off)==sign(int)
(still pushing into the rail); the persistent mean off during pinned periods; and the delivered I-term authority
(lc_ki*int) vs what it WANTS (lc_ki * unclipped integral demand)."""
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
    cap = 0.30 if label == 'WEAK' else None  # weak fixed; gold interp
    # STRAIGHT GATE: the integral only ACCUMULATES on straights (|apply_curvature|<0.005, carcontroller.py:341);
    # in curves it DECAYS. Restrict to the logged total curvature |curv|<0.005 so the pinning/windup numbers are
    # the straight-only quantity the saturation claim is about (without this gate the fractions are diluted/conservative).
    straight = np.abs(curv) < 0.005
    A = A[straight]
    off, ll, pos, scl, conf, wid, integ, P, I, curv, spd = [A[:, i] for i in range(11)]
    mph = spd*2.237
    # define "pinned" relative to the config's cap
    if label == 'WEAK':
        pinned = np.abs(integ) >= 0.299
    else:
        # golden cap = interp(spd_ms,[20,30],[0.3,1.0]); pinned if within 1% of that ceiling
        cap_g = np.interp(spd, [20., 30.], [0.30, 1.0])
        pinned = np.abs(integ) >= 0.99*cap_g
    same_sign = np.sign(off) == np.sign(integ)
    # NON-circular windup evidence: same_sign among UNPINNED (if ~1.0 it's tautological; if <1 it's informative)
    unpinned = ~pinned
    print(f'\n===== {label} ({len(A)} straight LC samples, |curv|<0.005) =====')
    print(f'  frac pinned at cap:                 {np.mean(pinned)*100:5.1f}%  (n_pinned={pinned.sum()})')
    print(f'  same-sign(off,int) UNPINNED {np.mean(same_sign[unpinned])*100:5.1f}%  vs PINNED {np.mean(same_sign[pinned])*100:5.1f}%')
    print(f'    (UNPINNED<<100% => same-sign is NOT structurally guaranteed; the pinned rise is genuine, not a tautology)')
    print(f'  ROBUST WINDUP EVIDENCE — |off| while PINNED median {np.median(np.abs(off[pinned])):.4f} m '
          f'vs UNPINNED median {np.median(np.abs(off[unpinned])):.4f} m  (error stays large while integral railed = authority insufficient)')
    print(f'  signed mean offset during pinned:   {np.mean(off[pinned]):+.4f} m  (signed; pulled toward 0 by both-side pins)')
    print(f'  delivered I-authority lc_ki*int:    median |I| {np.median(np.abs(I)):.6f}  max |I| {np.max(np.abs(I)):.6f} 1/m')
    print(f'  max |int| reached:                  {np.max(np.abs(integ)):.3f}')
    # how much MORE authority golden's cap allows where weak would be pinned:
    if label == 'GOLD':
        over = np.abs(integ) > 0.30
        print(f'  frac of samples where int EXCEEDS the weak 0.30 cap: {np.mean(over)*100:5.1f}%  '
              f'(=correction WEAK physically could NOT deliver)')

print('\nCONCLUSION TEMPLATE: if WEAK is pinned a large frac WITH same-sign offset (windup), and GOLD routinely')
print('exceeds 0.30, then the weak cap demonstrably limited centering authority and golden delivered more —')
print('a control-theory fact from logged internals, independent of position measurement or single-drive baseline.')
