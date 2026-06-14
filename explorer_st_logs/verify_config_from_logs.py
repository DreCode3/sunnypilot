#!/usr/bin/env python3
"""OBJECTIVE CONFIG PROOF from the controller's OWN logged PI internals (carlog 'LC:' lines in rlog).
The controller logs at 1Hz: off, ll, pos, scl, conf, wid, int, P, I, curv, spd.
  P = lc_kp * off   ->  lc_kp = P/off   (recover the proportional gain that ACTUALLY RAN)
  I = lc_ki * int   ->  lc_ki = I/int
  int cap behavior  ->  WEAK = fixed 0.3 ; GOLDEN = np.interp(spd,[20,30]m/s -> [0.3,1.0]) i.e. ->1.0 at >67mph
Proves (not assumes) which PI config each drive ran. No reliance on file mtime / checkpoint notes / git."""
import sys, glob, os, json, re
import numpy as np
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

GROUPS = {'WEAK(b1)': ['route_b1'], 'WEAK(b2)': ['route_b2'], 'GOLD(b8)': ['route_b8']}
PAT = re.compile(r'off=(-?[\d.]+) ll=(-?[\d.]+) pos=(-?[\d.]+) scl=([\d.]+) conf=([\d.]+) wid=([\d.]+) '
                 r'int=(-?[\d.]+) P=(-?[\d.eE+-]+) I=(-?[\d.eE+-]+) curv=(-?[\d.]+) spd=(-?[\d.]+)')


def scan(rid):
    rows = []; version = None
    for sd in sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/')):
        f = sd + 'rlog.zst'
        if not os.path.exists(f): continue
        try:
            for msg in LogReader(f):
                if msg.which() != 'logMessage': continue
                s = msg.logMessage
                if 'LC:' not in s: continue
                try:
                    j = json.loads(s); txt = j.get('msg', '');
                    if version is None: version = j.get('ctx', {}).get('version')
                except Exception:
                    txt = s
                m = PAT.search(txt)
                if m:
                    off, ll, pos, scl, conf, wid, integ, P, I, curv, spd = map(float, m.groups())
                    rows.append((off, ll, pos, scl, conf, wid, integ, P, I, curv, spd))
        except Exception:
            continue
    return rid, np.array(rows) if rows else np.empty((0, 11)), version


allr = [r for v in GROUPS.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=3) as ex:
    res = {rid: (arr, ver) for rid, arr, ver in ex.map(scan, allr)}

print('=== recovered PI gains from logged internals (lc_kp = P/off, lc_ki = I/int) ===')
print('  EXPECT: WEAK lc_kp~0.0001 / lc_ki~0.0002 / int_cap=0.30 ; GOLDEN lc_kp~0.0005 / int_cap->1.0 at hwy')
for label, rids in GROUPS.items():
    A = np.vstack([res[r][0] for r in rids if len(res[r][0])])
    if not len(A):
        print(f'\n  {label}: NO LC lines'); continue
    off, ll, pos, scl, conf, wid, integ, P, I, curv, spd = [A[:, i] for i in range(11)]
    # recover kp only where |off| big enough to be numerically meaningful (logged to 6 dp)
    okp = np.abs(off) > 0.02
    kp = P[okp] / off[okp]
    oki = np.abs(integ) > 0.02
    ki = I[oki] / integ[oki]
    print(f'\n  {label}: {len(A)} LC lines, version={res[rids[0]][1]}')
    print(f'    lc_kp = P/off  : median {np.median(kp):.6f}  IQR[{np.percentile(kp,25):.6f},{np.percentile(kp,75):.6f}]  (n={okp.sum()})')
    print(f'    lc_ki = I/int  : median {np.median(ki):.6f}  IQR[{np.percentile(ki,25):.6f},{np.percentile(ki,75):.6f}]  (n={oki.sum()})')
    # int cap behavior: max |int| reached, split by speed (mph)
    mph = spd * 2.237
    hi = mph > 60
    print(f'    max|int| overall {np.max(np.abs(integ)):.3f} ; at >60mph max|int| '
          f'{np.max(np.abs(integ[hi])) if hi.any() else float("nan"):.3f} (n_hi={hi.sum()}) '
          f'-> WEAK pins ~0.30, GOLDEN can exceed 0.30 toward 1.0')
    # fraction of integral samples at/above 0.30 vs above 0.30 (cap test)
    print(f'    frac|int|>=0.299 {np.mean(np.abs(integ)>=0.299):.2f} ; frac|int|>0.31 {np.mean(np.abs(integ)>0.31):.3f}')
