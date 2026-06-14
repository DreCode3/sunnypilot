#!/usr/bin/env python3
"""DECISIVE follow-up: is the between-era centering baseline (+0.06-0.12m in model `pos`, present even unengaged)
a PHYSICAL change (car sits differently) or a MEASUREMENT change (camera extrinsic calibration drift shifting the
`pos` zero-point)? `pos` = model laneLines, which depend on liveCalibration.rpyCalib (camera roll/pitch/YAW).
If camera YAW recalibrated between weak-era (b1/b2/b4/b5) and gold-era (b8), `pos` shifts with no real motion.

Reports per route: median rpyCalib (roll,pitch,yaw in deg), calStatus, and lane WIDTH (model wid) as a cross-check
(a yaw/scale calib change also distorts perceived lane width). ALSO proves b3-b7 PI config from LC P/off where engaged."""
import sys, glob, os, json, re
import numpy as np
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

ERA = {'WEAK-era b1': ['route_b1'], 'WEAK-era b2': ['route_b2'], 'WEAK-era b4': ['route_b4'],
       'WEAK-era b5': ['route_b5'], 'GOLD-era b8': ['route_b8']}
PAT = re.compile(r'off=(-?[\d.]+).* wid=([\d.]+) int=(-?[\d.]+) P=(-?[\d.eE+-]+) I=(-?[\d.eE+-]+) curv=(-?[\d.]+) spd=(-?[\d.]+)')


def scan(rid):
    rpy = []; calstat = []; lc = []
    for sd in sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/')):
        f = sd + 'rlog.zst'
        if not os.path.exists(f): continue
        try:
            for msg in LogReader(f):
                w = msg.which()
                if w == 'liveCalibration':
                    g = msg.liveCalibration
                    try:
                        rpy.append([float(x) for x in g.rpyCalib]); calstat.append(int(g.calStatus))
                    except Exception:
                        pass
                elif w == 'logMessage':
                    s = msg.logMessage
                    if 'LC:' in s:
                        try: txt = json.loads(s).get('msg', '')
                        except Exception: txt = s
                        m = PAT.search(txt)
                        if m:
                            off, wid, integ, P, I, curv, spd = map(float, m.groups())
                            lc.append((off, wid, integ, P, I, curv, spd))
        except Exception: continue
    return rid, (np.array(rpy) if rpy else np.empty((0, 3))), np.array(calstat), (np.array(lc) if lc else np.empty((0, 7)))


allr = [r for v in ERA.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    res = dict((rid, (rpy, cs, lc)) for rid, rpy, cs, lc in ex.map(scan, allr))

DEG = 180/np.pi
print('=== liveCalibration.rpyCalib (camera extrinsic, deg) + lane width by era ===')
print(f'{"route":<14}{"roll":>8}{"pitch":>8}{"YAW":>9}{"calStat":>9}{"n_cal":>7}{"laneWid":>9}{"lc_kp":>9}')
for label, rids in ERA.items():
    rid = rids[0]; rpy, cs, lc = res[rid]
    if len(rpy):
        med = np.median(rpy, axis=0)*DEG
        cstat = int(np.median(cs)) if len(cs) else -1
        roll, pitch, yaw = med
    else:
        roll = pitch = yaw = float('nan'); cstat = -1
    if len(lc):
        wid = np.median(lc[:, 1])
        ok = np.abs(lc[:, 0]) > 0.02  # |off|>0.02 for kp recovery
        kp = np.median(lc[ok, 3]/lc[ok, 0]) if ok.sum() > 5 else float('nan')
    else:
        wid = float('nan'); kp = float('nan')
    print(f'  {label:<12}{roll:>8.3f}{pitch:>8.3f}{yaw:>9.3f}{cstat:>9}{len(rpy):>7}{wid:>9.3f}{kp:>9.5f}')

print('\nINTERPRETATION:')
print('  - If GOLD-era YAW differs from WEAK-era YAW by enough to shift pos ~0.06-0.12m => MEASUREMENT/calib drift')
print('    (the pos centering signal is partly a calibration artifact, not real motion).')
print('  - If rpyCalib (esp YAW) and laneWidth are ~stable across eras => the era baseline is PHYSICAL (alignment/')
print('    tire/road), a real between-era centering change but still NOT the golden PI.')
print('  - lc_kp column proves b4/b5 config (0.0001=weak); b3/b6/b7 are ~0% engaged (no LC) = pure human baseline.')
