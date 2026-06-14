#!/usr/bin/env python3
"""VERIFY the synthesis/Reviewer-B claim: the b8 steering '-20%' is a SPEED confound (Simpson's paradox), not PI.
Checks: (1) is GOLD(b8) driven faster than WEAK(b1,b2) on straights? (2) does steer amplitude fall with speed?
(3) does the steer reduction survive speed-binning / speed-matching, or flip sign?"""
import sys, glob, os, math
import numpy as np
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

FS = 20.0; WIN = 16.0; HOP = 8.0; ENG_MIN = 0.9
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}


def slow_rms(x, lo=0.1, hi=0.5):
    n = len(x); t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1 / FS); p = np.abs(X) ** 2
    return math.sqrt(2 * np.sum(p[(f >= lo) & (f < hi)]) / (n * n))


def load(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    tll=[];yaw=[]; tc=[];veg=[];sa=[]; te=[];eng=[]
    for sd in segs:
        f = sd + 'rlog.zst'
        if not os.path.exists(f): continue
        try:
            for msg in LogReader(f):
                w = msg.which()
                if w == 'liveLocationKalman':
                    g = msg.liveLocationKalman
                    if g.angularVelocityCalibrated.valid:
                        tll.append(msg.logMonoTime*1e-9); yaw.append(float(g.angularVelocityCalibrated.value[2]))
                elif w == 'carState':
                    tc.append(msg.logMonoTime*1e-9); veg.append(float(msg.carState.vEgo)); sa.append(float(msg.carState.steeringAngleDeg))
                elif w == 'carControl':
                    te.append(msg.logMonoTime*1e-9); eng.append(1.0 if msg.carControl.latActive else 0.0)
        except Exception: continue
    if len(tll)<400 or len(tc)<400 or len(te)<400: return rid, []
    tll=np.array(tll);o=np.argsort(tll);tll,yaw=tll[o],np.array(yaw)[o]
    tc=np.array(tc);oc=np.argsort(tc);tc,veg,sa=tc[oc],np.array(veg)[oc],np.array(sa)[oc]
    te=np.array(te);oe=np.argsort(te);te,eng=te[oe],np.array(eng)[oe]
    tu=np.arange(tll[0],tll[-1],1/FS)
    yawu=np.interp(tu,tll,yaw);vegu=np.interp(tu,tc,veg);sau=np.interp(tu,tc,sa);engu=np.interp(tu,te,eng)
    aLat=yawu*vegu
    out=[]; n=int(WIN*FS); hop=int(HOP*FS)
    for i in range(0,len(tu)-n,hop):
        sl=slice(i,i+n)
        if np.mean(engu[sl])<ENG_MIN: continue
        if np.median(np.abs(aLat[sl]))>=0.6: continue   # straight
        spd=float(np.median(vegu[sl])*2.237)
        if not (40<=spd<=80): continue
        out.append((spd, slow_rms(sau[sl]), float(np.std(sau[sl]))))
    return rid, out


allr=[r for v in GROUPS.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    res=dict(ex.map(load, allr))
W={g:np.array([row for r in rs for row in res.get(r,[])]) for g,rs in GROUPS.items()}
for g in GROUPS: print(f'  {g}: {len(W[g])} engaged-straight windows')

print('\n=== (1) SPEED distributions (mph) ===')
for g in GROUPS:
    s=W[g][:,0]
    print(f'  {g}: median {np.median(s):.1f}  IQR [{np.percentile(s,25):.1f},{np.percentile(s,75):.1f}]  mean {np.mean(s):.1f}')

print('\n=== (2) steer vs speed WITHIN WEAK (does amplitude fall with speed?) ===')
s=W['WEAK'][:,0]; std=W['WEAK'][:,2]; slo=W['WEAK'][:,1]
A=np.polyfit(s,std,1); B=np.polyfit(s,slo,1)
print(f'  steer_STD  ~ {A[0]:+.4f} deg/mph  (corr {np.corrcoef(s,std)[0,1]:+.2f})')
print(f'  steer_SLOW ~ {B[0]:+.4f} /mph     (corr {np.corrcoef(s,slo)[0,1]:+.2f})')

print('\n=== (3) speed-BINNED steer (does the -20% flip?) ===')
print(f'{"bin":<10}{"WEAK std":>10}{"GOLD std":>10}{"chg":>7}{"WEAK slow":>11}{"GOLD slow":>11}{"chg":>7}{"nW/nG":>9}')
for lo,hi in [(40,50),(50,57),(57,65),(65,80)]:
    wb=W['WEAK'][(W['WEAK'][:,0]>=lo)&(W['WEAK'][:,0]<hi)]
    gb=W['GOLD'][(W['GOLD'][:,0]>=lo)&(W['GOLD'][:,0]<hi)]
    if len(wb)>=4 and len(gb)>=4:
        wstd,gstd=np.median(wb[:,2]),np.median(gb[:,2]); wslo,gslo=np.median(wb[:,1]),np.median(gb[:,1])
        print(f'  {lo}-{hi:<5}{wstd:>10.3f}{gstd:>10.3f}{100*(gstd-wstd)/wstd:>+6.0f}%{wslo:>11.3f}{gslo:>11.3f}{100*(gslo-wslo)/wslo:>+6.0f}%{f"{len(wb)}/{len(gb)}":>9}')
    else:
        print(f'  {lo}-{hi:<5}  n too few ({len(wb)}/{len(gb)})')

print('\n=== (4) POOLED (unmatched) vs SPEED-MATCHED steer comparison ===')
wstd_p,gstd_p=np.median(W['WEAK'][:,2]),np.median(W['GOLD'][:,2])
wslo_p,gslo_p=np.median(W['WEAK'][:,1]),np.median(W['GOLD'][:,1])
print(f'  POOLED:  steer_std WEAK {wstd_p:.3f} -> GOLD {gstd_p:.3f} ({100*(gstd_p-wstd_p)/wstd_p:+.0f}%) | steer_slow {wslo_p:.3f}->{gslo_p:.3f} ({100*(gslo_p-wslo_p)/wslo_p:+.0f}%)')
# speed-match: for each GOLD window, find WEAK windows within +-2.5mph, compare medians
matched_w_std=[]; matched_g_std=[]; matched_w_slo=[]; matched_g_slo=[]
for gs,gslo,gstd in W['GOLD']:
    m=W['WEAK'][np.abs(W['WEAK'][:,0]-gs)<=2.5]
    if len(m)>=2:
        matched_w_std.append(np.median(m[:,2])); matched_g_std.append(gstd)
        matched_w_slo.append(np.median(m[:,1])); matched_g_slo.append(gslo)
if matched_g_std:
    rw,rg=np.median(matched_w_std),np.median(matched_g_std)
    rws,rgs=np.median(matched_w_slo),np.median(matched_g_slo)
    print(f'  MATCHED: steer_std WEAK {rw:.3f} -> GOLD {rg:.3f} ({100*(rg-rw)/rw:+.0f}%) | steer_slow {rws:.3f}->{rgs:.3f} ({100*(rgs-rws)/rws:+.0f}%)  (n={len(matched_g_std)} GOLD win w/ WEAK partner)')
print('\nVERDICT: if pooled shows -20% but matched/binned flips to ~0 or +, the steering "win" is a SPEED artifact.')
