import sys, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
import numpy as np
from scipy import stats
from c4_analyze import (load, build_grid, engaged_straight_mask, contiguous_runs, FS)

def rev_runs(group_routes, dead, lo, hi):
    vals_revps = []; vals_rstd=[]
    for route in group_routes:
        g = build_grid(load(route))
        m = engaged_straight_mask(g)
        runs = contiguous_runs(m, int(2*FS))
        for (i,j) in runs:
            seg = g['ang'][i:j].astype(float)
            if len(seg) < int(1*FS): continue
            sp = np.mean(g['v'][i:j])
            if not (lo<=sp<hi): continue
            rate = np.diff(seg)*FS
            rstd = np.std(rate)
            sig = np.where(rate>dead,1,np.where(rate<-dead,-1,0))
            nz = sig[sig!=0]
            nrev = np.sum(np.diff(nz)!=0) if len(nz)>1 else 0
            dur = len(seg)/FS
            vals_revps.append(nrev/dur); vals_rstd.append(rstd)
    return np.array(vals_revps), np.array(vals_rstd)

print("Reversals/s sensitivity to deadband (matched window 50-65 mph):")
for dead in [0.5, 1.0, 2.0, 5.0]:
    rw,_ = rev_runs(['route_b1','route_b2'], dead, 50, 65)
    rg,_ = rev_runs(['route_b8'], dead, 50, 65)
    p = stats.mannwhitneyu(rg, rw, alternative='two-sided')[1] if len(rw)>1 and len(rg)>1 else float('nan')
    print(f"  dead={dead:4.1f}deg/s: WEAK med={np.median(rw):6.3f} (n={len(rw)})  "
          f"GOLD med={np.median(rg):6.3f} (n={len(rg)})  Δ={(np.median(rg)-np.median(rw))/np.median(rw)*100:+.1f}%  p={p:.3f}")

# Also: raw wheel-angle std at matched speed as a sanity check on activity
print("\nRaw steeringAngleDeg std within engaged-straight runs (50-65 mph):")
def angstd(group_routes, lo, hi):
    vals=[]; ns=[]
    for route in group_routes:
        g = build_grid(load(route)); m = engaged_straight_mask(g)
        for (i,j) in contiguous_runs(m, int(2*FS)):
            seg=g['ang'][i:j].astype(float); sp=np.mean(g['v'][i:j])
            if lo<=sp<hi: vals.append(np.std(seg)); ns.append(len(seg))
    return np.array(vals), np.array(ns)
for grp,routes in [('WEAK',['route_b1','route_b2']),('GOLD',['route_b8'])]:
    v,n = angstd(routes,50,65)
    print(f"  {grp}: median abs-angle std={np.median(v):.3f}deg  nruns={len(v)}")
