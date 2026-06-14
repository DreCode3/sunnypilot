import sys, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy import stats
from c4_analyze import (load, build_grid, engaged_straight_mask, contiguous_runs,
                        hunt_band_rms_per_run, steering_rate_metrics_per_run, speed_of_run, FS)

def collect(group_routes):
    H, R = [], []  # H: (rms, n, speed); R: (rstd, revps, n, speed)
    for route in group_routes:
        g = build_grid(load(route))
        m = engaged_straight_mask(g)
        runs = contiguous_runs(m, int(2*FS))
        for (i,j,rms,n) in hunt_band_rms_per_run(g['ang'], runs):
            H.append((rms, n, speed_of_run(g,i,j)))
        for (i,j,rstd,revps,n) in steering_rate_metrics_per_run(g['ang'], runs):
            R.append((rstd, revps, n, speed_of_run(g,i,j)))
    return H, R

def matched_test(lo, hi):
    Hw,Rw = collect(['route_b1','route_b2'])
    Hg,Rg = collect(['route_b8'])
    print(f"\n##### MATCHED WINDOW {lo}-{hi} mph #####")
    # hunt
    hw = [v[0] for v in Hw if lo<=v[2]<hi]
    hg = [v[0] for v in Hg if lo<=v[2]<hi]
    rsw = [v[0] for v in Rw if lo<=v[3]<hi]
    rsg = [v[0] for v in Rg if lo<=v[3]<hi]
    rvw = [v[1] for v in Rw if lo<=v[3]<hi]
    rvg = [v[1] for v in Rg if lo<=v[3]<hi]
    for name, w, g in [('huntRMS(0.5-1.5Hz,deg)', hw, hg),
                       ('rateStd(deg/s)', rsw, rsg),
                       ('reversals/s', rvw, rvg)]:
        w=np.array(w); g=np.array(g)
        if len(w)<2 or len(g)<2:
            print(f"  {name}: insufficient runs (nW={len(w)},nG={len(g)})"); continue
        mw=np.median(w); mg=np.median(g)
        try:
            U,p = stats.mannwhitneyu(g, w, alternative='two-sided')
        except Exception:
            p=np.nan
        gold_higher = mg>mw
        print(f"  {name:24s}: WEAK med={mw:.4f} (n={len(w)})  GOLD med={mg:.4f} (n={len(g)})  "
              f"Δ={(mg-mw)/mw*100:+.1f}%  MWU p={p:.3f}  GOLD={'WORSE' if gold_higher else 'BETTER'}")

if __name__=='__main__':
    matched_test(55.0, 62.5)
    matched_test(57.5, 62.5)
    matched_test(50.0, 65.0)
