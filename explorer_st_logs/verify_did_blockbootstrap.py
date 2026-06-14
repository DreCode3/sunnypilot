#!/usr/bin/env python3
"""RIGOROUS DiD with autocorrelation-honest inference (replaces the i.i.d.-bootstrap significance in
reassess_pi_off_verify.py, which overstated significance on 50Hz autocorrelated data).

DiD = (GOLD_eng - GOLD_uneng) - (WEAK_eng - WEAK_uneng), signed offset; +DiD = GOLD engaged centers MORE
(toward 0 from the left bias) than WEAK does, after subtracting each drive's OWN manual (un-engaged) baseline.

Two honest inference methods:
 (1) BOUT-LEVEL (primary): each maximal contiguous run of engaged-straight (or unengaged-straight) samples in
     the speed band = ONE independent unit (mean offset). Bootstrap RESAMPLES BOUTS -> respects autocorrelation
     fully. Also report the autocorr integral time + effective N.
 (2) MOVING-BLOCK sample bootstrap with block length L swept {1(iid),50(1s),100(2s),250(5s)} -> shows how the
     CI widens once autocorrelation is honored; the honest CI is at L >= autocorr time."""
import os
import numpy as np
CACHE = 'explorer_st_logs/_cache_reassess'
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
FS = 50.0
rng = np.random.default_rng(424242)
LO, HI = 35.0, 50.0           # speed band where BOTH eng & uneng data are dense for both groups
MIN_BOUT = 50                  # >=1s for a stable bout mean


def bouts_and_samples(rids, want_engaged):
    """return (list of bout-mean offsets, concatenated sample offsets) for straight+band+state across routes."""
    bout_means = []; all_samp = []
    for rid in rids:
        d = {k: np.load(f'{CACHE}/{rid}.npz')[k] for k in np.load(f'{CACHE}/{rid}.npz').files}
        spd = d['spd'].astype(float)*2.237; al = d['yaw'].astype(float)*d['spd'].astype(float)
        eng = d['latact'].astype(bool); pos = d['pos'].astype(float)
        mask = (np.abs(al) < 0.6) & (spd >= LO) & (spd < HI) & np.isfinite(pos)
        mask &= eng if want_engaged else ~eng
        # maximal contiguous runs of mask==True
        idx = np.where(mask)[0]
        if len(idx) == 0: continue
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for run in np.split(idx, splits):
            if len(run) >= MIN_BOUT:
                bout_means.append(float(np.mean(pos[run])))
            all_samp.append(pos[run])
    return np.array(bout_means), (np.concatenate(all_samp) if all_samp else np.array([]))


WE_b, WE_s = bouts_and_samples(GROUPS['WEAK'], True)
WU_b, WU_s = bouts_and_samples(GROUPS['WEAK'], False)
GE_b, GE_s = bouts_and_samples(GROUPS['GOLD'], True)
GU_b, GU_s = bouts_and_samples(GROUPS['GOLD'], False)

print(f'=== DiD on signed offset, {LO:.0f}-{HI:.0f}mph straights, engaged minus un-engaged baseline ===')
print(f'  bouts (>= {MIN_BOUT/FS:.0f}s):  WEAK eng {len(WE_b)} / uneng {len(WU_b)}   GOLD eng {len(GE_b)} / uneng {len(GU_b)}')
print(f'  samples:            WEAK eng {len(WE_s)} / uneng {len(WU_s)}   GOLD eng {len(GE_s)} / uneng {len(GU_s)}')
if min(len(WE_b), len(WU_b), len(GE_b), len(GU_b)) < 3:
    print('  TOO FEW BOUTS in one cell for bout-level inference; falling back to block bootstrap only')

# point estimate (sample means = unbiased; bouts can be unequal length so use sample means for the point est)
pt = (GE_s.mean() - GU_s.mean()) - (WE_s.mean() - WU_s.mean())
print(f'\n  POINT ESTIMATE DiD = {pt:+.4f} m   (+ = GOLD engaged closer to center than WEAK, baseline-differenced)')
print(f'    means: WEAK eng {WE_s.mean():+.4f} / man {WU_s.mean():+.4f}   GOLD eng {GE_s.mean():+.4f} / man {GU_s.mean():+.4f}')


def autocorr_time(x):
    x = x - x.mean(); n = len(x)
    if n < 200: return 1.0
    ac = np.correlate(x, x, 'full')[n-1:] / (np.arange(n, 0, -1) * np.var(x))
    # integrated autocorr time = 1 + 2*sum(ac up to first crossing of 0)
    s = 1.0
    for k in range(1, min(n, 2000)):
        if ac[k] <= 0: break
        s += 2*ac[k]
    return s
tau_we = autocorr_time(WE_s)
print(f'\n  autocorr integrated time (engaged WEAK offset): {tau_we:.0f} samples (~{tau_we/FS:.1f}s) '
      f'-> effective N ~ {len(WE_s)/tau_we:.0f} (nominal {len(WE_s)})')

# (1) BOUT-LEVEL bootstrap (each bout = independent unit)
def boot_bouts(nb=8000):
    out = []
    for _ in range(nb):
        ge = GE_b[rng.integers(0, len(GE_b), len(GE_b))].mean()
        gu = GU_b[rng.integers(0, len(GU_b), len(GU_b))].mean()
        we = WE_b[rng.integers(0, len(WE_b), len(WE_b))].mean()
        wu = WU_b[rng.integers(0, len(WU_b), len(WU_b))].mean()
        out.append((ge-gu)-(we-wu))
    return np.array(out)
if min(len(WE_b), len(WU_b), len(GE_b), len(GU_b)) >= 3:
    bb = boot_bouts()
    lo, hi = np.percentile(bb, 2.5), np.percentile(bb, 97.5)
    print(f'\n  (1) BOUT-LEVEL bootstrap 95% CI: [{lo:+.4f}, {hi:+.4f}]  {"SIGNIFICANT" if (lo>0 or hi<0) else "ns"}  '
          f'(p2-sided~{2*min(np.mean(bb<=0),np.mean(bb>=0)):.3f})')

# (2) MOVING-BLOCK bootstrap, block length sweep
def mb_mean(x, L):
    n = len(x); nb = int(np.ceil(n / L))
    starts = rng.integers(0, max(1, n - L + 1), nb)
    return np.concatenate([x[s:s+L] for s in starts])[:n].mean()
def boot_block(L, nb=4000):
    out = [((mb_mean(GE_s, L)-mb_mean(GU_s, L))-(mb_mean(WE_s, L)-mb_mean(WU_s, L))) for _ in range(nb)]
    return np.array(out)
print('\n  (2) MOVING-BLOCK sample bootstrap (CI widens as block length L honors autocorrelation):')
for L in [1, 50, 100, 250]:
    bk = boot_block(L)
    lo, hi = np.percentile(bk, 2.5), np.percentile(bk, 97.5)
    print(f'      L={L:>4} ({L/FS:>4.0f}s): 95% CI [{lo:+.4f}, {hi:+.4f}]  {"SIG" if (lo>0 or hi<0) else "ns"}')
print('\n  HONEST VERDICT uses bout-level + block L>=autocorr-time. iid (L=1) is the OVERSTATED one I used before.')
