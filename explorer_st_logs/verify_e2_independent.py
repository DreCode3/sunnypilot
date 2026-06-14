import numpy as np

ERAS = {
    'WEAK': ['b1', 'b2', 'b4', 'b5'],  # engaged WEAK-PI routes
    'GOLD': ['b8'],                     # golden PI
}
# human-driven (unengaged) routes available per era?
# b3,b6,b7 are ~all unengaged human. They are WEAK-era hardware/model.
# b8 contains unengaged segments too (latact==0). b1,b2,b4,b5 contain unengaged segments.
# Approach: within EACH route, split by latact. engaged=latact==1, unengaged=latact==0 (human, PI off).

MPS2MPH = 2.2369362921

def load(r):
    return np.load(f'explorer_st_logs/_cache_reassess/route_{r}.npz')

def edge_clean_mask(flag, pad):
    """Drop samples within `pad` samples of any transition in boolean flag."""
    flag = flag.astype(bool)
    change = np.zeros_like(flag)
    change[1:] = flag[1:] != flag[:-1]
    # also mark first/last as edges
    bad = np.zeros_like(flag)
    idx = np.where(change)[0]
    for i in idx:
        lo = max(0, i - pad)
        hi = min(len(flag), i + pad)
        bad[lo:hi] = True
    return ~bad

def segment_means(pos, mask, t, min_run_s=2.0):
    """Return list of per-contiguous-run mean offsets for samples in mask.
    Treats each contiguous run as one unit (avoid iid 50Hz overcount)."""
    means = []
    m = mask.copy()
    i = 0
    n = len(m)
    runs = []
    start = None
    for j in range(n):
        if m[j] and start is None:
            start = j
        elif not m[j] and start is not None:
            runs.append((start, j))
            start = None
    if start is not None:
        runs.append((start, n))
    for (a, b) in runs:
        if t[b-1] - t[a] >= min_run_s:
            means.append(np.nanmean(pos[a:b]))
    return means

def era_stats(routes, engaged, pad=25):
    """Pool per-route, edge-cleaned straight 40-80mph, latact==engaged.
    Returns pooled sample mean, run-level mean, and counts."""
    all_pos = []
    run_means = []
    nrun = 0
    nsamp = 0
    nroutes_contrib = 0
    for r in routes:
        d = load(r)
        spd = d['spd']; pos = d['pos']; lat = d['latact']; yaw = d['yaw']; t = d['t']
        mph = spd * MPS2MPH
        straight = np.abs(yaw * spd) < 0.6
        band = (mph >= 40) & (mph <= 80)
        eng = (lat == 1) if engaged else (lat == 0)
        base = straight & band & eng
        # edge-clean on the engagement flag transitions
        clean = edge_clean_mask(lat == 1, pad)
        mask = base & clean & np.isfinite(pos)
        if mask.sum() == 0:
            continue
        nroutes_contrib += 1
        all_pos.append(pos[mask])
        rm = segment_means(pos, mask, t)
        run_means.extend(rm)
        nrun += len(rm)
        nsamp += int(mask.sum())
    if not all_pos:
        return None
    pooled = np.concatenate(all_pos)
    return {
        'pooled_mean': float(np.mean(pooled)),
        'pooled_median': float(np.median(pooled)),
        'run_mean': float(np.mean(run_means)) if run_means else float('nan'),
        'run_median': float(np.median(run_means)) if run_means else float('nan'),
        'nsamp': nsamp,
        'nrun': nrun,
        'nroutes': nroutes_contrib,
    }

print('='*70)
print('E2 INDEPENDENT VERIFICATION — edge-cleaned straight 40-80mph offset')
print('pad=25 samples (0.5s) around engage/disengage transitions')
print('='*70)

results = {}
for era, routes in ERAS.items():
    for eng_label, eng in [('eng', True), ('uneng', False)]:
        s = era_stats(routes, eng)
        results[(era, eng_label)] = s
        if s:
            print(f'{era:5s} {eng_label:6s}: pooled_mean={s["pooled_mean"]:+.4f}m  '
                  f'pooled_median={s["pooled_median"]:+.4f}m  run_mean={s["run_mean"]:+.4f}  '
                  f'nsamp={s["nsamp"]} nrun={s["nrun"]} nroutes={s["nroutes"]}')
        else:
            print(f'{era:5s} {eng_label:6s}: NO DATA')

print('-'*70)
we = results[('WEAK','eng')]['pooled_mean']
wu = results[('WEAK','uneng')]['pooled_mean']
ge = results[('GOLD','eng')]['pooled_mean']
gu = results[('GOLD','uneng')]['pooled_mean']
eng_gap = ge - we
uneng_gap = gu - wu
did = (ge - gu) - (we - wu)
print(f'WEAK eng={we:+.4f}  uneng={wu:+.4f}')
print(f'GOLD eng={ge:+.4f}  uneng={gu:+.4f}')
print(f'ENGAGED gap (GOLD-WEAK)   = {eng_gap:+.4f} m')
print(f'UNENGAGED gap (GOLD-WEAK) = {uneng_gap:+.4f} m   (both human, PI OFF)')
print(f'within-era DiD = (GOLD_eng-GOLD_uneng)-(WEAK_eng-WEAK_uneng) = {did:+.4f} m')
print('-'*70)
print('Claimed: WEAK eng -0.079/uneng -0.093; GOLD eng -0.018/uneng -0.028')
print('Claimed: ENGAGED gap +0.061; UNENGAGED gap +0.064 pooled; DiD -0.003')
print(f'reproduces gate: uneng_gap>=+0.05 AND |DiD|<0.02 -> '
      f'{uneng_gap>=0.05 and abs(did)<0.02}')
