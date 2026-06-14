import sys, os
sys.path.insert(0, '.')
import numpy as np

MPH = 1609.344 / 3600.0  # m/s per mph
LO, HI = 35 * MPH, 50 * MPH
RNG = np.random.default_rng(12345)

def load(route):
    return np.load(f'explorer_st_logs/_c5_raw/{route}.npz')

def build_grid(d):
    """Resample onto modelV2 (offset) grid by nearest/interp; return per-sample frame."""
    t = d['mdl_t']
    order = np.argsort(t)
    t = t[order]
    off = d['mdl_off'][order]
    yr = d['yr_v'][order]  # already on same grid (extracted together)
    # vEgo and eng on 100Hz grids -> interp onto t
    veg = np.interp(t, d['veg_t'], d['veg_v'])
    # eng is 0/1; interp then threshold >0.5 (latActive)
    eng_i = np.interp(t, d['eng_t'], d['eng_v'])
    eng = (eng_i > 0.5)
    return t, off, yr, veg, eng

def segment_route(route):
    """Return list of (era, engaged_bool, t_array, off_array) for 35-50mph straights,
       split into contiguous bouts of constant (engaged) state."""
    d = load(route)
    t, off, yr, veg, eng = build_grid(d)
    straight = np.abs(yr * veg) < 0.6
    band = (veg >= LO) & (veg <= HI)
    valid = straight & band & np.isfinite(off)
    return t, off, eng, valid

def collect(routes, era):
    """Concatenate samples across routes, tagging era. Also return bout list.
       A bout = maximal contiguous run (in time, with small gap tolerance) of valid samples
       with constant engaged state, within a single route."""
    all_off = []
    all_eng = []
    bouts = []  # (era, engaged, mean_off, n)
    for route in routes:
        t, off, eng, valid = segment_route(route)
        # iterate samples in time order; group valid samples into bouts by (engaged, contiguity)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        # detect breaks: index gap >1 (non-valid sample between) OR time gap >0.5s OR engaged change
        cur = [idx[0]]
        def flush(group):
            g = np.array(group)
            o = off[g]; e = bool(eng[g[0]])
            all_off.append(o); all_eng.append(np.full(len(o), e))
            bouts.append(dict(era=era, eng=e, off=o, t=t[g]))
        for k in range(1, len(idx)):
            i, j = idx[k-1], idx[k]
            tgap = t[j] - t[i]
            if (j - i) > 1 or tgap > 0.5 or (eng[j] != eng[i]):
                flush(cur); cur = [j]
            else:
                cur.append(j)
        flush(cur)
    off_arr = np.concatenate(all_off) if all_off else np.array([])
    eng_arr = np.concatenate(all_eng) if all_eng else np.array([])
    return off_arr, eng_arr, bouts

def cell_means(bouts):
    """means by engaged state, pooled over all samples in the era."""
    eng_off = np.concatenate([b['off'] for b in bouts if b['eng']]) if any(b['eng'] for b in bouts) else np.array([])
    une_off = np.concatenate([b['off'] for b in bouts if not b['eng']]) if any(not b['eng'] for b in bouts) else np.array([])
    return eng_off, une_off

def did_point(gold_bouts, weak_bouts):
    g_e, g_u = cell_means(gold_bouts)
    w_e, w_u = cell_means(weak_bouts)
    did = (g_e.mean() - g_u.mean()) - (w_e.mean() - w_u.mean())
    return did, dict(GOLD_eng=g_e.mean(), GOLD_uneng=g_u.mean(),
                     WEAK_eng=w_e.mean(), WEAK_uneng=w_u.mean(),
                     n_g_e=len(g_e), n_g_u=len(g_u), n_w_e=len(w_e), n_w_u=len(w_u))

# ---------- bout-level bootstrap ----------
def bout_bootstrap(gold_bouts, weak_bouts, B=5000):
    # resample bouts within each (era, engaged) cell, recompute pooled cell mean (sample-weighted)
    cells = {
        ('G', True):  [b for b in gold_bouts if b['eng']],
        ('G', False): [b for b in gold_bouts if not b['eng']],
        ('W', True):  [b for b in weak_bouts if b['eng']],
        ('W', False): [b for b in weak_bouts if not b['eng']],
    }
    def cell_mean(blist):
        return np.concatenate([b['off'] for b in blist]).mean()
    dids = np.empty(B)
    for bi in range(B):
        vals = {}
        for key, blist in cells.items():
            n = len(blist)
            pick = [blist[i] for i in RNG.integers(0, n, n)]
            vals[key] = cell_mean(pick)
        dids[bi] = (vals[('G',True)] - vals[('G',False)]) - (vals[('W',True)] - vals[('W',False)])
    return dids, {k: len(v) for k, v in cells.items()}

# ---------- moving-block bootstrap ----------
def make_blocks(bouts, block_s, dt=0.05):
    """Within each contiguous bout, cut into fixed-length time blocks; each block keeps its samples.
       Returns dict[(eng)] -> list of arrays (blocks)."""
    blk_len = max(1, int(round(block_s / dt)))
    out = {True: [], False: []}
    for b in bouts:
        o = b['off']; e = b['eng']
        # split bout into consecutive chunks of blk_len
        for s in range(0, len(o), blk_len):
            out[e].append(o[s:s+blk_len])
    return out

def block_bootstrap(gold_bouts, weak_bouts, block_s, B=5000, dt=0.05):
    gb = make_blocks(gold_bouts, block_s, dt)
    wb = make_blocks(weak_bouts, block_s, dt)
    cells = {('G',True): gb[True], ('G',False): gb[False],
             ('W',True): wb[True], ('W',False): wb[False]}
    def resample_mean(blocks):
        n = len(blocks)
        pick = [blocks[i] for i in RNG.integers(0, n, n)]
        return np.concatenate(pick).mean()
    dids = np.empty(B)
    for bi in range(B):
        vals = {k: resample_mean(v) for k, v in cells.items()}
        dids[bi] = (vals[('G',True)] - vals[('G',False)]) - (vals[('W',True)] - vals[('W',False)])
    return dids, {k: len(v) for k, v in cells.items()}

# ---------- autocorr time ----------
def autocorr_time(bouts, dt=0.05, maxlag_s=10):
    """Integrated autocorr time (sum of positive autocorr) on demeaned offset within bouts, pooled."""
    segs = [b['off'] - b['off'].mean() for b in bouts if len(b['off']) > 5]
    if not segs:
        return np.nan
    maxlag = int(maxlag_s/dt)
    num = np.zeros(maxlag+1); den = np.zeros(maxlag+1)
    var = np.concatenate(segs).var()
    for s in segs:
        for lag in range(0, min(maxlag, len(s)-1)+1):
            num[lag] += np.sum(s[:len(s)-lag]*s[lag:])
            den[lag] += (len(s)-lag)
    ac = (num/den)/var
    # sum until first negative
    tau = 1.0
    for lag in range(1, maxlag+1):
        if ac[lag] <= 0: break
        tau += 2*ac[lag]
    return tau * dt, ac

def main():
    print(f'Band {LO/MPH:.0f}-{HI/MPH:.0f} mph = {LO:.2f}-{HI:.2f} m/s; straight |yr*v|<0.6\n')
    _, _, gold_bouts = collect(['route_b8'], 'G')
    _, _, weak_bouts = collect(['route_b1', 'route_b2'], 'W')

    did, parts = did_point(gold_bouts, weak_bouts)
    print('CELL MEANS (signed model offset, m; +=right of center? per convention):')
    for k in ['GOLD_eng','GOLD_uneng','WEAK_eng','WEAK_uneng']:
        print(f'  {k:12s} = {parts[k]:+.4f}  n={parts["n_"+{"GOLD_eng":"g_e","GOLD_uneng":"g_u","WEAK_eng":"w_e","WEAK_uneng":"w_u"}[k]]}')
    print(f'  GOLD within = {parts["GOLD_eng"]-parts["GOLD_uneng"]:+.4f}')
    print(f'  WEAK within = {parts["WEAK_eng"]-parts["WEAK_uneng"]:+.4f}')
    print(f'\nDiD point estimate = {did:+.4f} m\n')

    n_g = len(gold_bouts); n_w = len(weak_bouts)
    print(f'n bouts: GOLD={n_g} (eng {sum(b["eng"] for b in gold_bouts)}, uneng {sum(not b["eng"] for b in gold_bouts)}), '
          f'WEAK={n_w} (eng {sum(b["eng"] for b in weak_bouts)}, uneng {sum(not b["eng"] for b in weak_bouts)})')

    tau_g, _ = autocorr_time(gold_bouts)
    tau_w, _ = autocorr_time(weak_bouts)
    print(f'autocorr time: GOLD~{tau_g:.2f}s WEAK~{tau_w:.2f}s\n')

    def ci(dids, label):
        lo, hi = np.percentile(dids, [2.5, 97.5])
        sig = (lo > 0) or (hi < 0)
        print(f'  {label:24s} mean={dids.mean():+.4f} 95%CI=[{lo:+.4f}, {hi:+.4f}]  {"SIG" if sig else "n.s."}')
        return lo, hi, sig

    print('AUTOCORR-HONEST INFERENCE:')
    db, cellsb = bout_bootstrap(gold_bouts, weak_bouts)
    ci(db, 'bout-level bootstrap')
    print('       bout cells:', cellsb)
    for bs in [1, 2, 5]:
        dblk, cellsblk = block_bootstrap(gold_bouts, weak_bouts, bs)
        ci(dblk, f'moving-block {bs}s')

    # iid bootstrap for contrast (resample individual samples)
    print('\nFOR CONTRAST (iid sample bootstrap = the WRONG one):')
    g_e, g_u = cell_means(gold_bouts); w_e, w_u = cell_means(weak_bouts)
    def iid_mean(a): return a[RNG.integers(0, len(a), len(a))].mean()
    dids_iid = np.array([(iid_mean(g_e)-iid_mean(g_u))-(iid_mean(w_e)-iid_mean(w_u)) for _ in range(5000)])
    ci(dids_iid, 'iid bootstrap')

if __name__ == '__main__':
    main()
