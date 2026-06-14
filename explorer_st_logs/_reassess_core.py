import numpy as np

MPH = 0.44704

def load(route):
    d = np.load(f'explorer_st_logs/_reassess_mine_{route}.npz')
    return {k: d[k] for k in d.keys()}

def concat(routes):
    ds = [load(r) for r in routes]
    out = {}
    for k in ds[0].keys():
        out[k] = np.concatenate([d[k] for d in ds])
    # also carry route-id and a continuity-break marker via time gaps
    return out

WEAK = concat(['route_b1', 'route_b2'])
GOLD = concat(['route_b8'])

def select(d, straight_thr=0.6, band=(40,80), gate='latact'):
    spd, steer, press, latact, yaw, pos = d['spd'], d['steer'], d['press'], d['latact'], d['yaw'], d['pos']
    eng = latact >= 0.5
    if gate == 'latact_nopress':
        eng = eng & (press < 0.5)
    straight = np.abs(yaw * spd) < straight_thr
    inband = (spd >= band[0]*MPH) & (spd <= band[1]*MPH)
    fin = np.isfinite(steer) & np.isfinite(pos) & np.isfinite(spd) & np.isfinite(yaw)
    return eng & straight & inband & fin

# ---------- (a) CENTERING: mean offset, per-mph bin, speed-matched ----------
def centering_by_bin(weak_sel_d, gold_sel_d, straight_thr=0.6, band=(40,80), gate='latact'):
    sw = select(weak_sel_d, straight_thr, band, gate)
    sg = select(gold_sel_d, straight_thr, band, gate)
    res = []
    for lo in range(band[0], band[1], 2):
        hi = lo + 2
        wmask = sw & (weak_sel_d['spd'] >= lo*MPH) & (weak_sel_d['spd'] < hi*MPH)
        gmask = sg & (gold_sel_d['spd'] >= lo*MPH) & (gold_sel_d['spd'] < hi*MPH)
        nw, ng = wmask.sum(), gmask.sum()
        if nw < 50 or ng < 50:
            continue
        wp = weak_sel_d['pos'][wmask]; gp = gold_sel_d['pos'][gmask]
        res.append((lo, hi, nw, ng, np.mean(wp), np.mean(gp), np.median(wp), np.median(gp)))
    return res

# ---------- block bootstrap on matched-speed pooled mean ----------
def block_boot_mean(vals, blocksize=50, nboot=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(vals)
    nblocks = int(np.ceil(n / blocksize))
    means = []
    for _ in range(nboot):
        starts = rng.integers(0, max(1, n - blocksize), size=nblocks)
        samp = np.concatenate([vals[s:s+blocksize] for s in starts])[:n]
        means.append(np.mean(samp))
    return np.array(means)

def matched_centering(weak_d, gold_d, straight_thr=0.6, band=(40,80), gate='latact'):
    """Per-2mph-bin mean offset, then average bins weighted equally (speed-matched).
    Returns weak_mean, gold_mean, delta and a block-bootstrap CI on the delta of pooled-matched.
    """
    bins = centering_by_bin(weak_d, gold_d, straight_thr, band, gate)
    if not bins:
        return None
    # equal-weight across overlapping bins -> removes speed confound
    wmeans = np.array([b[4] for b in bins]); gmeans = np.array([b[5] for b in bins])
    w_matched = np.mean(wmeans); g_matched = np.mean(gmeans)
    return dict(bins=bins, weak_matched=w_matched, gold_matched=g_matched,
                delta=g_matched - w_matched)

# ---------- (b) OSCILLATION: matched-speed band-RMS of steeringAngleDeg ----------
def osc_windows(d, sel, win_s=16.0, fs=50.0, hp_band=(0.05,)):
    """Compute per-window detrended RMS of steeringAngleDeg over contiguous engaged-straight windows.
    Returns list of (mean_spd_mph, rms_deg)."""
    win = int(win_s*fs)
    steer = d['steer']; spd = d['spd']
    idx = np.where(sel)[0]
    if len(idx) == 0:
        return []
    # find contiguous runs in selected indices
    out = []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    for run in runs:
        if len(run) < win:
            continue
        for st in range(0, len(run)-win+1, win//2):  # 50% overlap
            seg = run[st:st+win]
            s = steer[seg]
            # detrend (remove slow drift / curve trend) -> oscillation only
            x = np.arange(len(s))
            coef = np.polyfit(x, s, 1)
            resid = s - np.polyval(coef, x)
            rms = np.sqrt(np.mean(resid**2))
            out.append((np.mean(spd[seg])/MPH, rms))
    return out

def matched_oscillation(weak_d, gold_d, straight_thr=0.6, band=(40,80), gate='latact', win_s=16.0):
    sw = select(weak_d, straight_thr, band, gate)
    sg = select(gold_d, straight_thr, band, gate)
    ow = osc_windows(weak_d, sw, win_s)
    og = osc_windows(gold_d, sg, win_s)
    # match per 2mph bin
    rows = []
    wall=[]; gall=[]
    for lo in range(band[0], band[1], 2):
        hi = lo+2
        w = [r for s,r in ow if lo<=s<hi]
        g = [r for s,r in og if lo<=s<hi]
        if len(w)<3 or len(g)<3:
            continue
        rows.append((lo,hi,len(w),len(g),np.median(w),np.median(g)))
        wall += [(lo,r) for r in w]; gall += [(lo,r) for r in g]
    # speed-matched pooled: median of per-bin medians
    if rows:
        wmed = np.median([r[4] for r in rows]); gmed = np.median([r[5] for r in rows])
    else:
        wmed=gmed=np.nan
    return dict(rows=rows, weak_matched=wmed, gold_matched=gmed,
                n_w_win=len(ow), n_g_win=len(og))

if __name__ == '__main__':
    print("="*70)
    print("CONCLUSION (a): CENTERING -- mean offset, less-left = better")
    print("="*70)
    c = matched_centering(WEAK, GOLD)
    print(f"\n  Speed-matched (equal-weight 2mph bins, 40-80mph, straight<0.6, latact):")
    print(f"    WEAK mean pos = {c['weak_matched']:+.4f} m   GOLD mean pos = {c['gold_matched']:+.4f} m")
    print(f"    delta (GOLD-WEAK) = {c['delta']:+.4f} m   ({'GOLD less left=BETTER' if c['delta']>0 else 'GOLD more left=WORSE'})")
    print(f"\n  Per-bin detail (lo hi nW nG meanW meanG medW medG):")
    for b in c['bins']:
        print(f"    {b[0]:2d}-{b[1]:2d}  nW={b[2]:5d} nG={b[3]:5d}  meanW={b[4]:+.4f} meanG={b[5]:+.4f}  medW={b[6]:+.4f} medG={b[7]:+.4f}")

    print("\n" + "="*70)
    print("CONCLUSION (b): OSCILLATION -- matched-speed detrended RMS steeringAngleDeg")
    print("="*70)
    o = matched_oscillation(WEAK, GOLD)
    print(f"\n  Windows: WEAK={o['n_w_win']} GOLD={o['n_g_win']} (16s, 50% overlap)")
    print(f"  Speed-matched (median of per-bin medians):")
    print(f"    WEAK RMS={o['weak_matched']:.4f} deg   GOLD RMS={o['gold_matched']:.4f} deg")
    print(f"    {'GOLD MORE osc=WORSE' if o['gold_matched']>o['weak_matched'] else 'GOLD LESS osc=BETTER'}")
    print(f"\n  Per-bin (lo hi nW nG medW medG):")
    for r in o['rows']:
        flag = ' GOLD worse' if r[5]>r[4] else ' GOLD better'
        print(f"    {r[0]:2d}-{r[1]:2d}  nW={r[2]:4d} nG={r[3]:4d}  medW={r[4]:.4f} medG={r[5]:.4f}{flag}")
