import numpy as np
from _reassess_core import WEAK, GOLD, select, MPH

# The window method discards too much. Use a per-sample band-RMS approach:
# high-pass the steering signal (remove <0.1Hz drift via rolling-median detrend),
# then compute oscillation in matched speed bins from ALL engaged-straight samples,
# using contiguous-run detrending to avoid curve-edge artifacts.

def osc_persample(d, sel, fs=50.0, detrend_win_s=4.0):
    """Detrend within contiguous runs using a centered moving average (removes slow drift),
    return residual steering and speed per kept sample."""
    steer = d['steer']; spd = d['spd']
    idx = np.where(sel)[0]
    if len(idx)==0:
        return np.array([]), np.array([])
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    win = int(detrend_win_s*fs)
    resid_all=[]; spd_all=[]
    for run in runs:
        if len(run) < win:
            continue
        s = steer[run]
        # moving average via cumulative sum
        k = win | 1  # odd
        pad = k//2
        sp = np.pad(s, pad, mode='edge')
        csum = np.cumsum(np.insert(sp,0,0))
        ma = (csum[k:]-csum[:-k])/k
        resid = s - ma
        # drop edges affected by padding
        if len(run) > 2*pad:
            resid_all.append(resid[pad:-pad]); spd_all.append(spd[run][pad:-pad])
    if not resid_all:
        return np.array([]), np.array([])
    return np.concatenate(resid_all), np.concatenate(spd_all)

def matched_osc_rms(weak_d, gold_d, straight_thr=0.6, band=(40,80), gate='latact', detrend_win_s=4.0):
    sw = select(weak_d, straight_thr, band, gate)
    sg = select(gold_d, straight_thr, band, gate)
    rw, vw = osc_persample(weak_d, sw, detrend_win_s=detrend_win_s)
    rg, vg = osc_persample(gold_d, sg, detrend_win_s=detrend_win_s)
    rows=[]; wrms_bins=[]; grms_bins=[]
    for lo in range(band[0], band[1], 2):
        hi=lo+2
        wm = (vw>=lo*MPH)&(vw<hi*MPH); gm=(vg>=lo*MPH)&(vg<hi*MPH)
        nw,ng = wm.sum(), gm.sum()
        if nw<200 or ng<200:  # >=4s of data
            continue
        wr = np.sqrt(np.mean(rw[wm]**2)); gr=np.sqrt(np.mean(rg[gm]**2))
        rows.append((lo,hi,nw,ng,wr,gr))
        wrms_bins.append(wr); grms_bins.append(gr)
    return dict(rows=rows,
                weak_matched=np.median(wrms_bins) if wrms_bins else np.nan,
                gold_matched=np.median(grms_bins) if grms_bins else np.nan,
                n_bins=len(rows))

print("OSCILLATION via per-sample moving-avg detrend (keeps all engaged-straight samples)")
print("="*70)
for dw in [4.0, 2.0, 8.0]:
    o = matched_osc_rms(WEAK, GOLD, detrend_win_s=dw)
    print(f"\ndetrend_win={dw}s : {o['n_bins']} overlapping bins")
    print(f"  matched WEAK={o['weak_matched']:.4f}deg  GOLD={o['gold_matched']:.4f}deg  "
          f"{'GOLD WORSE' if o['gold_matched']>o['weak_matched'] else 'GOLD BETTER'}")
    for r in o['rows']:
        flag=' G-worse' if r[5]>r[4] else ' G-better'
        print(f"    {r[0]:2d}-{r[1]:2d} nW={r[2]:5d} nG={r[3]:5d}  W={r[4]:.4f} G={r[5]:.4f}{flag}")
