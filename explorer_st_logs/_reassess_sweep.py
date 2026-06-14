import numpy as np
from _reassess_core import WEAK, GOLD, select, MPH, matched_centering
from _reassess_osc2 import matched_osc_rms

rng = np.random.default_rng(42)

def block_boot_delta_centering(weak_d, gold_d, straight_thr, band, gate, nboot=1000, blk=50):
    """Block bootstrap CI on speed-matched centering delta.
    Resample contiguous blocks within each group, recompute per-bin equal-weight mean delta."""
    sw = select(weak_d, straight_thr, band, gate)
    sg = select(gold_d, straight_thr, band, gate)
    # precompute per-bin index arrays
    def bins_idx(d, sel):
        out={}
        for lo in range(band[0], band[1], 2):
            m = sel & (d['spd']>=lo*MPH)&(d['spd']<(lo+2)*MPH)
            idx=np.where(m)[0]
            if len(idx)>=50: out[lo]=idx
        return out
    wb = bins_idx(weak_d, sw); gb=bins_idx(gold_d, sg)
    common=[lo for lo in wb if lo in gb]
    deltas=[]
    for _ in range(nboot):
        wmeans=[]; gmeans=[]
        for lo in common:
            wi=wb[lo]; gi=gb[lo]
            # block resample within bin's pos values (approx; honors local autocorr)
            wp=weak_d['pos'][wi]; gp=gold_d['pos'][gi]
            def bres(v):
                n=len(v); nbl=int(np.ceil(n/blk))
                st=rng.integers(0, max(1,n-blk), size=nbl)
                return np.concatenate([v[s:s+blk] for s in st])[:n]
            wmeans.append(np.mean(bres(wp))); gmeans.append(np.mean(bres(gp)))
        deltas.append(np.mean(gmeans)-np.mean(wmeans))
    deltas=np.array(deltas)
    return np.percentile(deltas,[2.5,50,97.5])

print("#"*72)
print("# ROBUSTNESS SWEEP")
print("#"*72)

straight_thrs = [0.4, 0.6, 0.8]
bands = [(40,80),(45,75),(50,70)]
gates = ['latact','latact_nopress']
osc_wins = [12,16,30]   # window-method sec  (also test detrend variants)
detrend_wins = [2.0,4.0,8.0]

print("\n===== (a) CENTERING delta (GOLD-WEAK), >0 = GOLD better (less left) =====")
print(f"{'straight':>8} {'band':>10} {'gate':>16}  {'W_mean':>8} {'G_mean':>8} {'delta':>8}  {'95%CI':>22} {'verdict'}")
cent_flips=0; cent_total=0
for st in straight_thrs:
    for bd in bands:
        for g in gates:
            c = matched_centering(WEAK, GOLD, st, bd, g)
            if c is None:
                print(f"{st:>8} {str(bd):>10} {g:>16}   <no overlapping bins>")
                continue
            ci = block_boot_delta_centering(WEAK, GOLD, st, bd, g, nboot=600)
            cent_total+=1
            better = c['delta']>0
            ci_excludes0 = (ci[0]>0) or (ci[2]<0)
            verdict = ('GOLD better' if better else 'GOLD WORSE') + (' *' if ci_excludes0 else ' (CI~0)')
            if not better: cent_flips+=1
            print(f"{st:>8} {str(bd):>10} {g:>16}  {c['weak_matched']:+8.4f} {c['gold_matched']:+8.4f} {c['delta']:+8.4f}  "
                  f"[{ci[0]:+.4f},{ci[2]:+.4f}] {verdict}")

print(f"\n  centering 'GOLD better' held in {cent_total-cent_flips}/{cent_total} configs (flips={cent_flips})")

print("\n===== (b) OSCILLATION matched RMS, GOLD>WEAK = GOLD worse =====")
print(f"{'straight':>8} {'band':>10} {'gate':>16} {'dwin':>5}  {'W_rms':>8} {'G_rms':>8} {'nbins':>5} {'maj':>9} {'verdict'}")
osc_flips=0; osc_total=0
for st in straight_thrs:
    for bd in bands:
        for g in gates:
            for dw in detrend_wins:
                o = matched_osc_rms(WEAK, GOLD, st, bd, g, detrend_win_s=dw)
                if o['n_bins']==0:
                    continue
                osc_total+=1
                worse = o['gold_matched']>o['weak_matched']
                # majority direction across bins
                nworse=sum(1 for r in o['rows'] if r[5]>r[4]); nb=o['n_bins']
                maj=f"{nworse}/{nb}"
                if not worse: osc_flips+=1
                verdict='GOLD worse' if worse else 'GOLD BETTER<<'
                print(f"{st:>8} {str(bd):>10} {g:>16} {dw:>5}  {o['weak_matched']:8.4f} {o['gold_matched']:8.4f} {nb:>5} {maj:>9} {verdict}")

print(f"\n  oscillation 'GOLD flat-to-worse' (median): held in {osc_total-osc_flips}/{osc_total} configs (flips to GOLD-better={osc_flips})")
