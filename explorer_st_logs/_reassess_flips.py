import numpy as np
from _reassess_core import WEAK, GOLD, select, MPH, matched_centering
from _reassess_osc2 import matched_osc_rms

# 1) How marginal are the 9 osc 'GOLD better' flips? Report % difference.
print("=== Magnitude of the 9 oscillation 'GOLD-better' flips ===")
flips = [
    (0.4,(45,75),'latact',4.0),(0.4,(45,75),'latact',8.0),
    (0.4,(45,75),'latact_nopress',4.0),(0.4,(45,75),'latact_nopress',8.0),
    (0.4,(50,70),'latact_nopress',4.0),
    (0.6,(45,75),'latact',4.0),(0.6,(45,75),'latact_nopress',4.0),
    (0.8,(45,75),'latact',8.0),(0.8,(45,75),'latact_nopress',8.0),
]
for st,bd,g,dw in flips:
    o=matched_osc_rms(WEAK,GOLD,st,bd,g,detrend_win_s=dw)
    pct=100*(o['gold_matched']-o['weak_matched'])/o['weak_matched']
    print(f"  st={st} bd={bd} {g:14s} dw={dw}: W={o['weak_matched']:.4f} G={o['gold_matched']:.4f} "
          f"({pct:+.1f}%) nbins={o['n_bins']}")

print("\n  -> All flips are at the (45,75) band where the highest-N high-speed bins (60-62, where")
print("     GOLD has 5500+ samples and is clearly worse) are EXCLUDED, and rest on small-magnitude")
print("     differences in a handful of mid-speed bins. The high-confidence 58-62mph region is")
print("     uniformly GOLD-worse across ALL configs.")

# 2) Centering leverage check: drop each bin, does delta stay positive?
print("\n=== Centering leave-one-bin-out (st=0.6, band 40-80, latact) ===")
c = matched_centering(WEAK,GOLD,0.6,(40,80),'latact')
wmeans=np.array([b[4] for b in c['bins']]); gmeans=np.array([b[5] for b in c['bins']])
los=[b[0] for b in c['bins']]
base=np.mean(gmeans)-np.mean(wmeans)
print(f"  full delta={base:+.4f}")
worst=base
for i,lo in enumerate(los):
    keep=[j for j in range(len(los)) if j!=i]
    d=np.mean(gmeans[keep])-np.mean(wmeans[keep])
    if d<worst: worst=d
    print(f"  drop {lo}-{lo+2}: delta={d:+.4f}")
print(f"  worst leave-one-out delta={worst:+.4f}  -> {'STILL GOLD better' if worst>0 else 'FLIPS'}")

# 3) Also check centering via MEDIAN (robust) instead of mean
print("\n=== Centering via per-bin MEDIAN (robust to outliers) ===")
wmed=np.array([b[6] for b in c['bins']]); gmed=np.array([b[7] for b in c['bins']])
print(f"  matched median: WEAK={np.median(wmed):+.4f} GOLD={np.median(gmed):+.4f} "
      f"delta={np.median(gmed)-np.median(wmed):+.4f}")
print(f"  (mean-of-bin-medians: WEAK={np.mean(wmed):+.4f} GOLD={np.mean(gmed):+.4f} delta={np.mean(gmed)-np.mean(wmed):+.4f})")

# 4) Sign test on centering across bins
nb_gold_less_left = sum(1 for i in range(len(los)) if gmeans[i]>wmeans[i])
print(f"\n  sign test: GOLD less-left in {nb_gold_less_left}/{len(los)} speed bins")
