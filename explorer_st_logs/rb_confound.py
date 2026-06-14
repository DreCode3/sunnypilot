#!/usr/bin/env python3
"""Reviewer B: speed confound + GPS location-match confound."""
import numpy as np, math
exec(open('explorer_st_logs/rb_analyze.py').read().split('print(f"=== ENGAGED')[0])
# now WK, GD, W, straights() available

WK=straights(WEAK); GD=straights(GOLD)

# --- 1. SPEED CONFOUND ---
print("=== SPEED CONFOUND: steer metrics binned by speed (mph) ===")
print(f"{'bin':<10}{'WEAK n':>7}{'WEAK std':>10}{'GOLD n':>7}{'GOLD std':>10}{'chg':>7}{'  WEAK band':>11}{'GOLD band':>10}{'chg':>7}")
for lo,hi in [(40,50),(50,57),(57,65),(65,80)]:
    wk=[w for w in WK if lo<=w['spd']<hi]; gd=[w for w in GD if lo<=w['spd']<hi]
    if len(wk)>=3 and len(gd)>=3:
        ws=np.median([w['steer_std'] for w in wk]); gs=np.median([w['steer_std'] for w in gd])
        wb=np.median([w['steer_band'] for w in wk]); gb=np.median([w['steer_band'] for w in gd])
        print(f"  {lo}-{hi:<6}{len(wk):>7}{ws:>10.3f}{len(gd):>7}{gs:>10.3f}{100*(gs-ws)/ws:>+6.0f}%{wb:>11.3f}{gb:>10.3f}{100*(gb-wb)/wb:>+6.0f}%")
    else:
        print(f"  {lo}-{hi:<6}{len(wk):>7}{'--':>10}{len(gd):>7}{'--':>10}  (too few)")

# regression: does steer_std depend on speed within WEAK alone? (establishes the speed sensitivity)
def slope(ws):
    s=np.array([w['spd'] for w in ws]); y=np.array([w['steer_std'] for w in ws])
    return np.polyfit(s,y,1)[0]
print(f"\n  steer_std vs speed slope: WEAK {slope(WK):+.4f} deg/mph, GOLD {slope(GD):+.4f}")
print(f"  speed gap GOLD-WEAK = {np.median([w['spd'] for w in GD])-np.median([w['spd'] for w in WK]):+.1f} mph")
exp_from_speed = slope(WK)*(np.median([w['spd'] for w in GD])-np.median([w['spd'] for w in WK]))
print(f"  predicted steer_std change from speed gap alone: {exp_from_speed:+.3f} deg "
      f"(vs observed {np.median([w['steer_std'] for w in GD])-np.median([w['steer_std'] for w in WK]):+.3f})")

# --- 2. GPS LOCATION OVERLAP ---
print("\n=== GPS BLOCK OVERLAP (engaged straight windows, ~275m blocks) ===")
def blkset(rs):
    bd={}
    for r in rs:
        for w in W[r]:
            if w['absalat']<0.6 and w['prs']<0.1 and 40<=w['spd']<=80:
                bd.setdefault(w['blk'],[]).append(w)
    return bd
bw=blkset(WEAK); bg=blkset(GOLD)
print(f"  WEAK distinct straight blocks: {len(bw)}   GOLD: {len(bg)}")
shared_any=[k for k in bw if k in bg]
print(f"  shared blocks (any direction): {len(shared_any)}")
def mbrg(ws):
    bx=np.mean([math.cos(math.radians(w['brg'])) for w in ws]);by=np.mean([math.sin(math.radians(w['brg'])) for w in ws])
    return math.degrees(math.atan2(by,bx))%360
shared_dir=[k for k in shared_any if adiff(mbrg(bw[k]),mbrg(bg[k]))<=45]
shared_dir_spd=[k for k in shared_dir if abs(np.median([w['spd'] for w in bw[k]])-np.median([w['spd'] for w in bg[k]]))<=8]
print(f"  shared + same-direction (<=45deg): {len(shared_dir)}")
print(f"  shared + same-dir + speed-matched (<=8mph): {len(shared_dir_spd)}")
print(f"  --> GOLD straight time on shared blocks: {sum(len(bg[k]) for k in shared_dir)} of {sum(len(v) for v in bg.values())} windows")

print("\n  Location-matched paired steer (shared+dir, NOT speed-gated):")
for m in ['steer_std','steer_band','aLat_band','pos_band']:
    pairs=[(np.median([w[m] for w in bw[k]]),np.median([w[m] for w in bg[k]])) for k in shared_dir]
    if len(pairs)>=3:
        wv=[p[0] for p in pairs];gv=[p[1] for p in pairs]
        gold_lower=sum(1 for p in pairs if p[1]<p[0])
        print(f"    {m:<11} WEAK {np.median(wv):.3f} -> GOLD {np.median(gv):.3f} ({100*(np.median(gv)-np.median(wv))/np.median(wv):+.0f}%) gold-lower {gold_lower}/{len(pairs)}")
    else:
        print(f"    {m:<11} only {len(pairs)} pairs")

print("\n  Location+speed-matched paired steer:")
for m in ['steer_std','steer_band']:
    pairs=[(np.median([w[m] for w in bw[k]]),np.median([w[m] for w in bg[k]]),
            np.median([w['spd'] for w in bw[k]]),np.median([w['spd'] for w in bg[k]])) for k in shared_dir_spd]
    if len(pairs)>=2:
        for p in pairs:
            print(f"    {m} blk: WEAK {p[0]:.3f}@{p[2]:.0f}mph -> GOLD {p[1]:.3f}@{p[3]:.0f}mph")
    else:
        print(f"    {m}: only {len(pairs)} location+speed pairs")
