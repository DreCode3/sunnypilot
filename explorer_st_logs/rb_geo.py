#!/usr/bin/env python3
"""Reviewer B: characterize WHY overlap is poor - bearings, geographic spread, and a
properly speed-matched pooled comparison (subsample GOLD to WEAK's speed distribution)."""
import numpy as np, math
exec(open('explorer_st_logs/rb_analyze.py').read().split('print(f"=== ENGAGED')[0])
WK=straights(WEAK); GD=straights(GOLD)

# geographic extent
def extent(rs):
    la=[];lo=[]
    for r in rs:
        for w in W[r]:
            la.append(w['blk'][0]*BLOCK);lo.append(w['blk'][1]*BLOCK)
    return la,lo
wla,wlo=extent(WEAK); gla,glo=extent(GOLD)
print("=== geographic extent of straight windows ===")
print(f"  WEAK lat[{min(wla):.3f},{max(wla):.3f}] lon[{min(wlo):.3f},{max(wlo):.3f}]")
print(f"  GOLD lat[{min(gla):.3f},{max(gla):.3f}] lon[{min(glo):.3f},{max(glo):.3f}]")

# bearing histograms
print("\n=== bearing distribution (straight windows) ===")
def bhist(ws,lbl):
    b=np.array([w['brg'] for w in ws])
    h,_=np.histogram(b,bins=np.arange(0,361,45))
    print(f"  {lbl}: "+" ".join(f"{int(e)}-{int(e)+45}:{c}" for e,c in zip(range(0,360,45),h)))
bhist(WK,'WEAK'); bhist(GD,'GOLD')

# On the 28 shared blocks, how many are GOLD opposite-direction (return trip)?
def blkset(rs):
    bd={}
    for r in rs:
        for w in W[r]:
            if w['absalat']<0.6 and w['prs']<0.1 and 40<=w['spd']<=80:
                bd.setdefault(w['blk'],[]).append(w)
    return bd
def adiff(a,b):
    dd=abs(a-b)%360; return dd if dd<=180 else 360-dd
def mbrg(ws):
    bx=np.mean([math.cos(math.radians(w['brg'])) for w in ws]);by=np.mean([math.sin(math.radians(w['brg'])) for w in ws])
    return math.degrees(math.atan2(by,bx))%360
bw=blkset(WEAK); bg=blkset(GOLD)
shared=[k for k in bw if k in bg]
oppo=sum(1 for k in shared if adiff(mbrg(bw[k]),mbrg(bg[k]))>=135)
samed=sum(1 for k in shared if adiff(mbrg(bw[k]),mbrg(bg[k]))<=45)
cross=len(shared)-oppo-samed
print(f"\n=== of {len(shared)} shared blocks: same-dir(<=45) {samed}, opposite(>=135) {oppo}, cross {cross} ===")
print("  -> if mostly opposite, b8 is the RETURN trip on same roads (same geometry, reverse dir)")

# SPEED-MATCHED POOLED: subsample to common speed support (overlap window) and reweight
print("\n=== SPEED-MATCHED pooled (restrict BOTH to 50-62mph overlap band) ===")
band=(50,62)
wk=[w for w in WK if band[0]<=w['spd']<band[1]]; gd=[w for w in GD if band[0]<=w['spd']<band[1]]
print(f"  n WEAK {len(wk)} (med {np.median([w['spd'] for w in wk]):.1f}mph)  GOLD {len(gd)} (med {np.median([w['spd'] for w in gd]):.1f}mph)")
for m,lbl in [('steer_std','steer STD'),('steer_band','steer 0.1-0.5'),('aLat_band','aLat band'),('pos_band','pos band'),('cmd_band','cmd band')]:
    if len(wk)>=5 and len(gd)>=5:
        ma=np.median([w[m] for w in wk]); mb=np.median([w[m] for w in gd])
        print(f"    {lbl:<13} {ma:.4f} -> {mb:.4f} ({100*(mb-ma)/ma:+.0f}%)")

# stratified (per-bin) reweight to WEAK speed dist
print("\n=== STRATIFIED: GOLD reweighted to WEAK speed histogram ===")
bins=[(40,50),(50,57),(57,65)]
def strat_median(ws_target_dist, ws_eval, m):
    # weight each bin by target distribution count, take pooled median within reweighting
    vals=[];
    for lo,hi in bins:
        tcount=len([w for w in ws_target_dist if lo<=w['spd']<hi])
        ev=[w[m] for w in ws_eval if lo<=w['spd']<hi]
        if ev and tcount:
            # replicate eval bin values proportional to target count
            reps=max(1,round(tcount/len(ev)*10))
            vals+= ev*reps
    return np.median(vals) if vals else float('nan')
for m,lbl in [('steer_std','steer STD'),('steer_band','steer band')]:
    wkv=strat_median(WK,WK,m)  # weak on its own dist
    gdv=strat_median(WK,GD,m)  # gold reweighted to weak dist
    print(f"    {lbl:<11} WEAK {wkv:.3f} -> GOLD(reweighted) {gdv:.3f} ({100*(gdv-wkv)/wkv:+.0f}%)")
