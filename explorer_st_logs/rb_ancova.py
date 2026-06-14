#!/usr/bin/env python3
"""Reviewer B: ANCOVA-style speed-adjusted comparison + per-bin bootstrap stability."""
import numpy as np, math
exec(open('explorer_st_logs/rb_analyze.py').read().split('print(f"=== ENGAGED')[0])
WK=straights(WEAK); GD=straights(GOLD)
rng=np.random.default_rng(1)

# Pool, regress metric ~ speed + group, test group coefficient.
def ancova(m):
    spd=np.array([w['spd'] for w in WK+GD])
    y=np.array([w[m] for w in WK+GD])
    g=np.array([0]*len(WK)+[1]*len(GD))  # 1=GOLD
    X=np.column_stack([np.ones_like(spd), spd-spd.mean(), g])
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    resid=y-X@beta
    # bootstrap CI on group coef
    bs=[]
    idx=np.arange(len(y))
    for _ in range(4000):
        s=rng.choice(idx,len(idx))
        b,_,_,_=np.linalg.lstsq(X[s],y[s],rcond=None)
        bs.append(b[2])
    return beta[2], np.percentile(bs,[2.5,97.5]), beta[1]
print("=== ANCOVA: metric ~ speed + GROUP (group coef = GOLD effect at matched speed) ===")
for m,lbl in [('steer_std','steer STD(deg)'),('steer_band','steer band(deg)'),
              ('aLat_band','aLat band'),('pos_band','pos band'),('cmd_band','cmd band')]:
    coef,ci,spdslope=ancova(m)
    base=np.median([w[m] for w in WK])
    sig='SIG' if (ci[0]>0)==(ci[1]>0) else 'n.s.'
    print(f"  {lbl:<15} GOLD effect {coef:+.4f} (={100*coef/base:+.0f}% of WEAK)  95%CI[{ci[0]:+.4f},{ci[1]:+.4f}] {sig}  spd_slope {spdslope:+.4f}")

# per-bin bootstrap of the steer_std ratio, to show instability
print("\n=== per-speed-bin steer_std ratio GOLD/WEAK with bootstrap CI ===")
for lo,hi in [(40,50),(50,57),(57,65)]:
    wk=[w['steer_std'] for w in WK if lo<=w['spd']<hi]
    gd=[w['steer_std'] for w in GD if lo<=w['spd']<hi]
    if len(wk)>=3 and len(gd)>=3:
        rs=[np.median(rng.choice(gd,len(gd)))/np.median(rng.choice(wk,len(wk))) for _ in range(4000)]
        lo_,md,hi_=np.percentile(rs,[2.5,50,97.5])
        print(f"  {lo}-{hi}mph (n {len(wk)}/{len(gd)}): ratio {md:.2f} 95%CI[{lo_:.2f},{hi_:.2f}]")
    else:
        print(f"  {lo}-{hi}mph: too few ({len(wk)}/{len(gd)})")
