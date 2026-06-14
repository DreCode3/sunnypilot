#!/usr/bin/env python3
"""Reviewer B: highway sharp-curve authority. In curves PI gates OFF (|cmd|>=0.005),
so cmd should track desc regardless of PI config. Test cmd-vs-desc undershoot at speed,
and whether b8 differs from b1/b2 in curve tracking (it should NOT if PI is the only change)."""
import numpy as np, math
FS=20.0
def load(r): return np.load(f'explorer_st_logs/_rb_cache/{r}.npz')

print("=== highway curve tracking: |desc|>0.01 & speed>35mph & engaged & NOT pressed ===")
print("  cmd vs desiredCurvature; undershoot = peak|cmd| / peak|desc| at apex")
for r in ['route_b1','route_b2','route_b8']:
    d=load(r);tu=d['tu'];cmd=d['cmd'];desc=d['desc'];veg=d['veg'];eng=d['eng'];prs=d['prs']
    spd=veg*2.237
    mask=(np.abs(desc)>0.01)&(spd>35)&(eng>0.5)&(prs<0.5)
    # contiguous curve segments
    segs=[];i=0
    while i<len(mask):
        if mask[i]:
            j=i
            while j<len(mask) and mask[j]: j+=1
            if (j-i)>=20: segs.append((i,j))  # >=1s
            i=j
        else: i+=1
    ratios=[];lags=[]
    for a,b in segs:
        pk_d=np.max(np.abs(desc[a:b])); pk_c=np.max(np.abs(cmd[a:b]))
        if pk_d>0.012:
            ratios.append(pk_c/pk_d)
    if ratios:
        print(f"  {r}: {len(segs)} hwy curve segs, apex cmd/desc ratio median {np.median(ratios):.2f} "
              f"(IQR {np.percentile(ratios,25):.2f}-{np.percentile(ratios,75):.2f}), "
              f"frac<0.9 (undershoot) {np.mean(np.array(ratios)<0.9):.0%}")

# Now find b8 highway curves where driver TOOK OVER (pressed) at >35mph and was the curve
# sharper than what cmd delivered? Characterize the undershoot at takeover.
print("\n=== b8 HIGHWAY takeovers (>35mph, in a curve |desc|>0.02) ===")
d=load('route_b8')
tu=d['tu'];sa=d['sa'];prs=d['prs'];eng=d['eng'];cmd=d['cmd'];desc=d['desc'];veg=d['veg'];lat=d['lat'];lon=d['lon']
spd=veg*2.237
press=prs>0.5
epis=[];i=0
while i<len(press):
    if press[i]:
        j=i
        while j<len(press) and press[j]: j+=1
        if (j-i)>=6: epis.append((i,j))
        i=j
    else: i+=1
cands=[]
for a,b in epis:
    pre=slice(max(0,a-40),a)
    if np.median(spd[pre])>35 and np.nanmax(np.abs(desc[pre]))>0.02:
        cands.append((np.nanmax(np.abs(desc[pre])),a,b))
cands.sort(reverse=True)
print(f"  highway curve takeovers: {len(cands)}")
print(f"  {'desc_pre':>9}{'spd':>6}{'cmd_apex':>10}{'desc_apex':>10}{'ratio':>7}  GPS@onset")
for dmax,a,b in cands[:8]:
    pre=slice(max(0,a-40),a)
    pkc=np.max(np.abs(cmd[pre]));pkd=np.max(np.abs(desc[pre]))
    print(f"  {dmax:>9.4f}{np.median(spd[pre]):>6.0f}{pkc:>10.4f}{pkd:>10.4f}{pkc/pkd:>7.2f}  {lat[a]:.4f},{lon[a]:.4f}")
