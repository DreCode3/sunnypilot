#!/usr/bin/env python3
"""Reviewer B: (3) POS noise floor analysis, (4) sharp-curve takeover characterization."""
import numpy as np, math
FS=20.0
def load(r): return np.load(f'explorer_st_logs/_rb_cache/{r}.npz')

# --- (3) POS NOISE FLOOR ---
# Compare in-band (0.1-0.5Hz) power to HF (2-9Hz) noise floor for pos vs steer.
# If pos HF noise >= pos in-band signal, pos is too noisy to resolve the slow weave.
def band(x,lo,hi):
    n=len(x);t=np.arange(n);x=x-np.polyval(np.polyfit(t,x,1),t)
    X=np.fft.rfft(x);f=np.fft.rfftfreq(n,d=1/FS);p=np.abs(X)**2
    return math.sqrt(2*np.sum(p[(f>=lo)&(f<hi)])/(n*n))
print("=== (3) POS vs STEER signal-to-noise on engaged straights ===")
print("  (in-band 0.1-0.5Hz 'signal' vs HF 2-9Hz 'noise floor', per 16s window, median over windows)")
for r in ['route_b1','route_b2','route_b8']:
    d=load(r);tu=d['tu'];eng=d['eng'];veg=d['veg'];yaw=d['yaw'];pos=d['pos'];sa=d['sa'];prs=d['prs']
    aLat=yaw*veg
    n=int(16*FS);hop=int(8*FS)
    pb=[];pn=[];sb=[];sn=[]
    for i in range(0,len(tu)-n,hop):
        sl=slice(i,i+n)
        if np.mean(eng[sl])<0.9: continue
        if np.median(np.abs(aLat[sl]))>=0.6: continue
        spd=np.median(veg[sl])*2.237
        if not (40<=spd<=80): continue
        if np.mean(prs[sl])>=0.1: continue
        pb.append(band(pos[sl],0.1,0.5));pn.append(band(pos[sl],2,9))
        sb.append(band(sa[sl],0.1,0.5));sn.append(band(sa[sl],2,9))
    if pb:
        print(f"  {r}: POS sig {np.median(pb):.4f} noise {np.median(pn):.4f} SNR {np.median(pb)/np.median(pn):.1f}x | "
              f"STEER sig {np.median(sb):.3f} noise {np.median(sn):.4f} SNR {np.median(sb)/np.median(sn):.0f}x  (n {len(pb)})")

# --- (4) SHARP CURVE TAKEOVER ---
print("\n=== (4) SHARP-CURVE TAKEOVER in b8 (driver said: less authority, had to take over) ===")
d=load('route_b8')
tu=d['tu'];sa=d['sa'];prs=d['prs'];eng=d['eng'];cmd=d['cmd'];desc=d['desc'];veg=d['veg'];yaw=d['yaw'];lat=d['lat'];lon=d['lon']
aLat=yaw*veg
# find override episodes: steeringPressed rising while in/near a curve (|desc| high)
press=prs>0.5
# episode = contiguous pressed region
epis=[]
i=0
while i<len(press):
    if press[i]:
        j=i
        while j<len(press) and press[j]: j+=1
        if (j-i)>=10:  # >=0.5s
            epis.append((i,j))
        i=j
    else:
        i+=1
print(f"  pressed episodes >=0.5s: {len(epis)}")
# rank by max |desiredCurvature| just BEFORE/at override (curve sharpness)
scored=[]
for (a,b) in epis:
    pre=slice(max(0,a-40),a)  # 2s before press onset
    dmax=np.nanmax(np.abs(desc[pre])) if a>0 else 0
    scored.append((dmax,a,b))
scored.sort(reverse=True)
print(f"  {'desc_pre':>9}{'spd':>6}{'cmd@on':>9}{'desc@on':>9}{'gap':>8}{'cmd/desc':>9}  GPS")
for dmax,a,b in scored[:6]:
    # window: 3s before to onset (the entry the controller was steering before takeover)
    ent=slice(max(0,a-60),a)
    spd=np.median(veg[ent])*2.237
    cmd_on=np.median(cmd[ent][-20:]) if a>=20 else cmd[a]
    desc_on=np.median(desc[ent][-20:]) if a>=20 else desc[a]
    gap=desc_on-cmd_on
    ratio=cmd_on/desc_on if abs(desc_on)>1e-5 else float('nan')
    print(f"  {dmax:>9.4f}{spd:>6.0f}{cmd_on:>+9.4f}{desc_on:>+9.4f}{gap:>+8.4f}{ratio:>9.2f}  {lat[a]:.4f},{lon[a]:.4f}")

# For the sharpest episode, dump the entry trajectory of cmd vs desc to see undershoot/rate-limit
dmax,a,b=scored[0]
print(f"\n  SHARPEST entry trace (curve desc_pre={dmax:.4f}, {lat[a]:.4f},{lon[a]:.4f}):")
print(f"  {'t-rel(s)':>9}{'desc':>9}{'cmd':>9}{'cmd-desc':>10}{'steer':>8}{'pressed':>8}")
for k in range(a-60,min(b,a+20),4):
    if k<0: continue
    print(f"  {(k-a)/FS:>9.1f}{desc[k]:>+9.4f}{cmd[k]:>+9.4f}{cmd[k]-desc[k]:>+10.4f}{sa[k]:>+8.1f}{int(prs[k]>0.5):>8}")
