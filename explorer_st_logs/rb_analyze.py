#!/usr/bin/env python3
"""Reviewer B core analysis from cached npz."""
import numpy as np, math
FS=20.0
WIN=16.0; HOP=8.0; ENG_MIN=0.9; BLOCK=0.0025
WEAK=['route_b1','route_b2']; GOLD=['route_b8']

def load(r):
    return np.load(f'explorer_st_logs/_rb_cache/{r}.npz')

def band_rms(x, lo=0.1, hi=0.5):
    n=len(x); t=np.arange(n)
    x=x-np.polyval(np.polyfit(t,x,1),t)
    X=np.fft.rfft(x); f=np.fft.rfftfreq(n,d=1/FS); p=np.abs(X)**2
    return math.sqrt(2*np.sum(p[(f>=lo)&(f<hi)])/(n*n))

def hf_rms(x, lo=2.0, hi=9.0):
    n=len(x); t=np.arange(n)
    x=x-np.polyval(np.polyfit(t,x,1),t)
    X=np.fft.rfft(x); f=np.fft.rfftfreq(n,d=1/FS); p=np.abs(X)**2
    return math.sqrt(2*np.sum(p[(f>=lo)&(f<hi)])/(n*n))

def bearing(la0,lo0,la1,lo1):
    dlat=la1-la0; dlon=(lo1-lo0)*math.cos(math.radians((la0+la1)/2))
    return math.degrees(math.atan2(dlon,dlat))%360

def adiff(a,b):
    dd=abs(a-b)%360
    return dd if dd<=180 else 360-dd

def windows(r):
    d=load(r)
    tu=d['tu'];lat=d['lat'];lon=d['lon'];yaw=d['yaw'];veg=d['veg'];sa=d['sa']
    prs=d['prs'];eng=d['eng'];cmd=d['cmd'];pos=d['pos'];desc=d['desc']
    aLat=yaw*veg
    n=int(WIN*FS); hop=int(HOP*FS); out=[]
    for i in range(0,len(tu)-n,hop):
        sl=slice(i,i+n)
        if np.mean(eng[sl])<ENG_MIN: continue
        la=lat[sl];lo=lon[sl]
        if not (np.all(la>33)&np.all(la<35.5)&np.all(lo>-85.5)&np.all(lo<-83.5)): continue
        absalat=float(np.median(np.abs(aLat[sl])))
        spd=float(np.median(veg[sl])*2.237)
        d0=dict(blk=(round(np.median(la)/BLOCK),round(np.median(lo)/BLOCK)),
                absalat=absalat, spd=spd, brg=bearing(la[0],lo[0],la[-1],lo[-1]),
                steer_band=band_rms(sa[sl]), steer_std=float(np.std(sa[sl])),
                steer_hf=hf_rms(sa[sl]),
                aLat_band=band_rms(aLat[sl]),
                cmd_band=band_rms(cmd[sl]), cmd_std=float(np.std(cmd[sl])),
                pos_band=band_rms(pos[sl]), pos_std=float(np.std(pos[sl])), pos_hf=hf_rms(pos[sl]),
                prs=float(np.mean(prs[sl])))
        out.append(d0)
    return out

W={r:windows(r) for r in WEAK+GOLD}
def straights(rs):
    return [w for r in rs for w in W[r] if w['absalat']<0.6 and 40<=w['spd']<=80 and w['prs']<0.1]

WK=straights(WEAK); GD=straights(GOLD)
print(f"=== ENGAGED STRAIGHTS (40-80mph, |aLat|<0.6, NOT pressed) ===")
print(f"n windows: WEAK(b1+b2) {len(WK)}  GOLD(b8) {len(GD)}")
print(f"{'metric':<14}{'WEAK':>10}{'GOLD':>10}{'change':>9}")
for m,lbl in [('steer_band','steer 0.1-0.5'),('steer_std','steer STD'),
              ('steer_hf','steer HF>2Hz'),('aLat_band','aLat 0.1-0.5'),
              ('cmd_band','cmd 0.1-0.5'),('cmd_std','cmd STD'),
              ('pos_band','pos 0.1-0.5'),('pos_std','pos STD'),('pos_hf','pos HF>2Hz')]:
    a=[w[m] for w in WK]; b=[w[m] for w in GD]
    ma,mb=np.median(a),np.median(b)
    print(f"  {lbl:<12}{ma:>10.4f}{mb:>10.4f}{100*(mb-ma)/ma:>+8.0f}%")

# bootstrap CI on steer_std and steer_band ratio (gold/weak)
rng=np.random.default_rng(0)
def boot_ratio(a,b,m):
    av=np.array([w[m] for w in a]);bv=np.array([w[m] for w in b])
    rs=[]
    for _ in range(4000):
        ra=rng.choice(av,len(av));rb=rng.choice(bv,len(bv))
        rs.append(np.median(rb)/np.median(ra))
    return np.percentile(rs,[2.5,50,97.5])
for m in ['steer_band','steer_std','aLat_band','pos_band','cmd_band']:
    lo,md,hi=boot_ratio(WK,GD,m)
    print(f"  ratio GOLD/WEAK {m:<11} median {md:.2f}  95%CI [{lo:.2f},{hi:.2f}]")

# speed distribution check
print("\n=== speed distribution of straight windows (mph) ===")
print(f"  WEAK: median {np.median([w['spd'] for w in WK]):.1f}  IQR [{np.percentile([w['spd'] for w in WK],25):.0f},{np.percentile([w['spd'] for w in WK],75):.0f}]")
print(f"  GOLD: median {np.median([w['spd'] for w in GD]):.1f}  IQR [{np.percentile([w['spd'] for w in GD],25):.0f},{np.percentile([w['spd'] for w in GD],75):.0f}]")
