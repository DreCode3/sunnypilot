import sys; sys.path.insert(0,'.'); sys.path.insert(0,'opendbc_repo')
import numpy as np
from scipy import signal, stats

FS=50.0
WEAK=['route_b1','route_b2','route_b4','route_b5']
GOLD=['route_b8']
sos = signal.butter(4, [0.1,0.5], btype='band', fs=FS, output='sos')
def load(r):
    d=np.load(f'explorer_st_logs/_cache_reassess/{r}.npz')
    return {k:np.asarray(d[k],dtype=float) for k in d.keys()}
def bp(x): return signal.sosfiltfilt(sos, np.nan_to_num(x-np.nanmean(x)))
WIN=20.0; WN=int(WIN*FS); STEP=int(5*FS)

def collect(routes):
    recs=[]
    for r in routes:
        d=load(r); t=d['t']; spd=d['spd']; yaw=d['yaw']; lat=d['latact']
        steer=d['steer']; pos=d['pos']; cmd=d['cmd']
        fst=bp(steer); fpos=bp(pos); fcmd=bp(cmd); faLat=bp(yaw*spd)
        straight=np.abs(yaw*spd)<1.0
        eng=(lat==1)&straight&np.isfinite(pos)&np.isfinite(cmd)&np.isfinite(yaw)&np.isfinite(steer)
        N=len(t); i=0
        while i+WN<=N:
            sl=slice(i,i+WN)
            if eng[sl].mean()>0.95:
                v=spd[sl].mean()
                recs.append((v*2.237,
                    np.sqrt(np.mean(fcmd[sl]**2)),
                    np.sqrt(np.mean(faLat[sl]**2)),
                    np.sqrt(np.mean(fst[sl]**2)),
                    np.sqrt(np.mean(fpos[sl]**2)),
                    fst[sl].max()-fst[sl].min(),
                    fpos[sl].max()-fpos[sl].min()))
            i+=STEP
    return np.array(recs)

w=collect(WEAK); g=collect(GOLD)
# 55-65 band (the speeds the driver actually drives golden)
for lo,hi,name in [(55,65,'55-65 driver-band'),(60,65,'60-65 best-powered')]:
    mw=(w[:,0]>=lo)&(w[:,0]<hi); mg=(g[:,0]>=lo)&(g[:,0]<hi)
    print(f'\n=== {name}: WEAK n={mw.sum()}  GOLD n={mg.sum()} ===')
    labels=['cmd_e4','aLat','steerDeg','pos_mm','p2p_steer','p2p_pos_mm']
    scales=[1e4,1,1,1e3,1,1e3]
    for j,(lab,sc) in enumerate(zip(labels,scales),start=1):
        ww=w[mw,j]*sc; gg=g[mg,j]*sc
        # Mann-Whitney one-sided: is GOLD < WEAK (improvement)?
        try:
            U,p_less=stats.mannwhitneyu(gg,ww,alternative='less')
        except Exception:
            p_less=np.nan
        print(f'  {lab:>11}: WEAK med={np.median(ww):8.3f} (mean {np.mean(ww):8.3f})  '
              f'GOLD med={np.median(gg):8.3f} (mean {np.mean(gg):8.3f})  '
              f'ratio={np.median(gg)/np.median(ww):.2f}  p(gold<weak)={p_less:.3f}')
    # tail: 90th pct
    print('  -- 90th percentile (tail / worst weave) --')
    for j,(lab,sc) in enumerate(zip(labels,scales),start=1):
        ww=np.percentile(w[mw,j]*sc,90); gg=np.percentile(g[mg,j]*sc,90)
        print(f'  {lab:>11}: WEAK p90={ww:8.3f}  GOLD p90={gg:8.3f}  ratio={gg/ww:.2f}')
