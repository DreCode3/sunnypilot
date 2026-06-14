import sys; sys.path.insert(0,'.'); sys.path.insert(0,'opendbc_repo')
import numpy as np
from scipy import signal

FS=50.0
WEAK=['route_b1','route_b2','route_b4','route_b5']
GOLD=['route_b8']
sos = signal.butter(4, [0.1,0.5], btype='band', fs=FS, output='sos')

def load(r):
    d=np.load(f'explorer_st_logs/_cache_reassess/{r}.npz')
    return {k:np.asarray(d[k],dtype=float) for k in d.keys()}

WIN=20.0; WN=int(WIN*FS); STEP=int(5*FS)

def bandrms(x):
    x=np.nan_to_num(x-np.nanmean(x))
    f=signal.sosfiltfilt(sos, x)
    return f

def weave(routes,label):
    recs=[]
    for r in routes:
        d=load(r)
        t=d['t']; spd=d['spd']; yaw=d['yaw']; lat=d['latact']
        steer=d['steer']; pos=d['pos']; cmd=d['cmd']
        # filter whole route ONCE (much faster, edge effects trimmed by window selection)
        fst  = bandrms(steer)
        fpos = bandrms(pos)
        fcmd = bandrms(cmd)
        fyaw = bandrms(yaw)
        aLat = yaw*spd; faLat = bandrms(aLat)
        straight=np.abs(yaw*spd)<1.0
        eng=(lat==1)&straight&np.isfinite(pos)&np.isfinite(cmd)&np.isfinite(yaw)&np.isfinite(steer)
        N=len(t); i=0
        while i+WN<=N:
            sl=slice(i,i+WN)
            if eng[sl].mean()>0.95:
                v=spd[sl].mean()
                recs.append((v*2.237,
                             np.sqrt(np.mean(fcmd[sl]**2)),
                             np.sqrt(np.mean(fyaw[sl]**2)),
                             np.sqrt(np.mean(faLat[sl]**2)),
                             np.sqrt(np.mean(fst[sl]**2)),
                             np.sqrt(np.mean(fpos[sl]**2))))
            i+=STEP
    recs=np.array(recs)
    print(f'=== {label}: {len(recs)} windows (20s, >95% engaged-straight) ===')
    bins=np.arange(25,75,5)
    print(f'{"spd":>7} {"n":>4} {"cmd_e4":>9} {"yaw_e3":>9} {"aLat":>9} {"steerDeg":>9} {"pos_mm":>9}')
    rows={}
    for i in range(len(bins)-1):
        m=(recs[:,0]>=bins[i])&(recs[:,0]<bins[i+1])
        if m.sum()==0: continue
        rows[bins[i]]=(m.sum(),
            np.median(recs[m,1])*1e4, np.median(recs[m,2])*1e3, np.median(recs[m,3]),
            np.median(recs[m,4]), np.median(recs[m,5])*1e3)
        print(f'{bins[i]:.0f}-{bins[i+1]:.0f} {m.sum():>4} '
              f'{np.median(recs[m,1])*1e4:>9.3f} '
              f'{np.median(recs[m,2])*1e3:>9.3f} '
              f'{np.median(recs[m,3]):>9.4f} '
              f'{np.median(recs[m,4]):>9.4f} '
              f'{np.median(recs[m,5])*1e3:>9.2f}')
    return recs, rows

rw,ow=weave(WEAK,'WEAK')
print()
rg,og=weave(GOLD,'GOLD')

# direct ratio at matched speed bins
print('\n=== GOLD/WEAK ratio (median band-RMS) at shared speed bins ===')
print(f'{"spd":>7} {"cmd":>7} {"yaw":>7} {"aLat":>7} {"steer":>7} {"pos":>7}')
for b in sorted(set(ow)&set(og)):
    w=ow[b]; g=og[b]
    print(f'{b:.0f}-{b+5:.0f} {g[1]/w[1]:>7.2f} {g[2]/w[2]:>7.2f} {g[3]/w[3]:>7.2f} {g[4]/w[4]:>7.2f} {g[5]/w[5]:>7.2f}')
