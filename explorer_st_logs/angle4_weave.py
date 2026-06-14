import numpy as np
import sys

CACHE='explorer_st_logs/_cache_reassess'
WEAK=['route_b1','route_b2','route_b4','route_b5']
GOLD=['route_b8']

def load(r):
    return np.load(f'{CACHE}/{r}.npz')

R_EARTH=6371000.0
def ll_to_m(lat,lon,lat0,lon0):
    x=np.radians(lon-lon0)*R_EARTH*np.cos(np.radians(lat0))
    y=np.radians(lat-lat0)*R_EARTH
    return x,y

def get_engaged(d, mph_lo=30, mph_hi=75, gentle=True):
    """Return mask of engaged samples in straights+gentle curves within speed band (main 50Hz grid)."""
    eng=d['latact']==1
    spd=d['spd']
    mph=spd*2.237
    m=eng & (mph>=mph_lo) & (mph<=mph_hi)
    if gentle:
        # lateral accel = yawRate*vEgo ; gentle = |aLat|<1.0
        aLat=np.abs(d['yaw']*spd)
        m=m & (aLat<1.0)
    return m

def bandpass(x, fs, lo, hi):
    """Zero-phase FFT bandpass."""
    x=np.asarray(x,float)
    x=x-np.nanmean(x)
    n=len(x)
    X=np.fft.rfft(x)
    f=np.fft.rfftfreq(n,1/fs)
    H=((f>=lo)&(f<=hi)).astype(float)
    return np.fft.irfft(X*H, n=n)
