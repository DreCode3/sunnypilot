#!/usr/bin/env python3
"""b8 RE-ANALYSIS with the metrics the DRIVER actually feels (position + wheel), not just aLat.
DISPARITY HYPOTHESIS: aLat = yawRate*vEgo = lateral ACCEL = pos'' -> band power weights oscillations by f^2,
so a SLOW position ping-pong (0.1-0.2Hz) is heavily attenuated. Driver feels POSITION amplitude + WHEEL motion.
Since b1/b2 AND b8 are BOTH CD210, model-dependent position IS comparable here (avoid only across OPM7<->CD210).

Per engaged straight 16s window (latActive>=0.9, |aLat|<0.6), compute oscillation in 4 channels:
  aLat   = yawRate*vEgo                          (lateral accel; f^2-weighted -- what I used before)
  steer  = carState.steeringAngleDeg            (WHEEL sawing; model-independent; what driver sees/feels)
  pos    = -(laneLines[1].y0+laneLines[2].y0)/2 (model lane offset = POSITION ping-pong; CD210 both -> comparable)
  cmd    = carControl.actuators.curvature        (commanded curvature; controller output incl PI)
Report std + slow-band(0.1-0.5Hz) RMS. CD210_WEAK=b1,b2 vs CD210_GOLD=b8."""
import sys, glob, os, math
import numpy as np
from collections import defaultdict
import concurrent.futures as cf
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

BLOCK = 0.0025; FS = 20.0; WIN = 16.0; HOP = 8.0; ENG_MIN = 0.9
GROUPS = {'CD210_WEAK': ['route_b1', 'route_b2'], 'CD210_GOLD': ['route_b8']}


def slow_rms(x, lo=0.1, hi=0.5):
    n = len(x); t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1 / FS); p = np.abs(X) ** 2
    return math.sqrt(2 * np.sum(p[(f >= lo) & (f < hi)]) / (n * n))


def bearing(la0, lo0, la1, lo1):
    dlat = la1 - la0; dlon = (lo1 - lo0) * math.cos(math.radians((la0 + la1) / 2))
    return math.degrees(math.atan2(dlon, dlat)) % 360


def adiff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def load(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    tl=[];lat=[];lon=[];yaw=[]; tc=[];veg=[];sa=[];prs=[]; te=[];eng=[];cmd=[]; tm=[];pos=[]
    for sd in segs:
        f = sd + 'rlog.zst'
        if not os.path.exists(f):
            continue
        try:
            for msg in LogReader(f):
                w = msg.which()
                if w == 'liveLocationKalman':
                    g = msg.liveLocationKalman
                    if not g.positionGeodetic.valid or not g.angularVelocityCalibrated.valid:
                        continue
                    v = g.positionGeodetic.value; av = g.angularVelocityCalibrated.value
                    tl.append(msg.logMonoTime*1e-9); lat.append(float(v[0])); lon.append(float(v[1])); yaw.append(float(av[2]))
                elif w == 'carState':
                    cs = msg.carState
                    tc.append(msg.logMonoTime*1e-9); veg.append(float(cs.vEgo)); sa.append(float(cs.steeringAngleDeg)); prs.append(1.0 if cs.steeringPressed else 0.0)
                elif w == 'carControl':
                    te.append(msg.logMonoTime*1e-9); eng.append(1.0 if msg.carControl.latActive else 0.0)
                    try: cmd.append(float(msg.carControl.actuators.curvature))
                    except Exception: cmd.append(0.0)
                elif w == 'modelV2':
                    ll = msg.modelV2.laneLines
                    if len(ll) > 2 and len(ll[1].y) > 0 and len(ll[2].y) > 0:
                        tm.append(msg.logMonoTime*1e-9); pos.append(-(float(ll[1].y[0]) + float(ll[2].y[0]))/2)
        except Exception:
            continue
    if len(tl) < 400 or len(tc) < 400 or len(te) < 400:
        return rid, []
    tl=np.array(tl);o=np.argsort(tl);tl,lat,lon,yaw=tl[o],np.array(lat)[o],np.array(lon)[o],np.array(yaw)[o]
    tc=np.array(tc);oc=np.argsort(tc);tc,veg,sa,prs=tc[oc],np.array(veg)[oc],np.array(sa)[oc],np.array(prs)[oc]
    te=np.array(te);oe=np.argsort(te);te,eng,cmd=te[oe],np.array(eng)[oe],np.array(cmd)[oe]
    tu = np.arange(tl[0], tl[-1], 1/FS)
    latu=np.interp(tu,tl,lat);lonu=np.interp(tu,tl,lon);yawu=np.interp(tu,tl,yaw)
    vegu=np.interp(tu,tc,veg);sau=np.interp(tu,tc,sa);prsu=np.interp(tu,tc,prs)
    engu=np.interp(tu,te,eng);cmdu=np.interp(tu,te,cmd)
    if len(tm) > 100:
        tm=np.array(tm);om=np.argsort(tm);tm,posv=tm[om],np.array(pos)[om];posu=np.interp(tu,tm,posv)
    else:
        posu=np.full_like(tu, np.nan)
    aLat = yawu*vegu
    wins=[]; n=int(WIN*FS); hop=int(HOP*FS)
    for i in range(0, len(tu)-n, hop):
        sl=slice(i,i+n)
        if np.mean(engu[sl]) < ENG_MIN: continue
        la=latu[sl];lo=lonu[sl]
        if not (np.all(la>33)&np.all(la<35.5)&np.all(lo>-85.5)&np.all(lo<-83.5)): continue
        d=dict(blk=(round(np.median(la)/BLOCK),round(np.median(lo)/BLOCK)),
                absalat=float(np.median(np.abs(aLat[sl]))), spd=float(np.median(vegu[sl])*2.237),
                brg=bearing(la[0],lo[0],la[-1],lo[-1]),
                aLat_slow=slow_rms(aLat[sl]), steer_slow=slow_rms(sau[sl]), steer_std=float(np.std(sau[sl])),
                cmd_slow=slow_rms(cmdu[sl]), cmd_std=float(np.std(cmdu[sl])))
        ps=posu[sl]
        if not np.any(np.isnan(ps)):
            d['pos_slow']=slow_rms(ps); d['pos_std']=float(np.std(ps))
        wins.append(d)
    return rid, wins


allr=[r for v in GROUPS.values() for r in v]
with cf.ThreadPoolExecutor(max_workers=6) as ex:
    res=dict(ex.map(load, allr))
WINS={g:[w for r in rs for w in res.get(r,[])] for g,rs in GROUPS.items()}
for r in allr: print(f'  {r}: {len(res.get(r,[]))} eng 16s-win')

def straights(g):
    return [w for w in WINS[g] if w['absalat']<0.6 and 40<=w['spd']<=80]

print('\n=== ABSOLUTE medians on engaged STRAIGHTS (40-80mph) — driver-felt channels ===')
print(f'{"metric":<12}{"CD210_WEAK(b1/b2)":>20}{"CD210_GOLD(b8)":>18}{"change":>10}')
WK=straights('CD210_WEAK'); GD=straights('CD210_GOLD')
print(f'  n windows: WEAK {len(WK)}  GOLD {len(GD)}')
for m,lbl in [('aLat_slow','aLat slow (f^2!)'),('steer_slow','steer slow'),('steer_std','steer STD(deg)'),
              ('pos_slow','POS slow'),('pos_std','POS STD(m)'),('cmd_slow','cmd slow'),('cmd_std','cmd STD')]:
    a=[w[m] for w in WK if m in w]; b=[w[m] for w in GD if m in w]
    if len(a)>=5 and len(b)>=5:
        ma,mb=np.median(a),np.median(b)
        print(f'  {lbl:<18}{ma:>14.4f}{mb:>18.4f}{100*(mb-ma)/ma:>+9.0f}%   (n {len(a)}/{len(b)})')
    else:
        print(f'  {lbl:<18}  n too few ({len(a)}/{len(b)})')

# shared-block same-direction pairing (few, but report)
def gblk(g):
    bd=defaultdict(list)
    for w in WINS[g]:
        if w['absalat']<0.6: bd[w['blk']].append(w)
    out={}
    for k,ws in bd.items():
        bx=np.mean([math.cos(math.radians(w['brg'])) for w in ws]);by=np.mean([math.sin(math.radians(w['brg'])) for w in ws])
        o={'brg':math.degrees(math.atan2(by,bx))%360,'spd':np.median([w['spd'] for w in ws]),'n':len(ws)}
        for m in ['aLat_slow','steer_slow','steer_std','pos_slow','pos_std','cmd_slow','cmd_std']:
            vals=[w[m] for w in ws if m in w]
            if vals: o[m]=float(np.median(vals))
        out[k]=o
    return out
BW,BG=gblk('CD210_WEAK'),gblk('CD210_GOLD')
shared=[k for k in BW if k in BG and adiff(BW[k]['brg'],BG[k]['brg'])<=60 and abs(BW[k]['spd']-BG[k]['spd'])<=8]
print(f'\n=== shared same-dir straight blocks (location-matched): {len(shared)} ===')
for m in ['aLat_slow','steer_slow','steer_std','pos_slow','pos_std','cmd_slow','cmd_std']:
    pairs=[(BW[k][m],BG[k][m]) for k in shared if m in BW[k] and m in BG[k]]
    if len(pairs)>=4:
        wv=[p[0] for p in pairs];gv=[p[1] for p in pairs]
        worse=sum(1 for p in pairs if p[1]>p[0])
        print(f'  {m:<12} WEAK {np.median(wv):.4f} -> GOLD {np.median(gv):.4f} ({100*(np.median(gv)-np.median(wv))/np.median(wv):+.0f}%)  gold-higher {worse}/{len(pairs)}')
    else:
        print(f'  {m:<12} only {len(pairs)} matched blocks')
