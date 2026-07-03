#!/usr/bin/env python3
"""Find the user's exact signature: lane-line confidence COLLAPSES as the road starts to
curve, and the model's commanded curvature drops toward 0 (car goes straight). Report each
event with seg/frameId/GPS so we can pull the video frames at the collapse.
"""
import glob, numpy as np, bisect, math
from openpilot.tools.lib.logreader import LogReader

def scan(route):
    # per modelV2 frame: t, seg, frameId, dc, innerLP ; plus vEgo, latActive, GPS
    MV=[]; CST=[]; CC=[]; GPS=[]
    import os
    def segnum(p): return int(os.path.basename(os.path.dirname(p)).split('--')[-1])
    for rl in sorted(glob.glob(f'explorer_st_logs/{route}/*/rlog.zst'), key=segnum):
        seg=segnum(rl)
        try: lr=LogReader(rl)
        except Exception: continue
        for m in lr:
            try: w=m.which()
            except Exception: continue
            t=m.logMonoTime
            if w=='modelV2':
                M=m.modelV2; p=list(M.laneLineProbs)
                MV.append((t,seg,int(M.frameId),float(M.action.desiredCurvature),
                           min(p[1:3]) if len(p)>=3 else 1.0, max(p[1:3]) if len(p)>=3 else 1.0))
            elif w=='carState': CST.append((t,float(m.carState.vEgo)))
            elif w=='carControl': CC.append((t,bool(m.carControl.latActive)))
            elif w=='liveLocationKalman':
                g=m.liveLocationKalman
                try:
                    pg=g.positionGeodetic.value
                    if len(pg)>=2 and g.positionGeodetic.valid: GPS.append((t,float(pg[0]),float(pg[1])))
                except Exception: pass
    MV.sort(); CST.sort(); CC.sort(); GPS.sort()
    ct=[x[0] for x in CST]; at=[x[0] for x in CC]; gt=[x[0] for x in GPS]
    def pv(a,ts,t):
        i=bisect.bisect_right(ts,t)-1; return a[i] if i>=0 else None
    # build aligned per-frame
    F=[]
    for (t,seg,fid,dc,ilp,xlp) in MV:
        v=pv(CST,ct,t); la=pv(CC,at,t); g=pv(GPS,gt,t)
        F.append(dict(t=t,seg=seg,fid=fid,dc=dc,ilp=ilp,xlp=xlp,
                      v=v[1] if v else 0,la=la[1] if la else False,
                      lat=g[1] if g else None,lon=g[2] if g else None))
    # detect collapse: innerLP goes >0.6 -> <0.2 within ~1.5s (30 frames) while active v>13
    events=[]
    n=len(F)
    i=0
    while i<n:
        f=F[i]
        if f['la'] and f['v']>13 and f['ilp']<0.2:
            # was there a >0.6 within the preceding 30 frames?
            lo=max(0,i-30); pre=F[lo:i]
            if pre and max(p['ilp'] for p in pre)>0.6:
                # curve context: max |dc| in the window [i-30, i+10] (model tried to curve?)
                hi=min(n,i+15); ctx=F[lo:hi]
                dcmax=max(abs(p['dc']) for p in ctx)
                # advance past this collapse (until ilp recovers >0.5) to cluster
                j=i
                while j<n and not (F[j]['ilp']>0.5): j+=1
                dur=(F[min(j,n-1)]['t']-f['t'])/1e9
                events.append(dict(seg=f['seg'],fid=f['fid'],lat=f['lat'],lon=f['lon'],
                                   v=f['v'],dur=dur,dcmax_ctx=dcmax,ilp_min=min(p['ilp'] for p in F[i:max(j,i+1)])))
                i=j
                continue
        i+=1
    return events

import sys
ROUTES = sys.argv[1:] or ['route_cf','route_ce']
print(f"{'route':10s} {'build':10s} {'events':7s} {'active_min':10s} {'events/min':10s}  deepest(ilp<0.1)")
BUILD={'route_7f':'OLD','route_b5':'OLD','route_c5':'OLD','route_c7':'OLD','route_c4':'OLD','route_c6':'OLD','route_ce':'NEW','route_cf':'NEW'}
for route in ROUTES:
    evs=scan(route)
    # active minutes: reuse scan internals cheaply — approximate from events' route by re-scanning F? Instead recompute active frames.
    import glob as _g, bisect as _b
    from openpilot.tools.lib.logreader import LogReader as _LR
    nact=0
    for rl in _g.glob(f'explorer_st_logs/{route}/*/rlog.zst'):
        CST=[];CC=[]
        try: lr=_LR(rl)
        except Exception: continue
        for m in lr:
            try: w=m.which()
            except Exception: continue
            if w=='carState': CST.append((m.logMonoTime,float(m.carState.vEgo)))
            elif w=='carControl': CC.append((m.logMonoTime,bool(m.carControl.latActive)))
        CST.sort();CC.sort();cct=[x[0] for x in CC]
        for (t,v) in CST:
            i=_b.bisect_right(cct,t)-1
            if v>13 and i>=0 and CC[i][1]: nact+=1
    amin=nact/100.0/60.0  # carState ~100Hz
    deep=sum(1 for e in evs if e['ilp_min']<0.1)
    rate=len(evs)/amin if amin>0 else 0
    print(f"  {route:8s} {BUILD.get(route,'?'):8s} {len(evs):5d}   {amin:8.1f}   {rate:8.2f}    {deep} events")
