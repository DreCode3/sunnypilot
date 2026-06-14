#!/usr/bin/env python3
"""VERIFY the centering finding independently of reassess_analyze.py (different method: whole-drive sample-level,
not windowed). Questions:
 1. SIGN: is WEAK persistently biased to one side while GOLD sits near zero? (mean offset signed, per group)
 2. Is it a model-pos artifact? cross-check with modpath (modelV2.position.y[0]) -- independent model channel.
 3. Sample-level, engaged + straight, speed-binned: |offset| and frac>0.3m by speed bin (no windowing/detrend).
 4. matched-n sanity for the windowed |mean offset| claim.
Sample-level avoids any window/detrend choice; if centering win survives here too, it's not a method artifact."""
import os, math
import numpy as np
CACHE = 'explorer_st_logs/_cache_reassess'
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
FS = 50.0


def cat(group):
    """concatenate sample-level engaged+straight arrays across the group's routes."""
    POS=[]; MP=[]; SPD=[]; STEER=[]; PRESS=[]; YAW=[]
    for rid in GROUPS[group]:
        f=f'{CACHE}/{rid}.npz'
        if not os.path.exists(f): continue
        d={k:np.load(f)[k] for k in np.load(f).files}
        eng=d['latact']>0; spd=d['spd'].astype(float); yaw=d['yaw'].astype(float)
        aLat=yaw*spd
        straight=np.abs(aLat)<0.6
        spdmph=spd*2.237
        m=eng & straight & (spdmph>=40) & (spdmph<=80) & np.isfinite(d['pos'])
        POS.append(d['pos'][m]); MP.append(d['modpath'][m]); SPD.append(spdmph[m])
        STEER.append(d['steer'][m].astype(float)); PRESS.append(d['press'][m].astype(float)); YAW.append(yaw[m])
    return (np.concatenate(POS), np.concatenate(MP), np.concatenate(SPD),
            np.concatenate(STEER), np.concatenate(PRESS), np.concatenate(YAW))


for g in GROUPS:
    pos,mp,spd,st,pr,yaw=cat(g)
    print(f'\n===== {g}: {len(pos)} engaged-straight samples ({len(pos)/FS/60:.1f} min) =====')
    print(f'  model offset (pos):  mean {np.mean(pos):+.4f}  median {np.median(pos):+.4f}  '
          f'|mean| {abs(np.mean(pos)):.4f}  mean|pos| {np.mean(np.abs(pos)):.4f}  frac|pos|>0.3 {np.mean(np.abs(pos)>0.3)*100:.1f}%')
    fmp=np.isfinite(mp)
    if fmp.sum()>100:
        print(f'  modelpath y0 (mp):   mean {np.mean(mp[fmp]):+.4f}  median {np.median(mp[fmp]):+.4f}  '
              f'mean|mp| {np.mean(np.abs(mp[fmp])):.4f}  frac|mp|>0.3 {np.mean(np.abs(mp[fmp])>0.3)*100:.1f}%')
    print(f'  steer angle (deg):   mean {np.mean(st):+.3f}  std {np.std(st):.3f}')
    print(f'  steeringPressed frac during engaged-straight: {np.mean(pr)*100:.2f}%')

print('\n===== SIGNED offset by SPEED BIN (sample-level; + = car right of center? check convention) =====')
print(f'{"bin":<9}{"WEAK mean":>11}{"WEAK |mean|":>12}{"WEAK>0.3":>10}   {"GOLD mean":>11}{"GOLD |mean|":>12}{"GOLD>0.3":>10}')
W=cat('WEAK'); G=cat('GOLD')
for lo,hi in [(40,50),(50,57),(57,65),(65,80)]:
    wm=(W[2]>=lo)&(W[2]<hi); gm=(G[2]>=lo)&(G[2]<hi)
    def stat(arr,msk):
        a=arr[msk]
        return (np.mean(a), abs(np.mean(a)), np.mean(np.abs(a)>0.3)*100) if msk.sum()>50 else (np.nan,np.nan,np.nan)
    wms,wma,wo=stat(W[0],wm); gms,gma,go=stat(G[0],gm)
    print(f'  {lo}-{hi:<4}{wms:>+11.4f}{wma:>12.4f}{wo:>9.1f}%   {gms:>+11.4f}{gma:>12.4f}{go:>9.1f}%   nW/nG {wm.sum()}/{gm.sum()}')

print('\n===== bootstrap 95% CI on GOLD-WEAK |mean offset| at MATCHED speed bin 57-65mph (sample-level) =====')
rng=np.random.default_rng(7)
wm=(W[2]>=57)&(W[2]<65); gm=(G[2]>=57)&(G[2]<65)
wa=np.abs(W[0][wm]); ga=np.abs(G[0][gm])
print(f'  WEAK mean|pos| {wa.mean():.4f} (n={len(wa)})  GOLD mean|pos| {ga.mean():.4f} (n={len(ga)})  diff {ga.mean()-wa.mean():+.4f}')
d=[ga[rng.integers(0,len(ga),len(ga))].mean()-wa[rng.integers(0,len(wa),len(wa))].mean() for _ in range(4000)]
print(f'  bootstrap 95% CI on (GOLD-WEAK) mean|pos|: [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]  '
      f'{"SIGNIFICANT" if np.percentile(d,97.5)<0 or np.percentile(d,2.5)>0 else "ns"}')
