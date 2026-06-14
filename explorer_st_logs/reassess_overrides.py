#!/usr/bin/env python3
"""Independent OVERRIDE-EVENT analysis (whole engaged drive, not windowed) — bears on 'felt confident/fewer takeovers'.
An override = steeringPressed rising edge while latActive. Count ONSETS (rising edges) per engaged-minute, and
total engaged-pressed fraction. Also split by speed bin. Cross-checks the workflow's override agent."""
import os
import numpy as np
CACHE = 'explorer_st_logs/_cache_reassess'
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
FS = 50.0

def analyze(group):
    eng_min_tot=0.0; onsets=0; pressed_samp=0; eng_samp=0
    bin_onset={b:[0,0.0] for b in [(40,50),(50,57),(57,65),(65,90)]}  # [onsets, eng_minutes]
    for rid in GROUPS[group]:
        f=f'{CACHE}/{rid}.npz'
        if not os.path.exists(f): continue
        d={k:np.load(f)[k] for k in np.load(f).files}
        eng=d['latact'].astype(bool); press=d['press'].astype(bool); spd=d['spd'].astype(float)*2.237
        e=eng  # engaged samples (any speed)
        eng_samp+=e.sum(); pressed_samp+=(e&press).sum()
        eng_min_tot+=e.sum()/FS/60
        # rising edges of press within engaged
        pe=press & eng
        rises=np.where((~pe[:-1]) & pe[1:])[0]+1
        onsets+=len(rises)
        for b in bin_onset:
            lo,hi=b
            inb=e&(spd>=lo)&(spd<hi)
            bin_onset[b][1]+=inb.sum()/FS/60
            peb=press&inb
            bin_onset[b][0]+=len(np.where((~peb[:-1])&peb[1:])[0])
    return dict(eng_min=eng_min_tot, onsets=onsets, onsets_per_min=onsets/eng_min_tot if eng_min_tot else 0,
                pressed_frac=pressed_samp/eng_samp if eng_samp else 0, bins=bin_onset)

print(f'{"group":<7}{"eng min":>9}{"overrides":>11}{"per eng-min":>13}{"pressed%":>10}')
res={}
for g in GROUPS:
    r=analyze(g); res[g]=r
    print(f'  {g:<5}{r["eng_min"]:>9.1f}{r["onsets"]:>11}{r["onsets_per_min"]:>13.2f}{r["pressed_frac"]*100:>9.2f}%')

print('\noverride ONSETS per engaged-minute by SPEED bin:')
print(f'{"bin":<9}{"WEAK /min":>12}{"(eng min)":>11}{"GOLD /min":>12}{"(eng min)":>11}')
for b in [(40,50),(50,57),(57,65),(65,90)]:
    w=res['WEAK']['bins'][b]; g=res['GOLD']['bins'][b]
    wpm=w[0]/w[1] if w[1]>0.1 else float('nan'); gpm=g[0]/g[1] if g[1]>0.1 else float('nan')
    print(f'  {b[0]}-{b[1]:<5}{wpm:>12.2f}{w[1]:>11.1f}{gpm:>12.2f}{g[1]:>11.1f}')
