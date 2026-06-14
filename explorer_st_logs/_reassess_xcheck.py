import numpy as np
from _reassess_core import WEAK, GOLD, select, MPH

# Cross-check centering using model-INDEPENDENT steeringAngleDeg mean (per speed bin).
# Centering bias should also show in the held steering angle on straights.
def matched_mean(d_w, d_g, chan, st=0.6, band=(40,80), gate='latact'):
    sw=select(d_w,st,band,gate); sg=select(d_g,st,band,gate)
    wb=[]; gb=[]
    for lo in range(band[0],band[1],2):
        hi=lo+2
        wm=sw&(d_w['spd']>=lo*MPH)&(d_w['spd']<hi*MPH)
        gm=sg&(d_g['spd']>=lo*MPH)&(d_g['spd']<hi*MPH)
        if wm.sum()<50 or gm.sum()<50: continue
        wb.append(np.mean(d_w[chan][wm])); gb.append(np.mean(d_g[chan][gm]))
    return np.mean(wb), np.mean(gb), np.mean(gb)-np.mean(wb)

print("Cross-check centering across channels (speed-matched, st=0.6, 40-80, latact):")
for chan,unit in [('pos','m model-offset'),('steer','deg wheel'),('cmd','1/m curv-cmd'),('yaw','rad/s yawrate')]:
    w,g,delta=matched_mean(WEAK,GOLD,chan)
    print(f"  {chan:5s} ({unit:16s}): WEAK={w:+.5f}  GOLD={g:+.5f}  delta={delta:+.5f}")

print("\nInterpretation:")
print("  pos: GOLD less-left (centering better) -- the headline.")
print("  steer/cmd: do they corroborate a directional shift consistent with less left-offset?")
