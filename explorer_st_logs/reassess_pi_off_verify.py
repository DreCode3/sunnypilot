#!/usr/bin/env python3
"""VERIFY the workflow's DECISIVE claim: ~75% of the GOLD-vs-WEAK centering shift is present with the PI OFF
(un-engaged, latact==0) -> the 'centering win' is mostly a between-drive baseline (road/lane/human-positioning),
NOT golden-PI authority.

Method: pos = model lane offset. ENGAGED (latact==1) = controller steers. UN-ENGAGED (latact==0) = HUMAN steers.
If GOLD already sits more-centered than WEAK when the HUMAN is driving (PI off), that offset is a baseline drive
difference. The clean PI effect is DIFFERENCE-IN-DIFFERENCES:
   DiD = (GOLD_eng - GOLD_uneng) - (WEAK_eng - WEAK_uneng)
i.e. how much MORE the golden controller re-centers relative to manual than the weak controller does, with each
drive's own manual baseline subtracted (cancels road/lane/human positioning per drive).
Report per speed bin where BOTH engaged & unengaged data exist for BOTH groups, with n. Bootstrap CI on DiD."""
import os
import numpy as np
CACHE = 'explorer_st_logs/_cache_reassess'
GROUPS = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
FS = 50.0
rng = np.random.default_rng(20260613)


def load_samples(group):
    """return dict of arrays for engaged-straight and unengaged-straight (any pressed state), 40-80mph."""
    eng_pos=[];eng_spd=[]; un_pos=[];un_spd=[]
    for rid in GROUPS[group]:
        f=f'{CACHE}/{rid}.npz'
        if not os.path.exists(f): continue
        d={k:np.load(f)[k] for k in np.load(f).files}
        spd=d['spd'].astype(float)*2.237; yaw=d['yaw'].astype(float)
        aLat=(d['yaw'].astype(float))*(d['spd'].astype(float))
        straight=np.abs(aLat)<0.6; band=(spd>=30)&(spd<=80); good=np.isfinite(d['pos'])
        eng=d['latact'].astype(bool)
        base=straight&band&good
        em=base&eng; um=base&(~eng)
        eng_pos.append(d['pos'][em]); eng_spd.append(spd[em])
        un_pos.append(d['pos'][um]); un_spd.append(spd[um])
    return (np.concatenate(eng_pos),np.concatenate(eng_spd),
            np.concatenate(un_pos),np.concatenate(un_spd))

WE,WEs,WU,WUs = load_samples('WEAK')
GE,GEs,GU,GUs = load_samples('GOLD')

print('engaged-straight / unengaged-straight sample counts (30-80mph):')
print(f'  WEAK: engaged {len(WE)} ({len(WE)/FS/60:.1f}min)  unengaged {len(WU)} ({len(WU)/FS/60:.1f}min)')
print(f'  GOLD: engaged {len(GE)} ({len(GE)/FS/60:.1f}min)  unengaged {len(GU)} ({len(GU)/FS/60:.1f}min)')

def binmean(pos,spd,lo,hi):
    m=(spd>=lo)&(spd<hi)
    return (np.mean(pos[m]) if m.sum()>=30 else np.nan), int(m.sum())

print('\n=== mean offset by speed bin: ENGAGED vs UNENGAGED, both groups ===')
print(f'{"bin":<8}{"WEAK eng":>10}{"(n)":>7}{"WEAK uneng":>12}{"(n)":>7}   {"GOLD eng":>10}{"(n)":>7}{"GOLD uneng":>12}{"(n)":>7}')
bins=[(30,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,80)]
for lo,hi in bins:
    we,wen=binmean(WE,WEs,lo,hi); wu,wun=binmean(WU,WUs,lo,hi)
    ge,gen=binmean(GE,GEs,lo,hi); gu,gun=binmean(GU,GUs,lo,hi)
    print(f'  {lo}-{hi:<4}{we:>+10.4f}{wen:>7}{wu:>+12.4f}{wun:>7}   {ge:>+10.4f}{gen:>7}{gu:>+12.4f}{gun:>7}')

print('\n=== KEY: GOLD-vs-WEAK offset gap, ENGAGED vs UNENGAGED, per bin (is the gap present PI-OFF?) ===')
print(f'{"bin":<8}{"ENG gap(G-W)":>14}{"UNENG gap(G-W)":>16}{"%present PI-off":>17}')
for lo,hi in bins:
    we,wen=binmean(WE,WEs,lo,hi); wu,wun=binmean(WU,WUs,lo,hi)
    ge,gen=binmean(GE,GEs,lo,hi); gu,gun=binmean(GU,GUs,lo,hi)
    if np.isfinite(we) and np.isfinite(ge) and np.isfinite(wu) and np.isfinite(gu):
        eng_gap=ge-we; uneng_gap=gu-wu
        pct=100*uneng_gap/eng_gap if abs(eng_gap)>1e-6 else np.nan
        print(f'  {lo}-{hi:<4}{eng_gap:>+14.4f}{uneng_gap:>+16.4f}{pct:>16.0f}%')
    else:
        print(f'  {lo}-{hi:<4}   (missing eng or uneng data in one group)')

print('\n=== DIFFERENCE-IN-DIFFERENCES (per-drive manual baseline subtracted) ===')
print('  DiD = (GOLD_eng - GOLD_uneng) - (WEAK_eng - WEAK_uneng) on SIGNED offset.')
print('  ⚠️ THE CI BELOW IS AN i.i.d. BOOTSTRAP = WRONG for 50Hz-autocorrelated data (overstates significance).')
print('  ⚠️ USE verify_did_blockbootstrap.py (bout-level + moving-block) FOR THE HONEST CI. There the DiD is NON-SIG.')
print('  SIGN: baseline bias is LEFT (offset<0), so centering = toward 0 = MORE POSITIVE. POSITIVE DiD = GOLD centers MORE.')
def did_band(lo,hi,nboot=4000):
    masks={}
    for nm,(p,s) in {'we':(WE,WEs),'wu':(WU,WUs),'ge':(GE,GEs),'gu':(GU,GUs)}.items():
        m=(s>=lo)&(s<hi); masks[nm]=p[m]
    if any(len(masks[k])<30 for k in masks):
        return None
    def did(we,wu,ge,gu):
        return (np.mean(ge)-np.mean(gu))-(np.mean(we)-np.mean(wu))
    pt=did(masks['we'],masks['wu'],masks['ge'],masks['gu'])
    bs=[did(masks['we'][rng.integers(0,len(masks['we']),len(masks['we']))],
            masks['wu'][rng.integers(0,len(masks['wu']),len(masks['wu']))],
            masks['ge'][rng.integers(0,len(masks['ge']),len(masks['ge']))],
            masks['gu'][rng.integers(0,len(masks['gu']),len(masks['gu']))]) for _ in range(nboot)]
    return pt,np.percentile(bs,2.5),np.percentile(bs,97.5),{k:len(masks[k]) for k in masks}
for lo,hi in [(40,45),(45,50),(50,55),(55,60)]:
    r=did_band(lo,hi)
    if r:
        pt,clo,chi,ns=r
        sig='SIG' if (clo>0 or chi<0) else 'ns'
        print(f'  {lo}-{hi}mph: DiD {pt:+.4f}m  95%CI[{clo:+.4f},{chi:+.4f}] {sig}  '
              f'(n we{ns["we"]}/wu{ns["wu"]}/ge{ns["ge"]}/gu{ns["gu"]})')
    else:
        print(f'  {lo}-{hi}mph: insufficient eng+uneng data in all 4 cells')
print('\n  Interpretation (CORRECTED SIGN): DiD≈0 => golden PI adds NO centering beyond manual baseline (drive/road confound).')
print('  DiD POSITIVE (GOLD eng closer to center=0 than WEAK eng, vs each manual baseline) => golden-PI centering effect.')
print('  BUT significance MUST come from the block/bout bootstrap (verify_did_blockbootstrap.py): there it is NON-SIG.')
print('  The point estimate (~+0.06m) leans toward a real golden-PI effect but is NOT statistically established here.')
