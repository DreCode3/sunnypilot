import sys, glob, os
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 100.0  # common grid Hz
MS2MPH = 2.2369362921

def load(route):
    d = np.load(f'explorer_st_logs/_c4/{route}.npz')
    return d

def build_grid(d):
    """Resample onto a uniform 100Hz grid per contiguous time region.
       Returns dict of arrays on grid: t, v(mph), ang(deg), press, lat(0/1), yaw(rad/s)."""
    cs_t = d['cs_t']; order = np.argsort(cs_t)
    cs_t = cs_t[order]
    cs_v = d['cs_v'][order]
    cs_ang = d['cs_ang'][order]
    cs_press = d['cs_press'][order]
    cc_t = d['cc_t']; o2 = np.argsort(cc_t); cc_t = cc_t[o2]; cc_lat = d['cc_lat'][o2]
    ll_t = d['ll_t']; o3 = np.argsort(ll_t); ll_t = ll_t[o3]; ll_yaw = d['ll_yaw'][o3]

    t0 = cs_t[0]; t1 = cs_t[-1]
    grid = np.arange(t0, t1, 1.0/FS)
    ang = np.interp(grid, cs_t, cs_ang)
    v_ms = np.interp(grid, cs_t, cs_v)
    press = np.interp(grid, cs_t, cs_press)  # interpolated; threshold >0.5
    lat = np.interp(grid, cc_t, cc_lat)
    yaw = np.interp(grid, ll_t, ll_yaw)
    # gap mask: invalidate grid points far from real carState samples (>0.2s gap)
    idx = np.searchsorted(cs_t, grid)
    idx = np.clip(idx, 1, len(cs_t)-1)
    gap = np.minimum(np.abs(grid - cs_t[idx]), np.abs(grid - cs_t[idx-1]))
    valid = gap < 0.2
    return dict(t=grid, v=v_ms*MS2MPH, v_ms=v_ms, ang=ang, press=press, lat=lat, yaw=yaw, valid=valid)

def engaged_straight_mask(g):
    eng = (g['lat'] > 0.5)
    straight = np.abs(g['yaw'] * g['v_ms']) < 0.6
    band = (g['v'] >= 40) & (g['v'] <= 80)
    notpressed = (g['press'] < 0.5)
    return eng & straight & band & g['valid'] & notpressed

def contiguous_runs(mask, min_len):
    """Yield (start,end) index ranges of contiguous True with length>=min_len."""
    runs = []
    i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs

# bandpass 0.5-1.5 Hz
SOS = butter(4, [0.5, 1.5], btype='band', fs=FS, output='sos')

def hunt_band_rms_per_run(ang, runs):
    """Return list of (rms, nsamp, mean_speed_placeholder) over each run."""
    out = []
    for (i, j) in runs:
        seg = ang[i:j].astype(float)
        seg = seg - seg.mean()
        if len(seg) < int(2*FS):  # need >=2s for stable 0.5Hz
            continue
        filt = sosfiltfilt(SOS, seg)
        rms = np.sqrt(np.mean(filt**2))
        out.append((i, j, rms, len(seg)))
    return out

def steering_rate_metrics_per_run(ang, runs):
    """rate std (deg/s) and reversals/s per run."""
    out = []
    for (i, j) in runs:
        seg = ang[i:j].astype(float)
        if len(seg) < int(1*FS):
            continue
        rate = np.diff(seg) * FS  # deg/s
        rstd = np.std(rate)
        # reversals: sign changes of rate, with small deadband to avoid noise-only flips
        dead = 0.5  # deg/s deadband
        sig = np.where(rate > dead, 1, np.where(rate < -dead, -1, 0))
        sig_nz = sig[sig != 0]
        nrev = np.sum(np.diff(sig_nz) != 0) if len(sig_nz) > 1 else 0
        dur = len(seg) / FS
        out.append((i, j, rstd, nrev/dur, len(seg)))
    return out

def speed_of_run(g, i, j):
    return np.mean(g['v'][i:j])

def report():
    groups = {'WEAK': ['route_b1', 'route_b2'], 'GOLD': ['route_b8']}
    data = {}
    for grp, routes in groups.items():
        # accumulate per-run records across routes in group
        runs_hunt = []   # (rms, nsamp, speed)
        runs_rate = []   # (rstd, revps, nsamp, speed)
        total_eng_straight_s = 0
        for route in routes:
            g = build_grid(load(route))
            m = engaged_straight_mask(g)
            total_eng_straight_s += m.sum()/FS
            runs = contiguous_runs(m, int(2*FS))
            for (i,j,rms,n) in hunt_band_rms_per_run(g['ang'], runs):
                runs_hunt.append((rms, n, speed_of_run(g,i,j)))
            for (i,j,rstd,revps,n) in steering_rate_metrics_per_run(g['ang'], runs):
                runs_rate.append((rstd, revps, n, speed_of_run(g,i,j)))
        data[grp] = dict(hunt=runs_hunt, rate=runs_rate, eng_s=total_eng_straight_s)
        print(f"{grp}: engaged-straight 40-80mph time = {total_eng_straight_s:.0f}s, "
              f"hunt-runs(>=2s)={len(runs_hunt)}, rate-runs={len(runs_rate)}")
    return data

def speed_dist(data):
    print("\n=== Speed distribution of engaged-straight runs (mph, sample-weighted) ===")
    for grp in ['WEAK','GOLD']:
        sp = []
        for (rstd,revps,n,s) in data[grp]['rate']:
            sp += [s]*int(n)
        sp = np.array(sp)
        if len(sp):
            print(f"  {grp}: n_samp={len(sp)}  median={np.median(sp):.1f}  "
                  f"mean={np.mean(sp):.1f}  p25={np.percentile(sp,25):.1f}  p75={np.percentile(sp,75):.1f}")

def per_bin(data):
    """Per 2.5-mph bin, sample-weighted median of run metrics, GOLD vs WEAK."""
    bins = np.arange(40, 82.5, 2.5)
    print("\n=== Per-bin matched comparison (sample-weighted across runs) ===")
    print(f"{'bin(mph)':>10} | {'huntRMS WEAK':>12} {'GOLD':>8} {'Δ%':>7} | "
          f"{'rateStd WEAK':>12} {'GOLD':>8} {'Δ%':>7} | {'rev/s WEAK':>11} {'GOLD':>8} {'Δ%':>7} | nW/nG")
    agg = {}  # metric -> list of (weight, weakval, goldval) for overall matched
    for b in range(len(bins)-1):
        lo, hi = bins[b], bins[b+1]
        row = {}
        for grp in ['WEAK','GOLD']:
            # hunt: weight by nsamp
            hr = [(rms,n) for (rms,n,s) in data[grp]['hunt'] if lo<=s<hi]
            rr = [(rstd,revps,n) for (rstd,revps,n,s) in data[grp]['rate'] if lo<=s<hi]
            def wmed(vals):
                if not vals: return (np.nan, 0)
                arr = np.array([v for v,_ in vals]); w = np.array([x for _,x in vals],float)
                # weighted median
                o = np.argsort(arr); arr=arr[o]; w=w[o]; cw=np.cumsum(w)
                if cw[-1]==0: return (np.nan,0)
                k = np.searchsorted(cw, cw[-1]/2.0)
                return (arr[min(k,len(arr)-1)], int(w.sum()))
            hm, hn = wmed(hr)
            rstd_vals = [(x[0], x[2]) for x in rr]
            rev_vals  = [(x[1], x[2]) for x in rr]
            rsm, rsn = wmed(rstd_vals)
            rvm, rvn = wmed(rev_vals)
            row[grp] = (hm, hn, rsm, rvm, rsn)
        def pct(w,g):
            if np.isnan(w) or np.isnan(g) or w==0: return np.nan
            return (g-w)/w*100
        hw,hnw,rsw,rvw,rsnw = row['WEAK']
        hg,hng,rsg,rvg,rsng = row['GOLD']
        if (hnw>0 and hng>0) or (rsnw>0 and rsng>0):
            print(f"{lo:6.1f}-{hi:4.1f} | {hw:12.4f} {hg:8.4f} {pct(hw,hg):7.1f} | "
                  f"{rsw:12.3f} {rsg:8.3f} {pct(rsw,rsg):7.1f} | "
                  f"{rvw:11.3f} {rvg:8.3f} {pct(rvw,rvg):7.1f} | {hnw}/{hng}")
            # for matched aggregate weight by min sample count in bin (overlap weight)
            wgt = min(hnw, hng)
            if wgt>0 and not (np.isnan(hw) or np.isnan(hg)):
                agg.setdefault('hunt',[]).append((wgt,hw,hg))
            wgt2 = min(rsnw, rsng)
            if wgt2>0 and not (np.isnan(rsw) or np.isnan(rsg)):
                agg.setdefault('ratestd',[]).append((wgt2,rsw,rsg))
                agg.setdefault('rev',[]).append((wgt2,rvw,rvg))
    print("\n=== Speed-matched aggregate (bins weighted by min(nW,nG) overlap) ===")
    for met in ['hunt','ratestd','rev']:
        if met not in agg: continue
        rows = agg[met]
        W = np.array([r[0] for r in rows],float)
        weak = np.array([r[1] for r in rows]); gold = np.array([r[2] for r in rows])
        wbar = np.sum(W*weak)/W.sum(); gbar = np.sum(W*gold)/W.sum()
        print(f"  {met:8s}: WEAK={wbar:.4f}  GOLD={gbar:.4f}  Δ={(gbar-wbar)/wbar*100:+.1f}%  (overlap bins={len(rows)})")

def matched_window(data, lo=57.5, hi=62.5):
    """Tight matched window where both groups have mass (per speed dist)."""
    print(f"\n=== Tight matched window {lo}-{hi} mph (run-level, robust) ===")
    for met, key, idx in [('hunt-band RMS','hunt',0), ('rate-std','rate',0), ('reversals/s','rate',1)]:
        for grp in ['WEAK','GOLD']:
            if key=='hunt':
                vals = [v[0] for v in data[grp]['hunt'] if lo<=v[2]<hi]
                ns   = [v[1] for v in data[grp]['hunt'] if lo<=v[2]<hi]
            else:
                vals = [v[idx] for v in data[grp]['rate'] if lo<=v[3]<hi]
                ns   = [v[2] for v in data[grp]['rate'] if lo<=v[3]<hi]
            if vals:
                med = np.median(vals)
                # sample-weighted mean too
                wm = np.sum(np.array(vals)*np.array(ns,float))/np.sum(ns) if np.sum(ns)>0 else np.nan
                print(f"  {met:16s} {grp}: median={med:.4f}  wmean={wm:.4f}  nruns={len(vals)}  nsamp={int(np.sum(ns))}")
        print()

if __name__ == '__main__':
    data = report()
    speed_dist(data)
    per_bin(data)
    matched_window(data, 57.5, 62.5)
    matched_window(data, 50.0, 65.0)
