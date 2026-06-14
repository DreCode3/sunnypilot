#!/usr/bin/env python3
"""Same-road, GPS-matched smooth_tau A/B: pair each b2 (0.25/0.12) straight to the b1 (0.12/0.04)
straight at the SAME physical location, compare hunt-band (0.5-1.5Hz) lateral-offset RMS.
This removes the road confound entirely (user: b2 is an earlier section of the same road as b1)."""
import sys, math, glob
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo'); sys.path.insert(0, 'explorer_st_logs')
from openpilot.tools.lib.logreader import LogReader
from explorer_st_logs.straight_hunting_mv2 import load_mv2, runs

BANDS = {'slow': (0.1, 0.5), 'hunt': (0.5, 1.5), 'hf': (1.5, 5.0)}


def band_var(x, fs):
    x = x - np.mean(x); n = len(x); X = np.fft.rfft(x); f = np.fft.rfftfreq(n, d=1.0 / fs); p = np.abs(X) ** 2
    return {k: 2.0 * np.sum(p[(f >= lo) & (f < hi)]) / (n * n) for k, (lo, hi) in BANDS.items()}


def load_gps(rid):
    segs = sorted(glob.glob(f'explorer_st_logs/{rid}/000000*--*/'), key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    t = []; lat = []; lon = []; msgtype = None
    for sd in segs:
        try:
            for msg in LogReader(sd + 'rlog.zst'):
                w = msg.which()
                if w == 'liveLocationKalman':
                    g = msg.liveLocationKalman
                    if not g.positionGeodetic.valid:
                        continue
                    v = g.positionGeodetic.value
                    t.append(msg.logMonoTime * 1e-9); lat.append(float(v[0])); lon.append(float(v[1])); msgtype = 'llk'
                elif w == 'gpsLocationExternal' and msgtype in (None, 'gle'):
                    g = msg.gpsLocationExternal
                    t.append(msg.logMonoTime * 1e-9); lat.append(float(g.latitude)); lon.append(float(g.longitude)); msgtype = 'gle'
        except Exception:
            continue
    if len(t) < 10:
        return None
    t = np.array(t); o = np.argsort(t)
    return dict(t=t[o], lat=np.array(lat)[o], lon=np.array(lon)[o], src=msgtype)


def haversine(la1, lo1, la2, lo2):
    R = 6371000.0; p1, p2 = math.radians(la1), math.radians(la2)
    dp = math.radians(la2 - la1); dl = math.radians(lo2 - lo1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def straights_with_gps(rid):
    d = load_mv2(rid); g = load_gps(rid)
    if d is None or g is None:
        print(f'  {rid}: mv2={d is not None} gps={g is not None}')
        return [], None
    t, dc, A, spd = d['t'], d['dc'], d['A'], d['spd']
    dcroll = np.convolve(np.abs(dc), np.ones(40) / 40, mode='same'); sm = (dcroll < 5e-4) & (spd > 13)
    out = []
    for a, b in runs(sm, 120):
        ts = t[a:b]
        if ts[-1] - ts[0] < 6:
            continue
        fs = 20.0; n = int((ts[-1] - ts[0]) * fs) + 1; tu = ts[0] + np.arange(n) / fs
        bv = band_var(np.interp(tu, ts, A[a:b]), fs)
        tmid = 0.5 * (ts[0] + ts[-1])
        if tmid < g['t'][0] or tmid > g['t'][-1]:
            continue
        out.append(dict(spd=float(np.median(spd[a:b]) * 2.237), n=n,
                        lat=float(np.interp(tmid, g['t'], g['lat'])), lon=float(np.interp(tmid, g['t'], g['lon'])),
                        hunt=bv['hunt'], slow=bv['slow']))
    return out, g['src']


b1, src1 = straights_with_gps('route_b1')
b2, src2 = straights_with_gps('route_b2')
print(f'GPS source: b1={src1} b2={src2}  |  straights w/GPS: b1={len(b1)} b2={len(b2)}')
if b1 and b2:
    # spatial overlap check
    mind = min(haversine(r2['lat'], r2['lon'], r1['lat'], r1['lon']) for r2 in b2 for r1 in b1)
    print(f'closest b1<->b2 straight: {mind:.0f} m  (small => same road)')
    matches = []
    for r2 in b2:
        best = min(b1, key=lambda r1: haversine(r2['lat'], r2['lon'], r1['lat'], r1['lon']))
        dist = haversine(r2['lat'], r2['lon'], best['lat'], best['lon'])
        if dist < 120:
            matches.append((best, r2, dist))
    print(f'\nmatched (<120m, same physical straight): {len(matches)} pairs')
    print(f'  {"location":<22}{"dist":>5}{"spd b1/b2":>11}{"hunt-RMS b1->b2 (mm)":>26}')
    for r1, r2, dist in matches:
        h1 = math.sqrt(r1['hunt']) * 1000; h2 = math.sqrt(r2['hunt']) * 1000
        print(f'  {r2["lat"]:.5f},{r2["lon"]:.5f}{dist:>5.0f}{r1["spd"]:>6.0f}/{r2["spd"]:<4.0f}{h1:>10.1f} -> {h2:<7.1f}{100*(h2-h1)/h1:>+5.0f}%')
    if matches:
        n1 = sum(r1['n'] for r1, _, _ in matches); n2 = sum(r2['n'] for _, r2, _ in matches)
        v1 = sum(r1['hunt'] * r1['n'] for r1, _, _ in matches) / n1
        v2 = sum(r2['hunt'] * r2['n'] for _, r2, _ in matches) / n2
        s1 = sum(r1['slow'] * r1['n'] for r1, _, _ in matches) / n1
        s2 = sum(r2['slow'] * r2['n'] for _, r2, _ in matches) / n2
        print(f'\n  POOLED over matched same-road straights:')
        print(f'    hunt 0.5-1.5Hz RMS: b1 {math.sqrt(v1)*1000:.1f}mm -> b2 {math.sqrt(v2)*1000:.1f}mm  ({100*(math.sqrt(v2)-math.sqrt(v1))/math.sqrt(v1):+.0f}%)')
        print(f'    slow 0.1-0.5Hz RMS: b1 {math.sqrt(s1)*1000:.1f}mm -> b2 {math.sqrt(s2)*1000:.1f}mm  ({100*(math.sqrt(s2)-math.sqrt(s1))/math.sqrt(s1):+.0f}%)')
