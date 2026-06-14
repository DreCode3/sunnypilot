#!/usr/bin/env python3
"""Map every local route -> date + GPS latitude span, flag Powder Springs(GA ~33.86) <-> Madison(AL ~34.70)
road trips (lat reaches north of ~34.2). Used to pin the 'golden lateral' road-trip dates."""
import sys, glob, os, datetime
import concurrent.futures as cf
from collections import defaultdict
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

segdirs = glob.glob('explorer_st_logs/**/0000*--*--*/', recursive=True)
routes = defaultdict(list)
for d in segdirs:
    base = os.path.basename(d.rstrip('/'))
    rid = base.rsplit('--', 1)[0]
    routes[rid].append(d)


def probe(item):
    rid, dirs = item
    dirs = sorted(dirs, key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))
    date = None; lats = []; lons = []
    sample = [dirs[0]]
    if len(dirs) > 2:
        sample.append(dirs[len(dirs) // 2])
    if len(dirs) > 1:
        sample.append(dirs[-1])
    for sd in sample:
        f = sd + 'rlog.zst'
        if not os.path.exists(f):
            f = sd + 'qlog.zst'
        if not os.path.exists(f):
            continue
        try:
            cnt = 0
            for msg in LogReader(f):
                w = msg.which()
                if w == 'gpsLocationExternal':
                    g = msg.gpsLocationExternal
                    if abs(g.latitude) < 0.01:
                        continue
                    if date is None and g.unixTimestampMillis > 0:
                        date = g.unixTimestampMillis
                    lats.append(g.latitude); lons.append(g.longitude); cnt += 1
                elif w == 'liveLocationKalman':
                    g = msg.liveLocationKalman
                    if not g.positionGeodetic.valid:
                        continue
                    v = g.positionGeodetic.value
                    lats.append(float(v[0])); lons.append(float(v[1])); cnt += 1
                if cnt >= 4:
                    break
        except Exception:
            continue
    if not lats:
        return rid, None
    ds = datetime.datetime.utcfromtimestamp(date / 1000).strftime('%Y-%m-%d') if date else '????-??-??'
    return rid, dict(date=ds, nseg=len(dirs), lat_min=min(lats), lat_max=max(lats), lon_min=min(lons), lon_max=max(lons))


with cf.ThreadPoolExecutor(max_workers=12) as ex:
    res = dict(ex.map(probe, routes.items()))

rows = [(rid, r) for rid, r in res.items() if r]
rows.sort(key=lambda x: x[1]['date'])
print(f'{"route_id":<26}{"date":<12}{"segs":>5}{"lat_min":>9}{"lat_max":>9}  flag')
for rid, r in rows:
    trip = 'GA<->AL ROAD TRIP' if r['lat_max'] > 34.2 else ('long' if r['nseg'] >= 40 else '')
    print(f'{rid:<26}{r["date"]:<12}{r["nseg"]:>5}{r["lat_min"]:>9.3f}{r["lat_max"]:>9.3f}  {trip}')
print(f'\n{len(rows)} routes probed; trips = lat_max>34.2 (toward Madison AL)')
