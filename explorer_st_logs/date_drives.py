#!/usr/bin/env python3
"""Date every local drive RELIABLY via initData.gitCommit (GPS time is garbage).
Maps each route -> the git commit it ran -> commit date -> inferred lateral-config era."""
import sys, glob, os, subprocess
import concurrent.futures as cf
from collections import defaultdict
sys.path.insert(0, '.'); sys.path.insert(0, 'opendbc_repo')
from openpilot.tools.lib.logreader import LogReader

segdirs = glob.glob('explorer_st_logs/**/0000*--*--*/', recursive=True)
routes = defaultdict(list)
for d in segdirs:
    base = os.path.basename(d.rstrip('/'))
    routes[base.rsplit('--', 1)[0]].append(d)


def getinit(item):
    rid, dirs = item
    d0 = sorted(dirs, key=lambda d: int(d.rstrip('/').rsplit('--', 1)[-1]))[0]
    f = d0 + 'rlog.zst'
    if not os.path.exists(f):
        return rid, None
    try:
        n = 0
        for msg in LogReader(f):
            if msg.which() == 'initData':
                i = msg.initData
                return rid, dict(commit=str(i.gitCommit), branch=str(i.gitBranch), dirty=bool(i.dirty), nseg=len(dirs))
            n += 1
            if n > 80:
                break
    except Exception as e:
        return rid, dict(commit='', branch='ERR:' + str(e)[:30], dirty=False, nseg=len(dirs))
    return rid, None


with cf.ThreadPoolExecutor(max_workers=12) as ex:
    res = dict(ex.map(getinit, routes.items()))

commits = {r['commit'] for r in res.values() if isinstance(r, dict) and r.get('commit')}
cdate = {}
for c in commits:
    try:
        out = subprocess.run(['git', 'show', '-s', '--format=%cd', '--date=short', c],
                             capture_output=True, text=True, timeout=10)
        cdate[c] = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else '(not-in-repo)'
    except Exception:
        cdate[c] = '(err)'

rows = []
for rid, r in res.items():
    if isinstance(r, dict) and r.get('commit'):
        rows.append((rid, r, cdate.get(r['commit'], '?')))
rows.sort(key=lambda x: (x[2], x[0]))


def era(date):
    if date < '2026-04-08':
        return 'pre-golden'
    if date <= '2026-04-11':
        return 'GOLDEN(Kp.0005,cap1.0,rg1.15)'
    if date < '2026-04-25':
        return 'strongPI,rg1.0,preMerge'
    if date < '2026-05-23':
        return 'strongPI,postMergeModel'
    return 'WEAK-PI(current)'


print(f'{"route_id":<26}{"date":<12}{"nseg":>5}{"dirty":>6}  {"commit":<10} era')
for rid, r, date in rows:
    e = era(date) if date.startswith('2026') else ''
    print(f'{rid:<26}{date:<12}{r["nseg"]:>5}{str(r["dirty"]):>6}  {r["commit"][:9]:<10} {e}')
print(f'\n{len(rows)} routes dated via initData.gitCommit')
