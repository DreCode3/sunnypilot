#!/usr/bin/env python3
import sys, json, collections, glob, os

START = 1782842287  # ~18:11 UTC clean OP start (epoch); adjust if needed
files = sorted(glob.glob("/data/log/swaglog.*"), key=os.path.getmtime, reverse=True)[:15]

levels = collections.Counter()
errs = []
warns = collections.Counter()
nfiles = 0
nmsgs = 0

def get_msg(m):
    for k in ("msg", "msg$s", "msg$d", "msg$f"):
        if k in m:
            return m[k]
    return ""

for path in files:
    try:
        data = open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        continue
    nfiles += 1
    for line in data.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        if m.get("created", 0) < START:
            continue
        nmsgs += 1
        lv = m.get("level", "?")
        levels[lv] += 1
        msg = str(get_msg(m))[:220]
        fn = m.get("filename", "")
        ln = m.get("lineno", "")
        if lv in ("ERROR", "CRITICAL"):
            errs.append((lv, fn, ln, msg))
        elif lv == "WARNING":
            warns[(fn, msg[:130])] += 1

print("files_scanned=%d  msgs_since_clean_start=%d" % (nfiles, nmsgs))
print("levels:", dict(levels))
print("--- ERROR/CRITICAL (count=%d) ---" % len(errs))
for e in errs[:50]:
    print("  ", e)
print("--- distinct WARNINGS ---")
for (fn, msg), c in warns.most_common(30):
    print("  [%dx] %s: %s" % (c, fn, msg))
