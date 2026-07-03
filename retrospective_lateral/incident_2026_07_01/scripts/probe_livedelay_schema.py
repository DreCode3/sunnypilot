#!/usr/bin/env python3
"""Probe liveDelay schema, initData params (bundle manifest, Lagd*), modelV2 fields.
One segment from an OLD route (c5) and one from a NEW route (ce)."""
import json
from openpilot.tools.lib.logreader import LogReader

SEGS = {
  "c5_old": "/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs/route_c5/000000c5--751078a818--1/rlog.zst",
  "ce_new": "/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs/route_ce/000000ce--f8a8d4f570--1/rlog.zst",
}

WANT_PARAMS = ("ModelManager_ActiveBundle", "LagdToggle", "LagdValueCache", "PlanplusControl")

for name, path in SEGS.items():
  print(f"\n================ {name} : {path}")
  lr = LogReader(path)
  got_ld = 0
  got_mv2 = False
  got_init = False
  it = iter(lr)
  while True:
    try:
      msg = next(it)
    except StopIteration:
      break
    except Exception as e:
      continue
    try:
      w = msg.which()
    except Exception:
      continue
    if w == "initData" and not got_init:
      got_init = True
      try:
        entries = msg.initData.params.entries
        for e in entries:
          k = e.key
          if any(wp.lower() in k.lower() for wp in ("modelmanager", "lagd", "planplus")):
            v = bytes(e.value)
            s = v.decode("utf-8", errors="replace")
            if len(s) > 900:
              s = s[:900] + "...<trunc>"
            print(f"  PARAM {k} = {s}")
      except Exception as ex:
        print("  initData param read fail:", ex)
    elif w == "liveDelay" and got_ld < 3:
      got_ld += 1
      d = msg.liveDelay.to_dict()
      print(f"  liveDelay #{got_ld}: {json.dumps(d, default=str)}")
      if got_ld == 1:
        print("  liveDelay dir:", [a for a in dir(msg.liveDelay) if not a.startswith("_")][:40])
    elif w == "modelV2" and not got_mv2:
      got_mv2 = True
      m = msg.modelV2
      print("  modelV2 keys:", [a for a in dir(m) if not a.startswith("_")][:60])
      print("  action:", m.action.to_dict())
      print("  orientation.z len:", len(m.orientation.z), " first3:", list(m.orientation.z)[:3])
      print("  orientationRate.z first3:", list(m.orientationRate.z)[:3])
      print("  velocity.x first3:", list(m.velocity.x)[:3])
    if got_ld >= 3 and got_mv2 and got_init:
      break
