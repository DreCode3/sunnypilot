#!/usr/bin/env python3
"""Localize the model-input fault: compare road-cam nv12 stats (new-build cf vs old-build 7f)
and audit the WIDE camera (ecamera + wideRoadEncodeIdx) that the Nevada split model consumes.
"""
import os, sys
import numpy as np
ROOT="/Users/dregilley/Documents/GitHub/sunnypilot"; sys.path.insert(0,ROOT); os.chdir(ROOT)
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader

def nv12_stats(hevc, idx, label):
    fr=FrameReader(hevc, pix_fmt="nv12")
    n=fr.frame_count
    idx=min(idx, n-1)
    buf=np.asarray(fr.get(idx), dtype=np.uint8).ravel()
    # nv12: W*H Y plane then W*H/2 interleaved UV. infer dims: try 1928x1208 & 1344x760
    for (w,h) in [(1928,1208),(1344,760),(2048,1216),(1164,874)]:
        if buf.size == w*h*3//2:
            Y=buf[:w*h].astype(np.float32); UV=buf[w*h:].astype(np.float32); dims=(w,h); break
    else:
        Y=buf[:buf.size*2//3].astype(np.float32); UV=buf[buf.size*2//3:].astype(np.float32); dims=("?", buf.size)
    print(f"  [{label}] frames={n} dims={dims} bytes={buf.size}")
    print(f"     Y : mean={Y.mean():6.1f} std={Y.std():5.1f} min={Y.min():.0f} max={Y.max():.0f} p1={np.percentile(Y,1):.0f} p99={np.percentile(Y,99):.0f}")
    print(f"     UV: mean={UV.mean():6.1f} std={UV.std():5.1f} min={UV.min():.0f} max={UV.max():.0f}")
    return dims, buf.size

print("=== ROAD CAM nv12 stats: new-build cf vs old-build 7f ===")
nv12_stats("explorer_st_logs/route_cf/000000cf--4d24fc7149--10/fcamera.hevc", 748, "cf(new) road f748")
nv12_stats("explorer_st_logs/route_cf/000000cf--4d24fc7149--10/fcamera.hevc", 300, "cf(new) road f300")
nv12_stats("explorer_st_logs/route_7f/0000007f--0436348eca--0/fcamera.hevc", 600, "7f(old) road f600")

print("\n=== WIDE CAM (ecamera) present + stats ===")
for seg,lbl in [("explorer_st_logs/route_cf/000000cf--4d24fc7149--10","cf(new) wide"),
                ("explorer_st_logs/route_7f/0000007f--0436348eca--0","7f(old) wide")]:
    ec=os.path.join(seg,"ecamera.hevc")
    if os.path.exists(ec):
        try: nv12_stats(ec, 300, lbl)
        except Exception as e: print(f"  [{lbl}] ecamera read FAILED: {e!r}")
    else:
        print(f"  [{lbl}] NO ecamera.hevc")

print("\n=== wideRoadEncodeIdx present in rlogs? (sim maps wide via this) ===")
for seg,lbl in [("explorer_st_logs/route_cf/000000cf--4d24fc7149--10","cf(new)"),
                ("explorer_st_logs/route_7f/0000007f--0436348eca--0","7f(old)")]:
    wr=0; rr=0
    for m in LogReader(os.path.join(seg,"rlog.zst")):
        try: w=m.which()
        except Exception: continue
        if w=="wideRoadEncodeIdx": wr+=1
        elif w=="roadEncodeIdx": rr+=1
    print(f"  [{lbl}] roadEncodeIdx={rr}  wideRoadEncodeIdx={wr}")
