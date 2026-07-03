#!/usr/bin/env python3
"""Decisive test: replay Nevada (new-engine math, anchor_validated) on the cf incident
frames vs the LOGGED on-device output. Time base = roadEncodeIdx.timestampEof (seconds),
matching the sim's frame timeline exactly. logged dc aligned by frameId.
"""
import os, sys
import numpy as np
ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); os.chdir(ROOT)
from openpilot.tools.lib.logreader import LogReader
from model_replay_sim import config as C

SEG = "explorer_st_logs/route_cf/000000cf--4d24fc7149--10"

mv={}   # frameId -> (dc, innerLP)
eof={}  # frameId -> timestampEof_s
cst=[]  # (mono_s, vEgo)
for m in LogReader(os.path.join(SEG, "rlog.zst")):
    try: w = m.which()
    except Exception: continue
    if w == "modelV2":
        M=m.modelV2; probs=list(M.laneLineProbs)
        mv[int(M.frameId)] = (float(M.action.desiredCurvature),
                              min(probs[1:3]) if len(probs)>=3 else (min(probs) if probs else 1.0))
    elif w == "roadEncodeIdx":
        e=m.roadEncodeIdx; eof[int(e.frameId)] = e.timestampEof*1e-9
    elif w == "carState":
        cst.append((m.logMonoTime*1e-9, float(m.carState.vEgo)))
cst.sort()
print(f"modelV2 frames={len(mv)} roadEncodeIdx={len(eof)} carState={len(cst)}")

# rebuild cache in SECONDS
cache = C.CACHE_ROOT / "route_cf.npz"
if cache.exists(): cache.unlink()
np.savez(cache, mono_time=np.array([c[0] for c in cst]), v_ego=np.array([c[1] for c in cst], np.float32))
print(f"wrote seconds-based cache {cache}  MAX_FRAME_DELTA_S={C.MAX_FRAME_DELTA_S}")

# window: warmup ~200 frames + incident cluster (frameId 12503..12770)
fids = sorted(fid for fid in mv if fid in eof and 12503 <= fid <= 12770)
mono_times = [eof[fid] for fid in fids]
logged_dc = np.array([mv[fid][0] for fid in fids])
logged_lp = np.array([mv[fid][1] for fid in fids])
print(f"window frames={len(fids)}  frameId [{fids[0]},{fids[-1]}]  eof [{mono_times[0]:.1f},{mono_times[-1]:.1f}]s")

BUNDLE = sys.argv[1] if len(sys.argv) > 1 else "Nevada"
print(f"### replaying bundle={BUNDLE}")
from model_replay_sim.infer import replay_window
import time as _t
t0=_t.time(); res=replay_window(BUNDLE,"route_cf",mono_times); dt=_t.time()-t0
rep_dc=np.asarray(res["desired_curvature"],float)
vego=np.asarray(res["v_ego"],float)
print(f"replay: {len(rep_dc)} frames in {dt:.1f}s ({dt/max(len(rep_dc),1)*1000:.0f} ms/frame)")

W=200  # warmup frames to drop
c_log=logged_dc[W:]; c_rep=rep_dc[W:]; c_lp=logged_lp[W:]; c_fid=fids[W:]; c_v=vego[W:]
fin=np.isfinite(c_rep)&np.isfinite(c_log)
print("\n===== COMPARE (post-warmup incident window) =====")
print(f"finite frames: {fin.sum()}")
if fin.sum():
    print(f"LOGGED   |dc|: mean={np.mean(np.abs(c_log[fin])):.5f} max={np.max(np.abs(c_log[fin])):.5f}  frac<0.0012={np.mean(np.abs(c_log[fin])<0.0012)*100:.0f}%")
    print(f"REPLAYED |dc|: mean={np.mean(np.abs(c_rep[fin])):.5f} max={np.max(np.abs(c_rep[fin])):.5f}  frac<0.0012={np.mean(np.abs(c_rep[fin])<0.0012)*100:.0f}%")
    print(f"logged inner laneProb: mean={np.mean(c_lp):.2f} min={np.min(c_lp):.2f}")
    if fin.sum()>2: print(f"corr(replayed,logged)={np.corrcoef(c_rep[fin],c_log[fin])[0,1]:.3f}")
    print("\nframeId | v(mph) | logged_dc | replayed_dc | logged_innerLP")
    for i in range(0,len(c_fid),4):
        print(f"   {c_fid[i]} | {c_v[i]*2.237:5.1f} | {c_log[i]:+.5f} | {c_rep[i]:+.5f} | {c_lp[i]:.2f}")
