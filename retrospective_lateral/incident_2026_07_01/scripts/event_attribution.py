#!/usr/bin/env python3
"""Per-event mechanism attribution for the 2026-07-01 incident drives (ce + cf).

For every driver-override cluster (same extraction as incident_analyses.overrides) and every
deep laneProb collapse (<0.15 for >=1.5 s while moving), extract the full context needed to
attribute it to a mechanism from handoff v3:
  A  blind model + silent AOL continuation (scene-driven perception loss)
  B  PI wrong-way transient after an offset reversal (weave amplifier)
  C  sustained wheel non-delivery vs ramping sent command (open-cause tail)
  D  manual driving (driver already hands-on; blinker = manual maneuver)
Signs (verified, handoff v3 §1.2): curvature + = RIGHT; laneLines +y = RIGHT;
midpoint offset + = car LEFT of center. steeringAngleDeg + = LEFT.

RUN: PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
       retrospective_lateral/incident_2026_07_01/scripts/event_attribution.py
"""
import glob, bisect, math, re, sys
from pathlib import Path
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader
from opendbc.car.vehicle_model import VehicleModel

VM = None

def load(route):
    global VM
    d = dict(mv=[], cs=[], co=[], cc=[], lc=[], al=[])
    for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst")):
        seg = int(Path(rl).parent.name.split("--")[-1])
        try:
            lr = LogReader(rl)
        except Exception:
            continue
        for m in lr:
            try:
                w = m.which()
            except Exception:
                continue
            t = m.logMonoTime * 1e-9
            if w == "modelV2":
                M = m.modelV2; ll = list(M.laneLines); p = list(M.laneLineProbs)
                ds = list(M.meta.desireState)
                mid = (ll[1].y[0] + ll[2].y[0]) / 2 if len(ll) >= 3 and len(ll[1].y) else np.nan
                d["mv"].append((t, seg, min(p[1:3]) if len(p) >= 3 else 1.0, mid,
                                float(M.action.desiredCurvature),
                                max(ds[3], ds[4]) if len(ds) >= 5 else 0.0,   # laneChange desire
                                max(ds[1], ds[2]) if len(ds) >= 3 else 0.0))  # turn desire
            elif w == "carState":
                c = m.carState
                d["cs"].append((t, float(c.vEgo), float(c.steeringAngleDeg), bool(c.steeringPressed),
                                float(c.steeringTorque), bool(c.leftBlinker or c.rightBlinker)))
            elif w == "carOutput":
                d["co"].append((t, float(m.carOutput.actuatorsOutput.curvature)))
            elif w == "carControl":
                d["cc"].append((t, bool(m.carControl.latActive), bool(m.carControl.enabled)))
            elif w == "logMessage":
                s = str(m.logMessage)
                if "LC: off=" in s:
                    mm = re.search(r"off=([-\d.]+) .* int=([-\d.]+) .* I=([-\d.]+) ", s)
                    if mm:
                        d["lc"].append((t, float(mm.group(1)), float(mm.group(2)), float(mm.group(3))))
            elif w == "selfdriveState":
                try:
                    a1 = str(m.selfdriveState.alertText1)
                    if a1:
                        d["al"].append((t, a1))
                except Exception:
                    pass
            elif w == "carParams" and VM is None:
                VM = VehicleModel(m.carParams)
    for k in d:
        d[k].sort()
    return d


def win(rows, ts, t0, t1):
    i0 = bisect.bisect_left(ts, t0); i1 = bisect.bisect_right(ts, t1)
    return rows[i0:i1]


def ach_curv(sa, v):
    """achieved curvature in the COMMAND frame (+ = right): negate VM (+angle = left)."""
    return -VM.calc_curvature(math.radians(sa), max(v, 1.0), 0.0)


def override_clusters(d):
    ov = [(t, tq, v) for (t, v, sa, sp, tq, blk) in d["cs"] if sp and abs(tq) > 1.8 and v > 13]
    clusters, cur = [], []
    for o in ov:
        if cur and o[0] - cur[-1][0] > 2.5:
            clusters.append(cur); cur = []
        cur.append(o)
    if cur:
        clusters.append(cur)
    return clusters


def deep_collapses(d):
    """laneProb < 0.15 sustained >= 1.5 s while moving (v>8)."""
    MV = d["mv"]; cst = [x[0] for x in d["cs"]]
    ev = []; i = 0; n = len(MV)
    while i < n:
        if MV[i][2] < 0.15:
            j = i
            while j < n and MV[j][2] < 0.7:
                j += 1
            dur_low = 0.0
            k = i
            while k < min(j, n) and MV[k][2] < 0.15:
                k += 1
            dur_low = MV[min(k, n-1)][0] - MV[i][0]
            if j < n and dur_low >= 1.5:
                jj = bisect.bisect_right(cst, MV[i][0]) - 1
                v = d["cs"][jj][1] if jj >= 0 else 0
                if v > 8:
                    ev.append((MV[i][0], MV[j][0], v))
            i = j if j > i else i + 1
            continue
        i += 1
    return ev


def analyze(route):
    d = load(route)
    mvt = [x[0] for x in d["mv"]]; cst = [x[0] for x in d["cs"]]
    cot = [x[0] for x in d["co"]]; cct = [x[0] for x in d["cc"]]
    lct = [x[0] for x in d["lc"]]; alt = [x[0] for x in d["al"]]

    print(f"\n{'='*110}\n#### {route} — OVERRIDE EVENTS\n{'='*110}")
    for cl in override_clusters(d):
        t0 = cl[0][0]; v0 = cl[0][2]; tq0 = cl[0][1]
        mvw = win(d["mv"], mvt, t0 - 8, t0 + 0.2)
        if not mvw:
            continue
        seg = mvw[-1][1]
        ccw = win(d["cc"], cct, t0 - 5, t0)
        lat_pct = 100 * np.mean([x[1] for x in ccw]) if ccw else np.nan
        en_pct = 100 * np.mean([x[2] for x in ccw]) if ccw else np.nan
        csw_pre = win(d["cs"], cst, t0 - 8, t0 - 0.5)
        pressed_pre = 100 * np.mean([x[3] for x in csw_pre]) if csw_pre else np.nan
        blk = any(x[5] for x in win(d["cs"], cst, t0 - 6, t0 + 1))
        lp3 = min(x[2] for x in win(d["mv"], mvt, t0 - 3, t0)) if win(d["mv"], mvt, t0-3, t0) else np.nan
        des_lc = max((x[5] for x in mvw), default=0.0)
        des_turn = max((x[6] for x in mvw), default=0.0)
        offs = [x[3] for x in win(d["mv"], mvt, t0 - 3, t0) if np.isfinite(x[3])]
        drift = (offs[-1] - offs[0]) if len(offs) > 2 else np.nan
        off_end = offs[-1] if offs else np.nan
        # command/delivery in last 2.5 s: model vs output vs wheel (baseline-relative)
        mw = win(d["mv"], mvt, t0 - 2.5, t0)
        cw = win(d["co"], cot, t0 - 2.5, t0)
        model_mean = np.mean([x[4] for x in mw]) if mw else np.nan
        out_mean = np.mean([x[1] for x in cw]) if cw else np.nan
        base = [ach_curv(x[2], x[1]) for x in win(d["cs"], cst, t0 - 5, t0 - 2.5)]
        b = np.median(base) if base else 0.0
        aw = [(x[0], ach_curv(x[2], x[1]) - b) for x in win(d["cs"], cst, t0 - 2.5, t0)]
        ach_move = (np.mean([a for (_, a) in aw[-20:]]) if aw else np.nan)
        # correct direction = opposite sign of offset (offset + = left of center -> steer right +)
        need_sign = np.sign(off_end) if np.isfinite(off_end) and abs(off_end) > 0.1 else 0.0
        pi_delta = out_mean - model_mean - 0.000035 if np.isfinite(out_mean) else np.nan
        lcw = win(d["lc"], lct, t0 - 4, t0 + 0.5)
        ints = [f"{x[2]:+.2f}" for x in lcw]
        alerts = [a for (ta, a) in win(d["al"], alt, t0 - 6, t0) if a]
        mode = "AOL" if (en_pct < 5 and lat_pct > 50) else ("ENGAGED" if en_pct > 50 else ("MANUAL" if lat_pct < 5 else "mixed"))
        print(f"\n[{route} OV t={t0:8.2f} seg{seg:2d} {v0*2.237:4.0f}mph]  mode={mode} (lat {lat_pct:.0f}%/eng {en_pct:.0f}%)  "
              f"preHandsOn={pressed_pre:.0f}%  blinker={'Y' if blk else 'n'}  desire lc/turn={des_lc:.2f}/{des_turn:.2f}")
        print(f"    minLP3s={lp3:.2f}  offset {offs[0] if offs else float('nan'):+.2f}->{off_end:+.2f} (drift {drift:+.2f} m)  "
              f"ovrTrq={tq0:+.1f}")
        print(f"    last2.5s: model={model_mean:+.6f} out={out_mean:+.6f} (PIdelta {pi_delta:+.6f})  "
              f"wheelMove(base-rel)={ach_move:+.6f}  needSign={'RIGHT+' if need_sign>0 else ('LEFT-' if need_sign<0 else '~')}")
        print(f"    PI integral (LC 1Hz): {ints if ints else 'n/a'}   alerts(-6s): {alerts if alerts else 'NONE'}")

    print(f"\n{'='*110}\n#### {route} — DEEP BLIND EVENTS (lp<0.15 for >=1.5 s, moving)\n{'='*110}")
    for (tc, tr, v) in deep_collapses(d):
        mvw = win(d["mv"], mvt, tc - 4, tc)
        seg = mvw[-1][1] if mvw else -1
        pre_cmd = np.mean([x[4] for x in mvw]) if mvw else np.nan
        blind = win(d["mv"], mvt, tc, tr)
        cmd_blind_end = np.mean([x[4] for x in blind[-10:]]) if len(blind) > 5 else np.nan
        offs = [x[3] for x in blind if np.isfinite(x[3])]
        drift = (offs[-1] - offs[0]) if len(offs) > 3 else np.nan
        ccw = win(d["cc"], cct, tc, tr)
        lat_pct = 100 * np.mean([x[1] for x in ccw]) if ccw else np.nan
        en_pct = 100 * np.mean([x[2] for x in ccw]) if ccw else np.nan
        blk = any(x[5] for x in win(d["cs"], cst, tc - 3, tc + 1))
        des_lc = max((x[5] for x in win(d["mv"], mvt, tc - 4, tr)), default=0.0)
        des_turn = max((x[6] for x in win(d["mv"], mvt, tc - 4, tr)), default=0.0)
        press = any(x[3] for x in win(d["cs"], cst, tc, tr))
        alerts = sorted(set(a for (ta, a) in win(d["al"], alt, tc - 1, tr + 1) if a))
        mode = "AOL" if (en_pct < 5 and lat_pct > 50) else ("ENGAGED" if en_pct > 50 else ("MANUAL" if lat_pct < 5 else "mixed"))
        print(f"\n[{route} BLIND t={tc:8.2f} seg{seg:2d} {v*2.237:4.0f}mph dur={tr-tc:5.2f}s]  mode={mode} "
              f"(lat {lat_pct:.0f}%/eng {en_pct:.0f}%)  blinker={'Y' if blk else 'n'} desire lc/turn={des_lc:.2f}/{des_turn:.2f} pressed={'Y' if press else 'n'}")
        print(f"    cmd before={pre_cmd:+.6f} -> end-of-blind={cmd_blind_end:+.6f} ({'DECAYED toward straight' if np.isfinite(pre_cmd) and abs(pre_cmd)>0.0008 and abs(cmd_blind_end)<0.6*abs(pre_cmd) else 'held/small'})  "
              f"offset drift during blind: {drift:+.2f} m")
        print(f"    alerts during: {alerts if alerts else 'NONE (silent)'}")


for route in ("route_ce", "route_cf"):
    analyze(route)
