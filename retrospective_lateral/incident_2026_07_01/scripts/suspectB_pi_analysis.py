#!/usr/bin/env python3
"""Suspect B: Ford PI lane-centering — helping, nothing, or actively wrong?

Subcommands:
  initdata [routes...]   dump initData.params keys of interest (LaneBiasIntegral,
                         enable_lane_positioning, FordCurveMode, FordPath4Enabled)
  signs [routes...]      empirical sign conventions from OLD-build logs (default c7,c5):
                         - laneLines[1]/[2] y[0] sign (which sign is the left line)
                         - GPS(NED heading-rate) vs desiredCurvature / steeringAngleDeg /
                           carState.yawRate / carOutput.curvature  (which sign turns right)
                         - laneline far-bend vs curvature (does +curv turn toward +laneline-y)
                         - position.y convention vs lanelines midpoint
                         - straight one-sided-offset episodes: what would the PI command
  event                  quantify PI contribution in the decisive route_ce event
                         (t0=212.535 mono s, seg 000000ce--f8a8d4f570--9, 62 mph):
                         measured carOutput-vs-desired delta + full carcontroller-chain
                         sim (PI off / weak / golden, integral variants) vs logged carOutput
"""
import sys, glob, bisect
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader  # noqa: E402
from openpilot.selfdrive.modeld.constants import ModelConstants  # noqa: E402

T_IDXS = np.array(ModelConstants.T_IDXS)

KEYS_OF_INTEREST = ("lanebias", "lane_positioning", "fordcurvemode", "fordpath4",
                    "disable_bp_long", "disableupdates")


def rlogs(route):
    return sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst"),
                  key=lambda p: int(p.split("--")[-1].split("/")[0]))


def safe_iter(path):
    try:
        it = LogReader(path)
    except Exception as e:
        print(f"  !! open fail {path}: {e}")
        return
    for m in it:
        try:
            m.which()
        except Exception:
            continue
        yield m


# ---------------------------------------------------------------- initdata
def cmd_initdata(routes):
    for route in routes:
        found = False
        for rl in rlogs(route):
            for m in safe_iter(rl):
                if m.which() != "initData":
                    continue
                found = True
                print(f"\n=== {route} initData (from {rl.split('/')[-2]}) ===")
                try:
                    entries = m.initData.params.entries
                except Exception as e:
                    print("  params read fail:", e); break
                hits = 0
                for e in entries:
                    k = str(e.key)
                    if any(s in k.lower() for s in KEYS_OF_INTEREST):
                        try:
                            v = bytes(e.value).decode(errors="replace")
                        except Exception:
                            v = repr(e.value)
                        print(f"  {k} = {v!r}")
                        hits += 1
                print(f"  ({hits} matching keys of {len(entries)} total params)")
                break
            if found:
                break
        if not found:
            print(f"\n=== {route}: NO initData found ===")


# ---------------------------------------------------------------- signs
def _extract_signs(rl):
    out = {"mv": [], "cs": [], "cc": [], "co": [], "llk": []}
    for m in safe_iter(rl):
        w = m.which()
        t = m.logMonoTime / 1e9
        if w == "modelV2":
            M = m.modelV2
            ll, lp = list(M.laneLines), list(M.laneLineProbs)
            if len(ll) < 3 or len(lp) < 3 or len(ll[1].y) < 16 or len(ll[2].y) < 16:
                continue
            py = list(M.position.y)
            if len(py) < 16:
                continue
            out["mv"].append((t,
                              float(ll[1].y[0]), float(ll[2].y[0]),
                              float(ll[1].y[15]) - float(ll[1].y[0]),   # far-bend of left line
                              float(ll[2].y[15]) - float(ll[2].y[0]),   # far-bend of right line
                              float(lp[1]), float(lp[2]),
                              float(M.action.desiredCurvature),
                              float(np.interp(0.2, T_IDXS[:len(py)], py)),
                              float(py[15])))
        elif w == "carState":
            c = m.carState
            out["cs"].append((t, float(c.vEgo), float(c.steeringAngleDeg), float(c.yawRate),
                              1.0 if c.steeringPressed else 0.0,
                              1.0 if (c.leftBlinker or c.rightBlinker) else 0.0))
        elif w == "carControl":
            out["cc"].append((t, 1.0 if m.carControl.latActive else 0.0,
                              1.0 if m.carControl.enabled else 0.0,
                              float(m.carControl.actuators.curvature)))
        elif w == "carOutput":
            out["co"].append((t, float(m.carOutput.actuatorsOutput.curvature)))
        elif w == "liveLocationKalman":
            L = m.liveLocationKalman
            v = list(L.calibratedOrientationNED.value)
            if len(v) == 3:
                out["llk"].append((t, float(v[2])))
    return {k: np.array(v) for k, v in out.items()}


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def cmd_signs(routes):
    for route in routes:
        files = rlogs(route)
        with ProcessPoolExecutor(max_workers=12) as ex:
            parts = list(ex.map(_extract_signs, files))
        D = {}
        for k in ("mv", "cs", "cc", "co", "llk"):
            arrs = [p[k] for p in parts if p[k].size]
            D[k] = np.concatenate(arrs) if arrs else np.zeros((0, 2))
        mv, cs, cc, co, llk = D["mv"], D["cs"], D["cc"], D["co"], D["llk"]
        print(f"\n================ {route}: {len(files)} segs, {len(mv)} modelV2 frames ================")
        if not len(mv) or not len(llk):
            print("  missing data, skip"); continue

        tm = mv[:, 0]
        # GPS heading rate (NED: + heading-rate = clockwise = RIGHT turn)
        hy = np.unwrap(llk[:, 1]); ht = llk[:, 0]
        hr = np.gradient(hy, ht)
        # smooth 0.5 s
        n = max(1, int(0.5 * len(ht) / max(ht[-1] - ht[0], 1)))
        hr = np.convolve(hr, np.ones(n) / n, mode="same")
        hr_m = np.interp(tm, ht, hr)

        v_m = np.interp(tm, cs[:, 0], cs[:, 1])
        ang_m = np.interp(tm, cs[:, 0], cs[:, 2])
        yaw_m = np.interp(tm, cs[:, 0], cs[:, 3])
        prs_m = np.interp(tm, cs[:, 0], cs[:, 4])
        blk_m = np.interp(tm, cs[:, 0], cs[:, 5])
        lat_m = np.interp(tm, cc[:, 0], cc[:, 1])
        co_m = np.interp(tm, co[:, 0], co[:, 1])

        lY0, rY0 = mv[:, 1], mv[:, 2]
        bendL, bendR = mv[:, 3], mv[:, 4]
        lp, rp = mv[:, 5], mv[:, 6]
        des = mv[:, 7]
        posy02, posy15 = mv[:, 8], mv[:, 9]
        mid = (lY0 + rY0) / 2
        goodp = np.minimum(lp, rp) > 0.6

        drv = (v_m > 15) & (lat_m > 0.5) & (prs_m < 0.5) & (blk_m < 0.5)
        print(f"[A] laneline y[0] sign (driving, probs>0.6, n={int((drv & goodp).sum())}):")
        print(f"    laneLines[1] ('left')  y[0]: mean {lY0[drv & goodp].mean():+.3f}  median {np.median(lY0[drv & goodp]):+.3f}")
        print(f"    laneLines[2] ('right') y[0]: mean {rY0[drv & goodp].mean():+.3f}  median {np.median(rY0[drv & goodp]):+.3f}")
        print(f"    lane width per code (rY0 - lY0): median {np.median((rY0 - lY0)[drv & goodp]):+.3f} m")

        # GPS ground truth: + hr = right turn
        k_gps = hr_m / np.maximum(v_m, 1.0)  # signed curvature, + = RIGHT turn (NED)
        print(f"[B] conventions vs GPS heading-rate (+ = RIGHT turn), driving frames n={int(drv.sum())}:")
        for name, sig in (("modelV2.action.desiredCurvature", des),
                          ("carControl.actuators.curvature ", np.interp(tm, cc[:, 0], cc[:, 3])),
                          ("carOutput.curvature            ", co_m),
                          ("carState.steeringAngleDeg      ", ang_m),
                          ("carState.yawRate               ", yaw_m)):
            c = _corr(sig[drv], k_gps[drv])
            lbl = "+ = RIGHT turn" if c > 0.3 else ("+ = LEFT turn" if c < -0.3 else "ambiguous")
            print(f"    corr({name}, k_gps) = {c:+.3f}   -> {lbl}")

        curvy = drv & goodp & (np.abs(des) > 0.0012)
        print(f"[C] does +curvature turn toward +laneline-y?  (curve frames n={int(curvy.sum())})")
        cL = _corr(des[curvy], bendL[curvy]); cR = _corr(des[curvy], bendR[curvy])
        print(f"    corr(desiredCurvature, laneline far-bend): left {cL:+.3f}  right {cR:+.3f}")
        print("    -> positive means +curv turns toward +y of the laneline frame "
              "(then PI 'apply += kp*midpoint' steers TOWARD lane center)")
        cP = _corr(des[curvy], posy15[curvy])
        print(f"    corr(desiredCurvature, position.y[15]) = {cP:+.3f}  (position-frame check)")
        cM = _corr(mid[drv & goodp & (np.abs(des) < 0.0005)], posy02[drv & goodp & (np.abs(des) < 0.0005)])
        print(f"    corr(laneline midpoint, position.y@0.2s) on straights = {cM:+.3f}  (same/opposite convention)")

        # straight one-sided offset episodes
        st = drv & goodp & (np.abs(des) < 0.0005) & (v_m > 20)
        runs = []
        i = 0
        while i < len(tm):
            if st[i] and abs(mid[i]) > 0.10:
                s0 = np.sign(mid[i]); j = i
                while j < len(tm) and st[j] and np.sign(mid[j]) == s0 and abs(mid[j]) > 0.05:
                    j += 1
                if tm[j - 1] - tm[i] >= 2.0:
                    runs.append((i, j, s0))
                i = j
            else:
                i += 1
        print(f"[D] straight-road one-sided-offset episodes (|mid|>0.10 m sustained >=2 s): {len(runs)}")
        for (i0, j0, s0) in runs[:8]:
            seg_mid = mid[i0:j0]; dur = tm[j0 - 1] - tm[i0]
            after = mid[j0:j0 + 60]
            print(f"    t={tm[i0]:.1f}s dur={dur:4.1f}s  mid {seg_mid[0]:+.2f}->{seg_mid[-1]:+.2f} "
                  f"(mean {seg_mid.mean():+.2f})  next3s_mean {after.mean() if len(after) else np.nan:+.2f}  "
                  f"PI P-term(golden)={0.0005 * seg_mid.mean():+.6f} 1/m")


# ---------------------------------------------------------------- event
EVENT_T0 = 212.535           # mono s, override onset (cluster idx 3)
EVENT_ROUTE = "route_ce"


def _extract_event(rl):
    out = {"mv": [], "mvarr": [], "cs": [], "cc": [], "co": []}
    for m in safe_iter(rl):
        w = m.which()
        t = m.logMonoTime / 1e9
        if w == "modelV2":
            M = m.modelV2
            ll, lp = list(M.laneLines), list(M.laneLineProbs)
            if len(ll) < 3 or len(lp) < 3 or len(ll[1].y) < 1:
                continue
            py = list(M.position.y); oz = list(M.orientationRate.z)
            out["mv"].append((t, float(ll[1].y[0]), float(ll[2].y[0]),
                              float(lp[1]), float(lp[2]),
                              float(M.action.desiredCurvature),
                              float(np.interp(0.2, T_IDXS[:len(py)], py)) if len(py) > 4 else np.nan))
            out["mvarr"].append(np.array(oz, dtype=np.float32))
        elif w == "carState":
            c = m.carState
            out["cs"].append((t, float(c.vEgoRaw), float(c.steeringAngleDeg), float(c.yawRate),
                              1.0 if c.steeringPressed else 0.0))
        elif w == "carControl":
            out["cc"].append((t, 1.0 if m.carControl.latActive else 0.0,
                              1.0 if m.carControl.enabled else 0.0,
                              float(m.carControl.actuators.curvature)))
        elif w == "carOutput":
            out["co"].append((t, float(m.carOutput.actuatorsOutput.curvature)))
    return out


def cmd_event():
    # window: warmup from t0-40 s, report t0-8..t0+4
    # t0=212.535 mono is in segment 2 (carState spans: seg1 101.6-161.6, seg2 161.7-221.7, seg3 221.7-281.6)
    files = [f for f in rlogs(EVENT_ROUTE) if int(f.split("--")[-1].split("/")[0]) in (1, 2, 3)]
    with ProcessPoolExecutor(max_workers=3) as ex:
        parts = list(ex.map(_extract_event, files))
    mv = np.array(sum((p["mv"] for p in parts), []))
    oz_all = sum((p["mvarr"] for p in parts), [])
    cs = np.array(sum((p["cs"] for p in parts), []))
    cc = np.array(sum((p["cc"] for p in parts), []))
    co = np.array(sum((p["co"] for p in parts), []))
    o = np.argsort(mv[:, 0]); mv = mv[o]; oz_all = [oz_all[i] for i in o]
    cs = cs[np.argsort(cs[:, 0])]; cc = cc[np.argsort(cc[:, 0])]; co = co[np.argsort(co[:, 0])]

    t0 = EVENT_T0
    tm = mv[:, 0]

    # ---- part 5a: raw measured delta carOutput vs model desired, pre-override 3 s
    print("=== decisive route_ce event: t0(mono)=%.3f s, seg --9, 62 mph ===" % t0)
    for lo, hi, lbl in ((t0 - 3, t0, "drift window t0-3..t0"), (t0 - 8, t0 - 3, "baseline t0-8..t0-3")):
        m = (co[:, 0] >= lo) & (co[:, 0] < hi)
        des_i = np.interp(co[m, 0], tm, mv[:, 5])
        act_i = np.interp(co[m, 0], cc[:, 0], cc[:, 3])
        d_model = co[m, 1] - des_i
        d_act = co[m, 1] - act_i
        print(f"[{lbl}] n={m.sum()}")
        print(f"   carOutput - modelV2.action.desiredCurvature: mean {d_model.mean():+.6f}  "
              f"mean|.| {np.abs(d_model).mean():.6f}  max|.| {np.abs(d_model).max():.6f}")
        print(f"   carOutput - carControl.actuators.curvature : mean {d_act.mean():+.6f}  "
              f"mean|.| {np.abs(d_act).mean():.6f}  max|.| {np.abs(d_act).max():.6f}")
        print(f"   mean desired {des_i.mean():+.6f}  mean carOutput {co[m,1].mean():+.6f}")

    # PI expectation for the offset
    m3 = (tm >= t0 - 3) & (tm < t0)
    mid = (mv[:, 1] + mv[:, 2]) / 2
    print(f"   laneline midpoint over drift window: {mid[m3][0]:+.3f} -> {mid[m3][-1]:+.3f} m "
          f"(mean {mid[m3].mean():+.3f}); minProb {np.minimum(mv[m3,3], mv[m3,4]).min():.2f}")
    print(f"   PI P-term should be: golden kp=5e-4 -> {0.0005*mid[m3].mean():+.6f} ; weak kp=1e-4 -> {0.0001*mid[m3].mean():+.6f}")
    print(f"   PI I-term bounds: ki=2e-4 * I(-0.3775 persisted) = {0.0002*-0.3775:+.6f} ; cap at 62mph(golden) ±{np.interp(27.7,[20,30],[0.3,1.0]):.2f} -> ±{0.0002*np.interp(27.7,[20,30],[0.3,1.0]):.6f}")

    # ---- full-chain sim
    try:
        from opendbc.car.ford.carcontroller import apply_ford_curvature_limits
        from opendbc.car.ford.values import CarControllerParams
        limits_fn = apply_ford_curvature_limits

        class _CP:  # minimal stub: Explorer ST is CAN FD
            from opendbc.car.ford.values import FordFlags
            flags = FordFlags.CANFD
        CPstub = _CP()
        angle_limits = CarControllerParams.CURVE_MODE_PARAMS[0]["angle_limits"]
        print("   [sim] using real apply_ford_curvature_limits + mode-0 angle limits")
    except Exception as e:
        limits_fn = None
        print("   [sim] real limits import failed (%s) — simple ±0.004 clip only" % e)

    t_start = tm[0] + 1.0
    grid = np.arange(t_start, t0 + 4.0, 0.05)
    v_g = np.interp(grid, cs[:, 0], cs[:, 1])
    prs_g = np.interp(grid, cs[:, 0], cs[:, 4]) > 0.5
    ang_g = np.interp(grid, cs[:, 0], cs[:, 2])
    yaw_g = np.interp(grid, cs[:, 0], cs[:, 3])
    act_g = np.interp(grid, cc[:, 0], cc[:, 3])
    co_g = np.interp(grid, co[:, 0], co[:, 1])
    mv_idx = np.searchsorted(tm, grid, side="right") - 1

    def run_sim(pi_mode, I0):
        """pi_mode: None | 'weak' | 'golden'. Returns sim carOutput + PI contribution trace."""
        kp = {"weak": 0.0001, "golden": 0.0005}.get(pi_mode, 0.0)
        ki = 0.0002 if pi_mode else 0.0
        ema_off = 0.0; I = I0; smooth_last = co_g[0]; apply_last = co_g[0]
        sim = np.zeros_like(grid); picontrib = np.zeros_like(grid)
        p_tr = np.zeros_like(grid); i_tr = np.zeros_like(grid)
        for k, t in enumerate(grid):
            v = v_g[k]; i = mv_idx[k]
            des = act_g[k]
            oz = oz_all[i]
            if v > 1.0 and len(oz) >= len(T_IDXS):
                lookup = float(np.interp(v, [15, 25], [0.5, 0.35]))
                pred = float(np.interp(lookup, T_IDXS, np.array(oz) / v))
            else:
                pred = des
            blend = float(np.interp(v, [7., 20., 27., 35.], [0.10, 0.30, 0.20, 0.10]))
            ac = pred * blend + des * (1 - blend)
            # deadband ~0 at this speed; EMA
            tau = float(np.interp(v, [4., 7., 25.], [0.12, 0.12, 0.04]))
            alpha = 1.0 - np.exp(-0.05 / tau)
            ac = float(alpha * ac + (1 - alpha) * smooth_last)
            smooth_last = ac
            # PI block
            pi = 0.0
            lY, rY, lp_, rp_, py02 = mv[i, 1], mv[i, 2], mv[i, 3], mv[i, 4], mv[i, 6]
            if pi_mode and v > 7.0:
                width = rY - lY
                conf = min(lp_, rp_, float(np.interp(width, [3.75, 4.25], [0.81, 0.59])))
                if conf > 0.6:
                    scale = float(np.interp(conf, [0.6, 0.8], [0.0, 1.0]))
                    raw = (py02 if np.isfinite(py02) else 0.0) * (1 - scale) + (lY + rY) / 2 * scale
                    a2 = 1.0 - np.exp(-0.05 / 1.5)
                    ema_off = a2 * raw + (1 - a2) * ema_off
                    if abs(ac) < 0.005 and not prs_g[k]:
                        I += ema_off * 0.05
                        cap = 0.30 if pi_mode == "weak" else float(np.interp(v, [20., 30.], [0.3, 1.0]))
                        I = float(np.clip(I, -cap, cap))
                        if ema_off * I < 0:
                            I *= 0.97
                    else:
                        I *= (0.98 if pi_mode == "weak" else 0.995)
                    pi = kp * ema_off + ki * I
                    p_tr[k] = kp * ema_off; i_tr[k] = ki * I
                    ac += pi
                else:
                    I *= 0.98
            ac += 0.000035
            cur = -yaw_g[k] / max(v, 0.1)
            if prs_g[k] and abs(ang_g[k]) > 45.0:
                ac = cur; smooth_last = cur; I = 0.0; ema_off = 0.0
            elif prs_g[k]:
                ac = 0.6 * cur + 0.4 * smooth_last; smooth_last = ac
            if limits_fn is not None:
                apply_last = limits_fn(ac, apply_last, cur, v, 0., True, CPstub,
                                       curvature_error=0.004, angle_limits=angle_limits)
            else:
                apply_last = float(np.clip(ac, cur - 0.004, cur + 0.004))
            sim[k] = apply_last; picontrib[k] = pi
        return sim, picontrib, I, p_tr, i_tr

    win = (grid >= t0 - 3) & (grid < t0)
    base = (grid >= t0 - 8) & (grid < t0 - 3)
    print("\n[sim vs logged carOutput] RMS over drift window (t0-3..t0) / baseline (t0-8..t0-3):")
    results = {}
    for lbl, mode, I0 in (("PI OFF          ", None, 0.0),
                          ("weak,  I0=-0.3775", "weak", -0.3775),
                          ("weak,  I0=0      ", "weak", 0.0),
                          ("golden,I0=-0.3775", "golden", -0.3775),
                          ("golden,I0=0      ", "golden", 0.0)):
        sim, pic, Iend, p_tr, i_tr = run_sim(mode, I0)
        rms_w = float(np.sqrt(np.mean((sim[win] - co_g[win]) ** 2)))
        rms_b = float(np.sqrt(np.mean((sim[base] - co_g[base]) ** 2)))
        results[lbl] = (sim, pic, p_tr, i_tr)
        print(f"   {lbl}: RMS drift {rms_w:.6f}  base {rms_b:.6f}  "
              f"PI-contrib drift mean {pic[win].mean():+.6f}  I_end {Iend:+.4f}")

    # timeline
    print("\n[timeline] t-t0 | v | desired(actuators) | carOutput | simOFF | simGOLD(I-.38) | mid | PIgold | P | I-term")
    simoff = results["PI OFF          "][0]; simg, picg, ptr, itr = results["golden,I0=-0.3775"]
    for t in np.arange(t0 - 6, t0 + 1.5, 0.5):
        k = int(np.searchsorted(grid, t))
        if k >= len(grid):
            break
        i = mv_idx[k]
        print(f"   {t-t0:+5.1f} | {v_g[k]:4.1f} | {act_g[k]:+.6f} | {co_g[k]:+.6f} | "
              f"{simoff[k]:+.6f} | {simg[k]:+.6f} | {(mv[i,1]+mv[i,2])/2:+.2f} | {picg[k]:+.6f} | "
              f"{ptr[k]:+.6f} | {itr[k]:+.6f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "signs"
    if cmd == "initdata":
        cmd_initdata(sys.argv[2:] or ["route_ce", "route_cf", "route_c7", "route_c5"])
    elif cmd == "signs":
        cmd_signs(sys.argv[2:] or ["route_c7", "route_c5"])
    elif cmd == "event":
        cmd_event()
    else:
        print(__doc__)
