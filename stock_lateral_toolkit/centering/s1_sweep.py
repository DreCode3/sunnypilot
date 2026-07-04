"""S1: CameraOffset counterfactual dose-response sweep on SP002 over the shared windows.

Per offset point: replay with camera_offset=delta + lane capture; measure
  d_center_m   = median lane-center(y0) shift vs the 0.0 replay  (predicted centering)
  band_ratio   = weave-band RMS(desired_curvature) / same at 0.0  (0.10-0.35 Hz)
  corr_vs_zero = corr(desired_curv(delta), desired_curv(0))       (curve proxy 1/2)
  low_band_ratio = <=0.05 Hz RMS ratio                            (curve proxy 2/2)
Noise band NB from the +/-0.005 m controls; determinism from a repeat at 0.0.
Gates: pre-registration M0 §5 (twinned in config).

RUN:
  caffeinate -i .venv311/bin/python stock_lateral_toolkit/centering/s1_sweep.py --replay
  .venv311/bin/python stock_lateral_toolkit/centering/s1_sweep.py --analyze
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC

S1_DIR = CC.RESULTS_DIR / "s1"
BUNDLE = "SP002"


def _pt_path(offset: float, window_id: int, rep: int = 0) -> Path:
    tag = f"{offset:+.3f}".replace(".", "p")
    return S1_DIR / f"sweep_{tag}_w{window_id}_r{rep}.npz"


def _replay_job(args) -> str:
    offset, window, rep = args
    from model_replay_sim.infer import replay_window
    r = replay_window(BUNDLE, window["route_id"], window["mono_times"],
                      camera_offset=float(offset), capture_outputs=("lane_lines",))
    out = _pt_path(offset, window["window_id"], rep)
    np.savez_compressed(out, mono_time=r["mono_time"],
                        desired_curvature=r["desired_curvature"],
                        lane_lines=r["captured"]["lane_lines"],
                        camera_offset_used=r["camera_offset_used"],
                        split_index=window["split_index"])
    return str(out)


def run_replays() -> None:
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()
    S1_DIR.mkdir(parents=True, exist_ok=True)
    points = list(CC.SWEEP_OFFSETS_M) + list(CC.CONTROL_OFFSETS_M)
    jobs = [(o, w, 0) for o in points for w in wins if not _pt_path(o, w["window_id"]).exists()]
    if not _pt_path(0.0, wins[0]["window_id"], rep=1).exists():
        jobs.append((0.0, wins[0], 1))            # determinism repeat
    print(f"{len(jobs)} replay jobs ({len(points)} offsets x {len(wins)} windows + repeat), "
          f"{CC.MAX_WORKERS_REPLAY} workers, ~0.41 s/frame")
    with ProcessPoolExecutor(max_workers=CC.MAX_WORKERS_REPLAY,
                             mp_context=mp.get_context("spawn")) as ex:
        for done in ex.map(_replay_job, jobs):
            print("done:", done)


def _point_metrics(offset: float, window_id: int, base: dict) -> dict:
    from model_replay_sim.metrics import weave_band_rms
    from stock_lateral_toolkit import signal_utils as SU
    z = np.load(_pt_path(offset, window_id))
    s = int(z["split_index"])
    curv = np.asarray(z["desired_curvature"], float)[s:]
    center = (z["lane_lines"][s:, 1, 0, 0] + z["lane_lines"][s:, 2, 0, 0]) / 2.0
    both = np.isfinite(curv) & np.isfinite(base["curv"])
    lo_self = SU.filter_continuous(curv, CC.FS_HZ, lowpass_hz=CC.LOW_BAND_HZ)
    lo_base = SU.filter_continuous(base["curv"], CC.FS_HZ, lowpass_hz=CC.LOW_BAND_HZ)
    def _rms(x):
        v = x[np.isfinite(x)]
        return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")
    return {
        "offset": float(offset), "window_id": window_id,
        "d_center_m": float(np.median(center) - base["center_med"]),
        "band_ratio": float(weave_band_rms(curv) / base["weave"]),
        "corr_vs_zero": float(np.corrcoef(curv[both], base["curv"][both])[0, 1]),
        "low_band_ratio": float(_rms(lo_self) / _rms(lo_base)),
        "dc_curv_shift": float(np.median(curv[both] - base["curv"][both])),
    }


def evaluate_gates(rows: list[dict], controls: dict, determinism_max_delta: float,
                   p_cam: float) -> dict:
    """Pure gate logic (unit-tested). rows = per-offset metrics AVERAGED over windows;
    controls = {control_offset: band_ratio}; p_cam = M-measured camera-relative deficit."""
    rows = sorted(rows, key=lambda r: r["offset"])
    off = np.array([r["offset"] for r in rows])
    dc = np.array([r["d_center_m"] for r in rows])
    rho = float(stats.spearmanr(off, dc)[0])
    slope = float(np.polyfit(off, dc, 1)[0])
    nb = max(CC.WEAVE_NOISE_FLOOR, 2.0 * max(abs(v - 1.0) for v in controls.values()))
    if determinism_max_delta > CC.DETERMINISM_TOL:
        nb = max(nb, 4.0 * determinism_max_delta)   # widen if replay is not bit-repeatable
    weave_ok = all(abs(r["band_ratio"] - 1.0) <= nb
                   and CC.WEAVE_HARD_CAP[0] <= r["band_ratio"] <= CC.WEAVE_HARD_CAP[1]
                   for r in rows)
    curve_ok = all(r["corr_vs_zero"] >= CC.CURVE_CORR_MIN
                   and CC.LOW_BAND_RATIO[0] <= r["low_band_ratio"] <= CC.LOW_BAND_RATIO[1]
                   for r in rows if r["offset"] != 0.0)
    monotonic = abs(rho) >= CC.MONOTONIC_SPEARMAN_MIN
    slope_ok = CC.SLOPE_UNIT_RANGE[0] <= abs(slope) <= CC.SLOPE_UNIT_RANGE[1]
    delta_star = float(p_cam / slope) if slope_ok and slope != 0 else float("nan")
    grid = [r["offset"] for r in rows]
    delta_star_grid = (min(grid, key=lambda o: abs(o - delta_star))
                       if np.isfinite(delta_star) else None)
    in_range = bool(np.isfinite(delta_star) and min(grid) <= delta_star <= max(grid))
    improvement_ok = bool(in_range and abs(slope * delta_star) >= 0.7 * abs(p_cam))
    return {"monotonic": monotonic, "spearman_rho": rho, "slope": slope, "slope_ok": slope_ok,
            "noise_band": nb, "weave_ok": weave_ok, "curve_ok": curve_ok,
            "delta_star_m": delta_star_grid, "delta_star_raw_m": delta_star,
            "delta_star_in_range": in_range, "improvement_ok": improvement_ok,
            "determinism_max_delta": determinism_max_delta,
            "all_pass": bool(monotonic and slope_ok and weave_ok and curve_ok
                             and in_range and improvement_ok)}


def analyze() -> None:
    from model_replay_sim.metrics import weave_band_rms
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()

    # determinism check: repeat at 0.0 on window 0
    z0 = np.load(_pt_path(0.0, wins[0]["window_id"], 0))
    z1 = np.load(_pt_path(0.0, wins[0]["window_id"], 1))
    det = float(np.nanmax(np.abs(z0["desired_curvature"] - z1["desired_curvature"])))
    print(f"determinism: max|repeat delta| = {det:.2e} (tol {CC.DETERMINISM_TOL})")

    bases = {}
    for w in wins:
        zb = np.load(_pt_path(0.0, w["window_id"]))
        s = int(zb["split_index"])
        curv = np.asarray(zb["desired_curvature"], float)[s:]
        center = (zb["lane_lines"][s:, 1, 0, 0] + zb["lane_lines"][s:, 2, 0, 0]) / 2.0
        bases[w["window_id"]] = {"curv": curv, "center_med": float(np.median(center)),
                                 "weave": weave_band_rms(curv)}

    per_offset, controls = [], {}
    for o in CC.SWEEP_OFFSETS_M:
        pts = [_point_metrics(o, w["window_id"], bases[w["window_id"]]) for w in wins]
        per_offset.append({k: float(np.mean([p[k] for p in pts])) if k != "window_id" else -1
                           for k in pts[0]} | {"offset": o, "per_window": pts})
    for o in CC.CONTROL_OFFSETS_M:
        pts = [_point_metrics(o, w["window_id"], bases[w["window_id"]]) for w in wins]
        controls[o] = float(np.mean([p["band_ratio"] for p in pts]))

    _cons = CC.RESULTS_DIR / "m1" / "m1_results_consolidated.json"   # amended 2026-07-04b: sizing uses consolidated P
    _m1 = _cons if _cons.exists() else (CC.RESULTS_DIR / "m1" / "m1_results.json")
    p_cam = json.loads(_m1.read_text())["median_offset_cam_m"] if _m1.exists() else float("nan")
    if not np.isfinite(p_cam):
        print("NOTE: M1 not done yet — gates evaluated with p_cam=nan (sizing gates will fail); "
              "re-run --analyze after Task 9.")
    gates = evaluate_gates([{k: r[k] for k in ("offset", "d_center_m", "band_ratio",
                                               "corr_vs_zero", "low_band_ratio")}
                            for r in per_offset], controls, det, p_cam)

    res = {"bundle": BUNDLE, "p_cam_m": p_cam, "gates": gates,
           "controls_band_ratio": {str(k): v for k, v in controls.items()},
           "dose_response": per_offset}
    (S1_DIR / "s1_report.json").write_text(json.dumps(res, indent=2))
    md = ["# S1 CameraOffset dose-response (SP002, shared windows)", "",
          "| offset (m) | d_center (m) | band_ratio | corr_vs_0 | low_band_ratio | dc_shift (1/m) |",
          "|---|---|---|---|---|---|"]
    for r in per_offset:
        md.append(f"| {r['offset']:+.2f} | {r['d_center_m']:+.4f} | {r['band_ratio']:.3f} "
                  f"| {r['corr_vs_zero']:.4f} | {r['low_band_ratio']:.3f} | {r['dc_curv_shift']:+.2e} |")
    md += ["", f"noise band NB = {gates['noise_band']:.3f}; determinism delta = {det:.2e}",
           f"slope = {gates['slope']:+.3f} (unit-range gate {CC.SLOPE_UNIT_RANGE}); "
           f"Spearman rho = {gates['spearman_rho']:+.3f}",
           f"delta* = {gates['delta_star_m']} m (raw {gates['delta_star_raw_m']}); "
           f"ALL GATES PASS = {gates['all_pass']}"]
    (S1_DIR / "s1_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if "--replay" in sys.argv:
        run_replays()
    elif "--analyze" in sys.argv:
        analyze()
    else:
        print("usage: s1_sweep.py --replay | --analyze")
