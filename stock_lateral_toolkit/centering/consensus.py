"""M2b: cross-model lane-center consensus on IDENTICAL frames.

Replays every bundle in config.BUNDLES_M2 over the shared Task-5 windows with
lane-line capture, then computes per-model lane-center series, cross-model pairwise
disagreement, the consensus (per-frame median across models), and the overlap
comparison against M1 video frames.

RUN:
  .venv311/bin/python stock_lateral_toolkit/centering/consensus.py --replay   # heavy (~20 min wall)
  .venv311/bin/python stock_lateral_toolkit/centering/consensus.py --stats    # seconds
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC

M2_DIR = CC.RESULTS_DIR / "m2"


def lane_center_series(lane_lines: np.ndarray, x_eval: float = 0.0):
    """(N,4,33,2) captured lane_lines -> (center_y, width) at forward distance x_eval.
    Calibrated y (+RIGHT); center>0 == car LEFT of that model's lane center (canonical)."""
    yl = lane_lines[:, 1, :, 0]
    yr = lane_lines[:, 2, :, 0]
    if x_eval == 0.0:
        yl0, yr0 = yl[:, 0], yr[:, 0]
    else:
        from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
        xg = np.asarray(ModelConstants.X_IDXS, dtype=float)
        yl0 = np.array([np.interp(x_eval, xg, row) for row in yl])
        yr0 = np.array([np.interp(x_eval, xg, row) for row in yr])
    return (yl0 + yr0) / 2.0, (yr0 - yl0)


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    both = np.isfinite(a) & np.isfinite(b)
    d = a[both] - b[both]
    return {"n": int(both.sum()),
            "median_delta_m": float(np.median(d)) if both.any() else float("nan"),
            "iqr_m": float(np.subtract(*np.percentile(d, [75, 25]))) if both.any() else float("nan"),
            "corr": float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 2 else float("nan")}


def _npz_path(bundle: str, window_id: int) -> Path:
    return M2_DIR / f"replay_{bundle}_w{window_id}.npz"


def _replay_job(args) -> str:
    bundle, window = args
    from model_replay_sim.infer import replay_window
    r = replay_window(bundle, window["route_id"], window["mono_times"],
                      capture_outputs=("lane_lines", "lane_lines_prob"))
    out = _npz_path(bundle, window["window_id"])
    np.savez_compressed(out,
                        mono_time=r["mono_time"], v_ego=r["v_ego"],
                        desired_curvature=r["desired_curvature"],
                        lane_lines=r["captured"]["lane_lines"],
                        lane_lines_prob=r["captured"]["lane_lines_prob"],
                        camera_offset_used=r["camera_offset_used"],
                        split_index=window["split_index"])
    return str(out)


def run_replays() -> None:
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()
    M2_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(b, w) for b in CC.BUNDLES_M2 for w in wins
            if not _npz_path(b, w["window_id"]).exists()]      # resumable
    print(f"{len(jobs)} replay jobs (bundles {CC.BUNDLES_M2} x {len(wins)} windows), "
          f"{CC.MAX_WORKERS_REPLAY} workers")
    with ProcessPoolExecutor(max_workers=CC.MAX_WORKERS_REPLAY,
                             mp_context=mp.get_context("spawn")) as ex:
        for done in ex.map(_replay_job, jobs):
            print("done:", done)


def run_stats() -> None:
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()
    res = {"bundles": list(CC.BUNDLES_M2), "windows": [w["window_id"] for w in wins],
           "per_window": {}, "vs_m1": {}}
    centers_all = {b: [] for b in CC.BUNDLES_M2}
    mono_all = []
    for w in wins:
        wid = w["window_id"]; split = int(w["split_index"])
        per = {}
        series = {}
        for b in CC.BUNDLES_M2:
            z = np.load(_npz_path(b, wid))
            c0, wd0 = lane_center_series(z["lane_lines"], 0.0)
            c10, _ = lane_center_series(z["lane_lines"], 10.0)
            series[b] = c0[split:]
            per[b] = {"own_center_median_y0_m": float(np.median(c0[split:])),
                      "own_center_median_y10_m": float(np.median(c10[split:])),
                      "width_median_m": float(np.median(wd0[split:])),
                      "min_inner_prob_median": float(np.median(
                          np.minimum(z["lane_lines_prob"][split:, 3], z["lane_lines_prob"][split:, 5])))}
            centers_all[b].append(c0[split:])
            if b == CC.BUNDLES_M2[0]:
                mono_all.append(np.asarray(z["mono_time"])[split:])
        pairs = {}
        bl = list(CC.BUNDLES_M2)
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                pairs[f"{bl[i]}-{bl[j]}"] = pair_stats(series[bl[i]], series[bl[j]])
        stacked = np.vstack([series[b] for b in bl])
        res["per_window"][str(wid)] = {"per_bundle": per, "pairs": pairs,
                                       "consensus_median_y0_m": float(np.median(np.median(stacked, axis=0)))}

    # overlap vs M1, matched by FRAME IDENTITY: the manifest's (seg_num, seg_id) maps
    # through the frame timeline to the eof timestamps the replay windows are keyed on.
    # (The manifest's own mono_time is the modelV2 PUBLISH time — eof + inference
    # latency — so a raw 1 ms mono match can never hit; frame identity is exact.)
    m1_csv = CC.RESULTS_DIR / "m1" / "per_frame_offsets.csv"
    if m1_csv.exists():
        from model_replay_sim.alignment import build_frame_timeline
        eof_by_seg = {(row.segment_num, row.segment_id): row.timestamp_eof_s
                      for row in build_frame_timeline(CC.ROUTE)}
        manifest = {int(r["frame_idx"]): r for r in
                    csv.DictReader(open(CC.RESULTS_DIR / "m1" / "frames_manifest.csv"))}
        m1_rows = [r for r in csv.DictReader(open(m1_csv)) if r["in_m2_window"] == "True"]
        mono_cat = np.concatenate(mono_all)
        for b in CC.BUNDLES_M2:
            cat = np.concatenate(centers_all[b])
            deltas = []
            for r in m1_rows:
                man = manifest[int(r["frame_idx"])]
                eof = eof_by_seg.get((int(man["seg_num"]), int(man["seg_id"])))
                if eof is None:
                    continue
                k = int(np.argmin(np.abs(mono_cat - eof)))
                if abs(mono_cat[k] - eof) < 1e-3:
                    deltas.append(float(cat[k]) - float(r["offset_cam_m"]))
            res["vs_m1"][b] = {"n_overlap": len(deltas),
                               "median_model_minus_video_m": float(np.median(deltas)) if deltas else None,
                               "mad_m": float(np.median(np.abs(np.array(deltas) - np.median(deltas)))) if deltas else None}
    else:
        print("NOTE: m1 per-frame offsets not present yet; vs_m1 section empty (re-run --stats after Task 9)")

    (M2_DIR / "m2_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    if "--replay" in sys.argv:
        run_replays()
    elif "--stats" in sys.argv:
        run_stats()
    else:
        print("usage: consensus.py --replay | --stats")
