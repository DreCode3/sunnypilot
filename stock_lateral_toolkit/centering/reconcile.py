"""M3: reconcile M1 (video), M2 (cross-model replay), and the logs against the
pre-registered tolerances (M0 §3), and decompose the logged offset:
    L (logged, model-frame) = P_cam (true, camera-relative) + dmid (definition bias).
Writes results/m3_verdict.json + results/m3_verdict.md.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/reconcile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def verdict(m1_median_cam: float, m2_sp002_vs_m1: float, m2_sp002_median_own: float,
            logged_median: float, consensus_pairs_max_abs: float) -> dict:
    """Pure verdict logic against M0 §3 (unit-tested; main() feeds real numbers)."""
    g1 = abs(m2_sp002_vs_m1) <= CC.TOL_M1_VS_M2_M
    g2 = abs(m2_sp002_median_own - logged_median) <= CC.TOL_M2_VS_LOGGED_M
    return {
        "gate_m1_vs_m2": "PASS" if g1 else "FAIL",
        "gate_m1_vs_m2_value_m": float(m2_sp002_vs_m1),
        "gate_m2_vs_logged": "PASS" if g2 else "FAIL",
        "gate_m2_vs_logged_value_m": float(m2_sp002_median_own - logged_median),
        "consensus_flag": "AGREE" if consensus_pairs_max_abs <= CC.TOL_CONSENSUS_DISAGREE_M
                          else "DEFINITIONS_DIFFER",
        "P_cam_m": float(m1_median_cam),
        "L_logged_m": float(logged_median),
        "dmid_logged_minus_video_m": float(logged_median - m1_median_cam),
        "proceed_to_R": bool(g1 and g2),
    }


def main():
    m1 = json.loads((CC.RESULTS_DIR / "m1" / "m1_results.json").read_text())
    m2 = json.loads((CC.RESULTS_DIR / "m2" / "m2_results.json").read_text())

    # logged L: median lane_center_y0 over the SAME eligibility the sampler used
    z = dict(np.load(CC.CACHE_NPZ))
    from stock_lateral_toolkit.centering.frame_sampler import eligibility
    ok, _ = eligibility(z)
    logged_median = float(np.nanmedian(np.asarray(z["lane_center_y0"], float)[ok]))

    sp = "SP002"
    own = [w["per_bundle"][sp]["own_center_median_y0_m"] for w in m2["per_window"].values()]
    pair_max = max(abs(p["median_delta_m"]) for w in m2["per_window"].values()
                   for p in w["pairs"].values())
    vs_m1 = m2["vs_m1"].get(sp, {})
    if vs_m1.get("median_model_minus_video_m") is None:
        raise SystemExit("M3 blocked: m2_results.json has no vs_m1 overlap — "
                         "run Task 9 then `consensus.py --stats` again.")

    v = verdict(m1_median_cam=float(m1["median_offset_cam_m"]),
                m2_sp002_vs_m1=float(vs_m1["median_model_minus_video_m"]),
                m2_sp002_median_own=float(np.median(own)),
                logged_median=logged_median,
                consensus_pairs_max_abs=float(pair_max))
    v["n_m1_frames"] = m1["n_frames"]
    v["n_overlap"] = vs_m1["n_overlap"]
    v["P_vehicle_m"] = m1.get("median_offset_vehicle_m")

    (CC.RESULTS_DIR / "m3_verdict.json").write_text(json.dumps(v, indent=2))
    lines = ["# M3 reconciliation verdict", "",
             "| check | value (m) | tolerance (m) | verdict |", "|---|---|---|---|",
             f"| M1 vs M2(SP002) | {v['gate_m1_vs_m2_value_m']:+.3f} | ±{CC.TOL_M1_VS_M2_M} | {v['gate_m1_vs_m2']} |",
             f"| M2(SP002) vs logged | {v['gate_m2_vs_logged_value_m']:+.3f} | ±{CC.TOL_M2_VS_LOGGED_M} | {v['gate_m2_vs_logged']} |",
             f"| cross-model max pair Δ | {pair_max:+.3f} | {CC.TOL_CONSENSUS_DISAGREE_M} | {v['consensus_flag']} |",
             "", f"Decomposition: L = {v['L_logged_m']:+.3f} = P_cam {v['P_cam_m']:+.3f} "
                 f"+ dmid {v['dmid_logged_minus_video_m']:+.3f}",
             f"P (vehicle) = {v['P_vehicle_m']}", f"proceed_to_R = {v['proceed_to_R']}"]
    (CC.RESULTS_DIR / "m3_verdict.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
