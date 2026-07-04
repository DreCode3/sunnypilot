"""M1d: enforce the pre-registered annotation trust rule, then turn accepted proposals
into per-frame physical offsets (canonical sign: + = vehicle LEFT of lane center) with
the declared per-frame sigma. Writes the route's m1_results.json (+ per-frame CSV) in
results/m1/ for the primary route, results/m1_<route>/ otherwise.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/m1_offsets.py [--route <name>]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering import ground_plane as G


def trust_check(reviewed_rows) -> tuple[bool, float]:
    """M0 §2: accepted = verdict 'accept' OR ('correct' with |corrected-u| <= PX_CORRECT_TOL)."""
    n = len(reviewed_rows)
    if n == 0:
        return False, 0.0
    good = 0
    for r in reviewed_rows:
        v = (r.get("verdict") or "").strip().lower()
        if v == "accept":
            good += 1
        elif v == "correct" and r.get("corrected_u_px", "") != "":
            if abs(float(r["corrected_u_px"]) - float(r["u_px"])) <= CC.PX_CORRECT_TOL:
                good += 1
    frac = good / n
    return frac >= CC.REVIEW_ACCEPT_MIN, frac


def _accepted_y(r) -> float | None:
    """Effective road-frame y for one proposal row after review semantics."""
    v = (r.get("verdict") or "").strip().lower()
    if v == "reject":
        return None
    if v == "correct" and r.get("corrected_u_px", "") != "":
        # re-project the corrected pixel on the same row (v_px changes negligibly)
        rpy = [float(r["cal_roll"]), float(r["cal_pitch"]), float(r["cal_yaw"])]
        Hm = G.road_homography(rpy, float(r["height"]), G.fcam_intrinsics())
        _, y = G.road_from_pixel(Hm, float(r["corrected_u_px"]), float(r["v_px"]))
        return y
    if r["auto_ok"] in (True, "True", "true", "1"):
        return float(r["y_road"])
    return None


def frame_offset_cam(rows) -> float | None:
    """Canonical camera-relative offset for ONE frame's proposal rows.
    Needs >= 2 accepted distances per side. y_cal = -y_road; offset = midpoint y_cal."""
    per_side: dict[str, list[float]] = {"left": [], "right": []}
    for r in rows:
        y = _accepted_y(r)
        if y is not None:
            per_side[r["side"]].append(float(y))
    if len(per_side["left"]) < 2 or len(per_side["right"]) < 2:
        return None
    y_left_cal = G.y_cal_from_y_road(float(np.median(per_side["left"])))
    y_right_cal = G.y_cal_from_y_road(float(np.median(per_side["right"])))
    return float((y_left_cal + y_right_cal) / 2.0)


def main(route: str | None = None):
    if route is None:
        route = sys.argv[sys.argv.index("--route") + 1] if "--route" in sys.argv else CC.ROUTE
    m1 = CC.m1_dir(route)
    props = list(csv.DictReader(open(m1 / "proposals.csv")))
    manifest = {int(r["frame_idx"]): r for r in csv.DictReader(open(m1 / "frames_manifest.csv"))}
    from model_replay_sim.context import route_context
    height = float(route_context(route).height)

    # merge review verdicts (by frame_idx+x_m+side) and manifest calib into proposal rows
    review = {}
    rev_path = m1 / "review_subset.csv"
    if rev_path.exists():
        for r in csv.DictReader(open(rev_path)):
            review[(int(r["frame_idx"]), float(r["x_m"]), r["side"])] = r
    reviewed_rows = []
    for r in props:
        key = (int(r["frame_idx"]), float(r["x_m"]), r["side"])
        if key in review:
            r["verdict"] = review[key]["verdict"]
            r["corrected_u_px"] = review[key]["corrected_u_px"]
            reviewed_rows.append(r)
        man = manifest[int(r["frame_idx"])]
        r["cal_roll"], r["cal_pitch"], r["cal_yaw"] = man["cal_roll"], man["cal_pitch"], man["cal_yaw"]
        r["height"] = height

    filled = [r for r in reviewed_rows if (r.get("verdict") or "").strip()]
    if len(filled) < len(reviewed_rows) or not reviewed_rows:
        raise SystemExit(f"USER GATE UNMET: review_subset.csv has {len(reviewed_rows) - len(filled)} "
                         f"unfilled verdicts (of {len(reviewed_rows)}). Fill it, then re-run.")
    ok, frac = trust_check(filled)
    print(f"trust rule: reviewed acceptance = {frac:.0%} (threshold {CC.REVIEW_ACCEPT_MIN:.0%})")
    full_manual = all((r.get("verdict") or "").strip() for r in props)
    if not ok:
        if full_manual:
            # M0 §2's prescribed fallback: on a subset-trust failure, a FULL manual pass
            # (every proposals.csv row verdicted) supersedes automation trust entirely —
            # the analysis below consumes only the verdicts, never bare auto_ok rows.
            print(f"TRUST RULE FAILED on the subset ({frac:.0%} < {CC.REVIEW_ACCEPT_MIN:.0%}) — "
                  f"proceeding on the FULL MANUAL PASS ({len(props)}/{len(props)} rows verdicted) "
                  "per the M0 §2 fallback.")
        else:
            raise SystemExit("TRUST RULE FAILED (M0 §2): automated annotations are NOT trusted. "
                             "Full manual annotation pass required — fill verdict+corrected_u_px "
                             "for EVERY row of proposals.csv and re-run.")

    by_frame: dict[int, list] = {}
    for r in props:
        by_frame.setdefault(int(r["frame_idx"]), []).append(r)
    sigma = G.sigma_frame_m()
    out_rows, offsets = [], []
    for fi, rows in sorted(by_frame.items()):
        off = frame_offset_cam(rows)
        if off is None:
            continue
        man = manifest[fi]
        out_rows.append(dict(frame_idx=fi, mono_time=float(man["mono_time"]),
                             offset_cam_m=off, sigma_m=sigma,
                             v_ego=float(man["v_ego"]), speed_bin=int(man["speed_bin"]),
                             heading_bin=int(man["heading_bin"]),
                             in_m2_window=man["in_m2_window"] == "True",
                             logged_center_y0=float(man["logged_center_y0"])))
        offsets.append(off)

    offsets = np.array(offsets)
    logged = np.array([r["logged_center_y0"] for r in out_rows])
    res = {
        "n_frames": int(len(offsets)),
        "median_offset_cam_m": float(np.median(offsets)),
        "mad_offset_cam_m": float(np.median(np.abs(offsets - np.median(offsets)))),
        "sigma_frame_m": sigma,
        "sem_random_m": float(sigma / max(np.sqrt(len(offsets)), 1)),
        "systematic_note": "lever-arm/mean-roll/mean-yaw systematics ~±0.04 m do not average out (M0 §2)",
        "cam_offset_from_centerline_m": CC.CAM_OFFSET_FROM_CENTERLINE_M,
        "cam_offset_measured": bool(CC.CAM_OFFSET_MEASURED),
        "median_offset_vehicle_m": (float(np.median(offsets)) + CC.CAM_OFFSET_FROM_CENTERLINE_M)
                                   if CC.CAM_OFFSET_MEASURED else None,
        "overlap_logged": {
            "n": int(len(logged)),
            "median_delta_logged_minus_video_m": float(np.median(logged - offsets)),
            "mad_delta_m": float(np.median(np.abs((logged - offsets) - np.median(logged - offsets)))),
        },
        "per_speed_bin": {str(sb): float(np.median([r["offset_cam_m"] for r in out_rows
                                                    if r["speed_bin"] == sb]))
                          for sb in sorted({r["speed_bin"] for r in out_rows})},
        "review_acceptance": frac,
        "review_mode": "full_manual" if full_manual else "subset_trust",
    }
    with open(m1 / "per_frame_offsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    (m1 / "m1_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if res["n_frames"] < CC.N_FRAMES_MIN:
        print(f"WARNING: n_frames < {CC.N_FRAMES_MIN} — M0 §2 floor unmet; "
              f"second-route pull required before M3 relies on M1.")
    if not CC.CAM_OFFSET_MEASURED:
        print("WARNING: lever arm not measured (Task 9 step 1) — vehicle-frame P unavailable; "
              "M3/R1 proceed camera-relative, final P blocked.")


if __name__ == "__main__":
    main()
