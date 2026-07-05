"""A-vs-C discrimination: is the weave an intrinsic model/path artifact (A) or a
closed-loop limit-cycle (C)?

Analysis-only. Reads existing route NPZ caches; writes only to
retrospective_lateral/results/. Imports nothing from the vehicle-control stack.

Primary discriminator (open-loop test): compare the weave-band amplitude of the
driving model's OWN output (model path, orientation-rate curvature, planner desired
curvature, and model-minus-lane deviation) when lateral control is ENGAGED
(latActive=1, closed loop) vs DISENGAGED (latActive=0, model running but not
actuating, open loop), on straight/gentle road, speed-matched.

  - A closed-loop limit-cycle (C) requires the loop closed, so opening the loop
    (disengaging) must KILL it: disengaged << engaged (ratio -> 0).
  - An intrinsic model/path artifact (A) is present in the model's output whether or
    not openpilot is steering: disengaged >= engaged (ratio >= ~1).

The per-route weave-band computation is parallelized across CPU cores.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.discrimination import offset_to_curvature, path_curvature_from_rate
from retrospective_lateral.code.signal_utils import filter_continuous, rms_masked

ML_BAND = C.DEFAULT_WEAVE_BAND_HZ
ML_SPEED_LO_MPH, ML_SPEED_HI_MPH, ML_SPEED_STEP_MPH = 10.0, 70.0, 5.0
ML_MIN_BIN_SAMPLES = 60
# Verdict thresholds on the disengaged/engaged weave-band RMS ratio (model-native signals):
ML_A_RATIO_MIN = 0.7   # ratio >= this => model weaves open-loop too => A (C refuted)
ML_C_RATIO_MAX = 0.3   # ratio <= this => weave collapses open-loop => C
# Signals whose disengaged value reflects the MODEL's own path prediction (not the
# human's instantaneous yaw); the A/C verdict rests on these.
ML_MODEL_NATIVE = ("model_y20", "desired_curvature", "model_minus_lane")


def _channels(arrays):
    keys = arrays.files if hasattr(arrays, "files") else arrays
    v = np.asarray(arrays["v_ego"], dtype=float)
    orient = (path_curvature_from_rate(arrays["orientation_rate_z0"], v)
              if "orientation_rate_z0" in keys else np.full_like(v, np.nan))
    return {
        "model_y20": offset_to_curvature(arrays["model_y20"], C.DISCRIM_LOOKAHEAD_M),
        "orientation_rate_curv": orient,
        "desired_curvature": np.asarray(arrays["desired_curvature"], dtype=float),
        "model_minus_lane": offset_to_curvature(
            np.asarray(arrays["model_y20"], dtype=float) - np.asarray(arrays["lane_center_y20"], dtype=float),
            C.DISCRIM_LOOKAHEAD_M),
    }


def weave_by_state(arrays, band=ML_BAND) -> dict:
    """Per-(signal, speed_bin, state) weave-band RMS for one route on straight road.

    state is "eng" (latActive>0.5) or "dis" (latActive<=0.5). Returns
    {(signal, speed_bin_lo, state): rms}. Speed bins let the caller speed-match.
    """
    keys = arrays.files if hasattr(arrays, "files") else arrays
    if "model_y20" not in keys or "v_ego" not in keys:
        return {}
    v = np.asarray(arrays["v_ego"], dtype=float)
    la = np.asarray(arrays["lat_active"], dtype=float) if "lat_active" in keys else np.ones_like(v)
    mph = v * C.MPS_TO_MPH
    road = filter_continuous(path_curvature_from_rate(arrays["yaw_rate"], v), C.FS_HZ, lowpass_hz=C.ROAD_LP_HZ)
    straight = np.isfinite(road) & (np.abs(road) < C.ROAD_CURV_ABS_MAX_1PM)
    band_sig = {k: filter_continuous(x, C.FS_HZ, band=band) for k, x in _channels(arrays).items()}
    out = {}
    lo = ML_SPEED_LO_MPH
    while lo < ML_SPEED_HI_MPH:
        sb = (mph >= lo) & (mph < lo + ML_SPEED_STEP_MPH) & straight
        for state, sel in (("eng", la > 0.5), ("dis", la <= 0.5)):
            mask = sb & sel
            if mask.sum() < ML_MIN_BIN_SAMPLES:
                continue
            for k, bs in band_sig.items():
                rr = rms_masked(bs, mask)
                if np.isfinite(rr):
                    out[(k, lo, state)] = rr
        lo += ML_SPEED_STEP_MPH
    return out


def route_state_profile(npz_path):
    """ProcessPoolExecutor worker: load a cache and return (route_id, weave_by_state)."""
    route_id = Path(npz_path).stem
    try:
        with np.load(npz_path) as data:
            arrays = {k: data[k] for k in data.files}
    except Exception:
        return route_id, {}
    return route_id, weave_by_state(arrays)


def classify_a_vs_c(ratio_by_signal: dict) -> tuple:
    """Verdict from disengaged/engaged weave-band ratios on model-native signals.

    ratio >= ML_A_RATIO_MIN  -> model_artifact_A (weave present open-loop; C refuted)
    ratio <= ML_C_RATIO_MAX  -> loop_limit_cycle_C (weave collapses open-loop)
    else -> ambiguous. Decision uses the median ratio over ML_MODEL_NATIVE signals.
    """
    vals = [ratio_by_signal[s] for s in ML_MODEL_NATIVE
            if s in ratio_by_signal and np.isfinite(ratio_by_signal[s])]
    if not vals:
        return ("insufficient_evidence", math.nan)
    med = float(np.median(vals))
    if med >= ML_A_RATIO_MIN:
        return ("model_artifact_A", med)
    if med <= ML_C_RATIO_MAX:
        return ("loop_limit_cycle_C", med)
    return ("ambiguous", med)


def build_model_vs_loop(cache_root: Path = C.DEFAULT_CACHE_ROOT,
                        report_root: Path = C.DEFAULT_REPORT_ROOT,
                        pi_set: str | None = "golden",
                        workers: int | None = None) -> dict:
    """Speed-matched engaged-vs-disengaged weave-band comparison over routes,
    optionally restricted to a pi_set (default golden = current config). Writes a
    per-signal CSV and returns the A/C verdict.
    """
    import pandas as pd

    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(Path(cache_root) / "route_*.npz")))
    if pi_set is not None:
        keep = []
        for p in paths:
            sidecar = Path(str(p)[:-4] + ".json")
            try:
                if json.loads(sidecar.read_text()).get("pi_set") == pi_set:
                    keep.append(p)
            except Exception:
                continue
        paths = keep

    per_route = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for route_id, prof in ex.map(route_state_profile, paths):
            if prof:
                per_route[route_id] = prof

    # Speed-matched pooling: for each (signal, speed_bin) take the median RMS across
    # routes for eng and dis, then keep only bins where BOTH states exist.
    bins = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # signal->bin->state->[rms]
    for prof in per_route.values():
        for (sig, lo, state), rr in prof.items():
            bins[sig][lo][state].append(rr)

    rows, ratio_by_signal = [], {}
    for sig in sorted(bins):
        eng_meds, dis_meds = [], []
        for lo in sorted(bins[sig]):
            e, d = bins[sig][lo].get("eng"), bins[sig][lo].get("dis")
            if e and d:
                eng_meds.append(float(np.median(e)))
                dis_meds.append(float(np.median(d)))
        if not eng_meds:
            continue
        eng = float(np.median(eng_meds))
        dis = float(np.median(dis_meds))
        ratio = dis / eng if eng > 0 else math.nan
        ratio_by_signal[sig] = ratio
        rows.append({"signal": sig, "matched_speed_bins": len(eng_meds),
                     "engaged_weave_rms_1e4": eng * 1e4, "disengaged_weave_rms_1e4": dis * 1e4,
                     "dis_over_eng": ratio, "model_native": sig in ML_MODEL_NATIVE})

    df = pd.DataFrame(rows)
    df.to_csv(report_root / "model_vs_loop_signals.csv", index=False)
    label, median_ratio = classify_a_vs_c(ratio_by_signal)
    return {"pi_set": pi_set, "routes": len(per_route), "verdict": label,
            "median_native_dis_over_eng": median_ratio}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="A-vs-C weave discrimination (open-loop test)")
    parser.add_argument("--cache-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", type=Path, default=C.DEFAULT_REPORT_ROOT)
    parser.add_argument("--pi-set", default="golden", help="restrict to a pi_set, or 'any'")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    pi_set = None if args.pi_set == "any" else args.pi_set
    result = build_model_vs_loop(args.cache_root, args.report_root, pi_set=pi_set, workers=args.workers)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
