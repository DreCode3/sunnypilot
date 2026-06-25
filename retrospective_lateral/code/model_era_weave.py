"""Cross-model open-loop weave comparison (no re-inference).

Analysis-only. Groups routes by the driving model recovered in model_labels.py and
compares the weave-band amplitude of each model's OWN output (engaged and, more
tellingly, disengaged / open-loop), speed-matched. Because the weave is intrinsic to
the model's path prediction (see the 2026-06-25 synthesis), the disengaged weave-band
RMS per model estimates how much each model wobbles regardless of the controller --
i.e. whether swapping the model would plausibly reduce the weave, and to which model.

Reuses retrospective_lateral.code.model_vs_loop.route_state_profile (parallel worker)
and retrospective_lateral.code.model_labels. Writes only to results/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.model_labels import load_labels
from retrospective_lateral.code.model_vs_loop import ML_MODEL_NATIVE, route_state_profile

# Models with enough labeled routes to compare (internal names from ModelManager_ActiveBundle
# plus hand-labels). Keys map an internal/hand name to a display group.
MODEL_GROUPS = {
    "C210M": "CD210", "CD210": "CD210",
    "NM": "Nevada", "OPM7": "OPM7", "OPM": "OPM7",
}


def aggregate_by_model(per_route, labels: dict, signals=ML_MODEL_NATIVE) -> list:
    """Speed-matched pooled engaged/disengaged weave-band RMS per model.

    per_route: {route_id: weave_by_state dict {(signal, speed_bin, state): rms}}.
    labels: {route_id: model_name}. Returns a list of per-(model, signal) rows with
    engaged/disengaged medians (speed-matched over bins present for both states) and
    the disengaged/engaged ratio, plus route counts.
    """
    # model -> signal -> bin -> state -> [rms across routes]
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    routes_per_model = defaultdict(set)
    for route_id, prof in per_route.items():
        model = MODEL_GROUPS.get(labels.get(route_id, ""))
        if model is None or not prof:
            continue
        routes_per_model[model].add(route_id)
        for (sig, lo, state), rr in prof.items():
            if sig in signals:
                acc[model][sig][lo][state].append(rr)

    rows = []
    for model in sorted(acc):
        for sig in signals:
            eng_meds, dis_meds = [], []
            for lo in sorted(acc[model][sig]):
                e = acc[model][sig][lo].get("eng")
                d = acc[model][sig][lo].get("dis")
                if e and d:
                    eng_meds.append(float(np.median(e)))
                    dis_meds.append(float(np.median(d)))
            if not eng_meds:
                continue
            eng = float(np.median(eng_meds))
            dis = float(np.median(dis_meds))
            rows.append({
                "model": model, "signal": sig, "routes": len(routes_per_model[model]),
                "matched_speed_bins": len(eng_meds),
                "engaged_weave_rms_1e4": eng * 1e4,
                "disengaged_weave_rms_1e4": dis * 1e4,
                "dis_over_eng": dis / eng if eng > 0 else float("nan"),
            })
    return rows


def build_model_era_weave(cache_root: Path = C.DEFAULT_CACHE_ROOT,
                          report_root: Path = C.DEFAULT_REPORT_ROOT,
                          workers: int | None = None) -> dict:
    import pandas as pd

    report_root = Path(report_root)
    labels = load_labels(report_root)
    if not labels:
        return {"error": "no route_model_labels.csv; run model_labels first"}

    paths = [str(Path(cache_root) / f"{rid}.npz") for rid in labels
             if MODEL_GROUPS.get(labels[rid]) is not None
             and (Path(cache_root) / f"{rid}.npz").exists()]
    per_route = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for route_id, prof in ex.map(route_state_profile, paths):
            if prof:
                per_route[route_id] = prof

    rows = aggregate_by_model(per_route, labels)
    df = pd.DataFrame(rows)
    df.to_csv(report_root / "model_era_weave.csv", index=False)

    # headline: per-model disengaged (open-loop, intrinsic) weave, averaged over native signals
    summary = {}
    if len(df):
        for model, sub in df.groupby("model"):
            summary[model] = {
                "routes": int(sub["routes"].max()),
                "disengaged_weave_rms_1e4": float(sub["disengaged_weave_rms_1e4"].median()),
                "engaged_weave_rms_1e4": float(sub["engaged_weave_rms_1e4"].median()),
            }
    return {"models_compared": list(summary), "open_loop_weave_by_model": summary}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-model open-loop weave comparison (no re-inference)")
    parser.add_argument("--cache-root", type=Path, default=C.DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", type=Path, default=C.DEFAULT_REPORT_ROOT)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    result = build_model_era_weave(args.cache_root, args.report_root, workers=args.workers)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
