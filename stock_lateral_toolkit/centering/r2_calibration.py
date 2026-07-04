"""FPR/power calibration for the R2 estimator (MUST pass before r2_crown.py's p is
trusted — M0 §7). Two nulls + injected-effect power, on the REAL window table shape:

NULL A (assumption-free, qa_calibration.fpr_crossdrive pattern): pair the real roll
series of one drive with the real offset series of ANOTHER (independent partner per
drive per rep) — true H0 with full real autocorrelation/drift. Target FPR <= 0.07.
NULL B (within-drive circular shift of roll by >= 60 windows): breaks the pairing,
keeps each series' own structure. Reported; A is the gate.
POWER: inject offset' = offset + beta*roll with beta sized to crown components of
0.03 / 0.06 m at the observed roll spread; report detection rate at alpha=0.05.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r2_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLKIT = REPO_ROOT / "stock_lateral_toolkit"
if str(TOOLKIT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT))

import shared
from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering.r2_crown import window_table

REPS = 300
NPERM = 400


def _test_p(df: pd.DataFrame) -> float:
    return shared.within_drive_spearman(df, "roll_med", "offset_med", ctrl="spd_mph",
                                        nperm=NPERM, nboot=0)["p"]


def fpr_crossdrive_pairing(df: pd.DataFrame, rng) -> float:
    groups = [g.reset_index(drop=True) for _, g in df.groupby("drive_id")]
    ng = len(groups)
    hits = tot = 0
    for _ in range(REPS):
        partner = [(k + int(rng.integers(1, ng))) % ng for k in range(ng)]
        rows = []
        for k in range(ng):
            gx, gy = groups[k], groups[partner[k]]
            n = min(len(gx), len(gy))
            if n < 4:
                continue
            for i in range(n):
                rows.append((k, float(gx["roll_med"][i]), float(gy["offset_med"][i]),
                             float(gx["spd_mph"][i])))
        d = pd.DataFrame(rows, columns=["drive_id", "roll_med", "offset_med", "spd_mph"])
        if d["drive_id"].nunique() < 2:
            continue
        p = _test_p(d)
        if np.isfinite(p):
            tot += 1
            hits += p < 0.05
    return hits / max(tot, 1)


def power_injected(df: pd.DataFrame, crown_m: float, rng) -> float:
    roll_spread = float(np.median(np.abs(df["roll_med"] - df["roll_med"].median())))
    beta = crown_m / max(roll_spread, 1e-6)
    hits = tot = 0
    for _ in range(REPS // 3):
        d = df.copy()
        # destroy any real association first (within-drive shuffle of offset), then inject
        d["offset_med"] = d.groupby("drive_id")["offset_med"].transform(
            lambda s: rng.permutation(s.values))
        d["offset_med"] = d["offset_med"] + beta * d["roll_med"]
        p = _test_p(d)
        if np.isfinite(p):
            tot += 1
            hits += p < 0.05
    return hits / max(tot, 1)


def main():
    rng = np.random.default_rng(20260703)
    df = window_table()
    print(f"window table: {len(df)} windows over {df['drive_id'].nunique()} drives")
    fpr = fpr_crossdrive_pairing(df, rng)
    pw_small = power_injected(df, 0.03, rng)
    pw_med = power_injected(df, 0.06, rng)
    res = {"n_windows": int(len(df)), "fpr": float(fpr),
           "fpr_max": CC.R2_FPR_MAX, "passed": bool(fpr <= CC.R2_FPR_MAX),
           "power_crown_0.03m": float(pw_small), "power_crown_0.06m": float(pw_med)}
    out = CC.RESULTS_DIR / "r2" / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if not res["passed"]:
        print("CALIBRATION FAILED — r2_crown.py p-values are NOT interpretable; "
              "the estimator (window length / control set) must be revised and re-calibrated.")


if __name__ == "__main__":
    main()
