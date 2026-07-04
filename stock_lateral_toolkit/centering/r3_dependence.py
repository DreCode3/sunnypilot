"""R3: the deficit's SHAPE. Signed logged offset broken down per drive x corridor
(GPS-cell cluster) x speed bin x heading direction across ALL stock caches.
Uniform offset -> vehicle/model-global cause; corridor-dependent (esp. tracking roll)
-> environmental (crown); speed-dependent -> dynamic.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r3_dependence.py
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

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering.r2_crown import load_drive, DRIVES

CORRIDOR_CELL_DEG = 0.01   # ~1 km blocks = "corridor" granularity


def breakdown() -> pd.DataFrame:
    rows = []
    for name in DRIVES:
        try:
            d = load_drive(name)
        except FileNotFoundError:
            continue
        elig = ((d["latActive"] > 0.5) & (d["pressed"] < 0.5)
                & (d["vEgo"] >= CC.V_MIN_MPS) & (d["innerProb"] > CC.LANE_PROB_MIN)
                & np.isfinite(d["offset"]) & np.isfinite(d["lat"]))
        # heading from GPS (coarse): quadrant of travel direction
        la, lo = d["lat"], d["lon"]
        head = np.degrees(np.arctan2(np.gradient(la) * 110540.0,
                                     np.gradient(lo) * 111320.0 * np.cos(np.radians(np.nanmean(la))))) % 360.0
        corridor = (np.round(la / CORRIDOR_CELL_DEG).astype(np.int64) * 100000
                    + np.round(lo / CORRIDOR_CELL_DEG).astype(np.int64))
        for i in np.flatnonzero(elig):
            sb = -1
            for k, (vlo, vhi) in enumerate(CC.SPEED_BINS_MPS):
                if vlo <= d["vEgo"][i] < vhi:
                    sb = k
            if sb < 0:
                continue
            rows.append((name, int(corridor[i]), sb, int(head[i] // 90.0) % 4,
                         float(d["offset"][i]), float(d["roll"][i])))
    return pd.DataFrame(rows, columns=["drive", "corridor", "speed_bin", "heading_q",
                                       "offset", "roll"])


def main():
    df = breakdown()
    g = (df.groupby(["corridor", "speed_bin", "heading_q"])
           .agg(offset_med=("offset", "median"), roll_med=("roll", "median"),
                n=("offset", "size"), drives=("drive", "nunique"))
           .reset_index())
    g = g[g["n"] >= 200]          # >= 10 s of frames per cell
    out_csv = CC.RESULTS_DIR / "r3_table.csv"
    g.to_csv(out_csv, index=False)

    per_corr = g.groupby("corridor")["offset_med"].median()
    per_speed = g.groupby("speed_bin")["offset_med"].median()
    per_head = g.groupby("heading_q")["offset_med"].median()
    res = {
        "n_cells": int(len(g)),
        "global_median_m": float(df["offset"].median()),
        "corridor_spread_m": float(per_corr.max() - per_corr.min()) if len(per_corr) > 1 else 0.0,
        "per_speed_bin_m": {str(k): float(v) for k, v in per_speed.items()},
        "per_heading_quadrant_m": {str(k): float(v) for k, v in per_head.items()},
        "corridor_dependent": bool(len(per_corr) > 1
                                   and (per_corr.max() - per_corr.min()) > CC.CORRIDOR_SPREAD_M),
        "corridor_offset_vs_roll_corr": float(np.corrcoef(
            g.groupby("corridor")["offset_med"].median(),
            g.groupby("corridor")["roll_med"].median())[0, 1]) if len(per_corr) > 3 else None,
    }
    (CC.RESULTS_DIR / "r3_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"table -> {out_csv}")


if __name__ == "__main__":
    main()
