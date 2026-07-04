"""R2: is the centering offset a CROWN RESPONSE? Within-drive offset~roll association
(speed-controlled, FPR-calibrated) + direction-paired same-road descriptive check.

RUN (after r2_calibration.py PASSES):
  .venv311/bin/python stock_lateral_toolkit/centering/r2_crown.py
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

import shared  # stock_lateral_toolkit/shared.py (within_drive_spearman etc.)
from stock_lateral_toolkit.centering import config as CC

DRIVES = ["stock_00", "stock_01", "stock_02", "stock_03", "stock_hiram_04", "stock_hiram_05"]
FS = 20.0  # extract_drive caches are modelV2-cadence

# R2's CALIBRATED drive set (2026-07-04). The FPR calibration (M0 §7) PASSES on this set
# (FPR 0.0, power 0.95@0.03m / 1.0@0.06m). Extending with stock_01 fails calibration
# (FPR 0.283 at the 0.07 gate; its series structure defeats the cross-drive pairing null
# even after the thin-drive floor), and per the pre-registered rule an estimator that
# fails calibration on a shape may not emit p-values on that shape — so the extension is
# DISCARDED, not re-tuned. Any future drive-set change must re-pass r2_calibration first.
# (DRIVES above stays the full set — R3's descriptive breakdown uses it, no p-values.)
R2_DRIVES = ["stock_02", "stock_hiram_04", "stock_hiram_05"]


def load_drive(name: str):
    z = np.load(CC.TOOLKIT_CACHE / f"{name}.npz", allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    d = {c: z["data"][:, i] for i, c in enumerate(cols)}
    return d


# Estimator-shape floor (revised 2026-07-04, recalibrated per M0 §7): a drive contributing
# fewer windows than this is dropped from the table. The cross-drive pairing null truncates
# each pair to min(len) rows, so a 1-window drive collapses partner groups to 1 row and
# destabilizes the permutation test (observed: FPR 0.226 with stock_00 at 1 window vs the
# 0.07 gate). The FPR gate is the pre-registered contract; the table shape is not.
MIN_WINDOWS_PER_DRIVE = 8


def window_table(drives=R2_DRIVES) -> pd.DataFrame:
    """Per 30 s window: median signed offset / roll / speed on eligible frames
    (engaged, unpressed, moving, finite offset+roll). Drives with fewer than
    MIN_WINDOWS_PER_DRIVE qualifying windows are excluded (see note above)."""
    rows = []
    w = int(CC.R2_WINDOW_S * FS)
    for name in drives:
        try:
            d = load_drive(name)
        except FileNotFoundError:
            print(f"NOTE: cache {name} missing, skipped")
            continue
        elig = ((d["latActive"] > 0.5) & (d["pressed"] < 0.5)
                & (d["vEgo"] >= CC.V_MIN_MPS)
                & np.isfinite(d["offset"]) & np.isfinite(d["roll"]))
        n = len(d["t"])
        for a in range(0, n - w, w):
            m = elig[a:a + w]
            if m.sum() < 0.5 * w:
                continue
            rows.append(dict(
                drive_id=name,
                offset_med=float(np.median(d["offset"][a:a + w][m])),
                roll_med=float(np.median(d["roll"][a:a + w][m])),
                spd_mph=float(np.median(d["vEgo"][a:a + w][m]) * 2.23694),
                lat_med=float(np.nanmedian(d["lat"][a:a + w][m])),
                lon_med=float(np.nanmedian(d["lon"][a:a + w][m])),
            ))
    df = pd.DataFrame(rows)
    if len(df):
        counts = df.groupby("drive_id")["offset_med"].transform("size")
        dropped = sorted(df.loc[counts < MIN_WINDOWS_PER_DRIVE, "drive_id"].unique())
        if dropped:
            print(f"NOTE: dropped thin drives (<{MIN_WINDOWS_PER_DRIVE} windows): {dropped}")
        df = df[counts >= MIN_WINDOWS_PER_DRIVE].reset_index(drop=True)
    return df


def direction_paired_hiram(df_04: dict, df_05: dict) -> dict:
    """Descriptive: per shared ~150 m GPS cell, (offset_04 - offset_05) vs (roll_04 - roll_05).
    Pure translation/model bias predicts offset deltas ~0 regardless of roll deltas;
    a crown response predicts offset deltas tracking roll deltas."""
    def cells(d):
        elig = (d["latActive"] > 0.5) & (d["pressed"] < 0.5) & (d["vEgo"] >= CC.V_MIN_MPS) \
               & np.isfinite(d["offset"]) & np.isfinite(d["roll"]) & np.isfinite(d["lat"])
        cell = (np.round(d["lat"] / 0.0015).astype(np.int64) * 100000
                + np.round(d["lon"] / 0.0015).astype(np.int64))
        out = {}
        for c in np.unique(cell[elig]):
            m = elig & (cell == c)
            if m.sum() >= 40:   # >= 2 s of frames
                out[int(c)] = (float(np.median(d["offset"][m])), float(np.median(d["roll"][m])))
        return out
    a, b = cells(df_04), cells(df_05)
    common = sorted(set(a) & set(b))
    d_off = np.array([a[c][0] - b[c][0] for c in common])
    d_roll = np.array([a[c][1] - b[c][1] for c in common])
    r = float(np.corrcoef(d_off, d_roll)[0, 1]) if len(common) > 5 else float("nan")
    return {"n_cells": len(common),
            "median_abs_offset_delta_m": float(np.median(np.abs(d_off))) if len(common) else float("nan"),
            "corr_offset_delta_vs_roll_delta": r}


def main():
    cal = CC.RESULTS_DIR / "r2" / "calibration.json"
    if not cal.exists():
        raise SystemExit("R2 blocked: run r2_calibration.py first (M0 §7 — no p-values "
                         "from an uncalibrated estimator).")
    cal_res = json.loads(cal.read_text())
    if not cal_res["passed"]:
        raise SystemExit(f"R2 blocked: calibration FAILED (FPR={cal_res['fpr']}); "
                         "fix the estimator before interpreting p-values.")

    df = window_table()
    res = {"n_windows": int(len(df)), "n_drives": int(df["drive_id"].nunique()),
           "calibration_fpr": cal_res["fpr"]}
    r = shared.within_drive_spearman(df, "roll_med", "offset_med", ctrl="spd_mph")
    res["within_drive"] = {k: (float(v) if np.isfinite(v) else None) for k, v in r.items()}
    res["crown_significant"] = bool(np.isfinite(r["p"]) and r["p"] < CC.CROWN_P_MAX
                                    and abs(r["r"]) >= CC.CROWN_R_MIN)
    # crown fraction: robust slope (Theil-Sen light: median of pairwise slopes on demeaned data)
    g = df.dropna(subset=["roll_med", "offset_med"])
    x = g["roll_med"].values - g["roll_med"].values.mean()
    y = g["offset_med"].values - g["offset_med"].values.mean()
    idx = np.random.default_rng(7).choice(len(x), size=(min(4000, len(x) * (len(x) - 1) // 2), 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    slopes = (y[idx[:, 0]] - y[idx[:, 1]]) / (x[idx[:, 0]] - x[idx[:, 1]] + 1e-12)
    res["slope_m_per_rad"] = float(np.median(slopes))
    res["crown_component_m"] = float(np.median(slopes) * np.median(np.abs(g["roll_med"])))

    try:
        res["direction_paired_hiram"] = direction_paired_hiram(
            load_drive("stock_hiram_04"), load_drive("stock_hiram_05"))
    except FileNotFoundError:
        res["direction_paired_hiram"] = None

    out = CC.RESULTS_DIR / "r2" / "r2_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
