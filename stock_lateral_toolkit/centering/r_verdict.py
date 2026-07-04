"""R verdict: MECHANICAL application of the pre-registered decision tree (M0 §4).
No numbers are chosen here — inputs come from m3_verdict.json / r1_results.json /
r2/r2_results.json / r3_results.json; thresholds from config (twins of the spec).

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def classify(proceed_to_r: bool, p_cam: float, dmid: float, dwidth: float, logged: float,
             crown_significant: bool, crown_component: float) -> dict:
    if not proceed_to_r:
        return {"class": "NO_VERDICT_METHOD_DISAGREEMENT", "camera_offset_is_the_lever": False,
                "note": "M3 method gate failed (M0 §3) — diagnose the discrepancy first."}
    p_big = abs(p_cam) >= CC.P_MEANINGFUL_M
    t_big = abs(dmid) >= CC.DMID_MEANINGFUL_M and abs(dwidth) <= CC.DWIDTH_COHERENT_M
    crown_fraction = (abs(crown_component) / abs(logged)) if (crown_significant and logged) else 0.0
    if crown_significant and crown_fraction > CC.CROWN_DOMINANT_FRACTION:
        return {"class": "CROWN_RESPONSE_DOMINANT", "crown_fraction": crown_fraction,
                "camera_offset_is_the_lever": False,
                "note": "S2 branch: crown-aware approach needs a NEW spec (out of this plan)."}
    if p_big and not t_big:
        cls = "TRAINED_PATH_PREFERENCE"
    elif p_big and t_big:
        cls = "MIXED_TRANSLATION_PREFERENCE"
    elif not p_big and t_big:
        cls = "MODEL_FRAME_DEFINITION_ONLY"
    else:
        cls = "NO_DEFICIT_MEASURABLE"
    return {"class": cls,
            "camera_offset_is_the_lever": cls in ("TRAINED_PATH_PREFERENCE",
                                                  "MIXED_TRANSLATION_PREFERENCE"),
            "crown_fraction": crown_fraction,
            "components": {"P_cam_m": p_cam, "T_dmid_m": dmid if t_big else 0.0,
                           "crown_m": crown_component if crown_significant else 0.0,
                           "L_logged_m": logged}}


def main():
    m3 = json.loads((CC.RESULTS_DIR / "m3_verdict.json").read_text())
    r1 = json.loads((CC.RESULTS_DIR / "r1_results.json").read_text())
    r2 = json.loads((CC.RESULTS_DIR / "r2" / "r2_results.json").read_text())
    r3 = json.loads((CC.RESULTS_DIR / "r3_results.json").read_text())
    v = classify(proceed_to_r=bool(m3["proceed_to_R"]),
                 p_cam=float(m3["P_cam_m"]),
                 dmid=float(r1["dmid_median_m"]),
                 dwidth=float(r1["dwidth_median_m"]),
                 logged=float(m3["L_logged_m"]),
                 crown_significant=bool(r2["crown_significant"]),
                 crown_component=float(r2.get("crown_component_m") or 0.0))
    v["inputs"] = {"m3": m3, "r1_dmid": r1["dmid_median_m"], "r2_sig": r2["crown_significant"],
                   "r3_corridor_spread_m": r3["corridor_spread_m"]}
    (CC.RESULTS_DIR / "r_verdict.json").write_text(json.dumps(v, indent=2))
    md = ["# R-phase verdict (mechanical, M0 §4)", "",
          f"**Class: {v['class']}**", "",
          f"- camera_offset_is_the_lever: {v['camera_offset_is_the_lever']}",
          f"- components: {json.dumps(v.get('components', {}))}",
          f"- crown fraction: {v.get('crown_fraction')}",
          f"- R3 corridor spread: {r3['corridor_spread_m']:.3f} m"]
    (CC.RESULTS_DIR / "r_verdict.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
