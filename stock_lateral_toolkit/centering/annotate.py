"""M1 semi-automated lane-marking annotation.

Proposals are IMAGE-EVIDENCE ONLY (no model outputs touch this file). Per frame and
per eval distance (8/12/16 m): sample luma along a metric lateral grid on the flat-road
plane, top-hat matched filter at the paint width, strongest peak per side, sub-cell
parabolic refinement, back-project to road-frame y (+LEFT).

Outputs (results/m1/):
  proposals.csv     one row per (frame, distance, side) w/ y_road_m, pixel, contrast, auto_ok
  overlays/frame_XXX.png   luma + green(ok)/red(low-contrast) markers + 0.5 m ruler
  review_subset.csv  every REVIEW_EVERY_N-th frame's rows, for the USER to verdict-fill

USER GATE (M0 §2): fill `verdict` (accept|reject|correct) and `corrected_u_px` (when
verdict=correct) in review_subset.csv. m1_offsets.py enforces >=85% acceptance.

RUN:
  .venv311/bin/python stock_lateral_toolkit/centering/annotate.py [--route <name>]            # propose + overlays
  .venv311/bin/python stock_lateral_toolkit/centering/annotate.py --review [--route <name>]   # write review_subset.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering import ground_plane as G

CAM_W, CAM_H = 1344, 760
GRID_STEP_M = 0.02


def luma_plane(nv12_flat, w: int = CAM_W, h: int = CAM_H) -> np.ndarray:
    return np.asarray(nv12_flat, dtype=np.uint8).ravel()[: w * h].reshape(h, w).astype(float)


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    u = np.clip(np.asarray(u, float), 0, w - 1.001)
    v = np.clip(np.asarray(v, float), 0, h - 1.001)
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    du = u - u0; dv = v - v0
    return ((img[v0, u0] * (1 - du) + img[v0, u0 + 1] * du) * (1 - dv)
            + (img[v0 + 1, u0] * (1 - du) + img[v0 + 1, u0 + 1] * du) * dv)


def metric_profile(img, H, x_m: float, y_grid: np.ndarray):
    uv = np.array([G.pixel_from_road(H, x_m, float(y)) for y in y_grid])
    return bilinear(img, uv[:, 0], uv[:, 1]), uv[:, 0], uv[:, 1]


def tophat_response(profile: np.ndarray, step_m: float, mark_width_m: float) -> np.ndarray:
    wm = max(1, int(round(mark_width_m / step_m)))
    inner = uniform_filter1d(profile, wm, mode="nearest")
    outer = uniform_filter1d(profile, 3 * wm, mode="nearest")
    return inner - outer


def propose_line(img: np.ndarray, H: np.ndarray, x_m: float, side: str) -> dict:
    """Strongest paint-like peak on `side` ('left' searches y_road in [+1.0,+3.4],
    'right' in [-3.4,-1.0]) at forward distance x_m. Returns y_road (+LEFT), the pixel,
    a MAD-normalized contrast, and auto_ok = contrast >= MIN_CONTRAST."""
    lo, hi = CC.LANE_SEARCH_BAND_M
    band = np.arange(lo, hi, GRID_STEP_M) if side == "left" else np.arange(-hi, -lo, GRID_STEP_M)
    prof, u, v = metric_profile(img, H, x_m, band)
    resp = tophat_response(prof, GRID_STEP_M, CC.MARK_WIDTH_M)
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med))) + 1e-9
    k = int(np.argmax(resp))
    contrast = float((resp[k] - med) / (1.4826 * mad))
    if 0 < k < len(resp) - 1:
        denom = resp[k - 1] - 2 * resp[k] + resp[k + 1]
        d = float(np.clip((resp[k - 1] - resp[k + 1]) / (2 * denom), -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    else:
        d = 0.0
    y_road = float(band[k] + d * GRID_STEP_M)
    uu, vv = G.pixel_from_road(H, x_m, y_road)
    return dict(x_m=float(x_m), side=side, y_road=y_road, u_px=float(uu), v_px=float(vv),
                contrast=contrast, auto_ok=bool(contrast >= CC.MIN_CONTRAST))


def _read_luma(seg_num: int, seg_id: int, route: str = CC.ROUTE) -> np.ndarray:
    from model_replay_sim.alignment import read_frame
    return luma_plane(np.asarray(read_frame(route, seg_num, seg_id), dtype=np.uint8).ravel())


def _overlay(img: np.ndarray, H: np.ndarray, proposals: list[dict], out_png: Path) -> None:
    from PIL import Image, ImageDraw
    rgb = Image.fromarray(np.stack([img.astype(np.uint8)] * 3, axis=-1))
    dr = ImageDraw.Draw(rgb)
    for x_m in CC.EVAL_DISTANCES_M:               # 0.5 m lateral ruler at each eval distance
        for y in np.arange(-3.5, 3.51, 0.5):
            u, v = G.pixel_from_road(H, x_m, float(y))
            color = (80, 160, 255) if abs(y) > 0.01 else (255, 255, 0)
            dr.line([(u, v - 4), (u, v + 4)], fill=color, width=1)
    for p in proposals:
        c = (0, 255, 0) if p["auto_ok"] else (255, 0, 0)
        u, v = p["u_px"], p["v_px"]
        dr.line([(u - 8, v), (u + 8, v)], fill=c, width=2)
        dr.line([(u, v - 8), (u, v + 8)], fill=c, width=2)
        dr.text((u + 10, v - 14), f"{p['side'][0]}{p['x_m']:.0f}m y={p['y_road']:+.2f}", fill=c)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_png)


def run_proposals(route: str = CC.ROUTE) -> None:
    m1 = CC.m1_dir(route)
    man_path = m1 / "frames_manifest.csv"
    rows = list(csv.DictReader(open(man_path)))
    from model_replay_sim.context import route_context
    height = float(route_context(route).height)
    out_rows = []
    for r in rows:
        rpy = [float(r["cal_roll"]), float(r["cal_pitch"]), float(r["cal_yaw"])]
        Hm = G.road_homography(rpy, height, G.fcam_intrinsics())
        img = _read_luma(int(r["seg_num"]), int(r["seg_id"]), route)
        props = [propose_line(img, Hm, x, s) for x in CC.EVAL_DISTANCES_M for s in ("left", "right")]
        _overlay(img, Hm, props, m1 / "overlays" / f"frame_{int(r['frame_idx']):03d}.png")
        for p in props:
            out_rows.append(dict(frame_idx=int(r["frame_idx"]), mono_time=r["mono_time"],
                                 seg_num=r["seg_num"], seg_id=r["seg_id"], **{k: p[k] for k in
                                 ("x_m", "side", "y_road", "u_px", "v_px", "contrast", "auto_ok")},
                                 verdict="", corrected_u_px=""))
    out = m1 / "proposals.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    n_ok = sum(r["auto_ok"] for r in out_rows)
    print(f"{len(out_rows)} proposals ({n_ok} auto_ok) -> {out}; overlays in {m1 / 'overlays'}/")


def write_review_subset(route: str = CC.ROUTE) -> None:
    m1 = CC.m1_dir(route)
    rows = list(csv.DictReader(open(m1 / "proposals.csv")))
    frames = sorted({int(r["frame_idx"]) for r in rows})
    review_frames = set(frames[:: CC.REVIEW_EVERY_N])
    sub = [r for r in rows if int(r["frame_idx"]) in review_frames]
    out = m1 / "review_subset.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sub[0].keys()))
        w.writeheader(); w.writerows(sub)
    print(f"{len(sub)} rows across {len(review_frames)} frames -> {out}")
    print("USER: open each frame's overlay PNG, fill `verdict` (accept|reject|correct) "
          "and `corrected_u_px` for corrections, save the CSV.")


if __name__ == "__main__":
    route_arg = sys.argv[sys.argv.index("--route") + 1] if "--route" in sys.argv else CC.ROUTE
    if "--review" in sys.argv:
        write_review_subset(route_arg)
    else:
        run_proposals(route_arg)
