from __future__ import annotations
import bisect
from dataclasses import dataclass
import numpy as np
from model_replay_sim import config as C
from model_replay_sim.alignment import build_frame_timeline
from model_replay_sim.context import route_context
from retrospective_lateral.code.signal_utils import filter_continuous

ROUTE_CANDIDATES = {
    "CD210":  [("route_b5", "local-CD210-active-bundle"), ("route_b8", "preferred-CD210"),
               ("route_c1", "preferred-CD210"), ("route_c2", "preferred-CD210"), ("route_c3", "preferred-CD210")],
    "Nevada": [("route_c4", "Nevada-active-bundle"), ("route_c5", "Nevada-active-bundle"),
               ("route_c6", "Nevada-active-bundle"), ("route_c7", "Nevada-active-bundle")],
    "OPM7":   [("route_7f", "hand-labeled-OPM7")],
}
_INTERNAL = {"CD210": {"C210M", "CD210"}, "Nevada": {"NM", "Nevada"}, "OPM7": {"OPM7"}}

@dataclass(frozen=True)
class AnchorCandidate:
    bundle: str
    route_id: str
    source: str
    local_fcamera_segments: int
    npz_exists: bool
    active_bundle_match: bool | None
    eligible_aligned_windows_20s: int
    needs_frame_pull: bool

def _replay_scene_eligible_mask(z) -> np.ndarray:
    n = len(z["t"]); g = lambda k, d: np.asarray(z.get(k, np.full(n, d)), float)
    v = g("v_ego", np.nan); mph = v * 2.2369362920544
    yr = g("yaw_rate", np.nan); pc = np.full(n, np.nan); ok = np.isfinite(yr) & np.isfinite(v) & (v > 3); pc[ok] = yr[ok] / v[ok]
    road = filter_continuous(pc, C.FS_HZ, lowpass_hz=C.ROAD_LP_HZ)
    return ((mph >= 10) & (mph <= 85) & np.isfinite(road) & (np.abs(road) < C.GENTLE_CURV_ABS_MAX_1PM)
            & ~(g("blinker", 0) > 0.5) & ~(g("lane_change_state", 0) > 0.5))

def _eligible_aligned_window_count(route_id: str, min_seconds: float = 20.0) -> int:
    npz = C.CACHE_ROOT / f"{route_id}.npz"
    if not npz.exists():
        return 0
    z = dict(np.load(npz)); mono = np.asarray(z["mono_time"], float)
    elig = _replay_scene_eligible_mask(z)
    try:
        tl = build_frame_timeline(route_id)
    except Exception:
        return 0
    if not tl:
        return 0
    eofs = sorted(r.timestamp_eof_s for r in tl)
    def covered(t: float) -> bool:
        i = bisect.bisect_left(eofs, t)
        return any(0 <= j < len(eofs) and abs(eofs[j] - t) <= C.MAX_FRAME_DELTA_S for j in (i - 1, i))
    good = elig & np.array([covered(float(t)) for t in mono], dtype=bool)
    need = int(min_seconds * C.FS_HZ); cnt = 0; i = 0; n = len(good)
    while i < n:
        if good[i]:
            j = i
            while j < n and good[j]:
                j += 1
            if (j - i) >= need:
                cnt += (j - i) // need
            i = j
        else:
            i += 1
    return cnt

def _active_match(bundle: str, route_id: str) -> bool | None:
    try:
        name = route_context(route_id).active_bundle_internal_name
    except Exception:
        return None
    return None if name is None else (name in _INTERNAL[bundle])

def anchor_candidates(bundle: str) -> list[AnchorCandidate]:
    out = []
    for route_id, source in ROUTE_CANDIDATES[bundle]:
        fcam = len(list((C.LOG_ROOT / route_id).glob("000000*--*--*/fcamera.hevc")))
        npz = (C.CACHE_ROOT / f"{route_id}.npz").exists()
        out.append(AnchorCandidate(bundle, route_id, source, fcam, npz,
                                    _active_match(bundle, route_id),
                                    _eligible_aligned_window_count(route_id) if fcam else 0,
                                    needs_frame_pull=(fcam == 0)))
    return out

def require_same_model_anchor_asset(bundle: str) -> AnchorCandidate:
    cands = anchor_candidates(bundle)
    if all(c.local_fcamera_segments == 0 for c in cands):
        raise AssertionError(f"{bundle}: missing local fcamera (needs frame pull)")
    if all(c.active_bundle_match is not True for c in cands):
        raise AssertionError(f"{bundle}: no active-bundle provenance for a strict same-model anchor")
    for c in cands:
        if c.active_bundle_match is True and c.local_fcamera_segments > 0 and c.eligible_aligned_windows_20s >= 1:
            return c
    raise AssertionError(f"{bundle}: no candidate with provenance + frames + an eligible aligned window")

def require_sanity_asset(bundle: str) -> AnchorCandidate:
    for c in anchor_candidates(bundle):
        if c.local_fcamera_segments > 0 and c.eligible_aligned_windows_20s >= 1:
            return c
    raise AssertionError(f"{bundle}: no sanity asset with frames + an eligible aligned window")
