"""Task 10: the CD210 same-model fidelity ANCHOR — the run-ready GATE.

Replays CD210 on a route_b5 window where CD210 actually drove and checks that the
replayed model ``action.desiredCurvature`` reproduces the LOGGED
``modelV2.action.desiredCurvature``. If the gate passes, the whole simulator
(Tasks 1-9) is validated end-to-end and run-ready.

GATE (config.py):
  passed = (corr >= ANCHOR_CORR_MIN=0.95) and (ANCHOR_BAND_RATIO[0] <= band_ratio <= ANCHOR_BAND_RATIO[1])

WARM-UP (critical correctness requirement):
  The policy ``features_buffer`` is a 100-frame (5 s @ 20 Hz) recurrent history seeded
  to ZEROS (infer.ReplayState / modeld.py:80-104). Comparing from frame 0 of the window
  would let a cold-start transient corrupt the verdict. So we replay a CONTIGUOUS span
  = [warm-up frames] + [comparison frames] built from REAL contiguous frame_ids, then
  compute ``anchor_metrics`` ONLY over the comparison portion (discard the first
  ``warmup_frames``). The warm-up frames are the real frames immediately preceding the
  comparison window so the recurrent state is built from real data. CD210 LAT_SMOOTH=0
  so the curvature post-step adds no extra warm-up.

Reuse (Tasks 3/6/8/9), do NOT reinvent:
  - alignment.build_frame_timeline  -> per-route FrameRow(segment_num, segment_id, frame_id, timestamp_eof_s, ...)
  - assets._replay_scene_eligible_mask + the route_b5 NPZ -> replay-scene eligibility
  - infer.replay_window("CD210", route, mono_times) -> replayed desiredCurvature series
  - parse.logged_action_curvature(route, frame_ids) -> logged modelV2.action.desiredCurvature
  - metrics.anchor_metrics(replayed, logged) -> {corr, band_ratio, band_rms_*, n_valid}

Analysis-only. The heavy replay imports tinygrad (via infer); the span selection and the
gate decision are pure and unit-testable without any model.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from model_replay_sim import config as C
from model_replay_sim.alignment import build_frame_timeline
from model_replay_sim.assets import _replay_scene_eligible_mask


# --------------------------------------------------------------------------- #
# Pure gate decision (unit-testable without a model)                          #
# --------------------------------------------------------------------------- #

def anchor_verdict(metrics: dict) -> bool:
    """The locked gate decision over an ``anchor_metrics`` dict.

    passed iff corr and band_ratio are BOTH finite AND inside their thresholds:
      corr >= C.ANCHOR_CORR_MIN  and  C.ANCHOR_BAND_RATIO[0] <= band_ratio <= [1].
    A nan corr or nan band_ratio (degenerate/constant series) is a FAIL, never a pass.
    """
    corr = metrics.get("corr", float("nan"))
    ratio = metrics.get("band_ratio", float("nan"))
    lo, hi = C.ANCHOR_BAND_RATIO
    corr_ok = bool(np.isfinite(corr)) and (corr >= C.ANCHOR_CORR_MIN)
    ratio_ok = bool(np.isfinite(ratio)) and (lo <= ratio <= hi)
    return bool(corr_ok and ratio_ok)


# --------------------------------------------------------------------------- #
# Span selection (pure: timeline + NPZ eligibility, no model)                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AnchorSpan:
    route_id: str
    mono_times: np.ndarray        # ordered timestamp_eof_s of the full [warmup+compare] span
    frame_ids: np.ndarray         # global frame_id per span frame (same order)
    seg_keys: tuple               # (segment_num, segment_id) per span frame (alignment guard)
    split_index: int              # mono_times[:split_index]=warm-up, [split_index:]=comparison
    warmup_frames: int
    compare_frames: int
    warmup_s: float
    compare_s: float


def _eligible_aligned_timeline_mask(route_id: str) -> tuple:
    """Boolean eligibility per timeline FrameRow + the ordered timeline.

    A frame is eligible iff its nearest NPZ sample (within C.MAX_FRAME_DELTA_S of its
    timestamp_eof_s) is replay-scene-eligible (assets._replay_scene_eligible_mask). The
    timeline itself is already frame-aligned (every row is a real decoded frame), so this
    is exactly the "frame-aligned AND replay-scene-eligible" predicate.
    """
    tl = build_frame_timeline(route_id)
    if not tl:
        raise AssertionError(f"{route_id}: empty frame timeline (no local frames?)")
    npz = C.CACHE_ROOT / f"{route_id}.npz"
    if not npz.exists():
        raise AssertionError(f"{route_id}: missing cached NPZ {npz}")
    z = dict(np.load(npz))
    mono = np.asarray(z["mono_time"], float)
    order = np.argsort(mono)
    mono_s = mono[order]
    elig = _replay_scene_eligible_mask(z)[order]

    def row_eligible(t: float) -> bool:
        i = bisect.bisect_left(mono_s, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(mono_s)]
        if not cands:
            return False
        j = min(cands, key=lambda k: abs(mono_s[k] - t))
        if abs(mono_s[j] - t) > C.MAX_FRAME_DELTA_S:
            return False
        return bool(elig[j])

    good = np.array([row_eligible(r.timestamp_eof_s) for r in tl], dtype=bool)
    return good, tl


def _longest_contiguous_run(good: np.ndarray, tl: tuple) -> tuple[int, int]:
    """Longest [start, end) run that is BOTH eligible (good) AND frame-id contiguous
    (consecutive global frame_ids, no decode/log gaps). Returns (start, end); (0,0) if none.
    """
    fids = [r.frame_id for r in tl]
    n = len(good)
    best = (0, 0)
    i = 0
    while i < n:
        if good[i]:
            j = i + 1
            while j < n and good[j] and fids[j] == fids[j - 1] + 1:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def select_anchor_span(route_id: str = "route_b5", warmup_s: float = 10.0,
                       min_compare_s: float = 30.0) -> AnchorSpan:
    """Find the longest contiguous frame-aligned + replay-scene-eligible run on ``route_id``
    and carve it into a warm-up prefix (>= warmup_s) and a comparison suffix (>= min_compare_s).

    The whole span is ONE contiguous block of consecutive frame_ids, so the recurrent
    ``features_buffer`` is built from real data across the warm-up before any comparison.
    Returns an :class:`AnchorSpan`. Raises if the route can't supply the required lengths.
    """
    good, tl = _eligible_aligned_timeline_mask(route_id)
    s, e = _longest_contiguous_run(good, tl)
    run_len = e - s
    need_warm = int(round(warmup_s * C.FS_HZ))
    need_cmp = int(round(min_compare_s * C.FS_HZ))
    if run_len < need_warm + need_cmp:
        run_s = (tl[e - 1].timestamp_eof_s - tl[s].timestamp_eof_s) if run_len else 0.0
        raise AssertionError(
            f"{route_id}: longest contiguous eligible run is {run_len} frames ({run_s:.1f}s) "
            f"but need >= {need_warm}+{need_cmp}={need_warm + need_cmp} frames "
            f"({warmup_s:.0f}s warm-up + {min_compare_s:.0f}s compare). Cannot supply anchor span.")

    # use the WHOLE contiguous run (longest compare for the weave band), warm-up = first
    # need_warm frames, comparison = the rest.
    rows = tl[s:e]
    split_index = need_warm
    mono_times = np.array([r.timestamp_eof_s for r in rows], dtype=np.float64)
    frame_ids = np.array([r.frame_id for r in rows], dtype=np.int64)
    seg_keys = tuple((r.segment_num, r.segment_id) for r in rows)

    warmup_frames = split_index
    compare_frames = len(rows) - split_index
    warmup_span_s = float(mono_times[split_index - 1] - mono_times[0]) if split_index >= 1 else 0.0
    compare_span_s = float(mono_times[-1] - mono_times[split_index])

    return AnchorSpan(
        route_id=route_id,
        mono_times=mono_times,
        frame_ids=frame_ids,
        seg_keys=seg_keys,
        split_index=split_index,
        warmup_frames=warmup_frames,
        compare_frames=compare_frames,
        warmup_s=warmup_span_s,
        compare_s=compare_span_s,
    )


# --------------------------------------------------------------------------- #
# The real anchor run (heavy: replays CD210 vision+policy)                     #
# --------------------------------------------------------------------------- #

def run_anchor(bundle: str, route_id: str = "route_b5", warmup_s: float = 10.0,
               min_compare_s: float = 30.0) -> dict:
    """Run a same-model fidelity anchor end-to-end for ANY bundle and return verdict + numbers.

    Generic over the bundle: replays ``bundle`` on a window of ``route_id`` where ``bundle``'s
    model actually drove and checks the replayed ``action.desiredCurvature`` reproduces the
    LOGGED ``modelV2.action.desiredCurvature`` on those frames. (For a non-anchor-validated
    bundle this is only meaningful if the route_id is one the bundle actually drove — the
    caller is responsible for that pairing; CD210/route_b5 is the validated pair.)

    Steps:
      1. select_anchor_span -> contiguous [warm-up + comparison] span (real frame_ids)
      2. replay_window(bundle, route, span.mono_times) -> replayed desiredCurvature
      3. split off the warm-up; verify the returned frame alignment matches the selected rows
      4. logged_action_curvature(route, comparison frame_ids) -> logged ground truth
      5. anchor_metrics(replayed_compare, logged_compare) + anchor_verdict -> gate

    Returns a flat dict with passed/corr/band_ratio/band_rms_*/counts/provenance + ``bundle``.
    """
    from model_replay_sim.infer import replay_window
    from model_replay_sim.parse import logged_action_curvature

    span = select_anchor_span(route_id, warmup_s=warmup_s, min_compare_s=min_compare_s)

    r = replay_window(bundle, route_id, span.mono_times)
    replayed = np.asarray(r["desired_curvature"], dtype=float)
    if replayed.shape[0] != span.mono_times.shape[0]:
        raise RuntimeError(
            f"replay returned {replayed.shape[0]} curvatures for {span.mono_times.shape[0]} "
            f"window frames — length mismatch")

    # ALIGNMENT GUARD: the (seg_num, seg_id) the replay actually ran must match the rows we
    # selected (so the logged frame_ids we use for ground truth are the right frames).
    # replay_window returns frames as [(segment_num, segment_id), ...].
    got_keys = tuple((int(a), int(b)) for (a, b) in r["frames"])
    if got_keys != span.seg_keys:
        n_mismatch = sum(1 for g, s in zip(got_keys, span.seg_keys) if g != s)
        raise RuntimeError(
            f"frame-alignment mismatch: replay ran different (seg_num, seg_id) than selected "
            f"({n_mismatch}/{len(span.seg_keys)} differ) — off-by-one in mono->frame mapping")

    split = span.split_index
    replayed_compare = replayed[split:]
    compare_frame_ids = span.frame_ids[split:]
    logged_compare = logged_action_curvature(route_id, compare_frame_ids)

    metrics = anchor_metrics_safe(replayed_compare, logged_compare)
    passed = anchor_verdict(metrics)

    # diagnostics (cheap, always computed): small-lag correlation to detect a residual shift
    lag_corr = _lag_offset_corr(replayed_compare, logged_compare, lags=(-2, -1, 0, 1, 2))

    prov = _provenance(bundle)
    n_logged_valid = int(np.isfinite(logged_compare).sum())
    n_replayed_valid = int(np.isfinite(replayed_compare).sum())

    return {
        "passed": bool(passed),
        "route_id": route_id,
        "bundle": bundle,
        "corr": metrics["corr"],
        "band_ratio": metrics["band_ratio"],
        "band_rms_replayed": metrics["band_rms_replayed"],
        "band_rms_logged": metrics["band_rms_logged"],
        "n_valid": metrics["n_valid"],
        "n_compare": int(compare_frame_ids.shape[0]),
        "n_logged_valid": n_logged_valid,
        "n_replayed_valid": n_replayed_valid,
        "warmup_frames": int(span.warmup_frames),
        "compare_frames": int(span.compare_frames),
        "warmup_s": float(span.warmup_s),
        "compare_s": float(span.compare_s),
        "lat_action_t": float(r.get("lat_action_t", float("nan"))),
        "frame_id_first": int(span.frame_ids[0]),
        "frame_id_last": int(span.frame_ids[-1]),
        "compare_frame_id_first": int(compare_frame_ids[0]),
        "compare_frame_id_last": int(compare_frame_ids[-1]),
        "lag_offset_corr": lag_corr,
        "thresholds": {"corr_min": C.ANCHOR_CORR_MIN, "band_ratio": list(C.ANCHOR_BAND_RATIO)},
        "provenance": prov,
        # raw series kept for the diagnostic report (small, ~hundreds of floats)
        "_replayed_compare": replayed_compare,
        "_logged_compare": logged_compare,
    }


def run_cd210_anchor(route_id: str = "route_b5", warmup_s: float = 10.0,
                     min_compare_s: float = 30.0) -> dict:
    """The CD210 same-model fidelity anchor — a thin wrapper over the generic
    :func:`run_anchor` pinned to the anchor-validated (CD210, route_b5) pair."""
    return run_anchor("CD210", route_id, warmup_s=warmup_s, min_compare_s=min_compare_s)


def anchor_metrics_safe(replayed, logged) -> dict:
    """Thin wrapper over metrics.anchor_metrics (imported lazily so the pure gate test
    needs no heavy deps if metrics later grows them — metrics is currently light)."""
    from model_replay_sim.metrics import anchor_metrics
    return anchor_metrics(replayed, logged)


def _lag_offset_corr(replayed, logged, lags=(-2, -1, 0, 1, 2)) -> dict:
    """Pearson r between replayed and logged at small integer frame lags, to detect a
    residual off-by-one shift (a sign that warp/alignment is shifted by 1-2 frames). A lag
    of +k correlates replayed[k:] with logged[:-k] (replayed leads logged by k frames)."""
    a = np.asarray(replayed, float)
    b = np.asarray(logged, float)
    out = {}
    for k in lags:
        if k == 0:
            x, y = a, b
        elif k > 0:
            x, y = a[k:], b[:-k]
        else:
            x, y = a[:k], b[-k:]
        both = np.isfinite(x) & np.isfinite(y)
        if both.sum() < 2 or np.std(x[both]) == 0 or np.std(y[both]) == 0:
            out[str(k)] = float("nan")
        else:
            out[str(k)] = float(np.corrcoef(x[both], y[both])[0, 1])
    return out


def _provenance(bundle: str = "CD210") -> dict:
    """tinygrad sha + onnx sha256 from Task 5 provenance.json + the gate thresholds."""
    import json
    p = {"tinygrad_sha": None, "models": [], "bundle_full_sha": C.BUNDLES[bundle]["full_sha"]}
    prov_path = C.RESULTS_ROOT / "onnx" / bundle / "provenance.json"
    if prov_path.exists():
        try:
            j = json.loads(prov_path.read_text())
            p["tinygrad_sha"] = j.get("tinygrad_sha")
            p["bundle_full_sha"] = j.get("full_sha", p["bundle_full_sha"])
            p["models"] = [{"name": m.get("name"), "onnx_sha256": m.get("onnx_sha256")}
                           for m in j.get("models", [])]
        except Exception:
            pass
    return p
