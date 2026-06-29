"""Cross-model weave COMPARISON driver — "on identical camera frames, which model weaves more?"

Replays each compared bundle on the SAME shared scene window (the anchor span of a chosen
``scene_route``, so every model sees the EXACT same frames + calibration) and compares the
post-warm-up weave-band RMS of each model's pre-controller ``action.desiredCurvature``.

TRUST GATE (the whole point of gating on the fidelity anchor):
  A weave number is only believable for a bundle whose same-model fidelity anchor PASSED
  (``C.BUNDLES[bundle]["anchor_validated"]``: CD210 True; OPM7/Nevada False). For an
  unvalidated bundle the replay's absolute weave amplitude may be systematically wrong
  (img_buffer_length / mlsim handling are UNVERIFIED), so the cross-model ratio against it
  is suspect. We DO compute and report the number (never silently drop it) but flag the
  whole result ``trustworthy: False`` and list the offending bundles in ``unvalidated``.

Pure/deterministic; no plotting. The heavy ``replay_window`` (tinygrad) is imported at module
scope so tests can ``monkeypatch`` it (and ``select_anchor_span``) with synthetic fakes.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from model_replay_sim import config as C
from model_replay_sim.anchor import select_anchor_span
from model_replay_sim.infer import replay_window
from model_replay_sim.metrics import weave_band_rms


def _anchor_validated(bundle: str) -> bool:
    return bool(C.BUNDLES.get(bundle, {}).get("anchor_validated", False))


def _ratio_key(num: str, den: str) -> str:
    return f"{num}/{den}"


def _pairwise_ratios(weave: dict) -> dict:
    """All ordered-pair weave ratios num/den over bundles with finite, positive weave.

    A ratio is nan if either side's weave is non-finite or the denominator is 0.
    """
    out: dict[str, float] = {}
    for num, den in itertools.permutations(weave.keys(), 2):
        wn, wd = weave[num], weave[den]
        if np.isfinite(wn) and np.isfinite(wd) and wd != 0:
            out[_ratio_key(num, den)] = float(wn / wd)
        else:
            out[_ratio_key(num, den)] = math.nan
    return out


def compare_on_scene(scene_route: str, bundles, warmup_s: float = 10.0,
                     min_compare_s: float = 30.0) -> dict:
    """Replay every bundle on ONE shared anchor window of ``scene_route`` and compare weave.

    Every model replays the EXACT same ``span.mono_times`` (identical frames + calibration),
    so any weave difference is purely the model, not the scene. The weave is the band-RMS of
    each model's post-warm-up ``desired_curvature`` (``curv[span.split_index:]``).

    Returns::

        {"scene_route", "n_compare", "split_index",
         "per_bundle": {bundle: {"weave_band_rms", "anchor_validated", "lat_action_t",
                                 "n_finite"}},
         "ratios": {"<num>/<den>": ratio, ...},          # all ordered pairs
         "unvalidated": [bundles with anchor_validated False],
         "trustworthy": bool}                            # True iff ALL compared bundles validated
    """
    bundles = list(bundles)
    if not bundles:
        raise ValueError("compare_on_scene needs at least one bundle")

    span = select_anchor_span(scene_route, warmup_s, min_compare_s)  # ONE shared window
    split = int(span.split_index)

    per_bundle: dict[str, dict] = {}
    weave: dict[str, float] = {}
    unvalidated: list[str] = []
    for b in bundles:
        r = replay_window(b, scene_route, span.mono_times)
        curv = np.asarray(r["desired_curvature"], dtype=float)[split:]  # post-warm-up only
        w = weave_band_rms(curv)
        validated = _anchor_validated(b)
        if not validated:
            unvalidated.append(b)
        weave[b] = w
        per_bundle[b] = {
            "weave_band_rms": w,
            "anchor_validated": validated,
            "lat_action_t": float(r.get("lat_action_t", float("nan"))),
            "n_finite": int(np.isfinite(curv).sum()),
        }

    return {
        "scene_route": scene_route,
        "n_compare": int(span.compare_frames),
        "split_index": split,
        "per_bundle": per_bundle,
        "ratios": _pairwise_ratios(weave),
        "unvalidated": unvalidated,
        "trustworthy": len(unvalidated) == 0,
    }


def _median_ratios(per_scene: dict) -> dict:
    """Median of each ratio key across scenes (nan-skipping; nan if no finite value)."""
    keys: set[str] = set()
    for sc in per_scene.values():
        keys.update(sc["ratios"].keys())
    out: dict[str, float] = {}
    for k in keys:
        vals = [sc["ratios"].get(k, math.nan) for sc in per_scene.values()]
        finite = [v for v in vals if np.isfinite(v)]
        out[k] = float(np.median(finite)) if finite else math.nan
    return out


def compare_models(scenes, bundles, warmup_s: float = 10.0,
                   min_compare_s: float = 30.0) -> dict:
    """Run :func:`compare_on_scene` for each scene and aggregate across scenes.

    Reports per-scene results, the per-bundle weave per scene, and the MEDIAN pairwise ratio
    across scenes (median for robustness to a single odd scene). Overall ``trustworthy`` is the
    AND across scenes; ``unvalidated`` is the union of unvalidated bundles seen.

    Returns::

        {"scenes", "bundles",
         "per_scene": {scene_route: <compare_on_scene result>, ...},
         "per_bundle_weave": {bundle: {scene_route: weave_band_rms, ...}, ...},
         "median_ratios": {"<num>/<den>": median_ratio, ...},
         "unvalidated": [...], "trustworthy": bool}
    """
    scenes = list(scenes)
    bundles = list(bundles)
    if not scenes:
        raise ValueError("compare_models needs at least one scene")

    per_scene: dict[str, dict] = {}
    for sc in scenes:
        per_scene[sc] = compare_on_scene(sc, bundles, warmup_s=warmup_s,
                                         min_compare_s=min_compare_s)

    per_bundle_weave: dict[str, dict] = {b: {} for b in bundles}
    for sc, res in per_scene.items():
        for b, pb in res["per_bundle"].items():
            per_bundle_weave[b][sc] = pb["weave_band_rms"]

    unvalidated = sorted({b for res in per_scene.values() for b in res["unvalidated"]})
    trustworthy = all(res["trustworthy"] for res in per_scene.values())

    return {
        "scenes": scenes,
        "bundles": bundles,
        "per_scene": per_scene,
        "per_bundle_weave": per_bundle_weave,
        "median_ratios": _median_ratios(per_scene),
        "unvalidated": unvalidated,
        "trustworthy": bool(trustworthy),
    }
