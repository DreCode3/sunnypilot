"""CLI for the model-replay simulator: the fidelity ANCHOR (gate) and the cross-model
weave COMPARISON.

ANCHOR (default / ``--anchor``): the run-ready GATE. Replays CD210 on a route_b5 window
where CD210 actually drove and checks the replayed ``action.desiredCurvature`` reproduces
the LOGGED ``modelV2.action.desiredCurvature``, with a warm-up prefix so the cold-start
recurrent transient is excluded from the verdict.

COMPARE (``--compare``): "on identical camera frames, which model weaves more?" Replays
each ``--bundles`` model on the SAME shared scene window(s) ``--scenes`` and compares the
post-warm-up weave-band RMS, GATED on each bundle's fidelity anchor (a result is flagged
NOT-TRUSTWORTHY if any compared bundle is not anchor-validated).

Usage:
    .venv311/bin/python -m model_replay_sim.run [--anchor] [--route route_b5] [--warmup-s 10] [--compare-s 30]
    .venv311/bin/python -m model_replay_sim.run --compare --scenes route_b5,route_7f --bundles CD210,OPM7

Read-only w.r.t. the repo. Writes JSON + markdown reports under ``C.RESULTS_ROOT`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from model_replay_sim import config as C
from model_replay_sim.anchor import run_cd210_anchor


def _tinygrad_repo_sha() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(C.TINYGRAD_PATH), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _fmt(x, nd=6):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _markdown(res: dict) -> str:
    p = res["provenance"]
    verdict = "PASS" if res["passed"] else "FAIL"
    lo, hi = res["thresholds"]["band_ratio"]
    lag = res.get("lag_offset_corr", {})
    rc = np.asarray(res.get("_replayed_compare", []), float)
    lg = np.asarray(res.get("_logged_compare", []), float)

    def head_tail(a, n=10):
        a = np.asarray(a, float)
        h = ", ".join(_fmt(v, 5) for v in a[:n])
        t = ", ".join(_fmt(v, 5) for v in a[-n:])
        return h, t

    rh, rt = head_tail(rc)
    lh, lt = head_tail(lg)

    lines = [
        f"# CD210 same-model fidelity anchor — {verdict}",
        "",
        f"- route: `{res['route_id']}`  bundle: `{res['bundle']}`",
        f"- **verdict: {verdict}**  (gate: corr >= {res['thresholds']['corr_min']} "
        f"AND band_ratio in [{lo}, {hi}])",
        "",
        "## Numbers",
        f"- corr               = **{_fmt(res['corr'])}**  (min {res['thresholds']['corr_min']})",
        f"- band_ratio         = **{_fmt(res['band_ratio'])}**  (in-band [{lo}, {hi}])",
        f"- band_rms_replayed  = {_fmt(res['band_rms_replayed'], 8)}",
        f"- band_rms_logged    = {_fmt(res['band_rms_logged'], 8)}",
        f"- n_compare          = {res['n_compare']} frames "
        f"({_fmt(res['compare_s'], 1)} s)",
        f"- n_valid (joint)    = {res['n_valid']}  "
        f"(replayed finite {res['n_replayed_valid']}, logged finite {res['n_logged_valid']})",
        f"- warmup_frames      = {res['warmup_frames']} ({_fmt(res['warmup_s'], 1)} s)",
        f"- lat_action_t       = {_fmt(res['lat_action_t'], 4)} s",
        f"- frame_ids: span [{res['frame_id_first']}, {res['frame_id_last']}], "
        f"compare [{res['compare_frame_id_first']}, {res['compare_frame_id_last']}]",
        "",
        "## Lag-offset diagnostic (Pearson r of replayed-vs-logged at integer frame lags)",
        "lag +k correlates replayed[k:] with logged[:-k] (replayed leads logged by k frames):",
        "",
        "| lag (frames) | corr |",
        "|---|---|",
    ]
    for k in ("-2", "-1", "0", "1", "2"):
        lines.append(f"| {k} | {_fmt(lag.get(k), 4)} |")
    lines += [
        "",
        "## First/last 10 values (comparison window)",
        f"- replayed head: [{rh}]",
        f"- replayed tail: [{rt}]",
        f"- logged   head: [{lh}]",
        f"- logged   tail: [{lt}]",
        "",
        "## Provenance",
        f"- bundle_full_sha: `{p.get('bundle_full_sha')}`",
        f"- tinygrad_sha (provenance.json): `{p.get('tinygrad_sha')}`",
        f"- tinygrad_repo HEAD (now): `{res.get('tinygrad_repo_sha_now')}`",
    ]
    for m in p.get("models", []):
        lines.append(f"- onnx `{m.get('name')}` sha256: `{m.get('onnx_sha256')}`")
    return "\n".join(lines) + "\n"


def _fmt_ratio(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "nan"
        return f"{float(x):.{nd}f}x"
    except Exception:
        return str(x)


def _compare_markdown(res: dict) -> str:
    """Markdown report for a compare_models result."""
    trust = "TRUSTWORTHY" if res["trustworthy"] else "NOT-TRUSTWORTHY"
    lines = [
        f"# Cross-model weave comparison — {trust}",
        "",
        f"- scenes:  `{', '.join(res['scenes'])}`",
        f"- bundles: `{', '.join(res['bundles'])}`",
        f"- **{trust}**" + ("" if res["trustworthy"]
                            else f"  — unvalidated bundles (no same-model fidelity anchor): "
                                 f"`{', '.join(res['unvalidated'])}`"),
        "",
        "Weave = band-RMS of post-warm-up model `action.desiredCurvature` in "
        f"{C.WEAVE_BAND_HZ[0]}-{C.WEAVE_BAND_HZ[1]} Hz @ {C.FS_HZ:.0f} Hz, on the SAME shared "
        "frames per scene. Higher = weaves more.",
        "",
        "## Per-bundle weave by scene (band-RMS, 1/m)",
        "",
        "| bundle | anchor_validated | " + " | ".join(res["scenes"]) + " |",
        "|---|---|" + "---|" * len(res["scenes"]),
    ]
    for b in res["bundles"]:
        validated = "yes" if C.BUNDLES.get(b, {}).get("anchor_validated", False) else "**NO**"
        cells = [_fmt(res["per_bundle_weave"].get(b, {}).get(sc), 8) for sc in res["scenes"]]
        lines.append(f"| {b} | {validated} | " + " | ".join(cells) + " |")

    lines += ["", "## Median pairwise weave ratio across scenes", "",
              "| pair (num/den) | median ratio |", "|---|---|"]
    for k in sorted(res["median_ratios"]):
        lines.append(f"| {k} | {_fmt_ratio(res['median_ratios'][k])} |")

    lines += ["", "## Per-scene detail", ""]
    for sc in res["scenes"]:
        s = res["per_scene"][sc]
        sc_trust = "trustworthy" if s["trustworthy"] else "NOT-trustworthy"
        lines.append(f"### {sc}  ({s['n_compare']} compare frames, split@{s['split_index']}; {sc_trust})")
        lines.append("")
        lines.append("| bundle | weave_band_rms | anchor_validated | n_finite |")
        lines.append("|---|---|---|---|")
        for b in res["bundles"]:
            pb = s["per_bundle"][b]
            v = "yes" if pb["anchor_validated"] else "**NO**"
            lines.append(f"| {b} | {_fmt(pb['weave_band_rms'], 8)} | {v} | {pb['n_finite']} |")
        lines.append("")
        lines.append("ratios: " + ", ".join(f"{k}={_fmt_ratio(v)}" for k, v in sorted(s["ratios"].items())))
        lines.append("")
    return "\n".join(lines) + "\n"


def _print_compare(res: dict) -> None:
    trust = "TRUSTWORTHY" if res["trustworthy"] else "NOT-TRUSTWORTHY"
    print("=" * 72)
    print(f"  CROSS-MODEL WEAVE COMPARISON   scenes={res['scenes']}  bundles={res['bundles']}")
    print("=" * 72)
    # per-bundle weave table (median across scenes for the headline column)
    print(f"  {'bundle':<10} {'anchor_validated':<18} weave_band_rms (per scene)")
    for b in res["bundles"]:
        validated = "yes" if C.BUNDLES.get(b, {}).get("anchor_validated", False) else "NO  <-- unvalidated"
        per = "  ".join(f"{sc}={_fmt(res['per_bundle_weave'].get(b, {}).get(sc), 8)}"
                        for sc in res["scenes"])
        print(f"  {b:<10} {validated:<18} {per}")
    print("-" * 72)
    print("  median pairwise weave ratio across scenes:")
    for k in sorted(res["median_ratios"]):
        print(f"    {k:<16} = {_fmt_ratio(res['median_ratios'][k])}")
    print("=" * 72)
    if res["trustworthy"]:
        print(f"  ***** {trust} ***** all compared bundles are anchor-validated.")
    else:
        print(f"  ##### {trust} ##### the following bundles have NO same-model fidelity")
        print(f"        anchor, so their weave (and any ratio against them) may be")
        print(f"        systematically wrong: {', '.join(res['unvalidated'])}")
    print("=" * 72, flush=True)


def _run_compare(args) -> int:
    from model_replay_sim.compare import compare_models
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    bundles = [b.strip() for b in args.bundles.split(",") if b.strip()]
    if not scenes:
        print("[compare] --scenes is required (e.g. --scenes route_b5,route_7f)", file=sys.stderr)
        return 2
    if not bundles:
        print("[compare] --bundles is required (e.g. --bundles CD210,OPM7)", file=sys.stderr)
        return 2

    print(f"[compare] replaying {bundles} on shared scenes {scenes} "
          f"(warmup>={args.warmup_s}s, compare>={args.compare_s}s)...", flush=True)
    cap = args.max_compare_frames if args.max_compare_frames and args.max_compare_frames > 0 else None
    res = compare_models(scenes, bundles, warmup_s=args.warmup_s, min_compare_s=args.compare_s,
                         max_compare_frames=cap)
    res["tinygrad_repo_sha_now"] = _tinygrad_repo_sha()
    _print_compare(res)

    out_dir = C.RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{'-'.join(scenes)}_{'-'.join(bundles)}"
    json_path = out_dir / f"compare_{tag}.json"
    md_path = out_dir / f"compare_{tag}.md"
    json_path.write_text(json.dumps(_jsonable(res), indent=2))
    md_path.write_text(_compare_markdown(res))
    print(f"[compare] wrote {json_path}")
    print(f"[compare] wrote {md_path}", flush=True)
    # exit 0 always for compare (a NOT-TRUSTWORTHY result is still a valid, reported run)
    return 0


def _run_anchor(args) -> int:
    print(f"[anchor] running CD210 fidelity anchor on {args.route} "
          f"(warmup>={args.warmup_s}s, compare>={args.compare_s}s)...", flush=True)
    res = run_cd210_anchor(route_id=args.route, warmup_s=args.warmup_s,
                           min_compare_s=args.compare_s)
    res["tinygrad_repo_sha_now"] = _tinygrad_repo_sha()

    verdict = "PASS" if res["passed"] else "FAIL"
    print("=" * 64)
    print(f"  VERDICT: {verdict}")
    print(f"  corr        = {_fmt(res['corr'])}   (>= {res['thresholds']['corr_min']})")
    lo, hi = res["thresholds"]["band_ratio"]
    print(f"  band_ratio  = {_fmt(res['band_ratio'])}   (in [{lo}, {hi}])")
    print(f"  band_rms    replayed={_fmt(res['band_rms_replayed'], 8)}  "
          f"logged={_fmt(res['band_rms_logged'], 8)}")
    print(f"  n_compare   = {res['n_compare']} frames ({_fmt(res['compare_s'], 1)}s)   "
          f"warmup={res['warmup_frames']} frames")
    print(f"  lag corr    = {{ " +
          ", ".join(f"{k}:{_fmt(v, 3)}" for k, v in res['lag_offset_corr'].items()) + " }")
    print("=" * 64, flush=True)

    out_dir = C.RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"anchor_cd210_{args.route}.json"
    md_path = out_dir / f"anchor_cd210_{args.route}.md"
    json_path.write_text(json.dumps(_jsonable(res), indent=2))
    md_path.write_text(_markdown(res))
    print(f"[anchor] wrote {json_path}")
    print(f"[anchor] wrote {md_path}", flush=True)

    return 0 if res["passed"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Model-replay simulator: fidelity anchor (gate) + cross-model weave comparison.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--anchor", action="store_true",
                      help="(default) run the CD210 same-model fidelity anchor")
    mode.add_argument("--compare", action="store_true",
                      help="cross-model weave comparison on shared scenes")
    # anchor args
    ap.add_argument("--route", default="route_b5", help="[anchor] route to run the anchor on")
    # compare args
    ap.add_argument("--scenes", default=None,
                    help="[compare] comma-separated scene routes, e.g. route_b5,route_7f")
    ap.add_argument("--bundles", default=None,
                    help="[compare] comma-separated bundles, e.g. CD210,OPM7")
    # shared
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--compare-s", type=float, default=30.0)
    ap.add_argument("--max-compare-frames", type=int, default=1200,
                    help="[compare] cap the compare window (select_anchor_span uses the whole "
                         "contiguous eligible run, e.g. route_7f ~6945 frames => ~47min/model). "
                         "1200 (~60s) matches the CD210-anchor scale. 0 = no cap.")
    args = ap.parse_args(argv)

    if args.compare:
        if args.scenes is None or args.bundles is None:
            ap.error("--compare requires --scenes and --bundles")
        return _run_compare(args)
    return _run_anchor(args)  # default mode = anchor


if __name__ == "__main__":
    sys.exit(main())
