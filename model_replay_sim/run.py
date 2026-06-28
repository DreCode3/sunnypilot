"""Task 10 CLI: run the CD210 same-model fidelity ANCHOR and write a report.

This is the run-ready GATE for the whole model-replay simulator. It replays CD210 on a
route_b5 window where CD210 actually drove and checks the replayed
``action.desiredCurvature`` reproduces the LOGGED ``modelV2.action.desiredCurvature``,
with a warm-up prefix so the cold-start recurrent transient is excluded from the verdict.

Usage:
    .venv311/bin/python -m model_replay_sim.run [--route route_b5] [--warmup-s 10] [--compare-s 30]

Read-only w.r.t. the repo. Writes a JSON + markdown report under
``C.RESULTS_ROOT/anchor_cd210_<route>.{json,md}`` (gitignored results dir).
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the CD210 same-model fidelity anchor.")
    ap.add_argument("--route", default="route_b5")
    ap.add_argument("--warmup-s", type=float, default=10.0)
    ap.add_argument("--compare-s", type=float, default=30.0)
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    sys.exit(main())
