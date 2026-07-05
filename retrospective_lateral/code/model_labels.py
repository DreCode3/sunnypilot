"""Recover the active driving-model bundle for each route from its boot rlog.

Analysis-only. The model identity is logged in every route's boot segment as the
param ModelManager_ActiveBundle inside initData.params (system/loggerd/logger.cc
readAll). This lets us group routes by driving model WITHOUT re-running any model.
Older routes predate that param; a hand-label fallback (from
explorer_st_logs/CHECKPOINT_2026-06-13_golden_PI_b8.md) covers the OPM7 era.

Writes only to retrospective_lateral/results/. Reads only boot rlogs (one segment
per route), parallelized across cores.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "opendbc_repo")

from retrospective_lateral.code import config as C
from retrospective_lateral.code.routes import discover_routes

# Hand-labeled fallback for older routes whose boot rlog has no ModelManager_ActiveBundle
# (source: explorer_st_logs/CHECKPOINT_2026-06-13_golden_PI_b8.md / analyze_b8.py).
HAND_LABELS = {
    "route_b1": "CD210", "route_b2": "CD210", "route_b8": "CD210",
    "route_7f": "OPM7", "route_95": "OPM7", "route_99": "OPM7", "route_98": "OPM7",
    "route_a0": "OPM7", "route_9b": "OPM7", "route_9d": "OPM7",
}


def parse_active_bundle(value) -> dict | None:
    """Parse a ModelManager_ActiveBundle param value into {index, internal_name,
    display_name}. Accepts bytes/str/dict. Returns None if not parseable."""
    try:
        if isinstance(value, dict):
            obj = value
        else:
            text = value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
            obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict) or "internalName" not in obj:
        return None
    return {
        "index": obj.get("index"),
        "internal_name": str(obj.get("internalName")),
        "display_name": str(obj.get("displayName", "")),
    }


def scan_route_label(args):
    """ProcessPoolExecutor worker: read one route's boot rlog and return
    (route_id, label_dict_or_None). Picklable args (route_id, boot_rlog_path)."""
    route_id, boot_rlog = args
    from openpilot.tools.lib.logreader import LogReader
    try:
        for msg in LogReader(str(boot_rlog)):
            if msg.which() == "initData":
                for entry in msg.initData.params.entries:
                    if entry.key == "ModelManager_ActiveBundle":
                        return route_id, parse_active_bundle(bytes(entry.value))
                break  # initData seen; no ActiveBundle in it
    except Exception:
        return route_id, None
    return route_id, None


def build_model_labels(log_root: Path = C.DEFAULT_LOG_ROOT,
                       report_root: Path = C.DEFAULT_REPORT_ROOT,
                       workers: int | None = None) -> dict:
    """Scan every route's boot rlog for its active model bundle (parallel), merge the
    hand-label fallback, and write route_model_labels.csv. Returns a summary."""
    import pandas as pd

    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    routes = discover_routes(Path(log_root))
    work = [(r.route_id, r.segments[0].rlog_path) for r in routes if r.segments]

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for route_id, label in ex.map(scan_route_label, work):
            if label is not None:
                rows.append({"route_id": route_id, "model": label["internal_name"],
                             "display_name": label["display_name"], "index": label["index"],
                             "source": "active_bundle"})
            elif route_id in HAND_LABELS:
                rows.append({"route_id": route_id, "model": HAND_LABELS[route_id],
                             "display_name": "", "index": None, "source": "hand_label"})
            else:
                rows.append({"route_id": route_id, "model": "unknown",
                             "display_name": "", "index": None, "source": "none"})

    df = pd.DataFrame(rows).sort_values("route_id")
    df.to_csv(report_root / "route_model_labels.csv", index=False)
    counts = df["model"].value_counts().to_dict()
    return {"routes": len(df), "models": counts}


def load_labels(report_root: Path = C.DEFAULT_REPORT_ROOT) -> dict:
    """Return {route_id: model} from a previously built route_model_labels.csv."""
    import pandas as pd
    path = Path(report_root) / "route_model_labels.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["route_id"], df["model"]))


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recover per-route driving-model labels from boot rlogs")
    parser.add_argument("--log-root", type=Path, default=C.DEFAULT_LOG_ROOT)
    parser.add_argument("--report-root", type=Path, default=C.DEFAULT_REPORT_ROOT)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    result = build_model_labels(args.log_root, args.report_root, workers=args.workers)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
