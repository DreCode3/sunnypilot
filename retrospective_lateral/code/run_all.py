from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from retrospective_lateral.code import config as C
from retrospective_lateral.code.report import write_markdown_report


def build_pipeline_steps(log_root: str, cache_root: str, report_root: str, routes: list[str], force: bool = False) -> list[list[str]]:
  extract = [".venv311/bin/python", "-m", "retrospective_lateral.code.extract", "--log-root", log_root, "--cache-root", cache_root]
  if force:
    extract.append("--force")
  for route in routes:
    extract.extend(["--route", route])
  metrics = [".venv311/bin/python", "-m", "retrospective_lateral.code.metrics", "--cache-root", cache_root, "--out", report_root]
  return [extract, metrics]


def write_pipeline_report(report_root: str | Path) -> Path:
  report_dir = Path(report_root)
  symptom_catalog = pd.read_csv(report_dir / "symptom_catalog.csv")
  scorecard_path = report_dir / "scorecard.csv"
  scorecard = pd.read_csv(scorecard_path) if scorecard_path.exists() else pd.DataFrame()
  return write_markdown_report(report_dir, symptom_catalog, scorecard)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-root", default=str(C.DEFAULT_LOG_ROOT))
  parser.add_argument("--cache-root", default=str(C.DEFAULT_CACHE_ROOT))
  parser.add_argument("--report-root", default=str(C.DEFAULT_REPORT_ROOT))
  parser.add_argument("--route", action="append", default=[])
  parser.add_argument("--force", action="store_true")
  args = parser.parse_args(argv)
  Path(args.cache_root).mkdir(parents=True, exist_ok=True)
  Path(args.report_root).mkdir(parents=True, exist_ok=True)
  for step in build_pipeline_steps(args.log_root, args.cache_root, args.report_root, args.route, force=args.force):
    print("+ " + shlex.join(step))
    subprocess.run(step, check=True)
  report_path = write_pipeline_report(args.report_root)
  print(f"wrote {report_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
