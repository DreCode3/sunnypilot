from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from retrospective_lateral.code import config as C


def build_pipeline_steps(log_root: str, cache_root: str, report_root: str, routes: list[str]) -> list[list[str]]:
  extract = [".venv311/bin/python", "-m", "retrospective_lateral.code.extract", "--log-root", log_root, "--cache-root", cache_root]
  for route in routes:
    extract.extend(["--route", route])
  metrics = [".venv311/bin/python", "-m", "retrospective_lateral.code.metrics", "--cache-root", cache_root, "--out", report_root]
  return [extract, metrics]


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-root", default=str(C.DEFAULT_LOG_ROOT))
  parser.add_argument("--cache-root", default=str(C.DEFAULT_CACHE_ROOT))
  parser.add_argument("--report-root", default=str(C.DEFAULT_REPORT_ROOT))
  parser.add_argument("--route", action="append", default=[])
  args = parser.parse_args(argv)
  Path(args.cache_root).mkdir(parents=True, exist_ok=True)
  Path(args.report_root).mkdir(parents=True, exist_ok=True)
  for step in build_pipeline_steps(args.log_root, args.cache_root, args.report_root, args.route):
    print("+ " + " ".join(step))
    subprocess.run(step, check=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
