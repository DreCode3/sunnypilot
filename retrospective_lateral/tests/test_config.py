from pathlib import Path

from retrospective_lateral.code import config as C


def test_default_paths_are_repo_relative():
  assert C.REPO_ROOT.name == "sunnypilot"
  assert C.DEFAULT_LOG_ROOT == C.REPO_ROOT / "explorer_st_logs"
  assert C.DEFAULT_CACHE_ROOT == C.REPO_ROOT / "retrospective_lateral" / "results" / "cache"
  assert C.DEFAULT_REPORT_ROOT == C.REPO_ROOT / "retrospective_lateral" / "results" / "reports"


def test_schema_and_metric_constants_are_explicit():
  assert isinstance(C.CACHE_SCHEMA_VERSION, str)
  assert C.CACHE_SCHEMA_VERSION.startswith("retrolat-v")
  assert C.FS_HZ == 20.0
  assert C.LOW_SPEED_MPH == (1.0, 10.0)
  assert C.WEAVE_SPEED_MPH == (10.0, 70.0)
  assert C.DEFAULT_WEAVE_BAND_HZ == (0.10, 0.35)
  assert C.HUNT_GUARD_BAND_HZ == (0.50, 1.50)
  assert C.MAX_INTERP_GAP_S == 0.5
  assert C.TELEMETRY_INTERP_GAP_S == 1.25
