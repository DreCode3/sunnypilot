from __future__ import annotations
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
TINYGRAD_PATH = REPO_ROOT/"tinygrad_repo"
RESULTS_ROOT = REPO_ROOT/"retrospective_lateral"/"results"/"model_replay"
CACHE_ROOT = REPO_ROOT/"retrospective_lateral"/"results"/"cache"
LOG_ROOT = REPO_ROOT/"explorer_st_logs"
FS_HZ = 20.0
WEAVE_BAND_HZ = (0.10, 0.35)
GENTLE_CURV_ABS_MAX_1PM = 0.005
ROAD_LP_HZ = 0.035
SELFTEST_ATOL = 1e-3
SELFTEST_RTOL = 1e-2
ANCHOR_CORR_MIN = 0.95
ANCHOR_BAND_RATIO = (0.85, 1.15)
MAX_FRAME_DELTA_S = 0.03
TINYGRAD_ENV = {"DEBUG": "0", "DEV": "CPU", "IMAGE": "0", "THREADS": "0"}
BUNDLES = {
  "CD210":  {"full_sha": "55f66e2246359c6593605399a0199d94d13ad90d", "repo": "commaai/openpilot",     "split": False, "internal_names": {"C210M", "CD210"}},
  "Nevada": {"full_sha": "3193eac5e385aa010694a8ac192ff38ffe000193", "repo": "commaai/openpilot",     "split": False, "internal_names": {"NM", "Nevada"}},
  "OPM7":   {"full_sha": "052692b25d63c5ddda276b5c2271383b6aff129f", "repo": "sunnypilot/sunnypilot", "split": True,  "internal_names": {"OPM7"}},
}
