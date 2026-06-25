from __future__ import annotations

from pathlib import Path

CACHE_SCHEMA_VERSION = "retrolat-v2"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = REPO_ROOT / "explorer_st_logs"
DEFAULT_CACHE_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "cache"
DEFAULT_REPORT_ROOT = REPO_ROOT / "retrospective_lateral" / "results" / "reports"

FS_HZ = 20.0
MAX_INTERP_GAP_S = 0.5
TELEMETRY_INTERP_GAP_S = 1.25

LOW_SPEED_MPH = (1.0, 10.0)
WEAVE_SPEED_MPH = (10.0, 70.0)
DEFAULT_WEAVE_BAND_HZ = (0.10, 0.35)
LOW_SPEED_INSPECT_BAND_HZ = (0.08, 0.80)
HUNT_GUARD_BAND_HZ = (0.50, 1.50)
ROAD_LP_HZ = 0.035
ROAD_CURV_ABS_MAX_1PM = 0.0015

ENGAGE_ERODE_S = 2.0
OVERRIDE_BUFFER_S = 1.0
BLINKER_BUFFER_S = 1.0
LANE_CHANGE_BUFFER_S = 1.0
LEAD_BUFFER_S = 2.0
LEAD_HEADWAY_S = 1.6

MIN_LOW_SPEED_WINDOW_S = 6.0
LOW_SPEED_WINDOW_S = 10.0
WEAVE_WINDOW_S = 30.0
MIN_WEAVE_ELIGIBLE_S = 20.0

GPS_CELL_M = 80.0
SPEED_BIN_MPH = 2.5
HEADING_BIN_DEG = 45.0

M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON_AT_EQUATOR = 111_320.0

# --- Offline model/path-vs-road discrimination (analysis-only) ---
DISCRIM_SCHEMA_VERSION = "discrim-v1"
DISCRIM_LOOKAHEAD_M = 20.0            # primary lookahead for offset->curvature conversion
DISCRIM_MIN_SPEED_MPS = 1.5          # below this, yawRate/v curvature is unreliable
DISCRIM_MAX_LAG_S = 2.0              # cross-correlation search half-window
DISCRIM_ROAD_COHERENCE_MIN = 0.6     # |corr| threshold for "moving together"
DISCRIM_ARTIFACT_RESIDUAL_RATIO = 0.5  # (model-minus-lane RMS)/(lane RMS) >= this => model adds motion
DISCRIM_REPRO_FRACTION_ROAD = 0.5    # cross-pass profile corr >= this => road-reproducible
DISCRIM_REPRO_MIN_SHARED_CELLS = 4   # min shared GPS cells to compare two passes
DISCRIM_FREQ_FLAT_HZ_PER_MPH = 0.002 # |d(peakHz)/d(mph)| below this => speed-independent (loop-like)
DISCRIM_TOP_N_PER_SYMPTOM = 40       # how many worst episodes per symptom to discriminate
MPS_TO_MPH = 2.2369362920544
