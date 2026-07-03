"""Shared constants for the stock lateral toolkit.

Seeded 2026-07-03 from the fork-era analysis library (values unchanged from
retrospective_lateral/code/config.py / learned_param_studies/code/shared.py —
these were validated by the FPR/power calibration harnesses; do not tweak
casually).
"""
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLKIT_ROOT.parent
CACHE_DIR = TOOLKIT_ROOT / "cache"

# NAS log archive (see memory reference_nas_storage — capital P matters on the mount)
NAS_LOG_ROOT = Path("/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs")

# DSP / weave band (the validated slow-weave band; hunt band kept for reference)
FS_HZ = 20.0                       # modelV2 cadence used by extract_drive.py
MAX_INTERP_GAP_S = 0.5
DEFAULT_WEAVE_BAND_HZ = (0.10, 0.35)
HUNT_BAND_HZ = (0.50, 1.50)

# Matched-comparison geometry
GPS_CELL_M = 80.0
HEADING_BIN_DEG = 45.0
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON_AT_EQUATOR = 111_320.0

# Windowing (>=30 s so the 0.10 Hz band edge is resolvable)
WEAVE_WINDOW_S = 30.0
MIN_WEAVE_ELIGIBLE_S = 20.0
