#!/usr/bin/env python3
"""Pre-registered analysis parameters (FROZEN before the controlled drive — see ANALYSIS_PLAN.md).
Editing these after seeing test data violates the pre-registration; the shakedown (existing data) may be
used to FINALIZE the perceptibility threshold BEFORE the test, then freeze it."""

FS = 50.0                      # resample grid (Hz) for continuous channels

# ---- frequency bands (Hz) ----
WEAVE_BAND = (0.10, 0.35)      # PRIMARY weave band (3-10 s)
SUB_BANDS = [(0.05, 0.10), (0.10, 0.20), (0.20, 0.35)]
HUNT_BAND = (0.50, 1.50)       # control band (should NOT carry the symptom)
ROAD_LP_HZ = 0.035             # low-pass to isolate slowly-varying ROAD curvature from weave

# ---- windows ----
WIN_S = 30.0
WIN_HOP_S = 30.0               # non-overlapping
MIN_ELIGIBLE_S = 20.0          # of the 30 s window

# ---- eligibility gates ----
ENGAGE_ERODE_S = 2.0           # drop engagement transitions
OVERRIDE_BUFFER_S = 1.0
BLINKER_BUFFER_S = 1.0
ROAD_CURV_ABS_MAX = 0.0015     # 1/m on lowpass(yawRate/vEgo): straight/gentle gate
LANE_PROB_MIN = 0.5
LANE_WIDTH_RANGE = (2.4, 4.6)  # m, sanity
SPEED_BAND_MPH = (40.0, 78.0)  # default; for the controlled test set this to the held setpoint band

# ---- episode metric (S3) ----
EPISODE_WIN_S = 8.0            # rolling window for "bad weave" state
EPISODE_STEER_RMS_DEG = 1.0    # steering 0.1-0.35 Hz RMS threshold for an episode (provisional)

# ---- matching / strata ----
# SPACE-ANCHORED: each 50Hz sample is snapped to a fixed global GPS grid of CELL_M, and the per-(pass x cell)
# band-RMS is computed directly over the cell's eligible samples (NO pass-relative time windows). The cell must
# hold enough seconds for the weave band, so it is LARGE. (Fixes the window(~600m)>>cell(80m) phase mismatch
# that produced zero shared strata.)
GPS_CELL_M = 500.0            # primary spatial cell (~20-30 s at highway speed); battery sweeps it
MIN_CELL_ELIGIBLE_S = 12.0   # a (pass,cell) counts only with >= this many eligible seconds (for the band)
SPEED_BIN_MPH = 2.5
CURV_BIN = 0.0015            # 1/m road-curvature stratum bin (~= the straight gate width: 1 bin on straights)
HEADING_OCTANTS = 8

# ---- statistics ----
N_BOOT = 5000
RNG_SEED = 20260613           # fixed for reproducibility (no Date/Random at runtime)
PRIMARY_UNIT = "pass"          # cluster-bootstrap & permutation over passes

# ---- PRE-REGISTERED decision thresholds (§11) ----
# PRIMARY metric is P2 (path curvature) — the symptom is visible PATH motion, and steering angle (P1) is the one
# channel attacked by the 1/v^2 speed confound. P1 is a corroborator (speed-residualized before it can gate).
PRIMARY_METRIC = "P2_curv_rms_1e4"
WIN_EFFECT_PCT = -15.0         # required % reduction on the PRIMARY (shakedown CV ~0.10 supports this; FROZEN)
WIN_SIGN_CONSISTENCY = 0.70    # fraction of matched strata agreeing in the claimed sign
WIN_PERM_P = 0.05
WIN_BATTERY_SURVIVAL = 0.80    # fraction of robustness variants that must keep sign+significance (§9 gate)
# ⚠️ MIN_PASSES_FOR_CLAIM is set by the PERMUTATION FLOOR, not just power: two-sided label-perm at n-vs-n has
# min p = 2/C(2n,n); n=3 -> 0.10 (cannot clear 0.05), n=4 -> 0.029, n=5 -> 0.008. So >=4 is the HARD floor;
# the §12 power calc (shakedown CV) wants ~6-8 for an 80%-powered -15%. DRIVE >=5 (ideally 6-8) passes/config.
MIN_PASSES_FOR_CLAIM = 4
TARGET_PASSES_FOR_POWER = 6
MIN_MATCHED_STRATA = 8

# ---- robustness battery (§9): each entry overrides one knob (run on BOTH P1 and P2) ----
ROBUSTNESS = {
    "band": [(0.08, 0.50), (0.10, 0.35), (0.12, 0.30), (0.10, 0.20), (0.20, 0.35)],
    "road_curv_abs_max": [0.0010, 0.0015, 0.0020],
    "gps_cell_m": [300.0, 500.0, 800.0],   # all >= the per-cell seconds needed for the band
    "speed_bin_mph": [2.0, 2.5, 5.0],
    "yaw_source": ["calibrated", "carstate"],
    "stat": ["median", "trim20"],
    "direction": ["both", "per"],          # crown-cancel check
}

# ---- lead-vehicle gate (§4/§6): drop samples with a lead within this headway (+- buffer) ----
LEAD_HEADWAY_S = 1.6
LEAD_BUFFER_S = 2.0
# ---- camera-calibration stability audit (§1/§6): flag if cal_yaw spread across configs exceeds this ----
CAL_YAW_MAX_SPREAD_DEG = 0.3
# ---- resample max-gap guard (§8): NaN any resampled sample farther than this from a real source sample ----
MAX_INTERP_GAP_S = 0.5

# ---- config identity (verified from telemetry, not trusted) ----
PI_SETS = {  # recovered lc_kp -> label
    "weak":   dict(lc_kp=0.0001, int_cap="fixed 0.30"),
    "golden": dict(lc_kp=0.0005, int_cap="interp 0.30->1.00"),
}
LC_KP_GOLDEN_MIN = 0.0003      # lc_kp above this => golden, else weak (5x separation makes this safe)
