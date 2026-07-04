"""Constants for the centering RCA workflow — NUMERIC TWINS of the pre-registered
values in docs/superpowers/specs/2026-07-03-centering-preregistration.md.
Pre-registration discipline: change the spec (dated addendum) BEFORE changing a
number here. Section tags (M0 §n) refer to that document."""
from pathlib import Path

CENTERING_ROOT = Path(__file__).resolve().parent
TOOLKIT_ROOT = CENTERING_ROOT.parent
REPO_ROOT = TOOLKIT_ROOT.parent
RESULTS_DIR = CENTERING_ROOT / "results"

ROUTE = "route_stock05"        # the only stock route with LOCAL camera files (12 segs)
CACHE_NPZ = REPO_ROOT / "retrospective_lateral" / "results" / "cache" / f"{ROUTE}.npz"
TOOLKIT_CACHE = TOOLKIT_ROOT / "cache"

# Second-route extension (Task 10 step 1b): extra routes with local camera files.
# Per-route paths keep the primary route's historical locations byte-identical.
ROUTES_EXTRA = ("route_stock04",)


def cache_npz(route: str) -> Path:
    return REPO_ROOT / "retrospective_lateral" / "results" / "cache" / f"{route}.npz"


def m1_dir(route: str) -> Path:
    return RESULTS_DIR / ("m1" if route == ROUTE else f"m1_{route}")


def windows_json(route: str) -> Path:
    return RESULTS_DIR / "m2" / ("windows.json" if route == ROUTE else f"windows_{route}.json")

# M0 §1 canonical sign: offset > 0  ==  vehicle LEFT of lane center
#                        ==  lane-center midpoint y > 0 in calibrated frame (y=+RIGHT).

# --- M0 §3 method-agreement tolerances (medians over overlap samples, meters) ---
TOL_M1_VS_M2_M = 0.06
TOL_M2_VS_LOGGED_M = 0.04
TOL_CONSENSUS_DISAGREE_M = 0.10

# --- M0 §2 M1 sampling + annotation ---
N_FRAMES_TARGET = 80
N_FRAMES_MIN = 40
MIN_FRAME_SEPARATION_S = 5.0
SAMPLER_SEED = 20260703
EVAL_DISTANCES_M = (8.0, 12.0, 16.0)
LANE_SEARCH_BAND_M = (1.0, 3.4)     # |y_road| band searched per side
MARK_WIDTH_M = 0.125                # nominal paint width
MIN_CONTRAST = 4.0                  # matched-filter peak vs row MAD
REVIEW_EVERY_N = 5                  # deterministic ~20% review subset
REVIEW_ACCEPT_MIN = 0.85            # M0 §2 trust rule
PX_CORRECT_TOL = 5.0                # reviewer correction <= this counts as accept
CAM_OFFSET_FROM_CENTERLINE_M = 0.0  # USER-MEASURED lever arm, + = camera RIGHT of centerline
CAM_OFFSET_MEASURED = False         # m1_offsets refuses vehicle-frame output while False
SIGMA_PIXEL_M = 0.03
SIGMA_ROLL_M = 0.03
SIGMA_YAW_M = 0.035
SIGMA_MISC_M = 0.02

# --- M1 eligibility (model-independent straightness via GPS heading rate) ---
V_MIN_MPS = 8.0
SPEED_BINS_MPS = ((8.0, 15.0), (15.0, 22.0), (22.0, 30.0))
HEADING_BIN_DEG = 90.0
STRAIGHT_CURV_MAX = 0.0005          # 1/m
LANE_PROB_MIN = 0.6

# --- M2 shared scene windows ---
BUNDLES_M2 = ("SP002", "Nevada", "CD210")
N_WINDOWS = 2
WARMUP_FRAMES = 200                 # 10 s @ 20 Hz, matches the anchor convention
COMPARE_FRAMES = 1200
MIN_COMPARE_FRAMES = 600            # window 2 fallback if the eligible run is short
MAX_WORKERS_REPLAY = 4              # each worker holds a compiled model; raise to 6 if RAM allows
LANE_CENTER_EVAL_X_M = (0.0, 10.0)

# --- M0 §5 S1 sweep ---
SWEEP_OFFSETS_M = tuple(round(-0.10 + 0.02 * i, 3) for i in range(11))  # -0.10..+0.10
CONTROL_OFFSETS_M = (-0.005, 0.005) # negative-control points -> noise band NB
WEAVE_NOISE_FLOOR = 0.03
WEAVE_HARD_CAP = (0.85, 1.15)
CURVE_CORR_MIN = 0.98
LOW_BAND_HZ = 0.05
LOW_BAND_RATIO = (0.95, 1.05)
SLOPE_UNIT_RANGE = (0.5, 1.5)
MONOTONIC_SPEARMAN_MIN = 0.90
DETERMINISM_TOL = 1e-9

# --- M0 §4 decision-tree thresholds ---
P_MEANINGFUL_M = 0.05
DMID_MEANINGFUL_M = 0.05
DWIDTH_COHERENT_M = 0.10
CROWN_P_MAX = 0.05
CROWN_R_MIN = 0.30
CORRIDOR_SPREAD_M = 0.05
CROWN_DOMINANT_FRACTION = 0.5

# --- R2 estimator ---
R2_WINDOW_S = 30.0
R2_FPR_MAX = 0.07
FS_HZ = 20.0
