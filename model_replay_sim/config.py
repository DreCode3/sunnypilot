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
# img_buffer_length: the device vision warp keeps a rolling buffer of this many frames and
# feeds the model cat(buffer[:6], buffer[-6:]) = the OLDEST+NEWEST frame, i.e. the img channels
# are frames t-(img_buffer_length-1) and t (selfdrive/modeld/compile_warp.py:101;
# buffer_length = 5 if is_20hz else 2, sunnypilot/modeld_v2/modeld.py:75). 5 is the modern
# 20Hz value; CD210=5 is ANCHOR-VALIDATED (sep-sweep: sep=4 -> band_ratio 1.008, corr 0.993).
# Nevada/OPM7 default to 5 (assumed 20Hz) -- NOT anchor-validated (no same-model anchor).
# anchor_validated: True for a bundle whose img_buffer_length=5 and post-step are proven by a
# same-model fidelity anchor. CD210: route_b5 corr 0.993, band_ratio 1.008. Nevada: route_c5
# corr 0.990, band_ratio 1.060 (validated 2026-06-29 after pulling route_c5 video). OPM7 has
# NO clean same-model anchor (confirmed-OPM7 routes rotated off the device; route_7f corr 0.925
# unconfirmed-build) -> its img_buffer_length/mlsim are UNVERIFIED; replaying it warns loudly
# (infer.replay_window) because its weave number may be systematically wrong.
BUNDLES = {
  "CD210":  {"full_sha": "55f66e2246359c6593605399a0199d94d13ad90d", "repo": "commaai/openpilot",     "split": False, "internal_names": {"C210M", "CD210"}, "img_buffer_length": 5, "anchor_validated": True},
  "Nevada": {"full_sha": "3193eac5e385aa010694a8ac192ff38ffe000193", "repo": "commaai/openpilot",     "split": False, "internal_names": {"NM", "Nevada"}, "img_buffer_length": 5, "anchor_validated": True},
  "OPM7":   {"full_sha": "052692b25d63c5ddda276b5c2271383b6aff129f", "repo": "sunnypilot/sunnypilot", "split": True,  "internal_names": {"OPM7"}, "img_buffer_length": 5, "anchor_validated": False},
  # SP002 = the DEFAULT bundled model of stock sunnypilot dev v2026.002.000 (device commit
  # 3f8e959..., a squashed release whose message pins master commit 31dc4d8e... — that master
  # commit ships the source ONNX as LFS). Drive-time rlog initData confirms no
  # ModelManager_ActiveBundle param -> the default bundle ran the stock drives. New layout:
  # vision + on_policy ONLY (off_policy retired after OPM7; big_* variants not used here).
  # ONNX oids differ from Nevada AND OPM7 -> genuinely new model.
  # ANCHOR-VALIDATED 2026-07-03 on route_stock05 (the drive it actually drove): corr 0.9986,
  # band_ratio 0.993, 2059/2059 frames, lag-0 peaked — best anchor in the program; also
  # confirms img_buffer_length=5/is_20hz and that this exact build made the stock drives.
  "SP002":  {"full_sha": "31dc4d8e520f3b309acd05da79cc6ab448affad2", "repo": "sunnypilot/sunnypilot", "split": True,
             "models": ["driving_vision.onnx", "driving_on_policy.onnx"],
             "internal_names": {"SP002"}, "img_buffer_length": 5, "anchor_validated": True},
}
