# Fresh-start inventory: analysis tooling + logging (2026-07-02)

Decision context: starting fresh on **stock sunnypilot dev** (shedding the fork's tech debt).
This inventories what of the existing analysis tooling and logging is **portable**,
**coupled-but-worth-porting**, or **obsolete** — so we carry forward the real assets and
leave the debt behind. Three parallel agents surveyed core tools, legacy tools, and logging.

## Headline: the tooling is in far better shape than "tech debt" implies
**~70% of the analysis library is portable**, and the coupling is **concentrated and shallow**:
- **steerRatio 17.2 is NOT hardcoded in any code** (the 267 grep hits were logged data values).
  Every tool reads each build's own `carParams.steerRatio` / `liveParameters`. Big de-risk.
- The custom-telemetry coupling (`LC:`/`CP:`/`CX1:` parsing) is a **bolt-on that no-ops on
  stock** (those log lines just won't exist) — the tools fall back to standard-cereal paths.
- The model-bundle/route coupling is **registry/config edits** (a dict entry, a route default),
  not structural.
- **Zero cereal-schema debt** in the logging (all `logMessage` strings), so recorded fork logs
  stay readable and a fresh build simply omits the custom lines.

## Portable core — the seed for a clean `stock_lateral_toolkit/`
Works on stock logs unchanged (standard cereal + each build's own carParams):

**Stats + DSP (the crown jewels — the discipline that reversed 3+ findings this year):**
- `learned_param_studies/code/shared.py` — weave band-RMS metric + calibrated stats battery
  (exact/MC permutation, Freedman-Lane partial/within-drive spearman, BH-FDR). Verified standalone.
- `learned_param_studies/code/qa_calibration.py` + `longitudinal_weave/calib_candidates.py`
  (+ `calibrate.py`) — Monte-Carlo FPR/power harnesses + autocorrelation-robust trend-test
  toolbox ("prove the estimator before trusting significance").
- `retrospective_lateral/code/signal_utils.py` — band/low-pass filtfilt, masked RMS/p2p,
  contiguous-region, GPS cells, spectral peak. Zero coupling.

**Extractors (standard cereal → npz):**
- `retrospective_lateral/stock_compare/extract_drive.py` — **already build-agnostic; proven on
  stock this session.** The reference extractor.
- `cleanroom_lateral_analysis/analysis/extract_signals.py` — cleanest capnp-direct extractor
  (no device runtime, no custom telemetry).
- `retrospective_lateral/code/extract.py` — richer (adds liveParameters/liveCalibration/calPerc);
  needs the `cp_*`/`cx1_*` entries dropped from `REQUIRED_CACHE_CHANNELS`.

**A/B + matched-comparison frameworks (pick ONE; they overlap):**
- `retrospective_lateral/stock_compare/analyze_compare.py` — matched GPS-cell+speed-bin
  weave/centering/curve, model-independent yaw weave on GPS-classified straights. **Ready today.**
- `cleanroom_lateral_analysis/analysis/analyze_weave.py` — best-engineered A/B (drive-aware
  bootstrap + exact permutation, GPS+speed+curvature strata); coupling is metadata-only.
- `controlled_test_analysis/code/{config,extract,analyze}.py` — most rigorous pre-registered
  space-anchored A/B with audit-halts. (Overlaps the cleanroom one — keep one.)

**Detectors / reporting (standard signals):**
- `retrospective_lateral/code/`: `config.py`, `routes.py`, `metrics.py`, `discrimination.py`,
  `corridor_repro.py`, `model_vs_loop.py`, `compare.py`, `report.py`, `qa.py`, `run_all.py`,
  most of `drilldown.py`.

**Model-replay simulator ENGINE (arch-general, targets the vision→policy split that IS current
sunnypilot modeld_v2):**
- `model_replay_sim/`: `alignment.py`, `warp.py`, `env.py`, `bundles.py`, `compile_bundle.py`,
  `metrics.py`, `parse.py`, `infer.py` (the recurrent engine, parametrized by input_shapes),
  `context.py`, and `anchor.py`'s general `run_anchor`/`anchor_verdict`.

## Coupled-but-worth-porting (small, high-value edits)
- `model_replay_sim/config.py` → add one `BUNDLES` entry for the **stock model** (commit SHA +
  repo + split). *Single biggest unlock for the simulator.*
- `model_replay_sim/anchor.py` → generalize `run_cd210_anchor` → `run_anchor(stock_bundle, route)`;
  fix `run.py`/`assets.py` route defaults.
- `retrospective_lateral/code/extract.py` → drop `cp_*`/`cx1_*` from `REQUIRED_CACHE_CHANNELS`;
  make the telemetry hook optional.
- `retrospective_lateral/code/model_labels.py` → keep boot-rlog bundle-hash recovery, drop the
  Nevada/CD210/OPM7 friendly-name map.
- `retrospective_lateral/stock_compare/analyze_compare.py` → add fresh corridors to `CORRIDORS`.
- `controlled_test_analysis/code/config.py` → delete `PI_SETS`/`LC_KP_GOLDEN_MIN`.

## Obsolete — leave behind
- `retrospective_lateral/code/telemetry.py` (LC/CP/CX1 parsing + golden/weak-PI recovery) and
  the controller-stage decomposition consuming it.
- `retrospective_lateral/code/model_era_weave.py` (multi-our-bundle era comparison).
- `model_replay_sim/compare.py` (cross-our-bundle weave driver) + the CD210/Nevada/OPM7 anchor
  *records* (the anchor *methodology* is kept; the specific anchors are dead).
- All of `retrospective_lateral/incident_2026_07_01/scripts/` (26 one-offs) — harvest idioms
  (VehicleModel achieved-curvature, override-cluster detection, modeld cadence audit,
  achieved-vs-commanded partition), discard the scripts.
- `controlled_test_analysis/deploy/cc_toggle_patch.py`, `learned_param_studies/code/test2_integrator.py`,
  `longitudinal_weave/analyze.py` + `final_trend.py` (device patch / refuted questions).

## Logging — clean to walk away from; minimal to re-port
- **LC / CP / CX1** are all `carlog.info` → standard `logMessage` (via `card.py`'s
  `ForwardingHandler(cloudlog)`). **No cereal schema, no new message types, no new params for
  telemetry.** Verified: upstream ford carcontroller emits none of it.
- CX1/CP are ~70% redundant with `carState`/`carControl`/`modelV2`. Only non-recoverable content
  is the fork's internal controller state (lane_centering_integral, pi_p/pi_i, offset+curvature
  EMAs, pre/post-rate-limit, anti-windup).
- **6 custom params** (`common/params_keys.h:41-46`): `enable_lane_positioning`, `FordCurveMode`,
  `FordPath4Enabled`, `LaneBiasIntegral`, `disable_BP_long_UI`, `disable_downhill_comp_UI` +
  the `/data/lc_pi_config` file hack. **Only `LaneBiasIntegral` persists live state** (PI-integral
  warm-start, written every 10 s).
- **Re-port (only if re-adding Ford centering tuning):** one trimmed 1 Hz `carlog.info` line with
  just the internal state + the `LaneBiasIntegral` param (register in params_keys.h). No schema
  surgery. **Abandon:** CX1 (29 fields, redundant, its EPAS question is answered) and the
  `/data/lc_pi_config` hack (make it a real Param if the weak/golden distinction survives).
- **Existing logs unaffected:** standard-message tools never break; only the fork's own parsers
  depend on the LC/CP/CX1 prefixes, and recorded fork routes keep those strings.

## Reclaimable disk
~3 GB of regenerable data/caches/venv in the legacy dirs (cleanroom 2.1 G, controlled_test 560 M,
longitudinal 327 M, learned_param 34 M) — the code is <200 KB / 3,669 LOC. The raw `.zst` rlogs
under `explorer_st_logs/` are the true source; everything else regenerates.

## Effort to a working stock-analysis toolkit
- **< half a day:** `signal_utils` + `extract` (minus cp_/cx1_) + `stock_compare/*` → band-limited
  weave, centering, curve-tracking, matched A/B on stock logs. (stock_compare already works.)
- **~0.5-1 day:** simulator on stock — add the stock `BUNDLES` entry, generalize `run_anchor`,
  re-run the fidelity anchor on a route the stock model drove.
- **~1 day:** full-library re-validation on a stock corpus.
- **Total ≈ 1.5-2 days, reusing ~70% of the library.** The two tools most worth carrying — the
  model-replay anchor + `stock_compare` A/B — are exactly the ones that quantify stock-vs-custom
  before/after, which is the whole point (the deepest weave lever is the model bundle, which a
  fresh base changes).
