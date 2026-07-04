# Centering RCA + Offline Solution Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine the definitive root cause of "stock centers worse" (logged +0.10–0.15 m left, model-frame) via video ground truth + cross-model consensus, and validate a CameraOffset compensation offline (dose-response counterfactual sweep in the anchor-validated replay simulator) before any test drive.

**Architecture:** Four phases from the authoritative spec (`docs/superpowers/specs/2026-07-03-f3-alert-only-and-centering-workflow-design.md`, Workflow 2): **M** builds three independent measurements of lateral offset (M1 video ground truth from local `route_stock05` camera files, M2 same-frame cross-model lane-line replay via `model_replay_sim/`, M3 reconciliation against logs) with tolerances pre-registered FIRST (M0); **R** applies a pre-registered decision tree (translation vs preference vs crown) to classify the cause; **S** wires CameraOffset into the replay sim (S0 — the shear already exists in `warp.py`, the plumbing gap is a `replay_window` override) and sweeps it −0.10…+0.10 m (S1); **D** is a protocol *document only* — the USER drives. All new analysis code lives in `stock_lateral_toolkit/centering/`, small single-purpose files, reusing the calibrated stats battery (`stock_lateral_toolkit/shared.py`) and the replay sim.

**Tech Stack:** Python 3.11 (`.venv311` — numpy/scipy/pandas present; **pillow installed in Task 2**; no cv2/matplotlib), tinygrad CPU replay (`model_replay_sim/`, SP002/Nevada/CD210 all anchor-validated), openpilot `LogReader`/`FrameReader`, `common/transformations` (pinhole camera + ground-plane homography), pytest.

---

## Verified facts this plan builds on (checked 2026-07-03 in this repo, branch `stock-2026.002.001-fresh-start`)

- **Production CameraOffset math** (`git show 3f8e95952aa342cab21c232462afb0da88e2263e:sunnypilot/modeld_v2/camera_offset_helper.py`):
  ```python
  @staticmethod
  def apply_camera_offset(model_transform, intrinsics, height, offset_param):
    cy = intrinsics[1, 2]
    shear = np.eye(3, dtype=np.float32)
    shear[0, 1] = offset_param / height
    shear[0, 2] = -offset_param / height * cy
    model_transform = (shear @ model_transform).astype(np.float32)
    return model_transform
  ```
  Applied to BOTH cams in `CameraOffsetHelper.update` with `height = liveCalibration.height[0] or 1.22`; the live param is EMA'd (`actual = 0.9*actual + 0.1*param`, steady state = param). Hook site at the stock commit: `sunnypilot/modeld_v2/modeld.py:291` (`camera_offset_helper.set_offset(params.get("CameraOffset", ...))`) and `:298` (`update(...)`). Param: `common/params_keys.h:236` `{"CameraOffset", {PERSISTENT | BACKUP, FLOAT, "0.0"}}`; it was 0.0 on all stock drives. No downstream centering — controlsd follows `modelV2.action.desiredCurvature` directly.
- **The sim already applies the shear**: `model_replay_sim/warp.py:71-82` reproduces `apply_camera_offset` verbatim; `warp.model_transform(ctx, wide, camera_offset=None)` applies it with `camera_offset` defaulting to `ctx.camera_offset` (read from boot-rlog params by `model_replay_sim/context.py:route_context`, `RouteContext.camera_offset`, default 0.0). **The gap:** `model_replay_sim/infer.py:replay_window` (lines 519-521) calls `model_transform(ctx, wide=False)` / `(ctx, wide=True)` with NO override parameter — counterfactuals are impossible without Task 3 (S0).
- **Lane lines come from the VISION model** for all three bundles in scope. Materialized metadata (`retrospective_lateral/results/model_replay/onnx/<bundle>/driving_vision_metadata.pkl`, inspected 2026-07-03): SP002/CD210/Nevada `driving_vision` `output_slices` all contain `lane_lines: slice(117, 645)` (528 floats) and `lane_lines_prob: slice(645, 653)` (8 floats); SP002's `driving_on_policy` has only `{plan, desire_state, pad}`. The production parser (`sunnypilot/modeld_v2/parse_model_outputs_split.py:120-124`) does `parse_mdn('lane_lines', in_N=0, out_N=0, out_shape=(4, 33, 2))` → parsed `lane_lines` shape `(1, 4, 33, 2)` (`[..., 0]`=y, `[..., 1]`=z, plus `lane_lines_stds` same shape) and `parse_binary_crossentropy('lane_lines_prob')` → sigmoid `(1, 8)`, probabilities at `[0, 1::2]` (`fill_model_msg.py:130`). Line index 1 = inner-left, 2 = inner-right (`fill_model_msg.py:63-66` uses `lane_lines[1].y[0]` / `lane_lines[2].y[0]`). `infer.ReplayState.step` already computes `vision_out = self.model.run_vision(...)` per frame — it just drops everything except `hidden_state`; Task 4 (M2a) captures it.
- **Sign convention (empirically pinned)**: calibrated/model frame is `[Forward, Right, Down]` (`common/transformations/README.md`). In `route_stock05` cache: median `lane_left_y0 = −1.461`, `lane_right_y0 = +1.715`, `lane_center_y0 = +0.103` — y is positive-RIGHT, so midpoint +0.10 ⇒ lane center is right of camera ⇒ **car sits ~0.10 m LEFT of the detected lane center on stock05** (matches the +0.15 m report figure, `docs/superpowers/reports/2026-07-02-stock-vs-custom-lateral-comparison.md` centering section: identical detected lane widths Δ≤2 cm; physical position unidentifiable from logs).
- **Assets**: `explorer_st_logs/route_stock05/` = 12 local segments `00000005--ef46fdca62--*` WITH fcamera/ecamera (the only local stock video). NAS (`/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock/`) rlogs: `00000000--c181384b0c` (sparse: segs 0,10,20,30,38), `00000001--e80958d9ca` (sparse: 0,9,18), `00000002--5da5840d8d` (FULL, 64 segs), `00000003--b37c613b41` (sparse: 0,4,7), `00000004--3d3385646d` (FULL, 26 segs), `00000005--ef46fdca62` (FULL, 12 segs). Toolkit caches exist for 04/05 (`stock_lateral_toolkit/cache/stock_hiram_{04,05}.npz`); retrospective cache `retrospective_lateral/results/cache/route_stock05.npz` (13142 frames @20 Hz) has `lane_left_y0/lane_right_y0/lane_center_y0`, `lane_prob_left/right`, `cal_yaw/cal_pitch/cal_roll` (radians; medians −0.0498/+0.0429/−0.00002), **`roll`** (93% finite, median +0.0244 rad — sourced from `liveLocationKalman.orientationNED.value[0]`, `retrospective_lateral/code/extract.py:360`, so stock rlogs DO carry it), `lat/lon`, `v_ego`, `lat_active`, `mono_time`.
- **Replay cost baseline**: ~0.41 s/frame/model on the M4 (route_7f 6945 frames ≈ 47 min, commit 39364bbc55). SP002 anchor on route_stock05: corr 0.9986, band_ratio 0.993, 2059 compare frames (`model_replay_sim/config.py:34-45`).
- **Camera**: comma four = (`mici`, `os04c10`) → `DEVICE_CAMERAS[("mici","os04c10")].fcam` = 1344×760, focal 1141.5 px, cx=672, cy=380 (`common/transformations/camera.py:50-52,19-26`). Ground-plane helper: `get_view_frame_from_road_frame(roll, pitch, yaw, height)` (`camera.py:85-89`; road frame = x fwd, y LEFT, z up; feed `liveCalibration.rpyCalib` directly — `device_from_road = R(rpy) @ diag([1,−1,−1])` is exactly `device_from_calib @ calib_from_road` since stock `cal_roll ≡ 0`).

## Dependency graph / parallelism

```
T1 (M0 prereg) ─► everything
T2 (scaffold) ─► T3..T17
T3 (S0 offset wiring) ─► T4 ─► T5 ─► { T7 (sampler), T10 (M2 replays), T17 (S1 sweep) }
T6 (ground_plane) ─► T8 (annotate) ─► T9 (m1 offsets) ─► T11 (M3), T14 (R1)
T10 ─► T11, T14        T12 (caches+roll) ─► T13 (R2), T15 (R3) ─► T16 (R verdict)
T16 + T17 ─► T18 (S2 decision) ─► T19 (D protocol doc)
```
After T5, three tracks run in parallel: M1 (T6–T9), M2 (T10), S1 (T17). Compute-heavy replays (T10, T17) are process-parallel (4 workers default); everything else is seconds-to-minutes on cached npz.

**Compute budget (honest, at 0.41 s/frame/model, windows ≈ 1400 + ~900 frames):**
- T10 M2: 3 bundles × ~2300 frames ≈ 47 min serial → ~18–20 min wall at 3–4 workers.
- T17 S1: 13 offset points × ~2300 frames + 1 determinism repeat ≈ 3.6 h serial → ~55–70 min wall at 4 workers (bump `MAX_WORKERS_REPLAY` to 6 if RAM comfortably allows; each worker holds a compiled SP002 vision+policy pkl).
- T8 annotation: minutes of compute + **~20–30 min of USER review time**.
- T12 extraction: ~5–15 min per full drive (12-way process pool, NAS reads).

**USER-GATED points (loud, do not skip):** T8 step 8 (annotation spot-review), T9 step 2 (tape-measure camera lever-arm), optional T10 step 1b (device video pull for a second route), all of T19's drives (user executes; F3 must be live first).

---

## File Structure

```
docs/superpowers/specs/2026-07-03-centering-preregistration.md        NEW  (T1) M0: tolerances, decision tree, desk-exit, D criteria
docs/superpowers/plans/2026-07-03-centering-road-ab-protocol.md       NEW  (T19) D-phase drive protocol (user executes)
model_replay_sim/infer.py                                             MOD  (T3,T4) window_transforms + camera_offset/capture_outputs in replay_window; ReplayState stashes last outputs
model_replay_sim/tests/test_infer_offset_capture.py                   NEW  (T3,T4) shear-vs-production + capture unit tests
stock_lateral_toolkit/extract_drive.py                                MOD  (T12) append `roll` column (liveLocationKalman.orientationNED)
stock_lateral_toolkit/centering/__init__.py                           NEW  (T2)
stock_lateral_toolkit/centering/config.py                             NEW  (T2) numeric twins of M0 (single source in code)
stock_lateral_toolkit/centering/windows.py                            NEW  (T5) shared scene-window selection (M1/M2/S1 use identical frames)
stock_lateral_toolkit/centering/ground_plane.py                       NEW  (T6) pixel<->road-plane homography (M1 math)
stock_lateral_toolkit/centering/frame_sampler.py                      NEW  (T7) stratified M1 frame sampling -> frames_manifest.csv
stock_lateral_toolkit/centering/annotate.py                           NEW  (T8) lane-mark proposals + overlay PNGs + review sheet
stock_lateral_toolkit/centering/m1_offsets.py                         NEW  (T9) trust rule + physical offsets + sigma budget -> m1_results.json
stock_lateral_toolkit/centering/smoke_replay.py                       NEW  (T3,T4) re-runnable integration gate for offset+capture
stock_lateral_toolkit/centering/consensus.py                          NEW  (T10) M2 cross-model replay driver + consensus stats -> m2_results.json
stock_lateral_toolkit/centering/reconcile.py                          NEW  (T11) M3 decomposition + verdict table -> m3_verdict.{json,md}
stock_lateral_toolkit/centering/r1_discriminator.py                   NEW  (T14) translation-vs-preference per-frame comparison
stock_lateral_toolkit/centering/r2_calibration.py                     NEW  (T13) FPR/power harness for the R2 estimator (must pass first)
stock_lateral_toolkit/centering/r2_crown.py                           NEW  (T13) offset~roll within-drive + direction-paired secondary
stock_lateral_toolkit/centering/r3_dependence.py                      NEW  (T15) corridor/speed/direction breakdown table
stock_lateral_toolkit/centering/r_verdict.py                          NEW  (T16) mechanical decision-tree application -> r_verdict.{json,md}
stock_lateral_toolkit/centering/s1_sweep.py                           NEW  (T17) CameraOffset dose-response sweep + gates -> s1_report.{json,md}
stock_lateral_toolkit/centering/tests/__init__.py                     NEW  (T2)
stock_lateral_toolkit/centering/tests/conftest.py                     NEW  (T2) repo root on sys.path
stock_lateral_toolkit/centering/tests/test_ground_plane.py            NEW  (T6)
stock_lateral_toolkit/centering/tests/test_windows.py                 NEW  (T5)
stock_lateral_toolkit/centering/tests/test_frame_sampler.py           NEW  (T7)
stock_lateral_toolkit/centering/tests/test_annotate.py                NEW  (T8) synthetic-image recovery test
stock_lateral_toolkit/centering/tests/test_m1_offsets.py              NEW  (T9) trust-rule + sign-convention tests
stock_lateral_toolkit/centering/tests/test_consensus.py               NEW  (T10) lane-center math on synthetic capture
stock_lateral_toolkit/centering/tests/test_reconcile.py               NEW  (T11)
stock_lateral_toolkit/centering/tests/test_r_verdict.py               NEW  (T16)
stock_lateral_toolkit/centering/tests/test_s1_gates.py                NEW  (T17) gate logic on synthetic dose-response
stock_lateral_toolkit/centering/results/                              NEW  (gitignored) all generated artifacts (manifests, overlays, npz, reports)
.gitignore                                                            MOD  (T2) add centering/results/
```

Conventions used by every module below:
- **Canonical offset sign** (M0 §1): `offset > 0` ⇔ vehicle sits LEFT of lane center ⇔ lane-center midpoint y > 0 in the calibrated frame (y = +RIGHT). This is identical to the logged `lane_center_y0` and to `stock_lateral_toolkit/extract_drive.py`'s `offset` column.
- All heavy artifacts go under `stock_lateral_toolkit/centering/results/` (gitignored); JSON verdicts/gates are small and human-diffable; each analysis script prints its verdict AND writes it.
- Tests: `.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests -v` (conftest puts repo root on `sys.path`; `stock_lateral_toolkit` is a PEP-420 namespace package — do NOT add `stock_lateral_toolkit/__init__.py`, other tools import its modules flat). Sim tests: `.venv311/bin/python -m pytest model_replay_sim/tests/test_infer_offset_capture.py -v`.

---

### Task 1: M0 — Pre-registration document (BEFORE any measurement)

**Files:**
- Create: `docs/superpowers/specs/2026-07-03-centering-preregistration.md`

No code. This document is the contract; Task 2's `config.py` mirrors its numbers. It must be committed before any T6+ analysis produces a number, so results cannot influence thresholds.

- [ ] **Step 1: Write the pre-registration file with EXACTLY this content**

```markdown
# Pre-registration: centering RCA + offline validation (2026-07-03)

Locked BEFORE any M/R/S measurement is computed. Amendments require a dated
addendum section explaining what was known at amendment time. Numeric twins
live in `stock_lateral_toolkit/centering/config.py`.

## §1 Definitions and sign convention
- Canonical offset: `offset > 0` ⇔ vehicle LEFT of lane center ⇔ lane-center
  midpoint y > 0 in the calibrated frame (y = +RIGHT; empirically pinned:
  route_stock05 median lane_left_y0 = −1.461, lane_right_y0 = +1.715).
- `L` = logged model-frame offset = median lane_center_y0 over eligible frames
  (engaged, no-press, straight, lane_prob ≥ 0.6, 8–30 m/s). Known ≈ +0.10 m on
  stock05; per-corridor values from R3.
- `P` = TRUE physical offset (M1 video, vehicle-frame after lever-arm).
- `Δmid` = SP002 perception bias = median(SP002 replayed lane-mid y − video
  lane-mid y) over M1-overlap frames (both camera-relative). Identity used for
  decomposition: **L = P_cam + Δmid** where P_cam = camera-relative P.
- Settling model for the S-phase lever: with CameraOffset δ, the model's
  perceived midpoint shifts by s·δ (slope s measured in S1); on road the loop
  settles where the perceived offset equals its preferred value, so
  settled_true_offset(δ) = P − s·δ, and the compensation target is δ* = P / s.

## §2 M1 measurement + per-frame uncertainty (declared, not fit)
Flat-road ground-plane back-projection at 8/12/16 m using fcam pinhole
intrinsics (1344×760, f=1141.5) + per-frame liveCalibration rpy + height.
1σ terms (m, lateral, per frame): pixel/annotation 0.03; unmodeled roll
(crown/suspension, calib roll≡0) 0.03 (≈ ε_roll·h with ε≈0.024 rad, h≈1.22);
calib-yaw residual 0.035 (≈ d·0.003 at 12 m); distortion/misc 0.02.
Combined per-frame σ ≈ 0.06; random part shrinks as 1/√N (N ≥ 40 required);
systematic part (lever-arm mismeasure, mean roll, mean yaw) ~±0.04 does NOT
shrink and bounds all agreement tolerances below.
- Annotation trust rule: USER reviews a deterministic ≥20% subset; batch is
  trusted iff ≥85% of reviewed proposals are accepted or corrected by ≤5 px.
  Otherwise: full manual annotation pass (no partial trust).
- Minimum sample: ≥40 usable frames, ≥3 speed bins, ≥2 heading quadrants.

## §3 Method-agreement tolerances (evaluated in M3, on overlap samples, medians)
- |M1 − M2_SP002| ≤ 0.06 m  (video vs replayed SP002 lane-mid, camera-relative)
- |M2_SP002 − logged| ≤ 0.04 m  (replay fidelity; anchor corr was 0.9986)
- Cross-model pairwise |Δcenter| medians are REPORTED; a pair > 0.10 m means
  the models define lane-center differently (supports definition-bias reading).
- If |M1 − M2_SP002| > 0.06 m: STOP. No R-phase verdict may be issued; the
  discrepancy is itself the finding and must be diagnosed first.

## §4 R-phase decision tree (mechanical; r_verdict.py implements verbatim)
Evidence inputs: P, Δmid, Δwidth (SP002−video lane width), L; R2 within-drive
offset~roll (calibrated permutation p, Spearman r); R3 per-corridor spread.
1. Method gate (§3) must pass.
2. Components:
   - PERCEPTION-TRANSLATION component T = Δmid if |Δmid| ≥ 0.05 AND
     |Δwidth| ≤ 0.10 (coherent shift, not scale); else T = 0.
   - TRUE-OFFSET component = P (physical, vehicle-frame).
   - CROWN component: present iff R2 p < 0.05 (FPR-calibrated estimator only)
     AND |r| ≥ 0.30 AND R3 corridor spread > 0.05 m aligned with roll sign.
3. Class (primary label; fractions reported alongside):
   - |P| ≤ 0.05 and |Δmid| ≥ 0.05  → MODEL-FRAME DEFINITION ONLY
     (car physically fine; logged offset is a perception artifact).
   - |P| ≥ 0.05 and |Δmid| < 0.05  → TRAINED PATH PREFERENCE
     (model sees the lane correctly and tolerates sitting off-center).
   - |P| ≥ 0.05 and |Δmid| ≥ 0.05 (same sign) → MIXED translation+preference.
   - CROWN RESPONSE label added when crown test fires; crown fraction =
     (regression slope × median |roll|) / |L|.
   - |P| ≤ 0.05 and |Δmid| < 0.05 → NO DEFICIT MEASURABLE (subjective report
     unexplained; D-phase optional, S-phase halted).
4. Solution branch: PHYSICAL TRANSLATION / TRAINED PREFERENCE / MIXED →
   CameraOffset compensation (S1-sized, δ* = P/s). CROWN RESPONSE dominant
   (crown fraction > 0.5) → CameraOffset NOT the designed solution; a
   crown-aware approach requires a NEW spec (explicitly out of this plan).

## §5 S1 desk-exit criteria (offline gates; ALL must pass to propose δ*)
- Dose-response: lane-center shift monotonic in offset (Spearman |ρ| ≥ 0.90
  across the 11 sweep points, both windows) and slope s within 0.5–1.5 of unit
  response (|d center / d offset| ∈ [0.5, 1.5]).
- Sizing: δ* = P/s must lie within the swept range [−0.10, +0.10] and its
  predicted improvement s·δ* ≥ 0.7·P (i.e., ≥70% of the M-measured deficit).
- Weave gate: |band_ratio(δ) − 1| ≤ NB for δ ∈ {δ*, neighbors}, where
  band_ratio = weave-band(0.10–0.35 Hz) RMS of replayed desiredCurvature at δ
  over that at 0, and NB = max(0.03, 2·max|band_ratio(±0.005 m control) − 1|),
  widened by any nonzero determinism-repeat spread. HARD CAP regardless of NB:
  band_ratio ∈ [0.85, 1.15] (the anchor tolerance).
- Curve/road gate: corr(desired_curv(δ), desired_curv(0)) ≥ 0.98 per window
  AND low-band (≤0.05 Hz) RMS ratio ∈ [0.95, 1.05]. (True sharp-curve tracking
  is NOT testable offline — eligible replay scenes are gentle; covered in D.)
- DC shift of desiredCurvature is reported (informational, no gate).

## §6 D-phase success criteria (road A/B, USER drives, F3 live first)
- Interleaved same-corridor A/B (CameraOffset 0 vs δ*), ≥3 passes per arm per
  direction, alternating, speed-matched (fixed cruise set-speed).
- PRIMARY (video, because the running model's own logged offset CANNOT show
  the improvement — at settle it reads its preferred value by construction):
  M1 pipeline re-run on frames sampled from each arm; treated-arm median
  physical |offset| ≤ 0.05 m OR reduced by ≥70% of P, sign as predicted.
- Weave: matched-cell band|yawRate| (0.10–0.35 Hz) delta not significant
  (calibrated test) and point estimate ≤ +15% vs control arm.
- Safety: zero F3 departure alerts attributable to the offset in either arm;
  minimum |distance to nearer lane line| (logged) not reduced by > 0.05 m.
- Subjective: user reports centering no worse (goal: better).

## §7 Analysis discipline
- Estimators with p-values must pass an FPR calibration harness first
  (`r2_calibration.py`; target FPR ≤ 0.07 at α=0.05, qa_calibration pattern).
- Lateral A/B rules per memory `feedback_lateral_ab_metrics` (GPS-cell +
  speed-bin matching, band-limited, robust medians; never pooled variance,
  never aLat-primary). Eligibility masks are re-derived per analysis.
- Every agent-produced number in gate reports is independently re-computed
  before a phase verdict is accepted (memory `feedback_verify_agent_outputs`).
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-07-03-centering-preregistration.md
git commit -m "centering M0: pre-registered tolerances, decision tree, desk-exit and road criteria

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Scaffold — package, config (numeric twins of M0), pillow, gitignore

**Files:**
- Create: `stock_lateral_toolkit/centering/__init__.py`, `stock_lateral_toolkit/centering/config.py`, `stock_lateral_toolkit/centering/tests/__init__.py`, `stock_lateral_toolkit/centering/tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Install pillow into the analysis venv (overlay PNGs; no cv2/matplotlib present)**

```bash
.venv311/bin/pip install pillow
.venv311/bin/python -c "import PIL; print('PIL', PIL.__version__)"
```
Expected: prints a PIL version.

- [ ] **Step 2: Create the package files**

`stock_lateral_toolkit/centering/__init__.py`:
```python
"""Centering RCA + offline-validation workflow (2026-07-03).
Pre-registration: docs/superpowers/specs/2026-07-03-centering-preregistration.md
Plan: docs/superpowers/plans/2026-07-03-centering-rca-offline-validation.md"""
```

`stock_lateral_toolkit/centering/tests/__init__.py`: empty file.

`stock_lateral_toolkit/centering/tests/conftest.py`:
```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

`stock_lateral_toolkit/centering/config.py`:
```python
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
```

- [ ] **Step 3: Gitignore the results tree**

Append to `.gitignore`:
```
stock_lateral_toolkit/centering/results/
```

- [ ] **Step 4: Sanity-check imports**

```bash
.venv311/bin/python -c "from stock_lateral_toolkit.centering import config as CC; print(CC.SWEEP_OFFSETS_M)"
```
Expected: `(-0.1, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.1)`

- [ ] **Step 5: Commit**

```bash
git add stock_lateral_toolkit/centering/__init__.py stock_lateral_toolkit/centering/config.py \
        stock_lateral_toolkit/centering/tests/__init__.py stock_lateral_toolkit/centering/tests/conftest.py .gitignore
git commit -m "centering: package scaffold + config (numeric twins of M0 pre-registration)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: S0 — CameraOffset counterfactual wiring in the replay sim (with unit test vs production math)

**Files:**
- Modify: `model_replay_sim/infer.py` (add `window_transforms`; extend `replay_window` signature)
- Create: `model_replay_sim/tests/test_infer_offset_capture.py`
- Create: `stock_lateral_toolkit/centering/smoke_replay.py` (integration gate, extended in Task 4)

**Resolved unknown (verified in this session):** the sim's warp path DOES apply the production shear — `model_replay_sim/warp.py:71-82` `apply_camera_offset` is a verbatim copy of `CameraOffsetHelper.apply_camera_offset` (both do `shear[0,1]=offset/height; shear[0,2]=-offset/height*cy; (shear @ M).astype(float32)` with `cy=intrinsics[1,2]`), and `warp.model_transform(ctx, wide, camera_offset=None)` applies it, defaulting to `ctx.camera_offset` (the route's logged param, 0.0 on all stock drives). **What's missing is only the override plumbing:** `infer.replay_window` builds `M_main/M_extra` via `model_transform(ctx, wide=...)` with no way to pass a counterfactual offset. Production's EMA (`0.9*actual + 0.1*param` per model frame, `camera_offset_helper.py`) converges to the param; a static-scene replay uses the steady value directly (already documented in `warp.model_transform`'s docstring).

- [ ] **Step 1: Write the failing tests**

`model_replay_sim/tests/test_infer_offset_capture.py`:
```python
"""S0/M2a tests: CameraOffset counterfactual plumbing + per-frame output capture.

The shear test compares the SIM's transform against the PRODUCTION
CameraOffsetHelper math (imported from sunnypilot/modeld_v2) — the fidelity
claim of the whole S1 sweep rests on this equivalence.
"""
import inspect
import types

import numpy as np
import pytest


def _fake_ctx(camera_offset=0.0):
    # rpy/height from route_stock05 (medians); device = comma four
    return types.SimpleNamespace(
        rpy_calib=[0.0, 0.0429, -0.0498],
        height=1.22,
        device_type="mici",
        road_sensor="os04c10",
        camera_offset=camera_offset,
    )


def test_window_transforms_shear_matches_production():
    from model_replay_sim.infer import window_transforms
    from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
    from openpilot.common.transformations.camera import DEVICE_CAMERAS

    ctx = _fake_ctx()
    dc = DEVICE_CAMERAS[("mici", "os04c10")]

    M0_main, M0_extra, used0 = window_transforms(ctx, camera_offset=0.0)
    Mx_main, Mx_extra, usedx = window_transforms(ctx, camera_offset=0.04)

    assert used0 == 0.0 and usedx == 0.04
    exp_main = CameraOffsetHelper.apply_camera_offset(M0_main, dc.fcam.intrinsics, 1.22, 0.04)
    exp_extra = CameraOffsetHelper.apply_camera_offset(M0_extra, dc.ecam.intrinsics, 1.22, 0.04)
    np.testing.assert_allclose(Mx_main, exp_main, atol=1e-6)
    np.testing.assert_allclose(Mx_extra, exp_extra, atol=1e-6)
    # offset must actually change the matrix
    assert not np.allclose(M0_main, Mx_main)


def test_window_transforms_default_uses_ctx_offset():
    from model_replay_sim.infer import window_transforms
    ctx = _fake_ctx(camera_offset=0.07)
    M_none, _, used = window_transforms(ctx, camera_offset=None)
    M_explicit, _, _ = window_transforms(ctx, camera_offset=0.07)
    assert used == pytest.approx(0.07)
    np.testing.assert_array_equal(M_none, M_explicit)


def test_replay_window_signature_has_new_params():
    from model_replay_sim.infer import replay_window
    params = inspect.signature(replay_window).parameters
    assert "camera_offset" in params and params["camera_offset"].default is None
    assert "capture_outputs" in params and params["capture_outputs"].default == ()
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_infer_offset_capture.py -v
```
Expected: FAIL / ERROR with `ImportError: cannot import name 'window_transforms'` (and the signature test failing with `KeyError`/`AssertionError`).

- [ ] **Step 3: Implement in `model_replay_sim/infer.py`**

(a) Add above `replay_window` (after `_route_window_v_ego`):
```python
def window_transforms(ctx, camera_offset: float | None = None):
    """The per-window forward warp matrices (M_main, M_extra) + the offset actually used.

    ``camera_offset=None`` -> the route's own logged param (``ctx.camera_offset``); an
    explicit float REPLACES it (S1 counterfactual). Production EMAs the live param
    (0.9*actual + 0.1*param per frame -> steady state == param, camera_offset_helper.py);
    a static-scene replay uses the steady value directly (see warp.model_transform).
    """
    from model_replay_sim.warp import model_transform
    used = float(camera_offset) if camera_offset is not None else float(getattr(ctx, "camera_offset", 0.0))
    M_main = model_transform(ctx, wide=False, camera_offset=used)
    M_extra = model_transform(ctx, wide=True, camera_offset=used)
    return M_main, M_extra, used
```

(b) Change the `replay_window` signature from
```python
def replay_window(bundle: str, route_id: str, mono_times) -> dict:
```
to
```python
def replay_window(bundle: str, route_id: str, mono_times,
                  camera_offset: float | None = None,
                  capture_outputs: tuple[str, ...] = ()) -> dict:
```
and extend its docstring with: `camera_offset: None = the route's logged param; float = counterfactual override (S1 sweep). capture_outputs: names of parsed model outputs captured per frame into result["captured"] (vision outputs win on name collision; see _collect_captured).`

(c) Replace the two transform lines
```python
    # one forward warp matrix per route (static scene; calib steady)
    M_main = model_transform(ctx, wide=False)
    M_extra = model_transform(ctx, wide=True)
```
with
```python
    # one forward warp matrix per route (static scene; calib steady)
    M_main, M_extra, camera_offset_used = window_transforms(ctx, camera_offset)
```
and remove `model_transform` from the function-local import line (keep `frame_to_sixchan`):
```python
    from model_replay_sim.warp import frame_to_sixchan
```

(d) Add `"camera_offset_used": camera_offset_used,` to the returned dict (next to `"lat_action_t"`). (Capture plumbing is Task 4; the `capture_outputs` parameter is added now so the signature is stable.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_infer_offset_capture.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Regression — the existing sim suite still passes**

```bash
.venv311/bin/python -m pytest model_replay_sim/tests -v
```
Expected: all pass (same count as before the change plus 3 new).

- [ ] **Step 6: Create the integration smoke gate (offset plumbing end-to-end, ~2 min)**

`stock_lateral_toolkit/centering/smoke_replay.py`:
```python
"""Re-runnable integration gate for the S0 offset plumbing (and, after Task 4,
the M2a capture). Replays a short real window of ROUTE on SP002.

GATE A (offset plumbing): camera_offset=None and camera_offset=0.0 must produce
IDENTICAL desired_curvature (the route's logged param IS 0.0), while
camera_offset=0.05 must produce a DIFFERENT series (the shear reached the warp).

RUN: .venv311/bin/python stock_lateral_toolkit/centering/smoke_replay.py
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC

N_FRAMES = 60  # short: model load dominates; replay ~25 s


def _window_monos():
    from model_replay_sim.anchor import select_anchor_span
    span = select_anchor_span(CC.ROUTE, warmup_s=1.0, min_compare_s=2.0)
    return [float(t) for t in span.mono_times[:N_FRAMES]]


def main():
    from model_replay_sim.infer import replay_window
    mono = _window_monos()

    r_none = replay_window("SP002", CC.ROUTE, mono, camera_offset=None)
    r_zero = replay_window("SP002", CC.ROUTE, mono, camera_offset=0.0)
    r_off = replay_window("SP002", CC.ROUTE, mono, camera_offset=0.05)

    assert r_none["camera_offset_used"] == 0.0, r_none["camera_offset_used"]
    same = np.nanmax(np.abs(r_none["desired_curvature"] - r_zero["desired_curvature"]))
    diff = np.nanmax(np.abs(r_none["desired_curvature"] - r_off["desired_curvature"]))
    print(f"GATE A: max|none-zero| = {same:.3e} (expect 0), max|none-0.05| = {diff:.3e} (expect > 0)")
    assert same == 0.0, "offset=0.0 must equal the route default (param was 0.0)"
    assert diff > 0.0, "offset=0.05 must change the replayed curvature"
    print("GATE A PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the smoke gate**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/smoke_replay.py
```
Expected: `GATE A PASS` (first run compiles nothing new — SP002 pkls exist from the anchor run).

- [ ] **Step 8: Commit**

```bash
git add model_replay_sim/infer.py model_replay_sim/tests/test_infer_offset_capture.py \
        stock_lateral_toolkit/centering/smoke_replay.py
git commit -m "model-sim S0: CameraOffset counterfactual override in replay_window (unit-tested vs production shear)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: M2a — capture lane-line outputs per frame in the replay

**Files:**
- Modify: `model_replay_sim/infer.py` (`ReplayState.step` stashes parsed outputs; `replay_window` collects them; new `_collect_captured`)
- Modify: `model_replay_sim/tests/test_infer_offset_capture.py` (add capture tests)
- Modify: `stock_lateral_toolkit/centering/smoke_replay.py` (add GATE B)

**Resolved unknown (metadata inspected this session):** for SP002 (and CD210/Nevada), `lane_lines` is a **vision-model** output — `driving_vision_metadata.pkl` `output_slices['lane_lines'] = slice(117, 645)`, `['lane_lines_prob'] = slice(645, 653)`. After `Parser.parse_vision_outputs` (already called by `BundleModel.run_vision`), the dict holds `lane_lines` `(1,4,33,2)` float32 means (`[...,0]`=y calibrated +right, `[...,1]`=z), `lane_lines_stds` `(1,4,33,2)`, `lane_lines_prob` `(1,8)` sigmoid (per-line prob at `[0,1::2]`). Lines: 0=outer-left, 1=inner-left, 2=inner-right, 3=outer-right. `ReplayState.step` already has these in its local `vision_out` — capture = stash + collect, no new model work.

- [ ] **Step 1: Add failing capture tests to `model_replay_sim/tests/test_infer_offset_capture.py`**

```python
def test_collect_captured_prefers_vision_then_policy():
    from model_replay_sim.infer import _collect_captured
    state = types.SimpleNamespace(
        last_vision_out={"lane_lines": np.arange(4 * 33 * 2, dtype=np.float32).reshape(1, 4, 33, 2),
                         "lane_lines_prob": np.full((1, 8), 0.5, dtype=np.float32)},
        last_policy_out={"plan": np.ones((1, 33, 15), dtype=np.float32),
                         "lane_lines": np.zeros((1, 4, 33, 2), dtype=np.float32)},  # collision: vision wins
    )
    out = _collect_captured(state, ("lane_lines", "lane_lines_prob", "plan"))
    assert out["lane_lines"].shape == (4, 33, 2)
    assert out["lane_lines"][0, 0, 1] == 1.0            # from vision (arange), not policy zeros
    assert out["lane_lines_prob"].shape == (8,)
    assert out["plan"].shape == (33, 15)


def test_collect_captured_missing_key_raises():
    from model_replay_sim.infer import _collect_captured
    state = types.SimpleNamespace(last_vision_out={"a": np.zeros((1, 2))}, last_policy_out={})
    with pytest.raises(KeyError):
        _collect_captured(state, ("lane_lines",))
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_infer_offset_capture.py -v
```
Expected: the two new tests FAIL with `ImportError: cannot import name '_collect_captured'`; the 3 Task-3 tests still pass.

- [ ] **Step 3: Implement capture in `model_replay_sim/infer.py`**

(a) In `ReplayState.step`, immediately after `policy_out = self.model.run_policy(...)` (and BEFORE the nan-v_ego early return so nan frames still capture), add:
```python
        # M2a capture hooks: the centering workflow reads these after each step.
        self.last_vision_out = vision_out
        self.last_policy_out = policy_out
```
Also add `self.last_vision_out: dict | None = None` / `self.last_policy_out: dict | None = None` at the end of `ReplayState.__init__`.

(b) Add module-level helper (next to `window_transforms`):
```python
def _collect_captured(state, keys) -> dict:
    """Pull the requested PARSED output arrays for the CURRENT frame from a stepped
    ReplayState, dropping the batch axis. Vision outputs win on name collision
    (lane_lines lives in the vision model for SP002/CD210/Nevada — verified from the
    materialized driving_vision_metadata.pkl output_slices); falls back to the policy
    outputs (e.g. 'plan'). Raises KeyError listing what IS available."""
    vis = getattr(state, "last_vision_out", None) or {}
    pol = getattr(state, "last_policy_out", None) or {}
    out = {}
    for k in keys:
        if k in vis:
            out[k] = np.array(vis[k][0], copy=True)
        elif k in pol:
            out[k] = np.array(pol[k][0], copy=True)
        else:
            raise KeyError(f"capture key {k!r} not found; vision has {sorted(vis)}, policy has {sorted(pol)}")
    return out
```

(c) In `replay_window`: before the frame loop add `cap_frames: list[dict] = []`; inside the loop, right after `curvs[i] = state.step(...)`, add:
```python
        if capture_outputs:
            cap_frames.append(_collect_captured(state, capture_outputs))
```
and extend the returned dict construction with:
```python
    result = { ... existing keys ... }
    if capture_outputs:
        result["captured"] = {k: np.stack([f[k] for f in cap_frames]) for k in capture_outputs}
    return result
```
(i.e., bind the existing literal to `result`, conditionally add `"captured"`, return `result`.)

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest model_replay_sim/tests/test_infer_offset_capture.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Extend the smoke gate with GATE B (real capture on real frames)**

In `stock_lateral_toolkit/centering/smoke_replay.py`, replace the body of `main()` with:
```python
def main():
    from model_replay_sim.infer import replay_window
    mono = _window_monos()

    r_none = replay_window("SP002", CC.ROUTE, mono, camera_offset=None)
    r_zero = replay_window("SP002", CC.ROUTE, mono, camera_offset=0.0)
    r_off = replay_window("SP002", CC.ROUTE, mono, camera_offset=0.05,
                          capture_outputs=("lane_lines", "lane_lines_prob"))

    assert r_none["camera_offset_used"] == 0.0, r_none["camera_offset_used"]
    same = np.nanmax(np.abs(r_none["desired_curvature"] - r_zero["desired_curvature"]))
    diff = np.nanmax(np.abs(r_none["desired_curvature"] - r_off["desired_curvature"]))
    print(f"GATE A: max|none-zero| = {same:.3e} (expect 0), max|none-0.05| = {diff:.3e} (expect > 0)")
    assert same == 0.0 and diff > 0.0
    print("GATE A PASS")

    ll = r_off["captured"]["lane_lines"]          # (N, 4, 33, 2)
    lp = r_off["captured"]["lane_lines_prob"]     # (N, 8)
    assert ll.shape == (N_FRAMES, 4, 33, 2), ll.shape
    assert lp.shape == (N_FRAMES, 8), lp.shape
    assert np.all((lp >= 0.0) & (lp <= 1.0)), "lane_lines_prob must be sigmoid output"
    width0 = ll[10:, 2, 0, 0] - ll[10:, 1, 0, 0]  # inner width at x=0, post img-buffer warmup
    print(f"GATE B: median inner lane width @x=0 = {np.median(width0):.2f} m (expect 2.5-5.0)")
    assert 2.5 < float(np.median(width0)) < 5.0
    print("GATE B PASS")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the smoke gate**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/smoke_replay.py
```
Expected: `GATE A PASS` then `GATE B PASS` with a lane width around 3.2–3.6 m (cache median `lane_width_y0` on this route is 3.258 m).

- [ ] **Step 7: Commit**

```bash
git add model_replay_sim/infer.py model_replay_sim/tests/test_infer_offset_capture.py \
        stock_lateral_toolkit/centering/smoke_replay.py
git commit -m "model-sim M2a: per-frame capture of parsed vision/policy outputs (lane_lines et al) in replay_window

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Shared scene windows (identical frames for M1 preference, M2 replays, S1 sweep)

**Files:**
- Create: `stock_lateral_toolkit/centering/windows.py`
- Test: `stock_lateral_toolkit/centering/tests/test_windows.py`

Why: every phase must use the SAME frames so numbers are comparable. Reuses the anchor module's eligibility machinery (`model_replay_sim/anchor.py:_eligible_aligned_timeline_mask` — frame-aligned AND replay-scene-eligible), then carves up to `N_WINDOWS` disjoint `(warmup + compare)` windows out of the eligible contiguous runs. The SP002 anchor found a ~2059-compare-frame run on route_stock05, so expect window 0 = 200+1200 frames and window 1 = the remainder (≥ 200+600).

- [ ] **Step 1: Write the failing test**

`stock_lateral_toolkit/centering/tests/test_windows.py`:
```python
import types

import numpy as np
import pytest


def _fake_timeline(n, t0=1000.0, dt=0.05):
    rows = []
    for i in range(n):
        rows.append(types.SimpleNamespace(frame_id=100 + i, timestamp_eof_s=t0 + i * dt,
                                          segment_num=i // 1200, segment_id=i % 1200))
    return tuple(rows)


def test_select_scene_windows_carves_disjoint_windows(monkeypatch):
    import model_replay_sim.anchor as anchor_mod
    from stock_lateral_toolkit.centering import windows as W

    tl = _fake_timeline(3000)
    good = np.ones(3000, dtype=bool)
    monkeypatch.setattr(anchor_mod, "_eligible_aligned_timeline_mask", lambda route: (good, tl))

    ws = W.select_scene_windows("route_fake", n_windows=2, warmup_frames=200,
                                compare_frames=1200, min_compare_frames=600)
    assert len(ws) == 2
    assert len(ws[0]["mono_times"]) == 1400 and ws[0]["split_index"] == 200
    assert len(ws[1]["mono_times"]) == 1400
    # disjoint and ordered
    assert ws[0]["frame_ids"][-1] < ws[1]["frame_ids"][0]


def test_select_scene_windows_short_run_fallback(monkeypatch):
    import model_replay_sim.anchor as anchor_mod
    from stock_lateral_toolkit.centering import windows as W

    tl = _fake_timeline(2300)  # 1400 + 900: second window uses the fallback size
    good = np.ones(2300, dtype=bool)
    monkeypatch.setattr(anchor_mod, "_eligible_aligned_timeline_mask", lambda route: (good, tl))

    ws = W.select_scene_windows("route_fake", 2, 200, 1200, min_compare_frames=600)
    assert len(ws) == 2
    assert len(ws[1]["mono_times"]) == 900          # warmup 200 + compare 700 remainder
    assert ws[1]["n_compare"] == 700


def test_select_scene_windows_no_run_raises(monkeypatch):
    import model_replay_sim.anchor as anchor_mod
    from stock_lateral_toolkit.centering import windows as W

    tl = _fake_timeline(100)
    monkeypatch.setattr(anchor_mod, "_eligible_aligned_timeline_mask",
                        lambda route: (np.ones(100, dtype=bool), tl))
    with pytest.raises(RuntimeError):
        W.select_scene_windows("route_fake", 1, 200, 1200, min_compare_frames=600)
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_windows.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'stock_lateral_toolkit.centering.windows'`.

- [ ] **Step 3: Implement `stock_lateral_toolkit/centering/windows.py`**

```python
"""Shared scene-window selection: N disjoint (warmup+compare) frame windows over the
replay-eligible contiguous runs of a route. Pure (no model). The SAME windows.json is
read by the M1 sampler (frame preference), M2 consensus replays, and the S1 sweep, so
all phases see identical frames.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/windows.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def _eligible_runs(route_id: str):
    """All maximal runs that are BOTH eligible and frame-id contiguous, longest first.
    (Same run predicate as anchor._longest_contiguous_run, but keeps every run.)"""
    import model_replay_sim.anchor as anchor_mod
    good, tl = anchor_mod._eligible_aligned_timeline_mask(route_id)
    fids = [r.frame_id for r in tl]
    runs = []
    i, n = 0, len(good)
    while i < n:
        if good[i]:
            j = i + 1
            while j < n and good[j] and fids[j] == fids[j - 1] + 1:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    runs.sort(key=lambda ab: ab[1] - ab[0], reverse=True)
    return runs, tl


def select_scene_windows(route_id: str, n_windows: int, warmup_frames: int,
                         compare_frames: int, min_compare_frames: int) -> list[dict]:
    runs, tl = _eligible_runs(route_id)
    out: list[dict] = []
    for a, b in runs:
        pos = a
        while len(out) < n_windows:
            remaining = b - pos
            if remaining >= warmup_frames + compare_frames:
                take = warmup_frames + compare_frames
            elif remaining >= warmup_frames + min_compare_frames:
                take = remaining
            else:
                break
            rows = tl[pos:pos + take]
            out.append({
                "window_id": len(out),
                "route_id": route_id,
                "mono_times": [float(r.timestamp_eof_s) for r in rows],
                "frame_ids": [int(r.frame_id) for r in rows],
                "split_index": int(warmup_frames),
                "n_compare": int(take - warmup_frames),
            })
            pos += take
        if len(out) >= n_windows:
            break
    if not out:
        raise RuntimeError(f"{route_id}: no eligible contiguous run of >= "
                           f"{warmup_frames + min_compare_frames} frames")
    return out


def windows_path() -> Path:
    return CC.RESULTS_DIR / "m2" / "windows.json"


def write_windows(windows: list[dict]) -> Path:
    p = windows_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(windows))
    return p


def load_windows() -> list[dict]:
    return json.loads(windows_path().read_text())


if __name__ == "__main__":
    ws = select_scene_windows(CC.ROUTE, CC.N_WINDOWS, CC.WARMUP_FRAMES,
                              CC.COMPARE_FRAMES, CC.MIN_COMPARE_FRAMES)
    p = write_windows(ws)
    for w in ws:
        m = w["mono_times"]
        print(f"window {w['window_id']}: {len(m)} frames ({w['n_compare']} compare) "
              f"mono [{m[0]:.1f}, {m[-1]:.1f}]")
    print(f"wrote {p}")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_windows.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Generate the real windows (pure, no model, ~1 min)**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/windows.py
```
Expected: 2 windows printed (window 0 with 1200 compare frames; window 1 with 600–1200) and `wrote .../results/m2/windows.json`. If only ONE window fits, record that in the run log — M2/S1 then run on 1 window and the pre-registered "≥2 windows" requirement is met by the OPTIONAL second-route video pull (Task 10 step 1b) instead.

- [ ] **Step 6: Commit**

```bash
git add stock_lateral_toolkit/centering/windows.py stock_lateral_toolkit/centering/tests/test_windows.py
git commit -m "centering: shared scene-window selection over replay-eligible runs

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: M1a — ground-plane projection math

**Files:**
- Create: `stock_lateral_toolkit/centering/ground_plane.py`
- Test: `stock_lateral_toolkit/centering/tests/test_ground_plane.py`

- [ ] **Step 1: Write the failing tests**

`stock_lateral_toolkit/centering/tests/test_ground_plane.py`:
```python
import numpy as np
import pytest


FX = FY = 1141.5   # DEVICE_CAMERAS[("mici","os04c10")].fcam focal (2688//2 x 1520//2)
CX, CY = 672.0, 380.0
H = 1.22


def test_identity_rpy_known_projection():
    """With rpy=(0,0,0), a road point (x=10 m, y=0) must land at u=cx,
    v = cy + fy*h/x (pinhole flat-road geometry) — this pins the whole
    convention stack (road frame y-LEFT, view frame permutation, intrinsics)."""
    from stock_lateral_toolkit.centering import ground_plane as G
    K = G.fcam_intrinsics()
    assert K[0, 0] == pytest.approx(FX) and K[1, 2] == pytest.approx(CY)
    Hm = G.road_homography([0.0, 0.0, 0.0], H, K)
    u, v = G.pixel_from_road(Hm, 10.0, 0.0)
    assert u == pytest.approx(CX, abs=0.5)
    assert v == pytest.approx(CY + FY * H / 10.0, abs=0.5)
    # a point 1.8 m to the LEFT (road y positive-left) must be LEFT in the image (u < cx)
    u_left, _ = G.pixel_from_road(Hm, 10.0, 1.8)
    assert u_left < CX - 100


def test_round_trip_with_real_calibration():
    from stock_lateral_toolkit.centering import ground_plane as G
    K = G.fcam_intrinsics()
    Hm = G.road_homography([0.0, 0.0429, -0.0498], H, K)   # stock05 median rpy
    for x in (6.0, 10.0, 18.0):
        for y in (-3.0, -1.7, 0.0, 1.7, 3.0):
            u, v = G.pixel_from_road(Hm, x, y)
            xr, yr = G.road_from_pixel(Hm, u, v)
            assert xr == pytest.approx(x, abs=1e-6)
            assert yr == pytest.approx(y, abs=1e-6)


def test_above_horizon_raises():
    from stock_lateral_toolkit.centering import ground_plane as G
    Hm = G.road_homography([0.0, 0.0, 0.0], H, G.fcam_intrinsics())
    with pytest.raises(ValueError):
        G.road_from_pixel(Hm, 672.0, 100.0)   # well above the horizon row


def test_sign_helper():
    from stock_lateral_toolkit.centering import ground_plane as G
    assert G.y_cal_from_y_road(1.8) == -1.8   # road +LEFT -> calibrated +RIGHT
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_ground_plane.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... ground_plane`.

- [ ] **Step 3: Implement `stock_lateral_toolkit/centering/ground_plane.py`**

```python
"""Flat-road ground-plane projection for M1 video ground truth.

Frames (common/transformations/README.md):
  calibrated frame: [x fwd, y RIGHT, z down] — model laneLines y lives here.
  road frame (get_view_frame_from_road_frame): [x fwd, y LEFT, z up].
Homography: KE = K @ get_view_frame_from_road_frame(rpyCalib, height) (3x4); road-plane
points [x, y, 0, 1] project through H = KE[:, [0, 1, 3]]. Feeding liveCalibration's
rpyCalib is exact here because device_from_road = R(rpy) @ diag([1,-1,-1]) ==
device_from_calib @ calib_from_road, and stock cal_roll == 0.

Everything in this module is CAMERA-relative (the same origin the model's laneLines
use). Vehicle-centerline conversion (the tape-measured lever arm) happens ONLY in
m1_offsets.py. Uncertainty budget: pre-registration doc §2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpilot.common.transformations.camera import DEVICE_CAMERAS, get_view_frame_from_road_frame


def fcam_intrinsics(device_type: str = "mici", sensor: str = "os04c10") -> np.ndarray:
    return DEVICE_CAMERAS[(device_type, sensor)].fcam.intrinsics


def road_homography(rpy_calib, height_m: float, intrinsics: np.ndarray) -> np.ndarray:
    """3x3 homography: road-plane [x_fwd, y_left, 1] -> pixel homogeneous [u, v, 1]."""
    roll, pitch, yaw = (float(v) for v in rpy_calib)
    ke = np.asarray(intrinsics, float) @ get_view_frame_from_road_frame(roll, pitch, yaw, float(height_m))
    return ke[:, [0, 1, 3]]


def pixel_from_road(H: np.ndarray, x_fwd_m: float, y_left_m: float) -> tuple[float, float]:
    p = H @ np.array([x_fwd_m, y_left_m, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def road_from_pixel(H: np.ndarray, u_px: float, v_px: float) -> tuple[float, float]:
    q = np.linalg.solve(H, np.array([u_px, v_px, 1.0]))
    if abs(q[2]) < 1e-12 or (q[0] / q[2]) <= 0.0:
        raise ValueError(f"pixel ({u_px:.0f},{v_px:.0f}) maps above the horizon / behind the camera")
    return float(q[0] / q[2]), float(q[1] / q[2])


def y_cal_from_y_road(y_left_m: float) -> float:
    """road frame y (+LEFT) -> calibrated frame y (+RIGHT)."""
    return -float(y_left_m)


def sigma_frame_m() -> float:
    """Pre-registered per-frame 1-sigma (M0 §2): quadrature of the declared terms."""
    from stock_lateral_toolkit.centering import config as CC
    return float(np.sqrt(CC.SIGMA_PIXEL_M ** 2 + CC.SIGMA_ROLL_M ** 2
                         + CC.SIGMA_YAW_M ** 2 + CC.SIGMA_MISC_M ** 2))
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_ground_plane.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add stock_lateral_toolkit/centering/ground_plane.py stock_lateral_toolkit/centering/tests/test_ground_plane.py
git commit -m "centering M1a: flat-road ground-plane projection (convention-pinned by tests)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: M1b — stratified frame sampler

**Files:**
- Create: `stock_lateral_toolkit/centering/frame_sampler.py`
- Test: `stock_lateral_toolkit/centering/tests/test_frame_sampler.py`

Strata = speed bin × heading quadrant, straight road only (GPS heading-rate, model-independent — the M1 flat-plane math and the "any band motion is weave" logic both need straights), engaged, unpressed, no blinker/lane-change, both inner lane probs ≥ 0.6. Frames INSIDE the Task-5 windows are preferred so M1↔M2 overlap is maximal. Route_stock05 has ~11 min of data at ~51% lat_active, so with 5 s de-correlation expect **~40–70 frames** — `N_FRAMES_MIN = 40` is the pre-registered floor (below it, the second-route pull in Task 10 step 1b becomes REQUIRED rather than optional).

- [ ] **Step 1: Write the failing test**

`stock_lateral_toolkit/centering/tests/test_frame_sampler.py`:
```python
import numpy as np


def _synth_cache(n=8000, dt=0.05):
    """Straight-line eastbound drive at 20 m/s with all gates open."""
    t = 1000.0 + np.arange(n) * dt
    lon0, lat0 = -84.7, 33.9
    lon = lon0 + (20.0 * np.arange(n) * dt) / (111320.0 * np.cos(np.radians(lat0)))
    return {
        "mono_time": t,
        "v_ego": np.full(n, 20.0),
        "lat": np.full(n, lat0) , "lon": lon,
        "lat_active": np.ones(n), "steering_pressed": np.zeros(n),
        "blinker": np.zeros(n), "lane_change_state": np.zeros(n),
        "lane_prob_left": np.full(n, 0.9), "lane_prob_right": np.full(n, 0.9),
    }


def test_sample_respects_separation_and_is_deterministic():
    from stock_lateral_toolkit.centering import frame_sampler as FS
    z = _synth_cache()
    in_win = np.zeros(len(z["mono_time"]), dtype=bool)
    in_win[2000:4000] = True
    a = FS.sample(z, in_win)
    b = FS.sample(z, in_win)
    assert a == b                                    # seeded => deterministic
    assert len(a) >= 40
    t = np.sort(z["mono_time"][a])
    assert np.min(np.diff(t)) >= 5.0 - 1e-9          # MIN_FRAME_SEPARATION_S
    # in-window preference: the window is 100 s long => it can hold ~20 samples,
    # and all of those slots must be used before out-of-window frames are taken
    n_in = int(np.sum(in_win[a]))
    assert n_in >= 18


def test_eligibility_blocks_curves():
    from stock_lateral_toolkit.centering import frame_sampler as FS
    z = _synth_cache()
    # bend the GPS track into a curve for the middle third
    n = len(z["lon"])
    theta = np.linspace(0, 2.0, n // 3)
    z["lat"] = z["lat"].copy()
    z["lat"][n // 3: 2 * n // 3] += np.cumsum(np.sin(theta)) * 1e-5
    ok, _head = FS.eligibility(z)
    assert ok[: n // 4].mean() > 0.8                 # straight part eligible
    assert ok[n // 3 + 200: 2 * n // 3 - 200].mean() < 0.5   # curved part mostly blocked
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_frame_sampler.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... frame_sampler`.

- [ ] **Step 3: Implement `stock_lateral_toolkit/centering/frame_sampler.py`**

```python
"""M1 frame sampler: stratified (speed bin x heading quadrant), de-correlated (>= 5 s
apart), model-independent-straight sample from the route's retrospective cache,
preferring frames INSIDE the shared M2 scene windows. Writes results/m1/frames_manifest.csv.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/frame_sampler.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def _smooth(x, w):
    return np.convolve(x, np.ones(w) / w, mode="same")


def road_curvature_gps(lat, lon, v_ego, mono_time, w=15):
    """Model-independent road curvature (1/m) + heading (deg) from GPS, the
    analyze_compare.py recipe (heading-rate / speed; sign irrelevant, |.| used)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    x = (lon - np.nanmean(lon)) * 111320.0 * np.cos(np.radians(np.nanmean(lat)))
    y = (lat - np.nanmean(lat)) * 110540.0
    dx = np.gradient(_smooth(x, w)); dy = np.gradient(_smooth(y, w))
    head = np.unwrap(np.arctan2(dy, dx))
    dt = np.maximum(np.gradient(np.asarray(mono_time, float)), 1e-3)
    headrate = np.gradient(_smooth(head, w)) / dt
    v = np.maximum(np.asarray(v_ego, float), 1.0)
    return np.clip(headrate / v, -0.02, 0.02), np.degrees(head) % 360.0


def eligibility(z):
    v = np.asarray(z["v_ego"], float)
    curv, head_deg = road_curvature_gps(z["lat"], z["lon"], v, z["mono_time"])
    ok = ((np.asarray(z["lat_active"]) > 0.5)
          & (np.asarray(z["steering_pressed"]) < 0.5)
          & (np.asarray(z["blinker"]) < 0.5)
          & (np.asarray(z["lane_change_state"]) <= 0.001)
          & (v >= CC.V_MIN_MPS)
          & (np.asarray(z["lane_prob_left"]) >= CC.LANE_PROB_MIN)
          & (np.asarray(z["lane_prob_right"]) >= CC.LANE_PROB_MIN)
          & (np.abs(curv) <= CC.STRAIGHT_CURV_MAX)
          & np.isfinite(np.asarray(z["lat"], float)) & np.isfinite(np.asarray(z["lon"], float)))
    return ok, head_deg


def stratum_of(v_mps: float, head_deg: float):
    sb = -1
    for i, (lo, hi) in enumerate(CC.SPEED_BINS_MPS):
        if lo <= v_mps < hi:
            sb = i
    hb = int(head_deg // CC.HEADING_BIN_DEG) % int(360 // CC.HEADING_BIN_DEG)
    return sb, hb


def sample(z, in_window_mask) -> list[int]:
    """Deterministic (seeded) round-robin stratified sample of cache indices."""
    ok, head = eligibility(z)
    mono = np.asarray(z["mono_time"], float)
    rng = np.random.default_rng(CC.SAMPLER_SEED)
    strata: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(ok):
        sb, hb = stratum_of(float(np.asarray(z["v_ego"])[i]), float(head[i]))
        if sb >= 0:
            strata.setdefault((sb, hb), []).append(int(i))

    chosen: list[int] = []
    chosen_t: list[float] = []

    def try_add(i: int) -> bool:
        t = mono[i]
        if all(abs(t - tt) >= CC.MIN_FRAME_SEPARATION_S for tt in chosen_t):
            chosen.append(i); chosen_t.append(t)
            return True
        return False

    keys = sorted(strata)
    pools = {k: rng.permutation(strata[k]).tolist() for k in keys}
    for prefer_window in (True, False):
        progress = True
        while progress and len(chosen) < CC.N_FRAMES_TARGET:
            progress = False
            for k in keys:
                for i in pools[k]:
                    if i in chosen or bool(in_window_mask[i]) != prefer_window:
                        continue
                    if try_add(i):
                        progress = True
                        break
                if len(chosen) >= CC.N_FRAMES_TARGET:
                    break
    return sorted(chosen)


def _in_window_mask(mono: np.ndarray) -> np.ndarray:
    from stock_lateral_toolkit.centering.windows import load_windows
    mask = np.zeros(len(mono), dtype=bool)
    try:
        wins = load_windows()
    except FileNotFoundError:
        print("WARNING: no windows.json (run windows.py first); sampling without preference")
        return mask
    for w in wins:
        wt = np.asarray(w["mono_times"], float)
        lo, hi = wt.min() - 0.03, wt.max() + 0.03
        mask |= (mono >= lo) & (mono <= hi)
    return mask


def main():
    z = dict(np.load(CC.CACHE_NPZ))
    mono = np.asarray(z["mono_time"], float)
    in_win = _in_window_mask(mono)
    idxs = sample(z, in_win)
    if len(idxs) < CC.N_FRAMES_MIN:
        print(f"WARNING: only {len(idxs)} frames (< pre-registered minimum {CC.N_FRAMES_MIN}); "
              f"the second-route video pull (Task 10 step 1b) is now REQUIRED, not optional")

    # map cache mono -> (segment_num, segment_id); frames without a decoded frame
    # within 0.03 s are dropped (logged but rare on this fully-local route)
    from model_replay_sim.alignment import map_window_to_frames
    _, head = road_curvature_gps(z["lat"], z["lon"], z["v_ego"], mono)   # once, O(n)
    rows, dropped = [], 0
    for i in idxs:
        try:
            al = map_window_to_frames(CC.ROUTE, [float(mono[i])])[0]
        except ValueError:
            dropped += 1
            continue
        sb, hb = stratum_of(float(z["v_ego"][i]), float(head[i]))
        rows.append(dict(frame_idx=len(rows), cache_index=i, mono_time=float(mono[i]),
                         seg_num=al.segment_num, seg_id=al.segment_id,
                         v_ego=float(z["v_ego"][i]), speed_bin=sb, heading_bin=hb,
                         in_m2_window=bool(in_win[i]),
                         lane_prob_left=float(z["lane_prob_left"][i]),
                         lane_prob_right=float(z["lane_prob_right"][i]),
                         cal_roll=float(z["cal_roll"][i]), cal_pitch=float(z["cal_pitch"][i]),
                         cal_yaw=float(z["cal_yaw"][i]),
                         logged_center_y0=float(z["lane_center_y0"][i])))
    out = CC.RESULTS_DIR / "m1" / "frames_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    n_in = sum(r["in_m2_window"] for r in rows)
    print(f"{len(rows)} frames ({dropped} dropped, {n_in} inside M2 windows) -> {out}")
    per = {}
    for r in rows:
        per[(r["speed_bin"], r["heading_bin"])] = per.get((r["speed_bin"], r["heading_bin"]), 0) + 1
    print("strata:", dict(sorted(per.items())))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_frame_sampler.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Run the sampler on the real route**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/frame_sampler.py
```
Expected artifact: `stock_lateral_toolkit/centering/results/m1/frames_manifest.csv` with ≥40 rows, ≥2 heading bins, strata counts printed. **Pre-registered interpretation rule (M0 §2):** if <40 rows, do NOT proceed to M3 verdicts on M1 alone; escalate to the second-route pull.

- [ ] **Step 6: Commit**

```bash
git add stock_lateral_toolkit/centering/frame_sampler.py stock_lateral_toolkit/centering/tests/test_frame_sampler.py
git commit -m "centering M1b: stratified de-correlated frame sampler (window-preferring)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: M1c — semi-automated lane-marking annotation (proposals + overlays + USER review gate)

**Files:**
- Create: `stock_lateral_toolkit/centering/annotate.py`
- Test: `stock_lateral_toolkit/centering/tests/test_annotate.py`

Design: proposals are **image-evidence only** (no model input — keeps M1 independent of the models under test). For each manifest frame, read the nv12 luma plane (same `FrameReader(pix_fmt="nv12")` path the anchor-validated replay uses — no RGB conversion uncertainty), sample intensity along **metric lateral grids** at 8/12/16 m via the Task-6 homography, run a top-hat matched filter sized to the paint width, take the strongest peak per side with sub-cell parabolic refinement, back-project to road-frame y. Output: `proposals.csv` + one overlay PNG per frame (luma + colored markers + a 0.5 m lateral ruler) for human review.

- [ ] **Step 1: Write the failing test (synthetic-image recovery)**

`stock_lateral_toolkit/centering/tests/test_annotate.py`:
```python
import numpy as np
import pytest


RPY = [0.0, 0.0429, -0.0498]
H = 1.22


def _paint_image(y_left=1.75, y_right=-1.72):
    """Synthetic 760x1344 luma: dark road with two bright 0.125 m paint stripes
    at road-frame lateral y_left / y_right, painted by forward projection."""
    from stock_lateral_toolkit.centering import ground_plane as G
    img = np.full((760, 1344), 60.0)
    Hm = G.road_homography(RPY, H, G.fcam_intrinsics())
    for x in np.arange(5.0, 25.0, 0.02):
        for yc in (y_left, y_right):
            for y in np.arange(yc - 0.0625, yc + 0.0625, 0.01):
                u, v = G.pixel_from_road(Hm, float(x), float(y))
                ui, vi = int(round(u)), int(round(v))
                if 0 <= vi < 760 and 0 <= ui < 1344:
                    img[vi, ui] = 220.0
    return img, Hm


def test_propose_line_recovers_painted_stripes():
    from stock_lateral_toolkit.centering import annotate as A
    img, Hm = _paint_image()
    for x_m in (8.0, 12.0, 16.0):
        left = A.propose_line(img, Hm, x_m, "left")
        right = A.propose_line(img, Hm, x_m, "right")
        assert left["auto_ok"] and right["auto_ok"]
        assert left["y_road"] == pytest.approx(1.75, abs=0.03)
        assert right["y_road"] == pytest.approx(-1.72, abs=0.03)


def test_propose_line_flags_blank_road():
    from stock_lateral_toolkit.centering import annotate as A
    from stock_lateral_toolkit.centering import ground_plane as G
    rng = np.random.default_rng(0)
    img = np.full((760, 1344), 60.0) + rng.normal(0, 2.0, (760, 1344))
    Hm = G.road_homography(RPY, H, G.fcam_intrinsics())
    p = A.propose_line(img, Hm, 12.0, "left")
    assert not p["auto_ok"]          # nothing mark-like -> low contrast


def test_luma_plane_shape():
    from stock_lateral_toolkit.centering import annotate as A
    flat = np.arange(1344 * 760 * 3 // 2, dtype=np.uint8)
    y = A.luma_plane(flat)
    assert y.shape == (760, 1344)
    assert y[0, 5] == 5.0
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_annotate.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... annotate`.

- [ ] **Step 3: Implement `stock_lateral_toolkit/centering/annotate.py`**

```python
"""M1 semi-automated lane-marking annotation.

Proposals are IMAGE-EVIDENCE ONLY (no model outputs touch this file). Per frame and
per eval distance (8/12/16 m): sample luma along a metric lateral grid on the flat-road
plane, top-hat matched filter at the paint width, strongest peak per side, sub-cell
parabolic refinement, back-project to road-frame y (+LEFT).

Outputs (results/m1/):
  proposals.csv     one row per (frame, distance, side) w/ y_road_m, pixel, contrast, auto_ok
  overlays/frame_XXX.png   luma + green(ok)/red(low-contrast) markers + 0.5 m ruler
  review_subset.csv  every REVIEW_EVERY_N-th frame's rows, for the USER to verdict-fill

USER GATE (M0 §2): fill `verdict` (accept|reject|correct) and `corrected_u_px` (when
verdict=correct) in review_subset.csv. m1_offsets.py enforces >=85% acceptance.

RUN:
  .venv311/bin/python stock_lateral_toolkit/centering/annotate.py            # propose + overlays
  .venv311/bin/python stock_lateral_toolkit/centering/annotate.py --review   # write review_subset.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering import ground_plane as G

CAM_W, CAM_H = 1344, 760
GRID_STEP_M = 0.02


def luma_plane(nv12_flat, w: int = CAM_W, h: int = CAM_H) -> np.ndarray:
    return np.asarray(nv12_flat, dtype=np.uint8).ravel()[: w * h].reshape(h, w).astype(float)


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    u = np.clip(np.asarray(u, float), 0, w - 1.001)
    v = np.clip(np.asarray(v, float), 0, h - 1.001)
    u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int)
    du = u - u0; dv = v - v0
    return ((img[v0, u0] * (1 - du) + img[v0, u0 + 1] * du) * (1 - dv)
            + (img[v0 + 1, u0] * (1 - du) + img[v0 + 1, u0 + 1] * du) * dv)


def metric_profile(img, H, x_m: float, y_grid: np.ndarray):
    uv = np.array([G.pixel_from_road(H, x_m, float(y)) for y in y_grid])
    return bilinear(img, uv[:, 0], uv[:, 1]), uv[:, 0], uv[:, 1]


def tophat_response(profile: np.ndarray, step_m: float, mark_width_m: float) -> np.ndarray:
    wm = max(1, int(round(mark_width_m / step_m)))
    inner = uniform_filter1d(profile, wm, mode="nearest")
    outer = uniform_filter1d(profile, 3 * wm, mode="nearest")
    return inner - outer


def propose_line(img: np.ndarray, H: np.ndarray, x_m: float, side: str) -> dict:
    """Strongest paint-like peak on `side` ('left' searches y_road in [+1.0,+3.4],
    'right' in [-3.4,-1.0]) at forward distance x_m. Returns y_road (+LEFT), the pixel,
    a MAD-normalized contrast, and auto_ok = contrast >= MIN_CONTRAST."""
    lo, hi = CC.LANE_SEARCH_BAND_M
    band = np.arange(lo, hi, GRID_STEP_M) if side == "left" else np.arange(-hi, -lo, GRID_STEP_M)
    prof, u, v = metric_profile(img, H, x_m, band)
    resp = tophat_response(prof, GRID_STEP_M, CC.MARK_WIDTH_M)
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med))) + 1e-9
    k = int(np.argmax(resp))
    contrast = float((resp[k] - med) / (1.4826 * mad))
    if 0 < k < len(resp) - 1:
        denom = resp[k - 1] - 2 * resp[k] + resp[k + 1]
        d = float(np.clip((resp[k - 1] - resp[k + 1]) / (2 * denom), -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    else:
        d = 0.0
    y_road = float(band[k] + d * GRID_STEP_M)
    uu, vv = G.pixel_from_road(H, x_m, y_road)
    return dict(x_m=float(x_m), side=side, y_road=y_road, u_px=float(uu), v_px=float(vv),
                contrast=contrast, auto_ok=bool(contrast >= CC.MIN_CONTRAST))


def _read_luma(seg_num: int, seg_id: int) -> np.ndarray:
    from model_replay_sim.alignment import read_frame
    return luma_plane(np.asarray(read_frame(CC.ROUTE, seg_num, seg_id), dtype=np.uint8).ravel())


def _overlay(img: np.ndarray, H: np.ndarray, proposals: list[dict], out_png: Path) -> None:
    from PIL import Image, ImageDraw
    rgb = Image.fromarray(np.stack([img.astype(np.uint8)] * 3, axis=-1))
    dr = ImageDraw.Draw(rgb)
    for x_m in CC.EVAL_DISTANCES_M:               # 0.5 m lateral ruler at each eval distance
        for y in np.arange(-3.5, 3.51, 0.5):
            u, v = G.pixel_from_road(H, x_m, float(y))
            color = (80, 160, 255) if abs(y) > 0.01 else (255, 255, 0)
            dr.line([(u, v - 4), (u, v + 4)], fill=color, width=1)
    for p in proposals:
        c = (0, 255, 0) if p["auto_ok"] else (255, 0, 0)
        u, v = p["u_px"], p["v_px"]
        dr.line([(u - 8, v), (u + 8, v)], fill=c, width=2)
        dr.line([(u, v - 8), (u, v + 8)], fill=c, width=2)
        dr.text((u + 10, v - 14), f"{p['side'][0]}{p['x_m']:.0f}m y={p['y_road']:+.2f}", fill=c)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_png)


def run_proposals() -> None:
    man_path = CC.RESULTS_DIR / "m1" / "frames_manifest.csv"
    rows = list(csv.DictReader(open(man_path)))
    from model_replay_sim.context import route_context
    height = float(route_context(CC.ROUTE).height)
    out_rows = []
    for r in rows:
        rpy = [float(r["cal_roll"]), float(r["cal_pitch"]), float(r["cal_yaw"])]
        Hm = G.road_homography(rpy, height, G.fcam_intrinsics())
        img = _read_luma(int(r["seg_num"]), int(r["seg_id"]))
        props = [propose_line(img, Hm, x, s) for x in CC.EVAL_DISTANCES_M for s in ("left", "right")]
        _overlay(img, Hm, props, CC.RESULTS_DIR / "m1" / "overlays" / f"frame_{int(r['frame_idx']):03d}.png")
        for p in props:
            out_rows.append(dict(frame_idx=int(r["frame_idx"]), mono_time=r["mono_time"],
                                 seg_num=r["seg_num"], seg_id=r["seg_id"], **{k: p[k] for k in
                                 ("x_m", "side", "y_road", "u_px", "v_px", "contrast", "auto_ok")},
                                 verdict="", corrected_u_px=""))
    out = CC.RESULTS_DIR / "m1" / "proposals.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    n_ok = sum(r["auto_ok"] for r in out_rows)
    print(f"{len(out_rows)} proposals ({n_ok} auto_ok) -> {out}; overlays in results/m1/overlays/")


def write_review_subset() -> None:
    rows = list(csv.DictReader(open(CC.RESULTS_DIR / "m1" / "proposals.csv")))
    frames = sorted({int(r["frame_idx"]) for r in rows})
    review_frames = set(frames[:: CC.REVIEW_EVERY_N])
    sub = [r for r in rows if int(r["frame_idx"]) in review_frames]
    out = CC.RESULTS_DIR / "m1" / "review_subset.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sub[0].keys()))
        w.writeheader(); w.writerows(sub)
    print(f"{len(sub)} rows across {len(review_frames)} frames -> {out}")
    print("USER: open each frame's overlay PNG, fill `verdict` (accept|reject|correct) "
          "and `corrected_u_px` for corrections, save the CSV.")


if __name__ == "__main__":
    if "--review" in sys.argv:
        write_review_subset()
    else:
        run_proposals()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_annotate.py -v
```
Expected: 3 passed. (The synthetic recovery test is the fidelity anchor of the whole M1 method: known ground truth in, ≤3 cm out.)

- [ ] **Step 5: Run proposals + overlays on the real frames (~2–5 min)**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/annotate.py
.venv311/bin/python stock_lateral_toolkit/centering/annotate.py --review
```
Expected artifacts: `results/m1/proposals.csv` (6 rows per frame), `results/m1/overlays/frame_*.png`, `results/m1/review_subset.csv`.

- [ ] **Step 6: 🧑 USER-GATED — annotation spot-review (DO NOT PROCEED WITHOUT IT)**

The USER opens the overlay PNGs for the frames listed in `review_subset.csv` (e.g. `open stock_lateral_toolkit/centering/results/m1/overlays/frame_000.png`) and fills `verdict` per row: `accept` (marker on the paint), `correct` + `corrected_u_px` (marker off by a readable amount), `reject` (no visible mark / marker on a shadow/crack/other car). This is the ONLY human-labeled input in the workflow; Task 9 enforces the pre-registered ≥85% trust rule on it.

- [ ] **Step 7: Commit**

```bash
git add stock_lateral_toolkit/centering/annotate.py stock_lateral_toolkit/centering/tests/test_annotate.py
git commit -m "centering M1c: image-evidence lane-mark proposals + review overlays (synthetic-recovery tested)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: M1d — trust rule + physical offsets + uncertainty budget

**Files:**
- Create: `stock_lateral_toolkit/centering/m1_offsets.py`
- Test: `stock_lateral_toolkit/centering/tests/test_m1_offsets.py`

- [ ] **Step 1: 🧑 USER-GATED — measure the camera lever arm**

Tape-measure the lateral distance from the device camera to the vehicle centerline (windshield-mount vs hood-center reference; + = camera RIGHT of centerline, calibrated-frame sign). Set in `stock_lateral_toolkit/centering/config.py`: `CAM_OFFSET_FROM_CENTERLINE_M = <measured>` and `CAM_OFFSET_MEASURED = True`. Record the value and method in the run log. Without this, `m1_offsets.py` emits camera-relative numbers only (sufficient for M3/R1, NOT for the final vehicle-frame `P`).

- [ ] **Step 2: Write the failing tests**

`stock_lateral_toolkit/centering/tests/test_m1_offsets.py`:
```python
import numpy as np
import pytest


def _rows(y_left=1.7, y_right=-1.9):
    """Proposals for one frame at 3 distances, both sides accepted. Road-frame y
    (+LEFT): left line +1.7, right line -1.9 => midpoint road -0.1 => canonical
    offset (car LEFT of center, calibrated y of midpoint) = +0.1."""
    rows = []
    for x in (8.0, 12.0, 16.0):
        rows.append(dict(frame_idx=0, x_m=x, side="left", y_road=y_left, contrast=9.0,
                         auto_ok=True, verdict="", corrected_u_px="", u_px=0.0, v_px=0.0))
        rows.append(dict(frame_idx=0, x_m=x, side="right", y_road=y_right, contrast=9.0,
                         auto_ok=True, verdict="", corrected_u_px="", u_px=0.0, v_px=0.0))
    return rows


def test_frame_offset_sign_convention():
    from stock_lateral_toolkit.centering import m1_offsets as M
    off = M.frame_offset_cam(_rows())
    assert off == pytest.approx(+0.10, abs=1e-9)   # car 0.10 m LEFT of center


def test_trust_rule():
    from stock_lateral_toolkit.centering import m1_offsets as M
    reviewed = [dict(verdict="accept", u_px=100.0, corrected_u_px="")] * 17 \
             + [dict(verdict="correct", u_px=100.0, corrected_u_px="103.0")] * 1 \
             + [dict(verdict="reject", u_px=100.0, corrected_u_px="")] * 2
    ok, frac = M.trust_check(reviewed)
    assert ok and frac == pytest.approx(0.90)
    reviewed_bad = [dict(verdict="reject", u_px=0.0, corrected_u_px="")] * 5 \
                 + [dict(verdict="accept", u_px=0.0, corrected_u_px="")] * 5
    ok2, frac2 = M.trust_check(reviewed_bad)
    assert not ok2 and frac2 == pytest.approx(0.50)


def test_frame_offset_requires_both_sides():
    from stock_lateral_toolkit.centering import m1_offsets as M
    rows = [r for r in _rows() if r["side"] == "left"]
    assert M.frame_offset_cam(rows) is None
```

- [ ] **Step 3: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_m1_offsets.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... m1_offsets`.

- [ ] **Step 4: Implement `stock_lateral_toolkit/centering/m1_offsets.py`**

```python
"""M1d: enforce the pre-registered annotation trust rule, then turn accepted proposals
into per-frame physical offsets (canonical sign: + = vehicle LEFT of lane center) with
the declared per-frame sigma. Writes results/m1/m1_results.json (+ per-frame CSV).

RUN: .venv311/bin/python stock_lateral_toolkit/centering/m1_offsets.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering import ground_plane as G


def trust_check(reviewed_rows) -> tuple[bool, float]:
    """M0 §2: accepted = verdict 'accept' OR ('correct' with |corrected-u| <= PX_CORRECT_TOL)."""
    n = len(reviewed_rows)
    if n == 0:
        return False, 0.0
    good = 0
    for r in reviewed_rows:
        v = (r.get("verdict") or "").strip().lower()
        if v == "accept":
            good += 1
        elif v == "correct" and r.get("corrected_u_px", "") != "":
            if abs(float(r["corrected_u_px"]) - float(r["u_px"])) <= CC.PX_CORRECT_TOL:
                good += 1
    frac = good / n
    return frac >= CC.REVIEW_ACCEPT_MIN, frac


def _accepted_y(r) -> float | None:
    """Effective road-frame y for one proposal row after review semantics."""
    v = (r.get("verdict") or "").strip().lower()
    if v == "reject":
        return None
    if v == "correct" and r.get("corrected_u_px", "") != "":
        # re-project the corrected pixel on the same row (v_px changes negligibly)
        rpy = [float(r["cal_roll"]), float(r["cal_pitch"]), float(r["cal_yaw"])]
        Hm = G.road_homography(rpy, float(r["height"]), G.fcam_intrinsics())
        _, y = G.road_from_pixel(Hm, float(r["corrected_u_px"]), float(r["v_px"]))
        return y
    if r["auto_ok"] in (True, "True", "true", "1"):
        return float(r["y_road"])
    return None


def frame_offset_cam(rows) -> float | None:
    """Canonical camera-relative offset for ONE frame's proposal rows.
    Needs >= 2 accepted distances per side. y_cal = -y_road; offset = midpoint y_cal."""
    per_side: dict[str, list[float]] = {"left": [], "right": []}
    for r in rows:
        y = _accepted_y(r)
        if y is not None:
            per_side[r["side"]].append(float(y))
    if len(per_side["left"]) < 2 or len(per_side["right"]) < 2:
        return None
    y_left_cal = G.y_cal_from_y_road(float(np.median(per_side["left"])))
    y_right_cal = G.y_cal_from_y_road(float(np.median(per_side["right"])))
    return float((y_left_cal + y_right_cal) / 2.0)


def main():
    m1 = CC.RESULTS_DIR / "m1"
    props = list(csv.DictReader(open(m1 / "proposals.csv")))
    manifest = {int(r["frame_idx"]): r for r in csv.DictReader(open(m1 / "frames_manifest.csv"))}
    from model_replay_sim.context import route_context
    height = float(route_context(CC.ROUTE).height)

    # merge review verdicts (by frame_idx+x_m+side) and manifest calib into proposal rows
    review = {}
    rev_path = m1 / "review_subset.csv"
    if rev_path.exists():
        for r in csv.DictReader(open(rev_path)):
            review[(int(r["frame_idx"]), float(r["x_m"]), r["side"])] = r
    reviewed_rows = []
    for r in props:
        key = (int(r["frame_idx"]), float(r["x_m"]), r["side"])
        if key in review:
            r["verdict"] = review[key]["verdict"]
            r["corrected_u_px"] = review[key]["corrected_u_px"]
            reviewed_rows.append(r)
        man = manifest[int(r["frame_idx"])]
        r["cal_roll"], r["cal_pitch"], r["cal_yaw"] = man["cal_roll"], man["cal_pitch"], man["cal_yaw"]
        r["height"] = height

    filled = [r for r in reviewed_rows if (r.get("verdict") or "").strip()]
    if len(filled) < len(reviewed_rows) or not reviewed_rows:
        raise SystemExit(f"USER GATE UNMET: review_subset.csv has {len(reviewed_rows) - len(filled)} "
                         f"unfilled verdicts (of {len(reviewed_rows)}). Fill it, then re-run.")
    ok, frac = trust_check(filled)
    print(f"trust rule: reviewed acceptance = {frac:.0%} (threshold {CC.REVIEW_ACCEPT_MIN:.0%})")
    if not ok:
        raise SystemExit("TRUST RULE FAILED (M0 §2): automated annotations are NOT trusted. "
                         "Full manual annotation pass required — fill verdict+corrected_u_px "
                         "for EVERY row of proposals.csv and re-run.")

    by_frame: dict[int, list] = {}
    for r in props:
        by_frame.setdefault(int(r["frame_idx"]), []).append(r)
    sigma = G.sigma_frame_m()
    out_rows, offsets = [], []
    for fi, rows in sorted(by_frame.items()):
        off = frame_offset_cam(rows)
        if off is None:
            continue
        man = manifest[fi]
        out_rows.append(dict(frame_idx=fi, mono_time=float(man["mono_time"]),
                             offset_cam_m=off, sigma_m=sigma,
                             v_ego=float(man["v_ego"]), speed_bin=int(man["speed_bin"]),
                             heading_bin=int(man["heading_bin"]),
                             in_m2_window=man["in_m2_window"] == "True",
                             logged_center_y0=float(man["logged_center_y0"])))
        offsets.append(off)

    offsets = np.array(offsets)
    logged = np.array([r["logged_center_y0"] for r in out_rows])
    res = {
        "n_frames": int(len(offsets)),
        "median_offset_cam_m": float(np.median(offsets)),
        "mad_offset_cam_m": float(np.median(np.abs(offsets - np.median(offsets)))),
        "sigma_frame_m": sigma,
        "sem_random_m": float(sigma / max(np.sqrt(len(offsets)), 1)),
        "systematic_note": "lever-arm/mean-roll/mean-yaw systematics ~±0.04 m do not average out (M0 §2)",
        "cam_offset_from_centerline_m": CC.CAM_OFFSET_FROM_CENTERLINE_M,
        "cam_offset_measured": bool(CC.CAM_OFFSET_MEASURED),
        "median_offset_vehicle_m": (float(np.median(offsets)) + CC.CAM_OFFSET_FROM_CENTERLINE_M)
                                   if CC.CAM_OFFSET_MEASURED else None,
        "overlap_logged": {
            "n": int(len(logged)),
            "median_delta_logged_minus_video_m": float(np.median(logged - offsets)),
            "mad_delta_m": float(np.median(np.abs((logged - offsets) - np.median(logged - offsets)))),
        },
        "per_speed_bin": {str(sb): float(np.median([r["offset_cam_m"] for r in out_rows
                                                    if r["speed_bin"] == sb]))
                          for sb in sorted({r["speed_bin"] for r in out_rows})},
        "review_acceptance": frac,
    }
    with open(m1 / "per_frame_offsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    (m1 / "m1_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if res["n_frames"] < CC.N_FRAMES_MIN:
        print(f"WARNING: n_frames < {CC.N_FRAMES_MIN} — M0 §2 floor unmet; "
              f"second-route pull required before M3 relies on M1.")
    if not CC.CAM_OFFSET_MEASURED:
        print("WARNING: lever arm not measured (Task 9 step 1) — vehicle-frame P unavailable; "
              "M3/R1 proceed camera-relative, final P blocked.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_m1_offsets.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Run on the real annotations (after the user review, Task 8 step 6)**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/m1_offsets.py
```
Expected artifact: `results/m1/m1_results.json` + `results/m1/per_frame_offsets.csv`. **Pre-registered interpretation (M0 §1/§2):** `median_offset_cam_m` is the camera-relative physical offset `P_cam`; `overlap_logged.median_delta_logged_minus_video_m` is the direct log-vs-video estimate of the definition bias `Δmid` (it must agree with R1's replay-based estimate within the M3 tolerances). The script HALTS (SystemExit) if the review is unfilled or the trust rule fails.

- [ ] **Step 7: Commit**

```bash
git add stock_lateral_toolkit/centering/m1_offsets.py stock_lateral_toolkit/centering/tests/test_m1_offsets.py
git commit -m "centering M1d: trust-rule-gated physical offsets + declared uncertainty budget

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: M2b — cross-model consensus replays + stats

**Files:**
- Create: `stock_lateral_toolkit/centering/consensus.py`
- Test: `stock_lateral_toolkit/centering/tests/test_consensus.py`

SP002 + Nevada + CD210 on IDENTICAL frames (the Task-5 windows) with `capture_outputs=("lane_lines","lane_lines_prob")`. All three bundles are anchor-validated (`model_replay_sim/config.py` BUNDLES) so no `_warn_if_unvalidated` warnings fire. **Compute: 3 bundles × ~2300 frames ≈ 47 min serial; ~18–20 min wall at 3–4 workers.** Parallelize across (bundle, window) jobs with `ProcessPoolExecutor` + spawn — NEVER inside tinygrad (replay is single-threaded by design, `TINYGRAD_ENV THREADS=0`).

- [ ] **Step 1 (optional, 🧑 USER-GATED): second-route video pull.** If the user wants a second ROUTE (stronger than a second window), pull `fcamera.hevc`/`ecamera.hevc` for `00000004--3d3385646d` from the device into `explorer_st_logs/route_stock04/<segdirs>/` alongside NAS rlogs (device SSH per `reference_device_config`; old drives may have rotated off — check first; ~70 MB/segment/camera × 26 segments). Then rerun Task 5 with `ROUTE="route_stock04"` appended (add a `ROUTES_EXTRA` entry in config and a second windows file). This plan's default path proceeds with route_stock05 only.

- [ ] **Step 2: Write the failing test (lane-center math on synthetic capture)**

`stock_lateral_toolkit/centering/tests/test_consensus.py`:
```python
import numpy as np
import pytest


def test_lane_center_series_x0_and_interp():
    from stock_lateral_toolkit.centering import consensus as CO
    n = 7
    ll = np.zeros((n, 4, 33, 2), dtype=np.float32)
    ll[:, 1, :, 0] = -1.6   # inner-left y (calibrated +right)
    ll[:, 2, :, 0] = +1.8   # inner-right
    center0, width0 = CO.lane_center_series(ll, x_eval=0.0)
    assert center0.shape == (n,)
    assert center0[0] == pytest.approx(0.1)
    assert width0[0] == pytest.approx(3.4)
    # linear-in-x lane lines: interp at x=10 must match exactly
    from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
    xg = np.asarray(ModelConstants.X_IDXS)
    ll2 = ll.copy()
    ll2[:, 1, :, 0] = -1.6 + 0.01 * xg
    ll2[:, 2, :, 0] = +1.8 + 0.01 * xg
    c10, _ = CO.lane_center_series(ll2, x_eval=10.0)
    assert c10[0] == pytest.approx(0.1 + 0.01 * 10.0, abs=1e-6)


def test_pair_stats():
    from stock_lateral_toolkit.centering import consensus as CO
    a = np.array([0.10, 0.11, 0.09, 0.10])
    b = np.array([0.02, 0.03, 0.01, 0.02])
    s = CO.pair_stats(a, b)
    assert s["median_delta_m"] == pytest.approx(0.08)
    assert s["n"] == 4
```

- [ ] **Step 3: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_consensus.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... consensus`.

- [ ] **Step 4: Implement `stock_lateral_toolkit/centering/consensus.py`**

```python
"""M2b: cross-model lane-center consensus on IDENTICAL frames.

Replays every bundle in config.BUNDLES_M2 over the shared Task-5 windows with
lane-line capture, then computes per-model lane-center series, cross-model pairwise
disagreement, the consensus (per-frame median across models), and the overlap
comparison against M1 video frames.

RUN:
  .venv311/bin/python stock_lateral_toolkit/centering/consensus.py --replay   # heavy (~20 min wall)
  .venv311/bin/python stock_lateral_toolkit/centering/consensus.py --stats    # seconds
"""
from __future__ import annotations

import csv
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC

M2_DIR = CC.RESULTS_DIR / "m2"


def lane_center_series(lane_lines: np.ndarray, x_eval: float = 0.0):
    """(N,4,33,2) captured lane_lines -> (center_y, width) at forward distance x_eval.
    Calibrated y (+RIGHT); center>0 == car LEFT of that model's lane center (canonical)."""
    yl = lane_lines[:, 1, :, 0]
    yr = lane_lines[:, 2, :, 0]
    if x_eval == 0.0:
        yl0, yr0 = yl[:, 0], yr[:, 0]
    else:
        from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
        xg = np.asarray(ModelConstants.X_IDXS, dtype=float)
        yl0 = np.array([np.interp(x_eval, xg, row) for row in yl])
        yr0 = np.array([np.interp(x_eval, xg, row) for row in yr])
    return (yl0 + yr0) / 2.0, (yr0 - yl0)


def pair_stats(a: np.ndarray, b: np.ndarray) -> dict:
    both = np.isfinite(a) & np.isfinite(b)
    d = a[both] - b[both]
    return {"n": int(both.sum()),
            "median_delta_m": float(np.median(d)) if both.any() else float("nan"),
            "iqr_m": float(np.subtract(*np.percentile(d, [75, 25]))) if both.any() else float("nan"),
            "corr": float(np.corrcoef(a[both], b[both])[0, 1]) if both.sum() > 2 else float("nan")}


def _npz_path(bundle: str, window_id: int) -> Path:
    return M2_DIR / f"replay_{bundle}_w{window_id}.npz"


def _replay_job(args) -> str:
    bundle, window = args
    from model_replay_sim.infer import replay_window
    r = replay_window(bundle, window["route_id"], window["mono_times"],
                      capture_outputs=("lane_lines", "lane_lines_prob"))
    out = _npz_path(bundle, window["window_id"])
    np.savez_compressed(out,
                        mono_time=r["mono_time"], v_ego=r["v_ego"],
                        desired_curvature=r["desired_curvature"],
                        lane_lines=r["captured"]["lane_lines"],
                        lane_lines_prob=r["captured"]["lane_lines_prob"],
                        camera_offset_used=r["camera_offset_used"],
                        split_index=window["split_index"])
    return str(out)


def run_replays() -> None:
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()
    M2_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(b, w) for b in CC.BUNDLES_M2 for w in wins
            if not _npz_path(b, w["window_id"]).exists()]      # resumable
    print(f"{len(jobs)} replay jobs (bundles {CC.BUNDLES_M2} x {len(wins)} windows), "
          f"{CC.MAX_WORKERS_REPLAY} workers")
    with ProcessPoolExecutor(max_workers=CC.MAX_WORKERS_REPLAY,
                             mp_context=mp.get_context("spawn")) as ex:
        for done in ex.map(_replay_job, jobs):
            print("done:", done)


def run_stats() -> None:
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()
    res = {"bundles": list(CC.BUNDLES_M2), "windows": [w["window_id"] for w in wins],
           "per_window": {}, "vs_m1": {}}
    centers_all = {b: [] for b in CC.BUNDLES_M2}
    mono_all = []
    for w in wins:
        wid = w["window_id"]; split = int(w["split_index"])
        per = {}
        series = {}
        for b in CC.BUNDLES_M2:
            z = np.load(_npz_path(b, wid))
            c0, wd0 = lane_center_series(z["lane_lines"], 0.0)
            c10, _ = lane_center_series(z["lane_lines"], 10.0)
            series[b] = c0[split:]
            per[b] = {"own_center_median_y0_m": float(np.median(c0[split:])),
                      "own_center_median_y10_m": float(np.median(c10[split:])),
                      "width_median_m": float(np.median(wd0[split:])),
                      "min_inner_prob_median": float(np.median(
                          np.minimum(z["lane_lines_prob"][split:, 3], z["lane_lines_prob"][split:, 5])))}
            centers_all[b].append(c0[split:])
            if b == CC.BUNDLES_M2[0]:
                mono_all.append(np.asarray(z["mono_time"])[split:])
        pairs = {}
        bl = list(CC.BUNDLES_M2)
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                pairs[f"{bl[i]}-{bl[j]}"] = pair_stats(series[bl[i]], series[bl[j]])
        stacked = np.vstack([series[b] for b in bl])
        res["per_window"][str(wid)] = {"per_bundle": per, "pairs": pairs,
                                       "consensus_median_y0_m": float(np.median(np.median(stacked, axis=0)))}

    # overlap vs M1 (matched by mono_time within 0.001 s — sampler drew from these windows)
    m1_csv = CC.RESULTS_DIR / "m1" / "per_frame_offsets.csv"
    if m1_csv.exists():
        m1_rows = [r for r in csv.DictReader(open(m1_csv)) if r["in_m2_window"] == "True"]
        mono_cat = np.concatenate(mono_all)
        for b in CC.BUNDLES_M2:
            cat = np.concatenate(centers_all[b])
            deltas = []
            for r in m1_rows:
                k = int(np.argmin(np.abs(mono_cat - float(r["mono_time"]))))
                if abs(mono_cat[k] - float(r["mono_time"])) < 1e-3:
                    deltas.append(float(cat[k]) - float(r["offset_cam_m"]))
            res["vs_m1"][b] = {"n_overlap": len(deltas),
                               "median_model_minus_video_m": float(np.median(deltas)) if deltas else None,
                               "mad_m": float(np.median(np.abs(np.array(deltas) - np.median(deltas)))) if deltas else None}
    else:
        print("NOTE: m1 per-frame offsets not present yet; vs_m1 section empty (re-run --stats after Task 9)")

    (M2_DIR / "m2_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    if "--replay" in sys.argv:
        run_replays()
    elif "--stats" in sys.argv:
        run_stats()
    else:
        print("usage: consensus.py --replay | --stats")
```
Note on `lane_lines_prob` indices: the raw captured vector is the sigmoid `(8,)`; per-line probabilities sit at odd indices (`fill_model_msg.py:130` uses `[0, 1::2]`), so inner-left = index 3, inner-right = index 5.

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_consensus.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Run the replays (heavy) then the stats**

```bash
caffeinate -i .venv311/bin/python stock_lateral_toolkit/centering/consensus.py --replay
.venv311/bin/python stock_lateral_toolkit/centering/consensus.py --stats
```
Expected artifacts: `results/m2/replay_{SP002,Nevada,CD210}_w{0,1}.npz`, `results/m2/m2_results.json`. **Pre-registered interpretation (M0 §3):** `vs_m1.SP002.median_model_minus_video_m` is the replay-based `Δmid` (must satisfy |M1−M2_SP002| ≤ 0.06 m in M3); pairwise `median_delta_m` > 0.10 m marks a genuine cross-model lane-center definition difference. Re-run `--stats` after Task 9 completes if M1 finished later (the replays themselves never need re-running).

- [ ] **Step 7: Commit**

```bash
git add stock_lateral_toolkit/centering/consensus.py stock_lateral_toolkit/centering/tests/test_consensus.py
git commit -m "centering M2b: cross-model lane-center consensus replays + disagreement stats

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: M3 — reconciliation + verdict table

**Files:**
- Create: `stock_lateral_toolkit/centering/reconcile.py`
- Test: `stock_lateral_toolkit/centering/tests/test_reconcile.py`

- [ ] **Step 1: Write the failing test (verdict logic only — inputs injected)**

`stock_lateral_toolkit/centering/tests/test_reconcile.py`:
```python
import pytest


def test_verdict_pass_and_decomposition():
    from stock_lateral_toolkit.centering import reconcile as R
    v = R.verdict(m1_median_cam=0.02, m2_sp002_vs_m1=0.04, m2_sp002_median_own=0.10,
                  logged_median=0.103, consensus_pairs_max_abs=0.06)
    assert v["gate_m1_vs_m2"] == "PASS"          # 0.04 <= 0.06
    assert v["gate_m2_vs_logged"] == "PASS"      # |0.10-0.103| <= 0.04
    assert v["proceed_to_R"] is True
    # decomposition: L = P_cam + dmid  ->  dmid = 0.103 - 0.02
    assert v["dmid_logged_minus_video_m"] == pytest.approx(0.083)
    assert v["consensus_flag"] == "AGREE"


def test_verdict_stop_on_method_disagreement():
    from stock_lateral_toolkit.centering import reconcile as R
    v = R.verdict(m1_median_cam=0.02, m2_sp002_vs_m1=0.09, m2_sp002_median_own=0.10,
                  logged_median=0.103, consensus_pairs_max_abs=0.15)
    assert v["gate_m1_vs_m2"] == "FAIL"
    assert v["proceed_to_R"] is False
    assert v["consensus_flag"] == "DEFINITIONS_DIFFER"
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_reconcile.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... reconcile`.

- [ ] **Step 3: Implement `stock_lateral_toolkit/centering/reconcile.py`**

```python
"""M3: reconcile M1 (video), M2 (cross-model replay), and the logs against the
pre-registered tolerances (M0 §3), and decompose the logged offset:
    L (logged, model-frame) = P_cam (true, camera-relative) + dmid (definition bias).
Writes results/m3_verdict.json + results/m3_verdict.md.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/reconcile.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def verdict(m1_median_cam: float, m2_sp002_vs_m1: float, m2_sp002_median_own: float,
            logged_median: float, consensus_pairs_max_abs: float) -> dict:
    """Pure verdict logic against M0 §3 (unit-tested; main() feeds real numbers)."""
    g1 = abs(m2_sp002_vs_m1) <= CC.TOL_M1_VS_M2_M
    g2 = abs(m2_sp002_median_own - logged_median) <= CC.TOL_M2_VS_LOGGED_M
    return {
        "gate_m1_vs_m2": "PASS" if g1 else "FAIL",
        "gate_m1_vs_m2_value_m": float(m2_sp002_vs_m1),
        "gate_m2_vs_logged": "PASS" if g2 else "FAIL",
        "gate_m2_vs_logged_value_m": float(m2_sp002_median_own - logged_median),
        "consensus_flag": "AGREE" if consensus_pairs_max_abs <= CC.TOL_CONSENSUS_DISAGREE_M
                          else "DEFINITIONS_DIFFER",
        "P_cam_m": float(m1_median_cam),
        "L_logged_m": float(logged_median),
        "dmid_logged_minus_video_m": float(logged_median - m1_median_cam),
        "proceed_to_R": bool(g1 and g2),
    }


def main():
    m1 = json.loads((CC.RESULTS_DIR / "m1" / "m1_results.json").read_text())
    m2 = json.loads((CC.RESULTS_DIR / "m2" / "m2_results.json").read_text())

    # logged L: median lane_center_y0 over the SAME eligibility the sampler used
    z = dict(np.load(CC.CACHE_NPZ))
    from stock_lateral_toolkit.centering.frame_sampler import eligibility
    ok, _ = eligibility(z)
    logged_median = float(np.nanmedian(np.asarray(z["lane_center_y0"], float)[ok]))

    sp = "SP002"
    own = [w["per_bundle"][sp]["own_center_median_y0_m"] for w in m2["per_window"].values()]
    pair_max = max(abs(p["median_delta_m"]) for w in m2["per_window"].values()
                   for p in w["pairs"].values())
    vs_m1 = m2["vs_m1"].get(sp, {})
    if vs_m1.get("median_model_minus_video_m") is None:
        raise SystemExit("M3 blocked: m2_results.json has no vs_m1 overlap — "
                         "run Task 9 then `consensus.py --stats` again.")

    v = verdict(m1_median_cam=float(m1["median_offset_cam_m"]),
                m2_sp002_vs_m1=float(vs_m1["median_model_minus_video_m"]),
                m2_sp002_median_own=float(np.median(own)),
                logged_median=logged_median,
                consensus_pairs_max_abs=float(pair_max))
    v["n_m1_frames"] = m1["n_frames"]
    v["n_overlap"] = vs_m1["n_overlap"]
    v["P_vehicle_m"] = m1.get("median_offset_vehicle_m")

    (CC.RESULTS_DIR / "m3_verdict.json").write_text(json.dumps(v, indent=2))
    lines = ["# M3 reconciliation verdict", "",
             "| check | value (m) | tolerance (m) | verdict |", "|---|---|---|---|",
             f"| M1 vs M2(SP002) | {v['gate_m1_vs_m2_value_m']:+.3f} | ±{CC.TOL_M1_VS_M2_M} | {v['gate_m1_vs_m2']} |",
             f"| M2(SP002) vs logged | {v['gate_m2_vs_logged_value_m']:+.3f} | ±{CC.TOL_M2_VS_LOGGED_M} | {v['gate_m2_vs_logged']} |",
             f"| cross-model max pair Δ | {pair_max:+.3f} | {CC.TOL_CONSENSUS_DISAGREE_M} | {v['consensus_flag']} |",
             "", f"Decomposition: L = {v['L_logged_m']:+.3f} = P_cam {v['P_cam_m']:+.3f} "
                 f"+ dmid {v['dmid_logged_minus_video_m']:+.3f}",
             f"P (vehicle) = {v['P_vehicle_m']}", f"proceed_to_R = {v['proceed_to_R']}"]
    (CC.RESULTS_DIR / "m3_verdict.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_reconcile.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Run reconciliation on real results (after T9 + T10)**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/reconcile.py
```
Expected artifact: `results/m3_verdict.{json,md}`. **Pre-registered interpretation (M0 §3):** `proceed_to_R = false` is a HARD STOP — no R-phase verdict may be issued; the method disagreement becomes the object of investigation (check calibration-era mismatch, annotation systematics, capture bugs) before anything else.

- [ ] **Step 6: Commit**

```bash
git add stock_lateral_toolkit/centering/reconcile.py stock_lateral_toolkit/centering/tests/test_reconcile.py
git commit -m "centering M3: method reconciliation + logged-offset decomposition against pre-registered tolerances

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Toolkit caches for all stock drives + `roll` column

**Files:**
- Modify: `stock_lateral_toolkit/extract_drive.py` (append `roll` column — APPEND-ONLY so existing positional consumers keep working)
- Regenerate/create: `stock_lateral_toolkit/cache/stock_{00,01,02,03}.npz`, `stock_hiram_{04,05}.npz`

`roll` source verified: stock rlogs carry `liveLocationKalman.orientationNED` (the retrospective cache's `roll` for route_stock05 is 93% finite via `retrospective_lateral/code/extract.py:360`). R2 (crown) and R3 (dependence) need it in the toolkit-format caches.

- [ ] **Step 1: Modify `stock_lateral_toolkit/extract_drive.py`**

(a) In `_extract_seg`, add a `rolls` collector. After `mv = []; cs = []; co = []; cc = []; gps = []` append `; rolls = []`. In the `liveLocationKalman` branch, extend to also collect roll:
```python
        elif w == "liveLocationKalman":
            try:
                loc = m.liveLocationKalman
                o = loc.orientationNED
                if o.valid and len(o.value) >= 1:
                    rolls.append((t, float(o.value[0])))
                pg = loc.positionGeodetic
                if pg.valid and len(pg.value) >= 2 and abs(pg.value[0]) > 1:
                    gps.append((t, float(pg.value[0]), float(pg.value[1])))
            except Exception:
                pass
```
(b) After `cs.sort(); ...` add `rolls.sort()` and `rollt = [x[0] for x in rolls]`.
(c) In the output loop, before `out.append(...)`:
```python
        rr = pv(rolls, rollt, t)
        roll = rr[1] if rr is not None and (t - rr[0]) < 1.0 else np.nan
```
and append `roll` as the LAST tuple element.
(d) In `main()`, change the cols line to:
```python
    cols = "t lat lon vEgo steerDeg yawRate pressed latActive enabled cmd_curv model_curv ach_curv offset innerProb roll".split()
```

- [ ] **Step 2: Regenerate ALL stock caches (NAS reads; ~5–15 min per full drive)**

```bash
NAS="/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock"
.venv311/bin/python stock_lateral_toolkit/extract_drive.py "$NAS/00000000--c181384b0c" stock_lateral_toolkit/cache/stock_00.npz
.venv311/bin/python stock_lateral_toolkit/extract_drive.py "$NAS/00000001--e80958d9ca" stock_lateral_toolkit/cache/stock_01.npz
.venv311/bin/python stock_lateral_toolkit/extract_drive.py "$NAS/00000002--5da5840d8d" stock_lateral_toolkit/cache/stock_02.npz
.venv311/bin/python stock_lateral_toolkit/extract_drive.py "$NAS/00000003--b37c613b41" stock_lateral_toolkit/cache/stock_03.npz
.venv311/bin/python stock_lateral_toolkit/extract_drive.py "$NAS/00000004--3d3385646d" stock_lateral_toolkit/cache/stock_hiram_04.npz
.venv311/bin/python stock_lateral_toolkit/extract_drive.py "$NAS/00000005--ef46fdca62" stock_lateral_toolkit/cache/stock_hiram_05.npz
```
Expected: per-drive frame counts printed. Drives 00/01/03 are SPARSE on the NAS (only some segments were pulled) — usable but thin; 02 (64 segs), 04, 05 are full.

- [ ] **Step 3: Verify the new column + roll plausibility**

```bash
.venv311/bin/python - <<'EOF'
import numpy as np
for n in ["stock_00","stock_01","stock_02","stock_03","stock_hiram_04","stock_hiram_05"]:
    z = np.load(f"stock_lateral_toolkit/cache/{n}.npz", allow_pickle=True)
    cols = list(z["cols"]); assert cols[-1] == "roll", (n, cols)
    roll = z["data"][:, cols.index("roll")]
    print(n, z["data"].shape, "roll finite:", round(float(np.mean(np.isfinite(roll))), 2),
          "median:", round(float(np.nanmedian(roll)), 4))
EOF
```
Expected: `roll` last column, finite fraction > 0.5, |median| < 0.05 rad for every drive.

- [ ] **Step 4: Regression — negative control still null**

```bash
.venv311/bin/python stock_lateral_toolkit/analyze_compare.py hiram_null_04v05
```
Expected: runs cleanly on the regenerated caches (the appended column does not disturb positional indexing — `analyze_compare.COLS` maps the first 14 columns) and the 04-vs-05 negative control remains null (no consistent "better" arm across speed bins; this re-validates the toolkit acceptance after the change).

- [ ] **Step 5: Commit**

```bash
git add stock_lateral_toolkit/extract_drive.py
git commit -m "toolkit: append roll (liveLocationKalman.orientationNED) to extract_drive caches for crown analysis

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
(Caches themselves are gitignored data artifacts — do not commit npz files.)

---

### Task 13: R2 — crown/roll analysis (FPR calibration FIRST, then the estimator)

**Files:**
- Create: `stock_lateral_toolkit/centering/r2_calibration.py`
- Create: `stock_lateral_toolkit/centering/r2_crown.py`

Estimator design (pre-registered in M0 §4/§7): per drive, 30 s windows on eligible frames → per-window medians of signed `offset` (canonical), `roll`, speed; association = `shared.within_drive_spearman(df, xcol='roll_med', ycol='offset_med', ctrl='spd_mph')` (Freedman-Lane permutation — the only calibrated within-drive gate in the toolkit; y = the lateral target per the qa_calibration note). **Its p-value may not be used until `r2_calibration.py` shows FPR ≤ 0.07 on THIS data shape** (cross-drive real-pairing null, the assumption-free pattern from `stock_lateral_toolkit/qa_calibration.py:fpr_crossdrive`). Secondary (descriptive, no p): direction-paired hiram 04-vs-05 per-GPS-cell signed-offset deltas vs roll deltas.

- [ ] **Step 1: Implement the shared window-table builder inside `r2_crown.py`**

`stock_lateral_toolkit/centering/r2_crown.py`:
```python
"""R2: is the centering offset a CROWN RESPONSE? Within-drive offset~roll association
(speed-controlled, FPR-calibrated) + direction-paired same-road descriptive check.

RUN (after r2_calibration.py PASSES):
  .venv311/bin/python stock_lateral_toolkit/centering/r2_crown.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLKIT = REPO_ROOT / "stock_lateral_toolkit"
if str(TOOLKIT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT))

import shared  # stock_lateral_toolkit/shared.py (within_drive_spearman etc.)
from stock_lateral_toolkit.centering import config as CC

DRIVES = ["stock_00", "stock_01", "stock_02", "stock_03", "stock_hiram_04", "stock_hiram_05"]
FS = 20.0  # extract_drive caches are modelV2-cadence


def load_drive(name: str):
    z = np.load(CC.TOOLKIT_CACHE / f"{name}.npz", allow_pickle=True)
    cols = [str(c) for c in z["cols"]]
    d = {c: z["data"][:, i] for i, c in enumerate(cols)}
    return d


def window_table(drives=DRIVES) -> pd.DataFrame:
    """Per 30 s window: median signed offset / roll / speed on eligible frames
    (engaged, unpressed, moving, finite offset+roll)."""
    rows = []
    w = int(CC.R2_WINDOW_S * FS)
    for name in drives:
        try:
            d = load_drive(name)
        except FileNotFoundError:
            print(f"NOTE: cache {name} missing, skipped")
            continue
        elig = ((d["latActive"] > 0.5) & (d["pressed"] < 0.5)
                & (d["vEgo"] >= CC.V_MIN_MPS)
                & np.isfinite(d["offset"]) & np.isfinite(d["roll"]))
        n = len(d["t"])
        for a in range(0, n - w, w):
            m = elig[a:a + w]
            if m.sum() < 0.5 * w:
                continue
            rows.append(dict(
                drive_id=name,
                offset_med=float(np.median(d["offset"][a:a + w][m])),
                roll_med=float(np.median(d["roll"][a:a + w][m])),
                spd_mph=float(np.median(d["vEgo"][a:a + w][m]) * 2.23694),
                lat_med=float(np.nanmedian(d["lat"][a:a + w][m])),
                lon_med=float(np.nanmedian(d["lon"][a:a + w][m])),
            ))
    return pd.DataFrame(rows)


def direction_paired_hiram(df_04: dict, df_05: dict) -> dict:
    """Descriptive: per shared ~150 m GPS cell, (offset_04 - offset_05) vs (roll_04 - roll_05).
    Pure translation/model bias predicts offset deltas ~0 regardless of roll deltas;
    a crown response predicts offset deltas tracking roll deltas."""
    def cells(d):
        elig = (d["latActive"] > 0.5) & (d["pressed"] < 0.5) & (d["vEgo"] >= CC.V_MIN_MPS) \
               & np.isfinite(d["offset"]) & np.isfinite(d["roll"]) & np.isfinite(d["lat"])
        cell = (np.round(d["lat"] / 0.0015).astype(np.int64) * 100000
                + np.round(d["lon"] / 0.0015).astype(np.int64))
        out = {}
        for c in np.unique(cell[elig]):
            m = elig & (cell == c)
            if m.sum() >= 40:   # >= 2 s of frames
                out[int(c)] = (float(np.median(d["offset"][m])), float(np.median(d["roll"][m])))
        return out
    a, b = cells(df_04), cells(df_05)
    common = sorted(set(a) & set(b))
    d_off = np.array([a[c][0] - b[c][0] for c in common])
    d_roll = np.array([a[c][1] - b[c][1] for c in common])
    r = float(np.corrcoef(d_off, d_roll)[0, 1]) if len(common) > 5 else float("nan")
    return {"n_cells": len(common),
            "median_abs_offset_delta_m": float(np.median(np.abs(d_off))) if len(common) else float("nan"),
            "corr_offset_delta_vs_roll_delta": r}


def main():
    cal = CC.RESULTS_DIR / "r2" / "calibration.json"
    if not cal.exists():
        raise SystemExit("R2 blocked: run r2_calibration.py first (M0 §7 — no p-values "
                         "from an uncalibrated estimator).")
    cal_res = json.loads(cal.read_text())
    if not cal_res["passed"]:
        raise SystemExit(f"R2 blocked: calibration FAILED (FPR={cal_res['fpr']}); "
                         "fix the estimator before interpreting p-values.")

    df = window_table()
    res = {"n_windows": int(len(df)), "n_drives": int(df["drive_id"].nunique()),
           "calibration_fpr": cal_res["fpr"]}
    r = shared.within_drive_spearman(df, "roll_med", "offset_med", ctrl="spd_mph")
    res["within_drive"] = {k: (float(v) if np.isfinite(v) else None) for k, v in r.items()}
    res["crown_significant"] = bool(np.isfinite(r["p"]) and r["p"] < CC.CROWN_P_MAX
                                    and abs(r["r"]) >= CC.CROWN_R_MIN)
    # crown fraction: robust slope (Theil-Sen light: median of pairwise slopes on demeaned data)
    g = df.dropna(subset=["roll_med", "offset_med"])
    x = g["roll_med"].values - g["roll_med"].values.mean()
    y = g["offset_med"].values - g["offset_med"].values.mean()
    idx = np.random.default_rng(7).choice(len(x), size=(min(4000, len(x) * (len(x) - 1) // 2), 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    slopes = (y[idx[:, 0]] - y[idx[:, 1]]) / (x[idx[:, 0]] - x[idx[:, 1]] + 1e-12)
    res["slope_m_per_rad"] = float(np.median(slopes))
    res["crown_component_m"] = float(np.median(slopes) * np.median(np.abs(g["roll_med"])))

    try:
        res["direction_paired_hiram"] = direction_paired_hiram(
            load_drive("stock_hiram_04"), load_drive("stock_hiram_05"))
    except FileNotFoundError:
        res["direction_paired_hiram"] = None

    out = CC.RESULTS_DIR / "r2" / "r2_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement `stock_lateral_toolkit/centering/r2_calibration.py`**

```python
"""FPR/power calibration for the R2 estimator (MUST pass before r2_crown.py's p is
trusted — M0 §7). Two nulls + injected-effect power, on the REAL window table shape:

NULL A (assumption-free, qa_calibration.fpr_crossdrive pattern): pair the real roll
series of one drive with the real offset series of ANOTHER (independent partner per
drive per rep) — true H0 with full real autocorrelation/drift. Target FPR <= 0.07.
NULL B (within-drive circular shift of roll by >= 60 windows): breaks the pairing,
keeps each series' own structure. Reported; A is the gate.
POWER: inject offset' = offset + beta*roll with beta sized to crown components of
0.03 / 0.06 m at the observed roll spread; report detection rate at alpha=0.05.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r2_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLKIT = REPO_ROOT / "stock_lateral_toolkit"
if str(TOOLKIT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT))

import shared
from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering.r2_crown import window_table

REPS = 300
NPERM = 400


def _test_p(df: pd.DataFrame) -> float:
    return shared.within_drive_spearman(df, "roll_med", "offset_med", ctrl="spd_mph",
                                        nperm=NPERM, nboot=0)["p"]


def fpr_crossdrive_pairing(df: pd.DataFrame, rng) -> float:
    groups = [g.reset_index(drop=True) for _, g in df.groupby("drive_id")]
    ng = len(groups)
    hits = tot = 0
    for _ in range(REPS):
        partner = [(k + int(rng.integers(1, ng))) % ng for k in range(ng)]
        rows = []
        for k in range(ng):
            gx, gy = groups[k], groups[partner[k]]
            n = min(len(gx), len(gy))
            if n < 4:
                continue
            for i in range(n):
                rows.append((k, float(gx["roll_med"][i]), float(gy["offset_med"][i]),
                             float(gx["spd_mph"][i])))
        d = pd.DataFrame(rows, columns=["drive_id", "roll_med", "offset_med", "spd_mph"])
        if d["drive_id"].nunique() < 2:
            continue
        p = _test_p(d)
        if np.isfinite(p):
            tot += 1
            hits += p < 0.05
    return hits / max(tot, 1)


def power_injected(df: pd.DataFrame, crown_m: float, rng) -> float:
    roll_spread = float(np.median(np.abs(df["roll_med"] - df["roll_med"].median())))
    beta = crown_m / max(roll_spread, 1e-6)
    hits = tot = 0
    for _ in range(REPS // 3):
        d = df.copy()
        # destroy any real association first (within-drive shuffle of offset), then inject
        d["offset_med"] = d.groupby("drive_id")["offset_med"].transform(
            lambda s: rng.permutation(s.values))
        d["offset_med"] = d["offset_med"] + beta * d["roll_med"]
        p = _test_p(d)
        if np.isfinite(p):
            tot += 1
            hits += p < 0.05
    return hits / max(tot, 1)


def main():
    rng = np.random.default_rng(20260703)
    df = window_table()
    print(f"window table: {len(df)} windows over {df['drive_id'].nunique()} drives")
    fpr = fpr_crossdrive_pairing(df, rng)
    pw_small = power_injected(df, 0.03, rng)
    pw_med = power_injected(df, 0.06, rng)
    res = {"n_windows": int(len(df)), "fpr": float(fpr),
           "fpr_max": CC.R2_FPR_MAX, "passed": bool(fpr <= CC.R2_FPR_MAX),
           "power_crown_0.03m": float(pw_small), "power_crown_0.06m": float(pw_med)}
    out = CC.RESULTS_DIR / "r2" / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if not res["passed"]:
        print("CALIBRATION FAILED — r2_crown.py p-values are NOT interpretable; "
              "the estimator (window length / control set) must be revised and re-calibrated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run calibration (expect ~10–20 min: REPS×NPERM permutations)**

```bash
caffeinate -i .venv311/bin/python stock_lateral_toolkit/centering/r2_calibration.py
```
Expected artifact: `results/r2/calibration.json` with `fpr <= 0.07` → `passed: true`, and power numbers (record them; if power at 0.06 m < 0.5, R2's null result is uninformative and must be reported as such). **If calibration fails, r2_crown.py refuses to run — that is by design.**

- [ ] **Step 4: Run the R2 estimator**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/r2_crown.py
```
Expected artifact: `results/r2/r2_results.json`. **Pre-registered interpretation (M0 §4):** crown evidence = `crown_significant: true` (calibrated p < 0.05 AND |r| ≥ 0.30); `crown_component_m` feeds the R-verdict fraction; `direction_paired_hiram` is descriptive corroboration only (no p).

- [ ] **Step 5: Commit**

```bash
git add stock_lateral_toolkit/centering/r2_calibration.py stock_lateral_toolkit/centering/r2_crown.py
git commit -m "centering R2: crown/roll estimator with mandatory FPR calibration + direction-paired check

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 14: R1 — translation-vs-preference discriminator

**Files:**
- Create: `stock_lateral_toolkit/centering/r1_discriminator.py`

The exact comparison (pre-registered): on every M1 frame that lies inside an M2 window, compare **per-line** positions in the calibrated frame (all camera-relative, so NO lever arm enters):
- `Δ_left  = SP002 lane_lines[f,1,0,0] − (−y_road_left_video)`  (video road-frame y is +LEFT → negate)
- `Δ_right = SP002 lane_lines[f,2,0,0] − (−y_road_right_video)`
- `Δmid = (Δ_left + Δ_right)/2` (the perception bias), `Δwidth = Δ_right − Δ_left` (scale/coherence check).
If SP002 misplaces the LANE (|Δmid| ≥ 0.05, |Δwidth| ≤ 0.10 → coherent shift), the logged offset is (at least partly) perception; if SP002 places the lane where the video says it is (|Δmid| < 0.05) while the logged offset L persists, the car TRULY sits off-center and the model tolerates it → trained path preference. Cross-check: `Δmid` here must agree with M1's log-based `overlap_logged.median_delta_logged_minus_video_m` (same quantity through a different path — replay vs logs; the SP002 anchor corr 0.9986 says they should be nearly identical).

- [ ] **Step 1: Implement `stock_lateral_toolkit/centering/r1_discriminator.py`**

```python
"""R1: does SP002 misplace the LANE (perception translation) or correctly place the
lane and target off its center (trained preference)? Per-frame per-line comparison of
replayed SP002 lane lines vs M1 video annotations on identical frames.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r1_discriminator.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering import ground_plane as G
from stock_lateral_toolkit.centering.m1_offsets import _accepted_y


def _video_lines_per_frame() -> dict[int, dict]:
    """frame_idx -> {'mono_time', 'y_left_cal', 'y_right_cal'} from accepted proposals
    (median across eval distances; >= 2 accepted distances per side, as in m1_offsets)."""
    m1 = CC.RESULTS_DIR / "m1"
    manifest = {int(r["frame_idx"]): r for r in csv.DictReader(open(m1 / "frames_manifest.csv"))}
    from model_replay_sim.context import route_context
    height = float(route_context(CC.ROUTE).height)
    review = {}
    rev = m1 / "review_subset.csv"
    if rev.exists():
        for r in csv.DictReader(open(rev)):
            review[(int(r["frame_idx"]), float(r["x_m"]), r["side"])] = r
    by_frame: dict[int, dict] = {}
    for r in csv.DictReader(open(m1 / "proposals.csv")):
        fi = int(r["frame_idx"])
        key = (fi, float(r["x_m"]), r["side"])
        if key in review:
            r["verdict"] = review[key]["verdict"]
            r["corrected_u_px"] = review[key]["corrected_u_px"]
        man = manifest[fi]
        r["cal_roll"], r["cal_pitch"], r["cal_yaw"] = man["cal_roll"], man["cal_pitch"], man["cal_yaw"]
        r["height"] = height
        y = _accepted_y(r)
        if y is None:
            continue
        e = by_frame.setdefault(fi, {"mono_time": float(man["mono_time"]), "left": [], "right": []})
        e[r["side"]].append(float(y))
    out = {}
    for fi, e in by_frame.items():
        if len(e["left"]) >= 2 and len(e["right"]) >= 2:
            out[fi] = {"mono_time": e["mono_time"],
                       "y_left_cal": G.y_cal_from_y_road(float(np.median(e["left"]))),
                       "y_right_cal": G.y_cal_from_y_road(float(np.median(e["right"])))}
    return out


def _sp002_lines() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenated post-warmup (mono, y_left, y_right) at x=0 from the M2 SP002 replays."""
    from stock_lateral_toolkit.centering.windows import load_windows
    monos, yls, yrs = [], [], []
    for w in load_windows():
        z = np.load(CC.RESULTS_DIR / "m2" / f"replay_SP002_w{w['window_id']}.npz")
        s = int(z["split_index"])
        monos.append(np.asarray(z["mono_time"])[s:])
        yls.append(z["lane_lines"][s:, 1, 0, 0])
        yrs.append(z["lane_lines"][s:, 2, 0, 0])
    return np.concatenate(monos), np.concatenate(yls), np.concatenate(yrs)


def main():
    video = _video_lines_per_frame()
    mono, yl, yr = _sp002_lines()
    d_left, d_right = [], []
    for fi, v in sorted(video.items()):
        k = int(np.argmin(np.abs(mono - v["mono_time"])))
        if abs(mono[k] - v["mono_time"]) > 1e-3:
            continue
        d_left.append(float(yl[k]) - v["y_left_cal"])
        d_right.append(float(yr[k]) - v["y_right_cal"])
    d_left = np.array(d_left); d_right = np.array(d_right)
    if len(d_left) < 10:
        raise SystemExit(f"R1 blocked: only {len(d_left)} overlap frames (< 10) — "
                         "increase in-window M1 sampling or add the second route.")
    dmid = (d_left + d_right) / 2.0
    dwidth = d_right - d_left
    rng = np.random.default_rng(3)
    boots = [np.median(rng.choice(dmid, len(dmid))) for _ in range(2000)]
    res = {
        "n_overlap": int(len(dmid)),
        "delta_left_median_m": float(np.median(d_left)),
        "delta_right_median_m": float(np.median(d_right)),
        "dmid_median_m": float(np.median(dmid)),
        "dmid_ci95_m": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "dwidth_median_m": float(np.median(dwidth)),
        "translation_component": bool(abs(np.median(dmid)) >= CC.DMID_MEANINGFUL_M
                                      and abs(np.median(dwidth)) <= CC.DWIDTH_COHERENT_M),
    }
    # cross-check vs the log-based estimate of the same quantity (m1_results.json)
    m1 = json.loads((CC.RESULTS_DIR / "m1" / "m1_results.json").read_text())
    res["dmid_logbased_m"] = m1["overlap_logged"]["median_delta_logged_minus_video_m"]
    res["dmid_replay_vs_logbased_delta_m"] = float(res["dmid_median_m"] - res["dmid_logbased_m"])
    out = CC.RESULTS_DIR / "r1_results.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (after T9 + T10)**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/r1_discriminator.py
```
Expected artifact: `results/r1_results.json`. **Pre-registered interpretation (M0 §4):** `translation_component: true` ⇒ perception-translation component `T = dmid_median_m`; `|dmid| < 0.05` with M3's `P` ≥ 0.05 ⇒ trained-preference reading. `dmid_replay_vs_logbased_delta_m` should be ≈0 (both are SP002-vs-video; a large gap means a capture or annotation bug — investigate before the verdict). The CI is descriptive only (never a significance gate — toolkit discipline).

- [ ] **Step 3: Commit**

```bash
git add stock_lateral_toolkit/centering/r1_discriminator.py
git commit -m "centering R1: per-line translation-vs-preference discriminator (replay vs video, log cross-check)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 15: R3 — dependence breakdown (corridor / speed / direction)

**Files:**
- Create: `stock_lateral_toolkit/centering/r3_dependence.py`

- [ ] **Step 1: Implement `stock_lateral_toolkit/centering/r3_dependence.py`**

```python
"""R3: the deficit's SHAPE. Signed logged offset broken down per drive x corridor
(GPS-cell cluster) x speed bin x heading direction across ALL stock caches.
Uniform offset -> vehicle/model-global cause; corridor-dependent (esp. tracking roll)
-> environmental (crown); speed-dependent -> dynamic.

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r3_dependence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC
from stock_lateral_toolkit.centering.r2_crown import load_drive, DRIVES

CORRIDOR_CELL_DEG = 0.01   # ~1 km blocks = "corridor" granularity


def breakdown() -> pd.DataFrame:
    rows = []
    for name in DRIVES:
        try:
            d = load_drive(name)
        except FileNotFoundError:
            continue
        elig = ((d["latActive"] > 0.5) & (d["pressed"] < 0.5)
                & (d["vEgo"] >= CC.V_MIN_MPS) & (d["innerProb"] > CC.LANE_PROB_MIN)
                & np.isfinite(d["offset"]) & np.isfinite(d["lat"]))
        # heading from GPS (coarse): quadrant of travel direction
        la, lo = d["lat"], d["lon"]
        head = np.degrees(np.arctan2(np.gradient(la) * 110540.0,
                                     np.gradient(lo) * 111320.0 * np.cos(np.radians(np.nanmean(la))))) % 360.0
        corridor = (np.round(la / CORRIDOR_CELL_DEG).astype(np.int64) * 100000
                    + np.round(lo / CORRIDOR_CELL_DEG).astype(np.int64))
        for i in np.flatnonzero(elig):
            sb = -1
            for k, (vlo, vhi) in enumerate(CC.SPEED_BINS_MPS):
                if vlo <= d["vEgo"][i] < vhi:
                    sb = k
            if sb < 0:
                continue
            rows.append((name, int(corridor[i]), sb, int(head[i] // 90.0) % 4,
                         float(d["offset"][i]), float(d["roll"][i])))
    return pd.DataFrame(rows, columns=["drive", "corridor", "speed_bin", "heading_q",
                                       "offset", "roll"])


def main():
    df = breakdown()
    g = (df.groupby(["corridor", "speed_bin", "heading_q"])
           .agg(offset_med=("offset", "median"), roll_med=("roll", "median"),
                n=("offset", "size"), drives=("drive", "nunique"))
           .reset_index())
    g = g[g["n"] >= 200]          # >= 10 s of frames per cell
    out_csv = CC.RESULTS_DIR / "r3_table.csv"
    g.to_csv(out_csv, index=False)

    per_corr = g.groupby("corridor")["offset_med"].median()
    per_speed = g.groupby("speed_bin")["offset_med"].median()
    per_head = g.groupby("heading_q")["offset_med"].median()
    res = {
        "n_cells": int(len(g)),
        "global_median_m": float(df["offset"].median()),
        "corridor_spread_m": float(per_corr.max() - per_corr.min()) if len(per_corr) > 1 else 0.0,
        "per_speed_bin_m": {str(k): float(v) for k, v in per_speed.items()},
        "per_heading_quadrant_m": {str(k): float(v) for k, v in per_head.items()},
        "corridor_dependent": bool(len(per_corr) > 1
                                   and (per_corr.max() - per_corr.min()) > CC.CORRIDOR_SPREAD_M),
        "corridor_offset_vs_roll_corr": float(np.corrcoef(
            g.groupby("corridor")["offset_med"].median(),
            g.groupby("corridor")["roll_med"].median())[0, 1]) if len(per_corr) > 3 else None,
    }
    (CC.RESULTS_DIR / "r3_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"table -> {out_csv}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (after T12)**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/r3_dependence.py
```
Expected artifact: `results/r3_table.csv` + `results/r3_results.json`. **Pre-registered interpretation (M0 §4):** `corridor_spread_m > 0.05` AND positive `corridor_offset_vs_roll_corr` corroborate the crown class; a flat table (spread ≤ 0.05, no speed trend) supports a vehicle/model-global cause. R3 alone never selects a class — it feeds `r_verdict.py`.

- [ ] **Step 3: Commit**

```bash
git add stock_lateral_toolkit/centering/r3_dependence.py
git commit -m "centering R3: corridor/speed/direction dependence breakdown of the signed offset

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 16: R verdict — mechanical decision-tree application

**Files:**
- Create: `stock_lateral_toolkit/centering/r_verdict.py`
- Test: `stock_lateral_toolkit/centering/tests/test_r_verdict.py`

- [ ] **Step 1: Write the failing test (every branch of the M0 §4 tree)**

`stock_lateral_toolkit/centering/tests/test_r_verdict.py`:
```python
def _base():
    return dict(proceed_to_r=True, p_cam=0.10, dmid=0.01, dwidth=0.02, logged=0.11,
                crown_significant=False, crown_component=0.0)


def test_trained_preference():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**_base())
    assert v["class"] == "TRAINED_PATH_PREFERENCE"
    assert v["camera_offset_is_the_lever"] is True


def test_definition_only():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "p_cam": 0.02, "dmid": 0.09})
    assert v["class"] == "MODEL_FRAME_DEFINITION_ONLY"
    assert v["camera_offset_is_the_lever"] is False


def test_mixed():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "p_cam": 0.08, "dmid": 0.06})
    assert v["class"] == "MIXED_TRANSLATION_PREFERENCE"
    assert v["camera_offset_is_the_lever"] is True


def test_crown_dominant_blocks_camera_offset():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "crown_significant": True, "crown_component": 0.08})
    assert "CROWN" in v["class"]
    assert v["camera_offset_is_the_lever"] is False   # crown fraction 0.08/0.11 > 0.5


def test_method_gate_blocks_everything():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "proceed_to_r": False})
    assert v["class"] == "NO_VERDICT_METHOD_DISAGREEMENT"


def test_no_deficit():
    from stock_lateral_toolkit.centering import r_verdict as RV
    v = RV.classify(**{**_base(), "p_cam": 0.01, "dmid": 0.01, "logged": 0.02})
    assert v["class"] == "NO_DEFICIT_MEASURABLE"
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_r_verdict.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... r_verdict`.

- [ ] **Step 3: Implement `stock_lateral_toolkit/centering/r_verdict.py`**

```python
"""R verdict: MECHANICAL application of the pre-registered decision tree (M0 §4).
No numbers are chosen here — inputs come from m3_verdict.json / r1_results.json /
r2/r2_results.json / r3_results.json; thresholds from config (twins of the spec).

RUN: .venv311/bin/python stock_lateral_toolkit/centering/r_verdict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC


def classify(proceed_to_r: bool, p_cam: float, dmid: float, dwidth: float, logged: float,
             crown_significant: bool, crown_component: float) -> dict:
    if not proceed_to_r:
        return {"class": "NO_VERDICT_METHOD_DISAGREEMENT", "camera_offset_is_the_lever": False,
                "note": "M3 method gate failed (M0 §3) — diagnose the discrepancy first."}
    p_big = abs(p_cam) >= CC.P_MEANINGFUL_M
    t_big = abs(dmid) >= CC.DMID_MEANINGFUL_M and abs(dwidth) <= CC.DWIDTH_COHERENT_M
    crown_fraction = (abs(crown_component) / abs(logged)) if (crown_significant and logged) else 0.0
    if crown_significant and crown_fraction > CC.CROWN_DOMINANT_FRACTION:
        return {"class": "CROWN_RESPONSE_DOMINANT", "crown_fraction": crown_fraction,
                "camera_offset_is_the_lever": False,
                "note": "S2 branch: crown-aware approach needs a NEW spec (out of this plan)."}
    if p_big and not t_big:
        cls = "TRAINED_PATH_PREFERENCE"
    elif p_big and t_big:
        cls = "MIXED_TRANSLATION_PREFERENCE"
    elif not p_big and t_big:
        cls = "MODEL_FRAME_DEFINITION_ONLY"
    else:
        cls = "NO_DEFICIT_MEASURABLE"
    return {"class": cls,
            "camera_offset_is_the_lever": cls in ("TRAINED_PATH_PREFERENCE",
                                                  "MIXED_TRANSLATION_PREFERENCE"),
            "crown_fraction": crown_fraction,
            "components": {"P_cam_m": p_cam, "T_dmid_m": dmid if t_big else 0.0,
                           "crown_m": crown_component if crown_significant else 0.0,
                           "L_logged_m": logged}}


def main():
    m3 = json.loads((CC.RESULTS_DIR / "m3_verdict.json").read_text())
    r1 = json.loads((CC.RESULTS_DIR / "r1_results.json").read_text())
    r2 = json.loads((CC.RESULTS_DIR / "r2" / "r2_results.json").read_text())
    r3 = json.loads((CC.RESULTS_DIR / "r3_results.json").read_text())
    v = classify(proceed_to_r=bool(m3["proceed_to_R"]),
                 p_cam=float(m3["P_cam_m"]),
                 dmid=float(r1["dmid_median_m"]),
                 dwidth=float(r1["dwidth_median_m"]),
                 logged=float(m3["L_logged_m"]),
                 crown_significant=bool(r2["crown_significant"]),
                 crown_component=float(r2.get("crown_component_m") or 0.0))
    v["inputs"] = {"m3": m3, "r1_dmid": r1["dmid_median_m"], "r2_sig": r2["crown_significant"],
                   "r3_corridor_spread_m": r3["corridor_spread_m"]}
    (CC.RESULTS_DIR / "r_verdict.json").write_text(json.dumps(v, indent=2))
    md = ["# R-phase verdict (mechanical, M0 §4)", "",
          f"**Class: {v['class']}**", "",
          f"- camera_offset_is_the_lever: {v['camera_offset_is_the_lever']}",
          f"- components: {json.dumps(v.get('components', {}))}",
          f"- crown fraction: {v.get('crown_fraction')}",
          f"- R3 corridor spread: {r3['corridor_spread_m']:.3f} m"]
    (CC.RESULTS_DIR / "r_verdict.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_r_verdict.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Run on real results (after T11, T13, T14, T15)**

```bash
.venv311/bin/python stock_lateral_toolkit/centering/r_verdict.py
```
Expected artifact: `results/r_verdict.{json,md}` — the root-cause class with its evidence pattern. Per `feedback_verify_agent_outputs`, re-derive the four input numbers by hand from their JSONs before accepting the class.

- [ ] **Step 6: Commit**

```bash
git add stock_lateral_toolkit/centering/r_verdict.py stock_lateral_toolkit/centering/tests/test_r_verdict.py
git commit -m "centering R: mechanical decision-tree verdict from M3/R1/R2/R3 evidence

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 17: S1 — CameraOffset counterfactual dose-response sweep

**Files:**
- Create: `stock_lateral_toolkit/centering/s1_sweep.py`
- Test: `stock_lateral_toolkit/centering/tests/test_s1_gates.py`

Grid: 11 sweep points −0.10…+0.10 step 0.02 **plus** 2 negative-control points ±0.005 m (they DEFINE the weave noise band NB — a 5 mm shear is physically negligible, so any weave delta it induces is pixel-resampling noise, the method's floor) **plus** 1 determinism repeat at 0.0. **Compute budget: (11+2)×~2300 frames ≈ 3.4 h serial + repeat ≈ 10 min → ~55–70 min wall at 4 workers** (SP002 only; can start right after T3+T4+T5, in parallel with all of M/R — it commits to nothing until the R verdict selects the branch).

- [ ] **Step 1: Write the failing test (gate logic on synthetic dose-response)**

`stock_lateral_toolkit/centering/tests/test_s1_gates.py`:
```python
import numpy as np
import pytest


def _synth(slope=1.0, weave_jitter=0.0):
    offsets = [round(-0.10 + 0.02 * i, 3) for i in range(11)]
    rows = []
    for off in offsets:
        rows.append(dict(offset=off, d_center_m=slope * off,
                         band_ratio=1.0 + weave_jitter * abs(off) / 0.10,
                         corr_vs_zero=0.999, low_band_ratio=1.0))
    controls = {(-0.005): 1.005, (0.005): 0.996}   # -> NB = max(0.03, 2*0.005) = 0.03
    return rows, controls


def test_gates_pass_on_clean_unit_response():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth()
    g = S.evaluate_gates(rows, controls, determinism_max_delta=0.0, p_cam=0.08)
    assert g["monotonic"] and g["slope_ok"] and g["weave_ok"] and g["curve_ok"]
    assert g["slope"] == pytest.approx(1.0, abs=0.01)
    assert g["delta_star_m"] == pytest.approx(0.08, abs=0.011)  # snapped to the grid
    assert g["all_pass"] is True


def test_weave_gate_fails_outside_noise_band():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth(weave_jitter=0.2)      # band_ratio up to 1.2 > hard cap
    g = S.evaluate_gates(rows, controls, 0.0, p_cam=0.08)
    assert not g["weave_ok"] and g["all_pass"] is False


def test_slope_gate_fails_on_flat_response():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth(slope=0.1)
    g = S.evaluate_gates(rows, controls, 0.0, p_cam=0.08)
    assert not g["slope_ok"] and g["all_pass"] is False


def test_delta_star_out_of_range_flags():
    from stock_lateral_toolkit.centering import s1_sweep as S
    rows, controls = _synth(slope=0.6)             # delta* = 0.09/0.6 = 0.15 > 0.10
    g = S.evaluate_gates(rows, controls, 0.0, p_cam=0.09)
    assert g["delta_star_in_range"] is False and g["all_pass"] is False
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_s1_gates.py -v
```
Expected: FAIL with `ModuleNotFoundError: ... s1_sweep`.

- [ ] **Step 3: Implement `stock_lateral_toolkit/centering/s1_sweep.py`**

```python
"""S1: CameraOffset counterfactual dose-response sweep on SP002 over the shared windows.

Per offset point: replay with camera_offset=delta + lane capture; measure
  d_center_m   = median lane-center(y0) shift vs the 0.0 replay  (predicted centering)
  band_ratio   = weave-band RMS(desired_curvature) / same at 0.0  (0.10-0.35 Hz)
  corr_vs_zero = corr(desired_curv(delta), desired_curv(0))       (curve proxy 1/2)
  low_band_ratio = <=0.05 Hz RMS ratio                            (curve proxy 2/2)
Noise band NB from the +/-0.005 m controls; determinism from a repeat at 0.0.
Gates: pre-registration M0 §5 (twinned in config).

RUN:
  caffeinate -i .venv311/bin/python stock_lateral_toolkit/centering/s1_sweep.py --replay
  .venv311/bin/python stock_lateral_toolkit/centering/s1_sweep.py --analyze
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stock_lateral_toolkit.centering import config as CC

S1_DIR = CC.RESULTS_DIR / "s1"
BUNDLE = "SP002"


def _pt_path(offset: float, window_id: int, rep: int = 0) -> Path:
    tag = f"{offset:+.3f}".replace(".", "p")
    return S1_DIR / f"sweep_{tag}_w{window_id}_r{rep}.npz"


def _replay_job(args) -> str:
    offset, window, rep = args
    from model_replay_sim.infer import replay_window
    r = replay_window(BUNDLE, window["route_id"], window["mono_times"],
                      camera_offset=float(offset), capture_outputs=("lane_lines",))
    out = _pt_path(offset, window["window_id"], rep)
    np.savez_compressed(out, mono_time=r["mono_time"],
                        desired_curvature=r["desired_curvature"],
                        lane_lines=r["captured"]["lane_lines"],
                        camera_offset_used=r["camera_offset_used"],
                        split_index=window["split_index"])
    return str(out)


def run_replays() -> None:
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()
    S1_DIR.mkdir(parents=True, exist_ok=True)
    points = list(CC.SWEEP_OFFSETS_M) + list(CC.CONTROL_OFFSETS_M)
    jobs = [(o, w, 0) for o in points for w in wins if not _pt_path(o, w["window_id"]).exists()]
    if not _pt_path(0.0, wins[0]["window_id"], rep=1).exists():
        jobs.append((0.0, wins[0], 1))            # determinism repeat
    print(f"{len(jobs)} replay jobs ({len(points)} offsets x {len(wins)} windows + repeat), "
          f"{CC.MAX_WORKERS_REPLAY} workers, ~0.41 s/frame")
    with ProcessPoolExecutor(max_workers=CC.MAX_WORKERS_REPLAY,
                             mp_context=mp.get_context("spawn")) as ex:
        for done in ex.map(_replay_job, jobs):
            print("done:", done)


def _point_metrics(offset: float, window_id: int, base: dict) -> dict:
    from model_replay_sim.metrics import weave_band_rms
    from stock_lateral_toolkit import signal_utils as SU
    z = np.load(_pt_path(offset, window_id))
    s = int(z["split_index"])
    curv = np.asarray(z["desired_curvature"], float)[s:]
    center = (z["lane_lines"][s:, 1, 0, 0] + z["lane_lines"][s:, 2, 0, 0]) / 2.0
    both = np.isfinite(curv) & np.isfinite(base["curv"])
    lo_self = SU.filter_continuous(curv, CC.FS_HZ, lowpass_hz=CC.LOW_BAND_HZ)
    lo_base = SU.filter_continuous(base["curv"], CC.FS_HZ, lowpass_hz=CC.LOW_BAND_HZ)
    def _rms(x):
        v = x[np.isfinite(x)]
        return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")
    return {
        "offset": float(offset), "window_id": window_id,
        "d_center_m": float(np.median(center) - base["center_med"]),
        "band_ratio": float(weave_band_rms(curv) / base["weave"]),
        "corr_vs_zero": float(np.corrcoef(curv[both], base["curv"][both])[0, 1]),
        "low_band_ratio": float(_rms(lo_self) / _rms(lo_base)),
        "dc_curv_shift": float(np.median(curv[both] - base["curv"][both])),
    }


def evaluate_gates(rows: list[dict], controls: dict, determinism_max_delta: float,
                   p_cam: float) -> dict:
    """Pure gate logic (unit-tested). rows = per-offset metrics AVERAGED over windows;
    controls = {control_offset: band_ratio}; p_cam = M-measured camera-relative deficit."""
    rows = sorted(rows, key=lambda r: r["offset"])
    off = np.array([r["offset"] for r in rows])
    dc = np.array([r["d_center_m"] for r in rows])
    rho = float(stats.spearmanr(off, dc)[0])
    slope = float(np.polyfit(off, dc, 1)[0])
    nb = max(CC.WEAVE_NOISE_FLOOR, 2.0 * max(abs(v - 1.0) for v in controls.values()))
    if determinism_max_delta > CC.DETERMINISM_TOL:
        nb = max(nb, 4.0 * determinism_max_delta)   # widen if replay is not bit-repeatable
    weave_ok = all(abs(r["band_ratio"] - 1.0) <= nb
                   and CC.WEAVE_HARD_CAP[0] <= r["band_ratio"] <= CC.WEAVE_HARD_CAP[1]
                   for r in rows)
    curve_ok = all(r["corr_vs_zero"] >= CC.CURVE_CORR_MIN
                   and CC.LOW_BAND_RATIO[0] <= r["low_band_ratio"] <= CC.LOW_BAND_RATIO[1]
                   for r in rows if r["offset"] != 0.0)
    monotonic = abs(rho) >= CC.MONOTONIC_SPEARMAN_MIN
    slope_ok = CC.SLOPE_UNIT_RANGE[0] <= abs(slope) <= CC.SLOPE_UNIT_RANGE[1]
    delta_star = float(p_cam / slope) if slope_ok and slope != 0 else float("nan")
    grid = [r["offset"] for r in rows]
    delta_star_grid = (min(grid, key=lambda o: abs(o - delta_star))
                       if np.isfinite(delta_star) else None)
    in_range = bool(np.isfinite(delta_star) and min(grid) <= delta_star <= max(grid))
    improvement_ok = bool(in_range and abs(slope * delta_star) >= 0.7 * abs(p_cam))
    return {"monotonic": monotonic, "spearman_rho": rho, "slope": slope, "slope_ok": slope_ok,
            "noise_band": nb, "weave_ok": weave_ok, "curve_ok": curve_ok,
            "delta_star_m": delta_star_grid, "delta_star_raw_m": delta_star,
            "delta_star_in_range": in_range, "improvement_ok": improvement_ok,
            "determinism_max_delta": determinism_max_delta,
            "all_pass": bool(monotonic and slope_ok and weave_ok and curve_ok
                             and in_range and improvement_ok)}


def analyze() -> None:
    from model_replay_sim.metrics import weave_band_rms
    from stock_lateral_toolkit.centering.windows import load_windows
    wins = load_windows()

    # determinism check: repeat at 0.0 on window 0
    z0 = np.load(_pt_path(0.0, wins[0]["window_id"], 0))
    z1 = np.load(_pt_path(0.0, wins[0]["window_id"], 1))
    det = float(np.nanmax(np.abs(z0["desired_curvature"] - z1["desired_curvature"])))
    print(f"determinism: max|repeat delta| = {det:.2e} (tol {CC.DETERMINISM_TOL})")

    bases = {}
    for w in wins:
        zb = np.load(_pt_path(0.0, w["window_id"]))
        s = int(zb["split_index"])
        curv = np.asarray(zb["desired_curvature"], float)[s:]
        center = (zb["lane_lines"][s:, 1, 0, 0] + zb["lane_lines"][s:, 2, 0, 0]) / 2.0
        bases[w["window_id"]] = {"curv": curv, "center_med": float(np.median(center)),
                                 "weave": weave_band_rms(curv)}

    per_offset, controls = [], {}
    for o in CC.SWEEP_OFFSETS_M:
        pts = [_point_metrics(o, w["window_id"], bases[w["window_id"]]) for w in wins]
        per_offset.append({k: float(np.mean([p[k] for p in pts])) if k != "window_id" else -1
                           for k in pts[0]} | {"offset": o, "per_window": pts})
    for o in CC.CONTROL_OFFSETS_M:
        pts = [_point_metrics(o, w["window_id"], bases[w["window_id"]]) for w in wins]
        controls[o] = float(np.mean([p["band_ratio"] for p in pts]))

    p_cam = json.loads((CC.RESULTS_DIR / "m1" / "m1_results.json").read_text())["median_offset_cam_m"] \
        if (CC.RESULTS_DIR / "m1" / "m1_results.json").exists() else float("nan")
    if not np.isfinite(p_cam):
        print("NOTE: M1 not done yet — gates evaluated with p_cam=nan (sizing gates will fail); "
              "re-run --analyze after Task 9.")
    gates = evaluate_gates([{k: r[k] for k in ("offset", "d_center_m", "band_ratio",
                                               "corr_vs_zero", "low_band_ratio")}
                            for r in per_offset], controls, det, p_cam)

    res = {"bundle": BUNDLE, "p_cam_m": p_cam, "gates": gates,
           "controls_band_ratio": {str(k): v for k, v in controls.items()},
           "dose_response": per_offset}
    (S1_DIR / "s1_report.json").write_text(json.dumps(res, indent=2))
    md = ["# S1 CameraOffset dose-response (SP002, shared windows)", "",
          "| offset (m) | d_center (m) | band_ratio | corr_vs_0 | low_band_ratio | dc_shift (1/m) |",
          "|---|---|---|---|---|---|"]
    for r in per_offset:
        md.append(f"| {r['offset']:+.2f} | {r['d_center_m']:+.4f} | {r['band_ratio']:.3f} "
                  f"| {r['corr_vs_zero']:.4f} | {r['low_band_ratio']:.3f} | {r['dc_curv_shift']:+.2e} |")
    md += ["", f"noise band NB = {gates['noise_band']:.3f}; determinism delta = {det:.2e}",
           f"slope = {gates['slope']:+.3f} (unit-range gate {CC.SLOPE_UNIT_RANGE}); "
           f"Spearman rho = {gates['spearman_rho']:+.3f}",
           f"delta* = {gates['delta_star_m']} m (raw {gates['delta_star_raw_m']}); "
           f"ALL GATES PASS = {gates['all_pass']}"]
    (S1_DIR / "s1_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    if "--replay" in sys.argv:
        run_replays()
    elif "--analyze" in sys.argv:
        analyze()
    else:
        print("usage: s1_sweep.py --replay | --analyze")
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv311/bin/python -m pytest stock_lateral_toolkit/centering/tests/test_s1_gates.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run the sweep (heavy — see budget) then the analysis**

```bash
caffeinate -i .venv311/bin/python stock_lateral_toolkit/centering/s1_sweep.py --replay
.venv311/bin/python stock_lateral_toolkit/centering/s1_sweep.py --analyze
```
Expected artifacts: `results/s1/sweep_*.npz` (27 files), `results/s1/s1_report.{json,md}` with the dose-response table. **Pre-registered interpretation (M0 §5):** the SIGN of the slope is measured here, never assumed (it also settles the user-facing "which way does + move the car" question); `delta_star_m` is the proposed device value ONLY IF `all_pass` AND Task 16's verdict says `camera_offset_is_the_lever`. Re-run `--analyze` (cheap) after M1 lands if it ran first.

- [ ] **Step 6: Commit**

```bash
git add stock_lateral_toolkit/centering/s1_sweep.py stock_lateral_toolkit/centering/tests/test_s1_gates.py
git commit -m "centering S1: CameraOffset dose-response sweep with pre-registered desk-exit gates

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 18: S2 — conditional solution decision (no code)

**Files:** none created — this is a decision checkpoint recorded in the run log / final report.

- [ ] **Step 1: Apply the pre-registered branching (M0 §4 item 4) to `results/r_verdict.json` + `results/s1/s1_report.json`:**

- IF `r_verdict.class` ∈ {`TRAINED_PATH_PREFERENCE`, `MIXED_TRANSLATION_PREFERENCE`} (i.e. `camera_offset_is_the_lever: true`) AND `s1_report.gates.all_pass: true` → **CameraOffset = `delta_star_m` IS the designed solution.** Proceed to Task 19 (road protocol) with that value. No new spec needed — S1 already sized and safety-gated it offline.
- IF `r_verdict.class == MODEL_FRAME_DEFINITION_ONLY` → the car is physically fine; **no control change is designed.** Record the finding, close the S-phase, and offer the user an OPTIONAL subjective-only D drive (the felt "centers worse" then needs a different explanation — e.g. comparison memory vs the fork's PI feel).
- IF `r_verdict.class == CROWN_RESPONSE_DOMINANT` → **a crown-aware approach is EXPLICITLY OUT OF THIS PLAN.** Write a new brainstorm/spec (fresh design per `feedback_fresh_optimization_approach`); do NOT deploy a CameraOffset as a crown band-aid (it would mis-center on uncrowned roads by construction).
- IF `s1_report.gates.all_pass: false` while the verdict says CameraOffset is the lever → the lever failed its own offline validation; STOP, report which gate failed (weave/curve/slope/sizing), and re-scope. Never carry a failed gate to the road.

- [ ] **Step 2: Record the decision** as a dated section appended to `docs/superpowers/specs/2026-07-03-centering-preregistration.md` (an addendum, per its amendment rule) naming the class, the chosen branch, and (if applicable) `delta_star_m`. Commit:

```bash
git add docs/superpowers/specs/2026-07-03-centering-preregistration.md
git commit -m "centering S2: record pre-registered solution branch decision

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 19: D — road A/B protocol document (USER drives; LAST)

**Files:**
- Create: `docs/superpowers/plans/2026-07-03-centering-road-ab-protocol.md`

- [ ] **Step 1: Write the protocol with EXACTLY this content** (fill the two `<from s1_report>` data references from `results/s1/s1_report.json` at write time — they are computed values, not open design points):

```markdown
# Centering road A/B protocol (D-phase) — USER EXECUTES

**HARD DEPENDENCY: F3 AOL safety net (Workflow 1) must be DEPLOYED AND OBSERVED
for at least one normal drive before these runs** (it is the net under them).
Device prep per reference_device_config: `DisableUpdates=1`, double-reboot gotcha.

## Arms
- A (control): CameraOffset = 0.0
- B (treatment): CameraOffset = <delta_star_m from s1_report.json>
Toggle ONLY while PARKED between passes, via SSH:
  python3 -c "from openpilot.common.params import Params; Params().put('CameraOffset', '<value>')"
  python3 -c "from openpilot.common.params import Params; print(Params().get('CameraOffset'))"
(read-back REQUIRED; modeld EMAs the param in ~0.5 s, but toggling parked removes
all doubt; per feedback_pi_param_drift, verify the param state before each pass).

## Corridor & schedule (feedback_lateral_ab_metrics: same location + speed match)
- Corridor: the hiram corridor (the stock 04/05 route), BOTH directions.
- ≥3 passes per arm per direction, interleaved A,B,A,B,A,B (never blocked AABB).
- Fixed cruise set-speed per corridor leg (pick the leg's normal speed; same both arms).
- Engaged, hands-off-but-ready, no manual corrections except safety; abort a pass on
  rain/heavy traffic/lead-follow (redo it).
- Log sheet per pass: wall time, arm, direction, set speed, weather, any overrides,
  any F3 alerts, subjective centering note (1-5).

## What gets logged (nothing extra to set up)
Normal rlogs+camera on device. After the session: pull rlogs AND fcamera.hevc for all
passes to explorer_st_logs/route_ab_<date>/ (video is REQUIRED — see below).

## Analysis (pre-registered, M0 §6)
1. Extract every pass: stock_lateral_toolkit/extract_drive.py -> per-pass npz.
2. PRIMARY — video ground truth: the running model CANNOT log its own improvement
   (at settle it reads its preferred offset by construction; S1's settling model),
   so re-run the M1 pipeline (frame_sampler with ROUTE pointed at the A/B route,
   annotate, m1_offsets) per arm. Success: treatment median physical |offset| ≤ 0.05 m
   OR reduced by ≥70% of P, sign as predicted by the S1 slope.
3. Weave: matched GPS-cell + speed-bin band|yawRate| (0.10–0.35 Hz) A vs B
   (analyze_compare-style cells). Success: delta not significant (calibrated test)
   AND point estimate ≤ +15% vs control.
4. Safety: zero F3 departure alerts attributable to the offset; logged min distance
   to the nearer lane line not reduced by > 0.05 m (median over matched cells).
5. Subjective: not worse (goal: better), recorded per pass before seeing numbers.
ALL of 2-5 must hold to adopt <delta_star_m> as the standing device value; any
failure -> revert CameraOffset to 0.0 and return to the R-phase evidence.

## Sanity notes
- The A-arm doubles as a fresh baseline P measurement (compare to the M-phase P).
- Do not mix in other param changes; this A/B tests exactly one lever.
- Per feedback_lateral_ab_metrics: comparisons have flipped 3x under poor control —
  if pass counts end up unbalanced or speeds unmatched (>2 m/s cell mismatch),
  collect more passes rather than relaxing the matching.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-07-03-centering-road-ab-protocol.md
git commit -m "centering D: interleaved road A/B protocol (user-executed, F3-gated, video-primary outcome)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Execution notes for workers

- Heavy replays (`consensus.py --replay`, `s1_sweep.py --replay`) are **resumable** — both skip existing npz files; re-running after an interruption is safe and cheap.
- Never run two replay drivers at once (they'd oversubscribe the 4-worker budget and the memory held by compiled models).
- Every `--stats`/`--analyze`/verdict script is idempotent and cheap; re-run freely as upstream artifacts land (M1 after user review; S1 analyze after M1).
- Order of user contact: Task 8 step 6 (review) and Task 9 step 1 (lever arm) can be requested together; Task 10 step 1 (video pull) only if the sampler under-delivers (<40 frames) or the user wants the stronger 2-route version; Task 19 is handed to the user as a document, not executed.
- If any gate FAILS, stop and report — the pre-registration makes a failed gate a *finding*, not an obstacle to route around.
