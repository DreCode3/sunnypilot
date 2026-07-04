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

## Addendum 2026-07-04 (execution)
Known at amendment time: proposals + overlays existed for 42 frames; no M/R verdict had
been issued; the S1 sweep's non-sizing gates had passed on SP002.
1. **Reviewer change (§2):** the user delegated the annotation spot-review to Claude
   (visual inspection of overlay PNGs), directed in-session. Review semantics unchanged.
2. **Trust-rule outcome (§2):** subset acceptance 45/54 = 83.3% < 85% → rule FAILED at
   the letter (8 of 9 rejects were rows the automation itself had flagged auto_ok=False;
   filter-passed acceptance was 45/46). The §2 fallback was executed as prescribed: a
   FULL manual pass over all 252 proposal rows (195 accept / 57 reject; rejects dominated
   by dashed-line sample distances landing in dash gaps). m1_offsets.py §2-fallback mode:
   with every row verdicted, analysis consumes only verdicts (no automation trust).
3. **Usable-frame floor (§2):** 22 frames with ≥2 accepted distances per side (< 40) →
   the pre-registered second-route escalation (route_stock04 video pull) is REQUIRED and
   was authorized by the user 2026-07-04. No M3/R verdict is issued on stock05 alone.
4. **Matching clarification (no tolerance change):** M1↔M2 overlap matching is by frame
   identity ((seg_num, seg_id) → frame-timeline eof timestamp), not raw mono_time — the
   manifest's mono_time is the modelV2 publish time (eof + inference latency), which a
   1 ms mono match structurally cannot hit.

## Addendum 2026-07-04b (user-approved amendments; known at amendment time: all M/R
## numbers below were already computed and are reported regardless of these changes)
1. **§2 minimum sample 40 → 34** for this analysis: pooled two-route n=34 after
   full-manual annotation attrition (curb/gutter + dash-gap scenery); sem_random
   0.010 m ≪ the ±0.04 m systematic bound that §2 itself declares binding.
2. **R1 overlap floor 10 → 7**, with the log-based Δmid (n=34, independent path)
   reported alongside as the primary cross-check.
3. **§3 M3 gate**: proceed_to_R decided by the M1-vs-M2 gate (+0.035 ≤ 0.06 PASS)
   plus the log-based decomposition. The M2-vs-logged gate is RETAINED as a report:
   like-for-like it passes on window 0 (+0.024) and fails on window 1 (+0.067) —
   recorded as a known sim limitation (absolute lane-center bias, suspected live-vs
   -steady calibration; steering dynamics unaffected, anchor corr 0.9986; S1 is
   differential and immune). Follow-up experiment: per-window-calibration replay.
4. **§5 sweep range extended to [−0.16, +0.10]** (step 0.02 unchanged) so the
   consolidated δ* ≈ −0.11 lies inside the swept grid; all §5 gates re-evaluated
   over the full extended grid. Sizing uses the CONSOLIDATED P (n=34).
