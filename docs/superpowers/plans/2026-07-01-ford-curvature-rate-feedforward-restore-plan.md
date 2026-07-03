> 🛑 **ON HOLD / LIKELY MOOT (2026-07-01).** The premise of this plan — "the EPS under-responds because Path B removed the curvature_rate feed-forward" — was **REFUTED** by collaborative QA + a VehicleModel re-check: on cf the wheel *does* follow the command (angle-ratio 0.90, like old builds); the ce evidence was n≈100, single-segment, curve-entry-confounded. **Do NOT implement this.** The investigation moved to "weak/slow lateral correction (new inference-engine temporal state and/or PI lane-centering)." See `docs/superpowers/reports/2026-07-01-lateral-safety-investigation-handoff-v2.md`. Kept for the record + the reusable Ford safety-suite / bounded-curvature_rate analysis (test_ford.py:286, fork `curvature_rate_cmd_checks`) if a feed-forward change is ever revisited.

# Plan — Restore bounded Ford `curvature_rate` feed-forward (fix the AOL silent-under-steer incident)

**Status:** DRAFT for 2-step QA (round 1 collaborative, round 2 adversarial). Do NOT implement until QA complete + all issues resolved. This touches the **panda safety layer** — treat with maximum rigor.

**Owner constraint (user, 2026-07-01):** the change **must pass the upstream opendbc Ford safety test suite** — existing tests still pass *or are rigorously extended*; `ford.h` adapted so a **bounded** `curvature_rate` is *provably within limits*. Fix on the new build (v2026.002.001), no rollback. Bench-validate before any re-drive.

---

## 1. Root cause (verified 2026-07-01) — what we're fixing
Under Always-On-Lateral the car repeatedly failed to steer curves ("lane lines vanished on the display, vehicle continued straight, not attempting to turn"), no alert, forced takeovers. Verified chain:

- **PRIMARY:** the migration's **Path B zeroed the `curvature_rate` feed-forward** (opendbc `carcontroller.py` sends `curvature_rate = 0`; upstream `ford.h` *enforces* it). The Ford EPS needs that feed-forward to physically actuate a curve. **Physical evidence** — system-only (driver steering excluded), matched speed 18–30 m/s, real curves (|cmd|>0.0015), median |achieved(`yawRate/vEgo`) ÷ commanded(`carOutput.curvature`)|:
  - OLD build (feed-forward present): route_c5 **1.22**, c7 **1.09**, b5 **0.99** → EPS follows the command.
  - NEW build (Path B): route_ce **0.12**, cf **0.72** → EPS under-responds; car barely turns.
- The model itself is fine: at the reported incident GPS it was confident (laneProb 0.94) and commanded the curve; during confidence dips the model **keeps** commanding the curve (only ~−30%, never zero) and plan≈desiredCurvature. Ruled out: model choice, calibration (identical old/new), frame/ISP corruption, on-device transient (offline sim reproduces logged output corr 0.980), recompiled bundle (canonical onnx reproduces), dramatic control-clip (command tracks model ~80%).
- **Lane-confidence collapses are pre-existing & secondary** (same/lower event rate new-vs-old: 1.3–3.6/min all builds) — a coincident display artifact, addressed by the defense-in-depth safeguard (§6), NOT the cause of the car going straight.

**Confidence & the one caveat to firm up (QA gate G0):** the direction is unambiguous (OLD ~1.0 vs NEW <1.0), but ce=0.12 rests on a small sample (n=103) and curve-entry-transient weighting. Before implementing, re-measure achieved/commanded with (a) larger matched samples, (b) steady-state-only vs entry split, (c) an independent achieved-curvature estimate (from `steeringAngleDeg` via steerRatio/wheelbase, cross-checked against `yawRate/vEgo`). Fix proceeds only if OLD≈1.0 and NEW materially <1.0 replicate.

## 2. Fix overview
Restore the fork's tuned, **bounded** `curvature_rate` feed-forward on the new build, admitting it through a ported, rate-limited safety check — mirroring how upstream already bounds `curvature`. Three coordinated edits + safety proof + validation gates.

## 3. Detailed changes

### 3a. `opendbc/car/ford/carcontroller.py` (the command)
- Restore the feed-forward **computation** if Path B/migration dropped it: the 7-sample `curvature_rate_deque` (0.3 s @ 20 Hz) → `apply_curvature_rate = (deque[-1]-deque[0])/Δt/vEgoRaw`, scaled by `curv_factor * curvature_rate_gain`, **clipped to ±0.001023** (within Ford field range). *(Present in pre-migration local `carcontroller.py:464-475`; VERIFY current on-device/migration file — Path B may have removed only the send, or the whole block.)*
- Revert the two Path-B sends: `create_lat_ctl2_msg`/`create_lat_ctl_msg` args `…, -apply_curv_send, 0., …` → `…, -apply_curv_send, -apply_curvature_rate, …` (device lines ~596/600).
- Keep `apply_curvature_rate = 0.0` in the not-steering / reset branches (matches safety "must be 0 when inactive").

### 3b. `opendbc/safety/modes/ford.h` (the safety gate) — the crux
Replace the blanket rejection `violation = (raw_curvature_rate != FORD_INACTIVE_CURVATURE_RATE) || …` (upstream lines 253/276, both LatCtl CAN + LatCtl2 CANFD) with the fork's **bounded** check, ported into the upstream structure:
- Add `#define FORD_CURVATURE_RATE_MIN -0.001024f`, `FORD_CURVATURE_RATE_MAX 0.00102375f`, `FORD_INACTIVE_CURVATURE_RATE_CANFD 1024U`, and `AngleSteeringLimits FORD_CURVATURE_RATE_LIMITS_CAN/CANFD` (speed-dependent rate-of-change lookup) — from fork ford.h:83-93,163-205.
- Add `curvature_rate_cmd_checks(desired_curvature_rate, steer_control_enabled, limits)` (fork ford.h:261-279): when steering, bound `curvature_rate` within `[last ± roc(speed)]` (frame-to-frame rate-of-change) **and** the absolute field range; when not steering, require `== 0`. Persist `desired_curvature_rate_last`.
- `path_angle`/`path_offset` stay forced-inactive (unchanged) — we only admit `curvature_rate`.
- Keep every other Ford safety check byte-identical (longitudinal, buttons, curvature `steer_angle_cmd_checks`, RX hooks).

### 3c. `opendbc/safety/tests/test_ford.py` (the suite)
- The blocking assertion is `test_ford.py:286`: `should_tx = path_offset == 0 and path_angle == 0 and curvature_rate == 0`. **Adapt** `curvature_rate == 0` → `curvature_rate within the admitted bounded range AND within the rate-of-change step`, so `should_tx` reflects the new (safe) behavior. Existing `path_offset`/`path_angle==0` assertions unchanged.
- Extend/port the fork's `curvature_rate` boundary + rate-of-change tests (sweep to prove out-of-range and too-fast-step `curvature_rate` are **rejected**, in-range accepted, and `!=0`-when-inactive rejected).
- All other Ford safety tests (`test_curvature_rate_limits`, longitudinal, buttons, curvature limits) pass **unmodified**.

## 4. Safety analysis (must be airtight for QA)
- **Absolute bound:** `curvature_rate ∈ ±0.001024 1/m²` — the Ford message's own spec'd field range; the EPS is designed to accept it. Old fork ran ±0.001023 for months without incident.
- **Rate-of-change bound:** `curvature_rate_cmd_checks` limits frame-to-frame change (speed-dependent `angle_rate` lookup), so the feed-forward can't step arbitrarily.
- **Worst-case actuation:** compute the max lateral jerk / curvature-change the bounded `curvature_rate` permits at the full speed range, and show it stays within comfort/safety limits and is dominated by the *existing* `curvature` `steer_angle_cmd_checks` (which we do NOT relax). Include the numeric derivation in the QA packet.
- **Fault behavior:** when `steer_control_enabled` is false OR limits violated → `curvature_rate` forced/required 0 (inactive), identical to today's fail-safe.
- **Divergence disclosure:** this modifies one upstream safety assertion (curvature_rate must be inactive → must be bounded). Documented, justified, test-covered. This is the intended, user-approved "adapt ford.h + extend tests" path.

## 5. Validation gates (all must pass before re-drive)
- **G0:** diagnosis re-confirmed (§1 caveat).
- **G1:** `opendbc` Ford safety suite green (adapted test_ford.py + all others unchanged); safety misra/build checks pass.
- **G2:** full opendbc + selfdrive test gauntlet green (the migration gauntlet).
- **G3:** panda safety build + (if available) HITL/replay safety check; confirm `curvature_rate` is bounded on real command traces (replay ce/cf commands through the safety model → no violations, and out-of-bound synthetic → blocked).
- **G4:** bench: build on device, confirm the command path emits the feed-forward (`apply_curvature_rate` nonzero on curves, 0 inactive); no panda faults offroad.
- **G5:** on-road shakedown (empty road, ready to take over): re-measure achieved/commanded → should return toward ~1.0; confirm curves are held; no silent straight-through.

## 6. Defense-in-depth (secondary, parallel) — low-confidence safeguard
Independent of the feed-forward: under AOL, when `laneLineProbs` collapse below threshold for >N frames, **alert the driver + degrade** (don't silently continue). Scope/UX TBD in a follow-up mini-plan; does not gate the primary fix but should ship before relying on AOL.

## 7. Rollback
Path B (`curvature_rate = 0` + upstream curvature-only `ford.h` + unmodified test) is the fallback and remains the known safe-if-degraded state. Keep the revert as a single commit. Device A/B slot _a (pre-migration) remains the ultimate rollback.

## 8. Risks / open questions for QA
- Is the **rate-of-change** bound (fork `FORD_CURVATURE_RATE_LIMITS`) still correct against the migration's `STEER_STEP`/timing? Re-derive.
- Does the migration `carcontroller.py` still contain the deque computation, or only the zeroed send? (Determines edit scope — verify on device.)
- **Alternative considered — curvature-only (no safety change):** compensate the under-response via curvature gain/lookahead so `curvature_rate` stays 0 and the suite passes untouched. Rejected as primary because (a) the under-response is the EPS not following the *command* (a gain boost fights a moving, speed-varying deficit 0.12–0.72 and risks overshoot/weave), (b) the feed-forward is the *verified-good* behavior. Keep as fallback if QA finds the safety divergence unacceptable.
- Confirm `-apply_curvature_rate` **sign/encoding** into `LatCtlCurv_NoRate_Actl` vs the safety decode (`raw_curvature_rate` bit layout) matches — an inverted sign would feed-forward the wrong way.
