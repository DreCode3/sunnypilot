# Handoff v3 — 2026-07-01 AOL Silent-Lateral Incident: root cause converged (session 2026-07-01/02)

**Supersedes handoff v2** (`2026-07-01-lateral-safety-investigation-handoff-v2.md`). This session
reproduced the entire v2 baseline, then **retired v2's leading suspect (A: new-engine temporal
state) and its strongest signal (the recovery-lag)**, decoded the decisive event frame-by-frame
with verified sign conventions, and converged on a two-mechanism root cause with one build
regression and one both-build amplifier. Car remains **PARKED — do not drive** until the fix
plan clears QA (§6).

## 0. TL;DR  (REVISED after QA rounds 1 AND 2 — see §9; the ranking changed in round 2)

The car's failures were **not** perception, **not** the new inference engine, **not** the camera,
**not** desire/lane-change gating, **not** a PI sign flip, and **not** device param drift.
The decisive event is a nudge-perturbation whose recovery failed in two stages:

1. **[THE TAKEOVER DIFFERENTIATOR — real, cause OPEN] Sustained wrong-way wheel vs a ramping
   sent command.** In the final ~1.7 s the sent command ramped −0.000004→−0.00099 while the
   wheel moved the WRONG way (+0.000086→+0.000337 right; 6 LSB on steeringAngleDeg, confirmed
   by yawRate and by the offset drifting −0.24 m/s). Driver-holding REFUTED (torque ≤0.19 Nm
   until the override at +2.0-2.9 Nm; route-wide hands-off p95 is 0.19). ce shows **2** such
   sustained (≥0.5 s) wrong-way divergences in ~1.4 min of straight-active data (the decisive
   event + trel 92.3-94.8); the OLD build shows **0** in ~14 min, and on matched wrong-way-
   integral events its wheel tracked the same-size slow ramps at lag 0.00-0.05 s corr
   0.95-0.99 — under **bit-equivalent CAN content** (both builds sent rate ≡ 0 at these
   magnitudes: the old rate FF was curve-gated `interp(|pred|,[0,0.001,0.002],[0,0,1])`,
   telemetry-proven; anti-overshoot/limits/ramp_type/precision identical at both pins).
   So the differentiator is NOT command content. Candidates: EPS-internal state/mode, vehicle
   mechanical drift between June and July 1 (tires/alignment — **check cheaply before
   anything else**), road surface/crown at those spots. It is a TAIL behavior (ce's own
   non-event ramps tracked cleanly, ratio ≈1.0), n=2 → §6 A/B Leg 1 (nudge-release arrest
   test) measures directly whether it reproduces.

2. **[CONFIRMED BOTH-BUILD AMPLIFIER — the weave] golden-PI phase-lagged feedback**
   (kp measured 0.000500 on ALL four routes; ki=0.0002; integral cap interp(v,[20,30],
   [0.3,1.0]) → ±0.84 @ 62 mph, c7 reaches 1.00; P on a 1.5 s-EMA offset). After every offset
   sign reversal the stale EMA + saturated integral push the wrong way ~1.5-2 s — in the
   decisive event it cancelled ~100 % of the model's correction at drift onset (+0.000248 vs
   −0.000252, QA-verified to the 4th decimal; integral 0.01→+0.816=cap→anti-correcting the
   recovery). **BUT the old build absorbed this exact state routinely**: 9 deduplicated c7
   highway episodes with wrong-way |integral| 0.61-0.95 (49 % of large-integral highway
   samples are wrong-way — the integral is chronically half a cycle behind) all arrested at
   ≤0.48 m (one 0.70 m), zero overrides — at speeds where the cap is LARGER than the decisive
   event's. The PI adds ~+0.1-0.2 m and delays arrest ~1.5 s; with a working wheel that stays
   sub-takeover. It IS the long-standing weave mechanism and is worth fixing (F2) — as a
   weave/comfort/margin fix, not as the incident's differentiator.

3. **[Perturbation origin — decisive event]** The never-arrested rightward velocity was
   **driver-injected**: two rightward nudges at t0−6.2..−5.2 s (steeringPressed, torque to
   −2.06 Nm > the 1.0 Nm Ford threshold; yaw-integrated impulse +0.433 m/s vs +0.021 from the
   model command). Post-lane-change recentering framing was wrong. The event is a natural
   instance of the §6 Leg-1 nudge-release arrest test — which the current build failed once.

4. **[Remaining true build delta — curve-entry only]** Path B's FF zeroing is a real change
   for curve-entry/exit transients (|curvature|>0.001-0.002 where the old gate opened).
   Irrelevant to the straight-road drift events.

Perception collapses are **scene-driven and model-inherent** (the sim reproduces the device's
blindness DURATIONS on clean scenes, both builds — cite durations, not corr: collapse-shaped
trajectories give high corr even on mismatches; ev1/ev4's maneuver-scene excuse is an
assumption the desire-less sim cannot test). The overpass blowout was a trigger, not a cause.
Perturbation rate at highway speed was 3-5× higher on ce than c7 (12.6 vs 2.6 pressed-episode
equivalents/min) — direction as stated before, number corrected. The safety gap regardless:
**AOL steers silently through blindness/departure with no alert** → §5.

## 1. What this session VERIFIED (all numbers reproduced/derived first-hand)

1. **v2 baseline reproduced exactly** — all five `incident_analyses.py` analyses match §3 of v2.
2. **Sign conventions anchored with GPS ground truth** (n=425 1-s bins, corr 0.97-0.99):
   `steeringAngleDeg` **+ = LEFT**; `carState.yawRate` **+ = LEFT**;
   `desiredCurvature`/`apply_curvature`/`carOutput.curvature` **+ = RIGHT**;
   laneLines frame **+y = RIGHT** (left line y0 ≈ −1.6, right ≈ +1.7);
   lane-center offset (midpoint) **+ = car LEFT of center**. The PI `# SIGN VERIFICATION NEEDED`
   is **resolved: sign CORRECT** (P steers toward center; `position.y` same convention).
   ⚠️ Re-derivation trap (QA-found): a naive corr(midpoint, position.y@0.2s) comes out
   NEGATIVE (−0.18..−0.26) from drift-continuation dynamics and will falsely "find" a sign bug;
   the decisive test is corr(desiredCurvature, position.y@0.2s) on curves = +0.84..+0.95.
3. **Engines are equivalent for Nevada** (code-level proof): fb (1,25,512) → old "split branch"
   `arange(100)[3::4]` ≡ new `feat_q[::4]` (t−96..t stride 4, current incl.); desire max-pool
   identical; img pair [t−4, t] identical; **no prev_desired_curv input** (the one real semantic
   change is moot). Old engine fed policy features ending t−1 (1 frame STALER than new — wrong
   direction to explain lag). Sim anchors bound residuals: c5 OLD 0.990, cf NEW 0.980.
4. **Recovery-lag headline RETIRED** (v2 §3.6): c5 (OLD Nevada) all-events median **3.15 s**
   (verified myself: 13 collapses) — inside the NEW range; the headline rested on c7 alone.
   50-58 % of collapses on EVERY route are lane-change-associated. Blinker-excluded pooled:
   OLD 2.08 s vs NEW 2.60 s, MWU p=0.57; 17 covariate-matched pairs (grey/vEgo/depth):
   median NEW−OLD = **−0.05 s**, p=0.96. Honesty note (QA): pooled ALL-events is OLD 2.30 vs
   NEW 4.05 s, p=0.10 — the controls kill the HEADLINE, they do not prove equality
   (n≈20/group post-exclusion; a true ~0.5 s effect would be undetectable).
5. **Sim discriminator** (`scripts/recovery_replay.py`, lane-prob capture added): the sim
   (old-engine math, canonical Nevada fp32) reproduces collapse→recovery near-exactly on
   old-build frames (c7 2.75→2.75 s corr 0.994; c5 2.35→2.35 s corr 0.996) AND on new-build
   frames with real wide cams (cf 2.6→2.7 s corr 0.992; cf 1.3→1.55 s corr 0.934; ce ev3
   6.4→6.35 s corr 0.987; ce ev5 2.6→2.65 s corr 0.98) → blindness durations are FRAME-driven
   on BOTH builds. First-pass ce used a road-frame proxy for the missing wide cam (ecamera
   since pulled from device): ev1 diverged (8.11 s logged vs 0.3 s sim, corr 0.303 — 23 mph
   blinker/intersection scene) and ev4 sim recovered 2× faster on a corr-0.985 trajectory
   (threshold sensitivity at laneProb≈0.7).
   **Wide-frame rerun (real ecamera) result**: ev2/ev3/ev5 match near-exactly (2.45→2.5
   corr 0.982; 6.4→6.35 corr 0.99; 2.6→2.6 corr 0.984). The two residual divergences are BOTH
   maneuver scenes the desire-less sim cannot faithfully replay: ev4 had a commanded lane
   change (desireState laneChange 0.998 at −2.0 s; lane probs legitimately drop mid-maneuver),
   ev1 is an intersection right-turn at 23 mph (blinker, turnRight desireState 0.30). Every
   clean (non-maneuver) event on BOTH builds matches the sim → no evidence of an engine/bundle
   numeric regression; blindness durations are frame+maneuver-driven.
6. **Decisive ce event decoded** (t0=212.535 s, seg 2, 62 mph, laneProb 0.96-0.99, dead-straight
   road — lane-fit median curvature +1e-5/m): after a completed LEFT lane change the car sat
   +0.5 m LEFT still drifting left (+0.118 m/s); the DRIVER made two rightward nudges at
   t0−6.2..−5.2 s (steeringPressed, torque −2.06 Nm; yaw-integrated impulse +0.433 m/s vs
   +0.021 from the model command — QA round 2) injecting the rightward velocity that was never
   arrested; car sailed through center (t0−3.3 s) to −0.85 m peak, hands-off from t0−5.2 until
   the override. The model's command was in the correcting direction the whole time at 1.2-5×
   the kinematic need — **adequate if delivered promptly**. It was not delivered: PI cancelled
   it at onset (§0.2), then the wheel moved the wrong way against the ramp (§0.1). v2 §3.5's
   "wheel follows the command exactly" is **WRONG for this event** (that claim came from
   |cmd|>0.0015 samples).
7. **PI ground truth from telemetry** (BOTH builds log `LC:`/`CX1:` in rlog logMessage — the
   old device ran the telemetry live too, dirty=True; c5 282 / c7 1000 LC lines, c7 6712 CX1
   rows across 27 segments): kp = P/off median **0.000500** on ALL FOUR routes (golden),
   ki=0.0002; integral saturates at the speed cap (ce ±0.816 @ 27.4 m/s; c7 reaches **1.000**
   @ 30.6 m/s), ce full-drive integral p5/p95 −0.52/+0.67. An event-window full-chain sim
   matches logged carOutput at RMS 1.8e-5 (≈CAN LSB) with golden PI, 6-9× worse with weak/off.
8. **Old build ran golden PI too — measured directly** (kp 0.000500 from its own LC telemetry;
   c7 integral hits the interp cap 1.000; corroborated by the delta-vs-offset envelope and the
   c7 warm-start LaneBiasIntegral −0.7455). The PI **config** is NOT the build difference; the
   −0.3775 persisted integral is in-range for golden @ speed and **washes out** (I0=0 vs
   I0=−0.3775 → identical event trajectories).
9. **Params identical across builds** (initData snapshots c5/c7/ce/cf): enable_lane_positioning=1,
   FordCurveMode=0, FordPath4Enabled=0, LagdToggle=1, LagdValueCache 0.345/0.371/0.395/0.395.
10. **Exonerated with high confidence** (7-agent evidence sweep, key numbers in
    workflow output `wf_581900ad-af4`):
    - **Engine cadence**: NEW is FASTER (exec 23.9-24.2 ms vs OLD 27-34 ms; p99 gap 51 ms;
      0 drops; 20.00 Hz everywhere).
    - **Smoothing/delay**: same bundle manifest (Nevada index 32 gen 12, override lat=".1");
      fitted smoothing tau = 0.10 s on all four routes; effective action_t 0.50-0.55 s (delta
      is learned-cache noise); cross-corr lag 0 frames.
    - **Desire/lane-change gating**: DH.update gated on latActive (BOTH builds, byte-identical);
      the one pure-AOL lane change on ce executed perfectly (desire 0.998, +0.05 s ramp).
      The "lane-change-associated overrides" are mostly manual lane changes AFTER takeover.
    - **Camera/AE**: pipeline functionally identical (AE loop include-only diff; road-cam
      register set effectively identical); NEW drives simply had ~5× larger illumination steps
      (scene); decisive event exposure was clean (|m−t| ≤ 0.082, integLines 4-8, gain 1.00).
      AGNOS 17.2 → 18.4 (kernel-side not audited — residual, low priority).
11. **EPS small-command transfer regression** (mine, matched criteria):
    c5 lag 0.30 s corr 0.971 gain 1.22 | c7 0.30 s / 0.948 / 1.15 | **ce 0.45 s / 0.802 / 0.98**
    (cf insufficient active data). Frequency-resolved cross-spectrum (Welch, 20 Hz grid):
    at 0.43-0.55 Hz ce shows **12-25° more phase lag** than both old routes at the same
    frequency (delay 0.27-0.29 s vs 0.18-0.21 s, coherence 0.93) — lag growing with frequency
    is the rate-FF-removal signature — BUT the mid-band (0.16-0.31 Hz) matches the old builds,
    ce has only ONE 63-s contiguous usable span, and its high-f gain is higher (not lower, as
    naive FF removal would predict). Honest grade: **directional but thin**. The event-local
    wheel non-delivery (§1.6) is the strongest in-log transient evidence; the controlled A/B
    drive (§6) is the decisive test — do not treat F1 as proven from logs alone.

## 2. Latent bugs found along the way (not the cause, fix eventually)

- **Desire pulse lost on dropped frame** (NEW build, code-level): a desire rising edge landing
  exactly on a vipc-dropped frame (prepare_only) is silently lost — new modeld.py consumes
  `prev_desire` before the prepare_only return while `desire_q` is only shifted on full runs.
  Didn't fire in these drives (0 drops). File under modeld_v2 cleanups.
- v2 carries: fork test_ford.py fails 8453 subtests vs its own ford.h (pre-existing).

## 3. Root-cause statement (for the fix plan) — REVISED after QA rounds 1+2

**Confirmed chain (decisive event)**: driver nudge injects +0.43 m/s rightward velocity →
model commands a correct, adequate correction (1.2-5× the kinematic need, ramping
−0.0003→−0.001) → (a) golden PI's stale EMA + cap-saturated integral cancel it for ~1.5-2 s
[CONFIRMED both-build amplifier: adds ~+0.1-0.2 m, delays arrest; the old build absorbed
identical wrong-way-integral states 9+ times on the highway at ≤0.48-0.70 m, no overrides] →
(b) in the final ~1.7 s the sent ramp was NOT delivered — the wheel moved the wrong way
≥0.5 s [THE DIFFERENTIATOR; cause OPEN; not command content (bit-equivalent CAN both builds);
not driver-holding (torque ≤0.19 Nm); tail behavior, 2 instances on ce vs 0 in 14 min OLD;
candidates: EPS-internal state, vehicle mechanical (tires/alignment), road surface] →
excursion 0.85 m → override.
The "old build arrested at ±0.3-0.4 m" offsets comparison (p95 0.34-0.36 vs NEW 0.51-0.71) is
route/condition-confounded; the CONTROLLED version of that comparison (matched wrong-way-
integral center-crossings) shows the old build arresting at ≤0.48 m with a tracking wheel —
which is exactly what pins the differentiator on delivery, not on the PI.

**Under AOL** the failure is silent (no alert while latActive-only) — that's the §5 safeguard.

## 4. Fix plan (REVISED after QA round 1 — goes through 2-step QA before ANY implementation)

Priority order changed: **F2 and F3 lead**; F1 is re-scoped and demoted.

- **F2 (tuning, no safety layer): de-lag the PI transient — now scoped as the WEAVE fix, not
  the incident fix.** Round 2 showed the old build absorbed identical PI wrong-way states
  sub-takeover with a working wheel, so F2 does NOT address the tail delivery failure; its
  honest goals are: reduce the chronic weave amplitude (the integral is wrong-way 49 % of the
  time it is large — it is chronically half a cycle behind), cut the ~1.5 s arrest delay, and
  reduce the +0.1-0.2 m excursion amplification. Best-supported option = **(c) integral cap
  back to 0.30 at all speeds** (QA2 verified it costs ≈0 steady-state centering: no route's
  highway data needs >0.3 to hold center — c7 centered median |int| 0.13, ce 0.02; the
  committed old pin even shipped cap 0.3 fixed; worst-case residual ≈0.05-0.1 m). Complements:
  (b) freeze P while sign(EMA) ≠ sign(raw offset); (a′) FASTER reversal decay — NOT a hard
  reset (would re-expose the structural +0.22 m left bias); (d) reduce kp. All cheaply
  falsifiable pre-drive by replaying the event decomposition. Success metric for the A/B:
  onset-phase net command no longer cancelled + wrong-way-integral episode excursions shrink
  vs the c7 baseline distribution (0.19-0.70 m) — NOT "prevents takeovers" (it can't, alone).
- **F3 (safety net): AOL blindness/departure safeguard** (§5, with the two QA fixes) — ship
  regardless.
- **F1 (re-scoped, demoted): curvature_rate FF.** Restoring the fork's implementation verbatim
  is a **no-op below 0.001 1/m** (curve-gated) — it CANNOT fix the decisive-event mechanism,
  and an A/B measuring small-command transfer on it would return a guaranteed null. Two honest
  variants: (F1a) restore the fork's gated FF for curve-entry/exit feel only (what Path B
  actually removed); (F1b) NEW design — command rate on small corrections too (lower/remove the
  gate) — this is untested new behavior, not a restore, and only worth pursuing if the A/B's
  small-perturbation arrest test shows the old build also fails slow small ramps (i.e. the
  §0.2 delivery failure is EPS-inherent and needs FF help) — panda-safety-touching, full Ford
  safety suite required.

## 5. AOL safeguard design (independent of root cause — draft for QA round 1)

Requirement: under AOL (latActive && !enabled), when lateral guidance is unreliable or a lane
departure develops, ALERT (chime + visual) and degrade predictably instead of silently drifting.

Design sketch (alert-first, no panda changes) — REVISED per QA round 1:
- **Maneuver suppression (required)**: 50-63 % of laneProb collapses on EVERY route are
  blinker/lane-change-associated, and lane changes legitimately sweep |offset| through
  0.7→1.7 m at good laneProb. BOTH monitors are suppressed while a blinker or lane-change/turn
  desire is active and for a ~3 s grace after — otherwise the alerts fire on essentially every
  lane change (instant alert fatigue).
- **Low-confidence monitor**: EMA(inner laneProb, ~0.5 s) < 0.3 for > 1.0 s while latActive and
  v > 10 m/s → warning alert "Lane confidence low — take control" (repeat chime while
  persisting). Threshold from this incident's data: collapses <0.15 lasted 2-8 s; a 1.0 s
  qualifier catches all takeover-relevant events, skips 0.3-0.5 s flickers.
- **Departure monitor — needs a drift-RATE term**: a plain |offset| > 0.7 m threshold fires
  only ~0.3 s before the decisive event's override (offset crossed 0.70 m at t0−0.3; the
  earlier "1.2 s" claim was wrong), i.e. AFTER the driver reacted once the 0.5 s sustain is
  added. Lowering to 0.45 m alone is alert-fatigue territory (NEW-build good-perception p95
  offsets are 0.50-0.70 m). Use instead: laneProb > 0.6 AND |offset| > 0.4 m AND
  offset-velocity > 0.15 m/s (same direction), sustained 0.5 s → fires at ~t0−1.7 s in the
  decisive event. Keep an absolute backstop (inner-line distance < 0.35 m).
- **Degrade**: keep steering best-effort (silent disengage is worse), alert escalates; optional
  param-gated fallback to disengage-lateral-with-loud-alert after N s of continuous blindness.
- **Placement**: sunnypilot MADS/AOL alert path (selfdrived events extension) so alerts fire
  without `enabled`; monitors read modelV2 + carState (blinkers) + carControl. NO ford.h /
  panda changes.
- Test plan: unit tests on the monitor state machine; replay harness over ce/cf incident
  windows must fire alerts at the right times (assert departure alert ≥ 1.5 s before the
  driver's actual overrides on the decisive event; low-conf alert ≤ 1.5 s after collapse
  onset on non-maneuver collapses); **zero false alerts** on c5/c7/b5/7f replays including
  their ~50 % lane-change collapse events.

## 6. Next steps (in order)

1. ~~Finish the ce wide-frame rerun~~ **DONE** — divergences resolved as desire-replay
   limitations on maneuver scenes (see §1.5); no engine numeric regression.
2. ~~2-step QA~~ **DONE** — both rounds ran and were folded in (§9); every load-bearing QA
   claim re-verified first-hand before adoption.
2b. ~~Cheap physical checks~~ **CLEARED by user (2026-07-02)**: vehicle just had its 60k-mile
   dealer service and the user's partner is an automotive diagnostic technician — mechanical
   condition confirmed good. Mechanism C's candidates narrow to **EPS-internal behavior and
   road surface** → A/B Leg 1 (nudge-release arrest) is now the single decisive test for C.
3. **Controlled A/B drive — REDESIGNED** (user drives, car otherwise parked): same corridor,
   matched 18-25 m/s + one highway leg (the PI cap is speed-dependent), CX1 telemetry on,
   DisableUpdates=1, one variable at a time:
   - **Leg 1 (no code change): small-perturbation arrest test on the CURRENT build** — from
     centered, introduce a small deliberate offset (brief nudge), release, measure
     time-to-arrest + excursion size + whether the wheel tracks the slow corrective ramp.
     This measures the §0.2 delivery failure directly and provides the baseline.
   - **Leg 2: F2 arm** (integral cap 0.30 — smallest reversible change) vs Leg 1 baseline.
     Prediction to test: post-perturbation excursion shrinks to sub-takeover size.
   - **Leg 3 (only if Leg 1 shows the wheel fails slow small ramps): F1b arm** (ungated
     small-command rate FF — NEW design, needs Ford safety suite first). NOTE: an A/B of the
     old gated FF (F1a) on small commands is a guaranteed null — do not run that as designed
     in the original §6.3.
   Metrics: time-to-arrest, excursion peak, offset-hold p95, EPS gain in the 0.08-0.8 Hz band
   (report gain, not the fragile lag headline), all at matched speed/corridor.
4. Implement F3 safeguard (after its QA), then F2/F1 per A/B outcomes.

## 7. Reproduce-this-session commands

- Baseline: `PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/incident_analyses.py {achieved,recovery,overrides,offsets,underpass}`
- Sim discriminator: `... scripts/recovery_replay.py [route_c7 route_c5 route_ce route_cf]`
  (writes JSON to `retrospective_lateral/incident_2026_07_01/replay_cache/`).
- Engine diff: `git diff 2021_explorer_st-mici clean-v2026.002.001 -- sunnypilot/modeld_v2/`;
  old-build opendbc pin: `git ls-tree 5e0785b98756 opendbc_repo` → fbcd9f3df; new: 849b72a1.
  Control-chain delta: `git -C opendbc_repo diff fbcd9f3df 849b72a1a -- opendbc/car/ford/`.
- PI telemetry: grep rlog logMessage for `"LC: off="` / `"CX1:"` (NEW build only).
- Agent sweep outputs: workflow `wf_581900ad-af4` journal (7 agents, all high confidence).

## 8. Discipline notes carried forward

- The investigation flipped again this session — TWICE (v2's Suspect A + recovery-lag headline
  are gone; then my own "Path-B FF removal is the leading build regression" was refuted by QA
  round 1: the old build's rate FF was curve-gated ≡0 at small commands, telemetry-proven).
  The flips were caught by: adding c5 to the comparison, verifying sign conventions from GPS,
  reading the actual telemetry, and the collaborative QA re-deriving everything independently.
  **Keep verifying every confident claim, especially your own.**
- **Single-segment checks over-generalize**: "old build has no LC/CX1 telemetry" came from
  scanning ONE segment (c7 seg 1, not lat-active); route-wide the telemetry is there (6712 CX1
  rows). Scan the whole route before declaring absence.
- Sign conventions on this car (verified, §1.2) — cite them instead of re-deriving.
- The car stays PARKED except the §6.3 instrumented A/B.
- Device: read-only SSH fine; DisableUpdates=1 stays; params at `/data/params/d/`.

## 9b. Per-event attribution (all system-active takeovers + AOL blind events; script
`scripts/event_attribution.py`, run 2026-07-02)

Mechanisms: **A** = blind model + silent continuation · **B** = PI wrong-way transient (weave
amplifier) · **C** = wheel under-/non-delivery of the sent command · **MANEUVER** = during a
system-commanded lane change. "Silent" = no alert beyond the maneuver chime.

| t (mono) | mode | mph | context | what the data shows | mechanism | alerted? |
|---|---|---|---|---|---|---|
| ce 137.9 | mixed | 61 | post lp-dip 0.83 | integral winding 0.02→0.35, PIΔ +1.6e-4 opposing needed-left; wheel did move left; early driver catch | B (mild) | none |
| ce 202.8 | ENGAGED | 55 | commanded LC, desire 1.0 | model −1.74e-3, sent −1.49e-3, wheel moved ~0 over 2.5 s; steer-saturated fired | **C** during MANEUVER | "take control" ✓ |
| ce 206.3 | ENGAGED | 56 | commanded LC, lp 0.24 | mid-maneuver line reassignment; drift +0.74 m; rescue | MANEUVER + A | "Changing Lanes" only |
| ce 212.5 | ENGAGED | 62 | the decisive event | driver-nudge perturbation → PI cancels onset (int +0.82=cap) → wheel wrong-way final 1.7 s → −0.85 m | **B + C** | silent |
| ce 256.8 | AOL | 63 | commanded LC executing | partial delivery, drift only −0.08 m; driver impatience/hesitant execution | MANEUVER (mild) | "Steer Left"/"Changing Lanes" |
| ce 260.2 | AOL | 61 | commanded LC, lp→0.00 | BLIND 4.9 s during maneuver (= BLIND 257.3), drift −0.67 m while blind | **A** during MANEUVER | "Changing Lanes" only — no blindness alert |
| ce 265.1 | AOL | 58 | post-LC | integral wound to +0.71, PIΔ +3.6e-4 opposing; wheel moving but fighting PI | **B** | silent |
| ce 273.1 | AOL | 60 | post-blind lp 0.17 | weave swing −0.42→+0.21 m, wheel delivering; driver caught the swing | A recovery + B | silent |
| ce 293.2 | AOL | 52 | lp dip 0.20 | offset swing +0.23 m; integral −0.36→−0.47; wheel −6.4e-4 vs sent +2.0e-4 (brief counter-move) | B (+C?) | silent |
| cf 676.1 | AOL | 53 | overpass CURVE ENTRY | model +1.2e-3 (the curve), sent +1.0e-3, wheel +3.5e-4 (~1/3); steer-saturated fired. NOTE |cmd|>1e-3 = the regime where the old build's rate-FF gate OPENED → the one event where Path-B (F1a) is plausibly relevant | **C** (curve-entry) | "take control" ✓ |
| cf 682.2 | AOL | 55 | blowout BLIND 2.6 s | deep-blind, total swing +1.79 m across the episode | **A** | **silent** |
| cf 686.2 | AOL | 55 | post-blowout, lp 0.03 | still blind/recovering, offset −1.01→+0.46; flat command | **A** | **silent** |

Key aggregate facts: (1) the four "lane-change-associated" overrides were **system-commanded
lane changes** (desire 1.00) being rescued — NOT manual maneuvers (corrects the round-1 desire
agent's inference for the override set; its code-level finding that desire gating works under
AOL stands). (2) The stock **steer-saturated alert caught the two strong-command
under-deliveries** (202.8, 676.1) but ALL small-command drift/blind failures were silent —
sharpening F3's requirement: the gap is specifically small-command drift + blindness, which
steerSaturated cannot see. (3) All remaining ce/cf deep-blind events occurred during MANUAL
driving at intersections/turns (expected blindness, no system role).

## 9. QA log

- **Round 1 (collaborative, 2026-07-01/02)**: 11 findings. MAJOR: (1) F1's causal mechanism
  refuted — old build's rate FF curve-gated ≡0 below 0.001 1/m, verified from old-build CX1
  telemetry (I re-verified first-hand: gate code at fbcd9f3df line 463; c7 6712 CX1 rows,
  rate ≡0 for 99.1 % of |cmd|<0.0005 samples); (2) §1.11 transfer "regression" fragile —
  reproducible signature is a GAIN deficit (ce 0.55-0.66 vs old 1.05-1.26) within old-build
  chunk scatter, cf counter-signal 1.10-1.24, lag headline not robust; PI-confound excluded
  (PI-P in-band variance 2-5 % of cmd variance). CONFIRMED to the 4th decimal: the decisive-
  event decomposition, all six sign conventions, golden-PI-on-both-builds (via direct kp
  measurement 0.000500 — stronger than my envelope test), recovery-lag retirement (their
  MWU p=0.76, mine p=0.57 — same conclusion), params. FIXED in this doc: §0/§3/§4/§5/§6
  restructured; "old build predates telemetry" deleted; departure-alert timing corrected
  (0.7 m crossing was t0−0.3 s, not −1.2 s); maneuver suppression added to safeguard;
  A/B redesigned around small-perturbation arrest.
- **Round 2 (adversarial, 2026-07-02)**: 6 findings; scripts `qa2_*.py` in the session
  scratchpad. **BREAKS (adopted, re-verified first-hand)**: the mechanism RANKING — the old
  build routinely carried the identical PI wrong-way state at highway speed (my independent
  extraction: 9 deduplicated c7 episodes, wrong-way |int| 0.61-0.95, all arrested ≤0.48 m —
  one 0.70 m — zero overrides; 49 % of large-integral highway samples are wrong-way) with the
  wheel tracking matched slow ramps at lag 0.00-0.05 s corr 0.95-0.99 under bit-equivalent
  CAN; ce has 2 sustained (≥0.5 s) wrong-way wheel divergences in 1.4 min vs 0 in 14 min OLD
  → the delivery failure is the takeover differentiator; PI relabeled both-build amplifier;
  F2's "prevents takeovers" prediction dropped. **WEAKENS (adopted)**: decisive-event
  perturbation was DRIVER-INJECTED (nudges to −2.06 Nm at t0−6.2..−5.2, impulse +0.433 m/s —
  corroborated by the geometry agent's independent timeline); §1.5 corr numbers are
  shape-inflated (cite durations); perturbation-rate figure corrected to 3-5× at highway
  speed. **HOLDS (attacks failed)**: delivery-failure measurement (driver-holding refuted,
  torque ≤0.19 Nm), model-command adequacy (1.2-5× need), F2(c) cap-0.30 safety (no route
  needs >0.3 steady integral; committed old pin shipped 0.3 fixed), §6 A/B design. NOTE: QA2's
  exact "twin event" timestamps did not reproduce under my time base (their trel vs logMono);
  I verified the underlying claim independently via LC telemetry before adopting.
