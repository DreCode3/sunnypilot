# Discriminator Analysis — Design v3 (logic only, NOT yet run)
*(v3 = after cooperative QA + adversarial QA. Both reconciliation logs at bottom.)*

## Purpose
Localize the cause of the REAL, GPS-confirmed in-curve **right over-turn** on the 2021 Explorer ST
(curvature control, CAN Q3), from existing logs. Driver ground truth: right curves cut right into the
adjacent lane → override; left fine; **alignment perfect (no manual pull)**; effect appears road-independent.

## Scope — what the logs CAN and CANNOT decide (revised by adversarial QA)
The logs **can** (decidable here):
1. **Rule banking (H3) IN** (bank present on approach ⇒ likely contributor). ⚠️ Ruling H3 **OUT is weak** —
   approach-straight roll under-reads apex bank (superelevation increases into curves); only a confident rule-IN.
2. **Test road-INDEPENDENCE as a POPULATION test** (many distinct bends each over-turn; ~55 distinct right
   bends / ~250+ events across ~31 CX1 routes) — NOT a per-bend repeatability test (~1.2 passes/bend).
3. **Detect/rule-out H2 (plant gain) via the magnitude-scaling regression** on `dCE = Δψ_GPS − Δψ_cmd`
   (slope ∝ |curvature| ⇒ plant gain). ⚠️ This does NOT positively identify H1 (corrected — see Discriminator):
   `dCE` is a pure plant metric, so H1 (model over-commands, plant faithful) gives dCE≈0. H1 is only inferred
   BY ELIMINATION (plant faithful + confirmed drift) or needs the weak κ_road reference. **Clean positive H1-vs-H2
   separation still requires the controlled PI-off drive (or a trustworthy κ_road).** The original scope concession
   stands; adversarial-QA's "logs can separate H1/H2" was an over-claim.
4. **Quantify the H0 yaw-sensor offset** and confirm it's a measurement artifact.

## Hypotheses (with their log signatures)
- **H1 PERCEPTION** (RESIDUAL camera/lane bias beyond calibration): `des` mis-aimed → road-INDEPENDENT,
  **ASYMMETRIC** L/R, bank-independent, **ADDITIVE** (dCE_int has nonzero intercept, ~zero slope vs Δψ_cmd),
  car executes the wrong command. ⚠️ `rpyCalib` yaw ≈ −2.96° already absorbs the −3.05° mount, so a nominal
  −3° is H0/H2-consistent; H1 needs a *residual* mis-aim. Positive log signature = the regression INTERCEPT.
- **H2 CURVATURE-GAIN** (steerRatio/plant): **MULTIPLICATIVE** — dCE_int ∝ Δψ_cmd (nonzero SLOPE through
  origin), **SYMMETRIC** L/R, road-INDEPENDENT, bank-independent, zero straight-line pull (matches "no pull").
- **H3 BANKING**: over-turn correlates with **road bank measured on approach straights** (per-route-zeroed);
  road-specific. Rule-IN strong, rule-OUT weak.
- **H0 YAW-SENSOR OFFSET** (nuisance): constant additive offset in `meas=−yawRate/v` (GPS~0 on straights).
  Estimate & subtract; must NOT touch event selection or direction labels.

## Signals (feasibility-verified across both QA rounds)
- CX1 ≈20 Hz, **CX1-bearing routes only (~31, ≈route_91+)**: `cmd`,`des`,`meas`(=−yawRate/v, ✓3e-7),`lOff`,
  `ang`,`yr`,`v`,`ovr`. Conventions **+=RIGHT confirmed 100%** (left 45/45 neg, right 46/46 pos).
- GPS `liveLocationKalman` 20 Hz: `velocityNED` (Kalman state — PRIMARY for GPS_kappa; gate on `velocityNED.valid`,
  ~33% invalid during start-of-route GPS warmup — drop it), `positionGeodetic` (cross-check only),
  `angularVelocityCalibrated[2]` (3rd estimator, same sensor family as meas), `orientationNED` roll.
- `liveCalibration.rpyCalib` yaw ≈ −2.96°. `modelV2.roadEdges`/`laneLines` (model-derived — caveat).

## Derived quantities
1. **GPS_kappa(t)** PRIMARY = `κ_right = −dψ/ds`, ψ=atan2(vN,vE), ds=speed·dt (velocity-heading-rate;
   per-sample σ≈8.8e-4 but see #2). Position double-diff = cross-check only. Assert sign vs cmd on known curves.
2. **Δψ = ∫κ ds over the curve = total heading change** — HEADLINE (validated: endpoint-difference, so error
   ≈1e-4 1/m-equiv and does NOT random-walk with window; delay-robust; ~4:1 per-event SNR for a 3e-4 effect).
3. **yaw_offset(route)** = **purpose-written** per-route median of `meas` on cmd-based straights
   (|cmd|<5e-4, v>15, not-override). ⚠️ Do NOT reuse `compute_bias_baseline` (it's event-local mean(meas−cmd),
   thresh 0.0008, v≥7 — different definition). Sanity vs known +0.00035.
4. **meas_corr = meas − yaw_offset(route)**.
5. **road_needed_kappa** (WEAK, secondary only): roadEdges (model-derived) / low-|lOff| passes (~3/bend). dMODEL caveated.
6. **road_bank** = `orientationNED` roll on approach/exit STRAIGHTS, **minus a per-route roll-zero** (long-straight
   median — roll has NO constant zero: measured +1.5/−0.83/+1.13/−0.35/+0.93° across routes). NOT apex roll
   (apex roll ≈ bank + body-lean; corr(aLat, roll_apex−roll_approach)=−0.77 → body-lean dominates). Where possible
   use **reverse-direction passes of the same bend** (road bank flips sign with travel direction; isolates bank from lean).

## Event selection (v3 — C-A fix)
- CX1-bearing routes only; coverage pass; **unified path resolver** (cx1-dir + gps-prefix) with non-empty asserts
  (find_rlogs ≠ load_cx1 layout — verified bug); per-file try/except (a corrupt zstd segment exists).
- `detect_curve_events(..., require_dir_match_pct=0.0)` — ⚠️ **strip the meas-based dir_match gate**: it drops
  events where sign(cmd)≠sign(meas) and meas carries the +0.00035 H0 offset → asymmetric pruning (measured 6% L vs
  3% R) that would leak H0 into the symmetry test. **Label direction by peak_cmd/GPS only; meas never touches selection.**
- Moderate+sharp; gate on **inner** laneLineProbs ≥0.6. **Do NOT drop overridden events** — analyze the
  **pre-override window** to onset (dropping them removes the worst right curves → masks the effect); report override asymmetry.
- Speed-bin matched (20-35/35-45/45-55), **N≥~20 clean events/(dir×bin)** (achievable: 45-55R already 21/16 bends in 6 routes).

## THE discriminator metrics
- `Δψ_GPS`, `Δψ_cmd`, `Δψ_des`, `Δψ_meas_corr` per event (global lag = 0.25s + lookahead, pooled once; no per-event xcorr).
- `dCE_int = Δψ_GPS − Δψ_cmd` (execution vs command) + per-event SE; require |effect|>2·SE.
- **⭐ PLANT-GAIN REGRESSION (I-A, CORRECTED): fit `dCE_int = g·Δψ_cmd + b`, per direction, speed-binned.**
  ⚠️ CORRECTION to adversarial-QA I-A: `dCE = Δψ_GPS − Δψ_cmd` is a PURE PLANT metric (execution vs command;
  κ_road never enters it). So:
  - **g≠0 (slope)** ⇒ MULTIPLICATIVE plant gain ⇒ **H2** (report g as the gain error). VALID.
  - **b≠0 (intercept)** ⇒ additive PLANT bias — **NOT H1.** (The agent's "intercept = H1 perception" is wrong.)
  - **H1 (model over-commands, plant faithful) produces dCE ≈ 0 and is INVISIBLE to this regression.**
  H1 is identified only (a) BY ELIMINATION: dCE≈0 (plant faithful) + independently-confirmed drift (lOff) ⇒ the
  over-command is the model; or (b) by regressing the FULL symptom `(Δψ_GPS − Δψ_road) vs Δψ_cmd` — slope=H2,
  intercept=H1 — which REQUIRES the weak κ_road reference. So this regression rules H2 in/out; positive H1
  confirmation still needs κ_road (weak) or the PI-off drive.
- `dCMD = Δψ_cmd − Δψ_des` (≈0 expected). `dMODEL` (secondary/caveated). `road_bank` regression for H3.

## Decision logic (v3)
| Pattern | Verdict | Confidence note |
|---|---|---|
| dCE_int vs Δψ_cmd: **slope≠0** | **H2 plant gain** | log-decidable if SNR ok |
| dCE_int ≈ 0 (slope&intercept≈0) **+ confirmed lOff drift** | **H1 model/perception** (by ELIMINATION) | NOT a positive signature; confirm via κ_road or PI-off drive |
| dCE_int intercept≠0, slope≈0 | additive PLANT bias (a distinct cause) | NOT H1 |
| over-turn ↑ with approach-straight **road-bank** (zeroed) | **H3 banking** contributes | rule-IN strong; rule-OUT weak |
| asymmetric L/R **after** stripping the meas dir-gate | supports H1/geometry over H2 | guard vs H0 selection leak |
| effect vanishes after yaw_offset correction | **H0** artifact | GPS already rules out real drift |
Mixtures allowed (e.g., H3 amplifying H1). If the regression CIs are inconclusive, the deliverable is a sized
prior + the PI-off drive as the tiebreaker — NOT a forced call.

## Residual risks (post-both-QA)
- H1/H2 regression SNR (3e-4 effect; needs N≥20 + the 15x range to pull g/b apart) — may still defer to drive.
- road_bank rule-OUT weak (apex more banked than approach); reverse-direction passes the cleaner test if available.
- dMODEL reference model-derived ⇒ not perception-independent.
- yaw_offset stability within route; velocityNED warmup; corrupt-segment handling.

---
## Reconciliation log — Stage 1 COOPERATIVE QA (agent a98953ac)
C1 dCE<GPS-noise → velocity GPS_kappa + integral Δψ + SE gate. C2 CX1-only routes. C3 unified path resolver+asserts.
I1 GPS_kappa sign (−dψ/ds)+assert (cmd/des/meas +=right CONFIRMED). I2 cmd<0@33.92615 RESOLVED (it's the OLD LEFT
event, not the right bend). I3 road_needed thin/model-derived → dMODEL demoted. I4 bank from approach straights not
apex. I5 yaw_offset cmd-based straights. I6 global lag not per-event. M1 rpyCalib absorbs camera yaw → H1=residual.
M4 N≥20. M5 pre-override window, don't drop overrides.

## Reconciliation log — Stage 2 ADVERSARIAL QA (agent af736f0f)
**C-A (CRITICAL, accepted):** event detector's meas-based `dir_match` gate is H0-contaminated → asymmetric event
pruning (6% L vs 3% R) leaks into the GPS symmetry test → run with `require_dir_match_pct=0.0`, label dir by cmd/GPS.
**I-A (PARTIALLY accepted, then CORRECTED on review):** add the `dCE_int = g·Δψ_cmd + b` regression — but ONLY as
an H2 PLANT-GAIN detector (slope). The agent's claim that intercept=H1 is WRONG: dCE=GPS−cmd is a pure plant metric,
so H1 (model over-commands, plant faithful) gives dCE≈0 and is invisible here; the intercept is an additive PLANT bias.
The additive/multiplicative idea separates H1/H2 only when applied to the FULL symptom (GPS−κ_road) vs cmd, which needs
the weak κ_road. So the agent OVER-claimed "logs can separate H1/H2"; original scope concession stands (H1 positive
confirmation needs κ_road or the PI-off drive). H1 from logs = by-elimination only. [Corrected by main author, not the agent.]
**I-B (IMPORTANT, accepted):** roll has no stable zero (per-route zeroing) + approach-straight under-reads apex bank
→ H3 rule-IN strong / rule-OUT weak; prefer reverse-direction same-bend passes. body-lean dominates apex roll (−0.77).
**M-A (accepted):** write the per-route yaw_offset estimator; don't reuse compute_bias_baseline (different def).
**M-B (accepted):** gate on velocityNED.valid + drop GPS warmup + per-file try/except (corrupt segment exists).
**Validated as correct:** integral Δψ (no random-walk, ~4:1 SNR), road-independence (55 bends/250+ events) as a
POPULATION test, N≥20 achievable, avoiding apex roll, demoting dMODEL.
