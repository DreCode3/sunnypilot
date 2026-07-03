# 2026-06-24 Retrospective Lateral — Fresh Full-Analysis Root-Cause Review

Subject: 2021 Ford Explorer ST sunnypilot lateral driving-performance issues
1. Low-speed steering-wheel swing at 1–10 mph.
2. Straight/gentle-section weave at 10–70 mph.

Method: `superpowers:systematic-debugging` (Phase 1 root-cause investigation → Phase 2 pattern analysis → Phase 3 hypothesis testing). This is an **independent re-review**: prior conclusions, tuning values, and report verdicts were treated as hypotheses and re-derived from the CSV/JSON/NPZ artifacts before acceptance. No prior conclusion was inherited without re-checking row counts and key group statistics.

Constraints honored: no vehicle-control code touched; `opendbc_repo/` and `panda/` untouched (verified clean); no remote comma actions; all `retrospective_lateral/results/` files remain gitignored and unstaged; analysis was read-only Python/pandas over existing outputs.

---

## 1. Executive Summary

**Both symptoms are real, and the dominant evidence places the *origin* of the oscillation upstream of the steering controller — at the model / path-planner desired-curvature stage — for both symptom classes. The custom PI lane-centering controller, the EMA/smooth filter, the rate limiter, and the final Ford curvature command do not create or amplify the oscillation; they faithfully pass it through and, on the median, *attenuate* it.**

This conclusion survived every adversarial test I ran:

- **Stage magnitude ladder (both symptoms):** the band-limited oscillation is already at full size at the model/planner stage and does **not** grow across controller stages. Weave: `desired` 2.09e-4 → `cp_final_command` 1.66e-4 (lower) → realized `path` 2.20e-4. Low-speed: `desired` 10.4e-4 is the ladder peak, `cp_final` 8.4e-4 (below desired); `cp_final/desired` median **0.875**, attenuating in **60/68** rows.
- **Golden-PI natural experiment (NEW, strengthens weave):** 38 weave episodes exist on 3 proven current-config "golden PI" routes (`route_b8/be/bf`, `lc_kp=0.0005`). A *stronger* controller neither amplified nor cured the weave; its oscillation still originates at model/desired with `command/desired = 1.000`, statistically indistinguishable from weak/unknown configs. The handoff's "0 golden rows" caveat is true **only for low-speed**, not weave.
- **Lead-context split:** `cp_final < desired` in all 6 lead×subset cells; no controller amplification appears even in near-lead tails. Lead is a co-occurrence marker, not a trigger (event-aligned lead onset shows **no** upstream growth: −9.45% at `model_y20`).
- **Low-speed metric is not degenerate:** curvature shows no `1/v` inflation and agrees with speed-independent steering-degrees and model-path-meters — the swing is real path/wheel motion.
- **Speed:** the dominant severity modulator (Spearman −0.80), but the weave is not *only* speed — its spectral peak is pinned at ~0.20 Hz independent of speed.

**What the evidence does NOT yet establish:** whether the upstream oscillation is (a) a model/perception/path-planning **artifact** (the model invents a wobble), (b) the model **faithfully following a real wandering lane-center/corridor**, or (c) a **fixed-period loop element** (delay/filter/update-cadence) producing a limit cycle that appears upstream via feedback. This is the single most important unresolved question, and it is the reason a vehicle-control change is not yet justified.

**Recommendation: DO NOT CHANGE DRIVING CODE YET.** The evidence is ready for an *offline model/path-vs-road discrimination* analysis plan, not a vehicle-control implementation plan. The correction *direction* (upstream model/path/perception, not PI/final-command tuning) is evidence-backed; the specific *fix* is not yet identifiable from these logs. Section 8 gives the gate that would flip this to "ready for implementation plan."

---

## 2. Confidence Snapshot (this review's calibrated estimates)

| Claim | Confidence | Change vs handoff | Why |
| --- | ---: | --- | --- |
| Controller (PI / EMA / rate-limit / final command) is **not** the primary source of either symptom. | **High, ~85%** | ↑ | Attenuation ladder + golden-PI natural experiment + lead-split + robustness all agree; only 4/75 low-speed + 2/26 rows are "controller-gain-not-excluded," none top-severity. |
| 10–70 mph weave originates upstream in model/lane/path desired-curvature. | **High, ~85%** | ↑ (from 80–85%) | Golden-PI natural experiment refutes the PI-config confound for weave; 1482/1482 model_or_desired; survives lead/speed/yaw/band robustness. |
| Low-speed swing originates upstream of the final command. | **Medium-high, ~78%** | ≈ | 58/75 upstream buckets; metric-degeneracy refuted; only 4/75 controller-gain-not-excluded. Held back by 0 golden coverage, 7 CP-missing rows, low SNR. |
| Low-speed immediate carrier is near/short-horizon lane-center + desired geometry. | **Medium, ~55%** | ≈ (slightly lower) | Mixed: 12/26 source rows but the **largest single class (11/26) is the desired-high/model-absent counterexample**; median y10 finite only 0.46; model/lane corr median 0.13. |
| Upstream signal is a model artifact vs a real road feature. | **Undecidable (~50/50)** | new framing | Stage localization cannot separate "model invents wobble" from "model follows a real wandering corridor." Correction direction depends on this. |
| PI config / steerRatio / branch era / dirty-state is a causal lever. | **Low, ~10–15%** | ≈ | Golden experiment + only 5 sparse, mixed weak-vs-golden matched contexts (−45% to +35%). |

---

## 3. Artifact Verification (row counts re-checked before any conclusion)

All 27 core artifacts match the handoff's documented shapes exactly. Spot results:

| Artifact | Verified shape | Check |
| --- | ---: | --- |
| `symptom_catalog.csv` | 1557 × 22 | 75 `low_speed_wheel_swing` + 1482 `weave_10_70` = 1557 ✓ |
| `cache/manifest.json` | 147 routes | 145 NPZ present; `route_67`, `route_68` have **no NPZ** (silent failure); `route_8d` had 1 corrupt segment but still produced an NPZ ✓ |
| `drilldown_stage_gain_lag_summary.csv` | 234 × 27 | ✓ |
| `drilldown_low_speed_model_horizon_audit.csv` | 75 × 61 | root_cause buckets 32/26/7/6/4 ✓ |
| `drilldown_low_speed_source_geometry_26row_summary.csv` | 26 × 152 | classifier 5/7/11/2/1; controller 24/2 ✓ |
| `drilldown_low_speed_source_geometry_26row_counterexamples.csv` | 40 × 12 | 18 episodes, 40 bins ✓ |
| `low_speed_visual_inventory.csv` | 6 × 11 | only `route_8d` has local camera (2/6 episodes) ✓ |

Independently re-derived handoff statistics (all reproduced **exactly**):
- 26-row classifier: strong=5, partial=7 (=12 "source"); **`desired_high_model_absent_counterexample`=11 (largest single class)**; y10 ratio>1 = 15/26; y20 ratio>1 = 17/26; controller downstream/attenuated = 24/26; rows with ≥1 counterexample bin = 18/26; total bins = 40.
- 75-row low-speed root-cause buckets: `forward_model_or_corridor_common_mode`=32, `horizon_limited_upstream_lane_or_near_field_desired`=26 (=**58/75 upstream**), `cp_final_missing_controller_stage_unresolved`=7, `low_lane_quality_confounded`=6, `controller_gain_not_excluded`=4.
- Low-speed lead context: **0/75 near-lead** (lead_far=72, no_lead=3) — radar was read; swing is not lead-following.
- Config: low-speed `pi_set` = weak(17)/unknown(58)/**golden(0)**; weave = unknown(949)/weak(495)/**golden(38)**.

---

## 4. Adversarial Cross-Checks (5 run; all support the upstream thesis)

| # | Adversarial question (trying to *refute* upstream-origin) | Verdict | Key number |
| --- | --- | --- | --- |
| 1 | Does upstream-origin survive a lead-context split? | **Supported** | `cp_final < desired` in all 6 lead×subset cells; near-lead attenuates *less* (0.826 vs 0.758) but never amplifies. |
| 2 | Is the weave just the speed confound? | **Supported, with nuance** | Speed real (r≈−0.7/−0.8) but spectral peak fixed ~0.20 Hz **independent of speed**; worst energy is **mid-speed (30–50 mph)**, <30 mph holds only 10.8% of energy. |
| 3 | Does model/lane lead steering at the worst low-speed row (route_6b)? | **Partially supported** | Ladder attenuates (cp_final/desired=0.88); desired/lane lead steering ~1.2–1.5 s; but sign-flips across lookaheads = low SNR; **labeling error found** (see §6). |
| 4 | Is the low-speed curvature metric degenerate (yawRate/v)? | **Refuted (degeneracy)** → thesis supported | No inverse-speed correlation; curvature agrees with steering-deg and model-path-m → real motion. |
| 5 | Are episodes config-confounded (no current-config data)? | **Supported; confound REFUTED for weave** | 38 golden weave rows show same upstream oscillation; confound real only for low-speed (0 golden). |

Notable secondary observation (cross-check 1): the **EMA/smooth stage is the single controller sub-stage that adds band energy** (`ema/predicted` up to 1.13), even though the net controller chain still attenuates vs desired. If any controller-side knob were ever examined, `smooth_tau` is the only one with a mechanistic footprint — but note iter3 `smooth_tau (0.25,0.12)` was already tested on-device and reverted with no benefit (EXPLORER_ST.md).

---

## 5. Ranked Root-Cause Conclusions

### 5.1 Low-Speed Steering-Wheel Swing (1–10 mph)

**Observed symptom:** large left/right wheel motion at single-digit speeds (75 episodes, 37 routes, median 4.55 mph, steering peak-to-peak median 10.2°, p90 14.5°, max 20.5°). Strongest rows: `route_6b@2317/2329` (20.5°), `route_8d@1436` (20.1°), `route_3d@1616` (18.1°).

**Inferred stage:** upstream of the final command. The motion is present in `desired_curvature` (and `act_curvature` ≈ desired) before the controller; `cp_final` is below desired in 60/68 rows. The often-quoted "first supported stage = model_y20" label is partly a **detector artifact** (absolute-correlation + amplitude threshold, and `model_y20` is first in the ladder) — so the *load-bearing* evidence is the magnitude-attenuation ladder, not the stage label.

| Rank | Hypothesis | Confidence | Evidence for | Evidence against / limits |
| ---: | --- | --- | --- | --- |
| 1 | Swing originates upstream of the controller; controller attenuates, does not amplify. | **High (~78%)** | 58/75 upstream buckets; cp_final/desired median 0.875 (60/68 attenuated); 24/26 downstream/attenuated; metric-degeneracy refuted (real motion); route_6b ladder attenuates. | 0 golden-PI coverage (config-confound unresolved by data); 7 CP-missing rows; 4 `controller_gain_not_excluded` routes (94/9c/aa/af, +10–18%, none top-severity). |
| 2 | Immediate carrier is near/short-horizon **lane-center + desired** geometry; forward model horizon collapses at low speed (enabling context). | **Medium (~55%)** | 12/26 strong/partial lane-center source; lane-center y10/y20 amplitude ratio >1 in 15/26 and 17/26; at route_6b lane-center extends 192 m while `model_y20`/`y30` are 16%/0% finite and lead steering. | **Largest single class (11/26) is the desired-high/model-absent counterexample**; median focus `model_y10` finite only 0.46, model/lane corr median 0.13; route_6b sign-flips across lookaheads (low SNR); camera-verified for only route_8d. |
| 3 | Model forward-horizon loss is context/enabler, not the direct amplitude trigger. | Medium-high | Two-mode split: 41/75 forward-horizon (route_8d-like common-mode) vs 34/75 horizon-limited (route_6b-like near-field); high steering occurs with near/short model present. | Long-horizon absence could still alter planner path selection; not testable from logged outputs alone. |
| 4 | PI / final-command tuning is the primary cause. | **Low (~12%)** | — | Controller attenuates desired; symptom present under weak & unknown PI; only 4/75 rows even leave controller gain "not excluded," none top-severity. |
| 5 | Plant/EPAS/road-only first source. | Low | — | Desired/model already move before path/steering; rows are not steering-only. |

**Bottom line (low-speed):** high confidence it is upstream of the controller; **medium** confidence on the specific upstream mechanism. The cleanest physical story: at 1–10 mph the model's forward path horizon collapses (path = speed × time ⇒ ~12 m at 3 mph), the planner derives `desired_curvature` from noisier near-field lane-center geometry, and the actuator faithfully follows it into a visible wheel swing — **but** ~40% of focus bins are counterexamples (high desired ≠ swing), so this is not a single clean signature.

### 5.2 Straight/Gentle-Section Weave (10–70 mph)

**Observed symptom:** slow side-to-side path wander on straights/gentle curves (1482 episodes, 104 routes, median 49.4 mph, path slow-band RMS median 1.97e-4, p95 4.57e-4, max 13.1e-4, spectral peak ~0.20 Hz). Worst rows are lower/mid-speed, constrained corridors (e.g. `route_b5@1038`: 31.6 mph, close lead 0.54 s headway, curb/right-edge).

**Inferred stage:** upstream — model/orientation-rate/desired. The oscillation is full-size at the model/planner stage and is not amplified downstream (final ≤ desired ≈ realized path).

| Rank | Hypothesis | Confidence | Evidence for | Evidence against / limits |
| ---: | --- | --- | --- | --- |
| 1 | Slow-band motion is already present in model/desired and the controller passes/attenuates it (controller is not the source). | **High (~85%)** | 1482/1482 model_or_desired; ladder desired 2.09e-4 → cp_final 1.66e-4 < path 2.20e-4; command/desired 1.000; **golden-PI natural experiment** (38 rows, indistinguishable); survives lead-split, CP/CX1 availability, CAN/calibrated-yaw, narrow/wide band. | command_band_rms may be pinned to desired in the catalog (so it cannot detect a perfect-tracking controller); EMA stage adds some band energy internally (net still attenuates). |
| 2 | Speed is the dominant severity modulator (not an independent root). | High descriptively (~80%) | Spearman(speed, path-weave) −0.80; median path RMS 4.19e-4 (10–20 mph) → 0.97e-4 (60–70 mph). | Speed is also part of detector eligibility (partly definitional); worst *energy* is mid-speed (30–50 mph), not the slowest bin. |
| 3 | The upstream signal cannot yet be separated into model-artifact vs real-road-feature vs fixed-period loop limit-cycle. | **Undecidable** | Stage evidence is correlational; lane-geometry coupling (model↔lane-center corr 0.83 @20 m, 0.93 @30 m) is consistent with *either* a faithful follow of a real corridor *or* a model wobble. Fixed ~0.20 Hz independent of speed hints at a temporal-period (loop) source. | No same-corridor repeat-pass test yet; spectral-peak flatness partly a band-detection artifact. |
| 4 | Near-lead/headway amplifies weave. | Low–medium (co-occurrence), Low (trigger) | 815/1482 near-lead; near-lead path RMS 2.15 vs no-lead 1.25e-4; worst rows near-lead. | Event-aligned lead onset shows **no** upstream growth (−9.45% model_y20); no-lead rows are higher-speed (speed-confounded); near-lead attenuates *less* but never amplifies. |
| 5 | Lane/corridor perception quality contributes to severity. | Medium | Some top rows have low lane-prob (0.42–0.56). | Result survives lane-prob ≥0.5 and ≥0.8 robustness filters. |
| 6 | PI / steerRatio / branch era / dirty-state is primary. | Low (~12%) | — | Golden experiment; only 5 sparse mixed matched contexts (−45% to +35%); symptom spans all configs. |

**Bottom line (weave):** high confidence it originates upstream of the controller and the controller is not the lever. The **open question is whether the model is inventing the wobble or faithfully tracking a real wandering corridor** — that must be resolved offline before any model/path change.

---

## 6. Discrepancies & Corrections Found in Prior Reports (review value-add)

1. **Stale lead claim in the two 2026-06-23 reports.** `root-cause-hypotheses.md` and `drilldown-stage-analysis.md` state lead is "not_extracted / the major missing input." That was fixed in the later `retrolat-v3` rebuild: radar *was* read, and the verified catalog shows low-speed is **0/75 near-lead** (not lead-following). Read those two reports as pre-lead-extraction snapshots.
2. **"first supported stage" is partly circular.** Multiple digests and my own check flag that the `model_or_desired` "first supported family" label can be pulled early by the `model_y20` amplitude threshold. It is corroborating, not load-bearing; the magnitude-attenuation ladder is the robust evidence.
3. **route_6b "counterexample" terminology collision.** The handoff and lane/replay reports call `route_6b@2317/2329` "the counterexample," but the 26-row classifier labels `@2329.1` a **`strong_lane_center_y10_model_source`** (a source row, model_y10 80% finite, model/lane corr 0.99). The genuine `desired_high_model_absent` counterexample on that route is **`@2691.4`**. "Counterexample" was used in two senses (counterexample-to-the-20–30 m-common-mode-framing vs counterexample-class-row). Worth fixing in any future write-up.
4. **`lane-geometry-audit.md` upgrades to "high confidence" on correlation alone**, while its own Next Step concedes it cannot separate perception/corridor motion from a real road feature the model follows. Treat its high-confidence model/path claim as correlational.
5. **"0 golden" is a low-speed-only caveat** (see §4 #5) — not a global one; weave has a usable 38-row golden natural experiment.
6. **Silent data gaps:** `route_67`/`route_68` produced no NPZ and are absent from every downstream table (no error promoted to the catalog). Low-speed `steering_rate_rms_deg_s` is all-zero and `spectral_peak_hz` all-NaN (channels unpopulated), so the steering-rate channel could not corroborate the low-speed metric.

None of these overturn the central conclusion; they tighten its scope.

---

## 7. Correction Recommendation Matrix

Target-layer key: **A** = analysis-only (no driving code). **M** = model selection / model-side path (`modeld`, model bundle, `LAT_SMOOTH_SECONDS`). **P** = lateral planning input (lane-centering target/offset). **C** = controller (PI / EMA `smooth_tau` / rate limits / final Ford command — vehicle-control, do not touch yet).

| # | Correction idea | Target layer | Evidence supporting | Risk | Smallest validation step | Ready for implementation planning? |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Discriminate **model-artifact vs real-road-feature** for the upstream oscillation | **A** | All stage evidence is upstream but cannot separate source; this is the gating question | None (read-only) | Model/lane/path overlay replay on top weave + low-speed rows; **same-GPS-corridor repeat-pass** comparison (does the same road weave the same across passes? consistent ⇒ road/perception-of-road; varying ⇒ model/loop artifact) | **Yes — do this first (analysis)** |
| 2 | Probe the **fixed ~0.20 Hz, speed-independent** period for a loop time-constant source (steerActuatorDelay, `LAT_SMOOTH_SECONDS`, livePose/camera-odometry timing, model cadence) | **A** | Fixed temporal frequency independent of speed is unlike a passive spatial road wavelength | None | Offline spectral + per-stage phase/lag on top rows; check period vs known loop delays | **Yes — analysis** |
| 3 | Try an **alternate driving model** / model-side path smoothing | **M** | Oscillation originates at model output; model swaps live in `/data/media`, survive updates | Medium — changes behavior broadly; could worsen curves | Offline replay of candidate model on logged scenes **before** any drive; then one controlled corridor A/B | **No** — needs #1 result first |
| 4 | Filter / re-target the **lane-centering input** (lane offset feeding the PI) | **P** | Low-speed carrier is near-field lane-center geometry | Medium — centering-vs-weave tradeoff; touches lateral behavior | Offline replay of the proposed input transform on logged rows | **No** — needs #1 result first |
| 5 | Adjust **`smooth_tau` EMA** on `desiredCurvature` | **C** | EMA is the one controller sub-stage that adds band energy | Low-med — adds lag/delay; **iter3 (0.25,0.12) already tested & reverted, no benefit** | Offline replay only | **No** — evidence says controller isn't the source; already tried |
| 6 | Tune **PI gain / int cap / rate limits / final Ford command** | **C** | — (evidence is *against*) | High — could worsen centering and curve behavior; safety-adjacent | None warranted now | **No — explicitly do not change** |

### 7.1 Separated recommendation tiers

**Evidence-backed correction direction:** Direct any future correction effort at the **model / path / perception input side (upstream of the controller)** and **away from PI / final-command tuning**. This direction — not a specific fix — is what the data supports: the attenuation ladder, the golden-PI natural experiment, the lead-split, and the band/yaw robustness all show the controller faithfully passing/attenuating an oscillation that already exists at model/desired.

**Implementation hypotheses (to test offline — NOT to implement):** (a) a model/path-planning artifact removable by a model swap or model-side path smoothing; (b) a lane-centering-input wobble removable by filtering/re-targeting the lane offset; (c) a fixed-period loop limit-cycle from a delay/filter/timing element. These are mutually distinguishable by the offline tests below; do not act on any until one is confirmed.

**Smallest offline / log-analysis validation step (do this next):**
1. Model/lane/path **overlay replay** on the ~8 worst weave rows and ~8 worst low-speed rows: does `model_y*`/path move *with* lane-center and road edges (faithful follow) or *independently* (artifact)?
2. **Same-corridor repeat-pass** test: find GPS cells traversed on ≥2 passes/days (the location-matched context table already has 164 multi-route weave contexts) and compare the weave waveform — repeatable-by-location ⇒ road/perception-of-road; pass-to-pass-variable ⇒ model/loop artifact.
3. Extend the low-speed geometry grid to **per-packet model horizon + 5/10/15 m** (partly built) to settle whether route_6b's near-field mode is common or rare.
4. Probe the **0.20 Hz period** against known loop delays.
All four are analysis-only, keep generated files under the ignored `results/`, and require no driving-code change.

**Smallest on-road validation step (only if offline cannot decide, and only after the above):** one controlled corridor with **PI fixed at golden**, varying exactly **one upstream factor (the driving model bundle)** at a time. Hold branch/steerRatio/tire/load/route constant. Capture CP/CX1 telemetry, camera, lead state, lane quality. Include a **mid-speed straight (30–50 mph, where weave energy peaks)** and a **2–8 mph creep** for low-speed, with explicit no-lead / lead-far / lead-near passes. Any device action must follow the AGENTS.md offroad-safe detached workflow (`IsOffroad=true` gating, `flock`, fast-forward-only target-branch fetch, status file, reboot only on success).

**What should NOT be changed yet (no evidence supports it; risk is real):**
- PI gains (`lc_kp`), integrator cap/decay, P/I terms.
- Rate limits, the EMA `smooth_tau` (already tried heavier and reverted), the final Ford curvature/curvature-rate command path and its sign convention.
- `steerRatio`, `steerActuatorDelay` as a *fix* (it may be a *clue* per #2, but changing it is not yet warranted).
- `panda/ford.h`, `opendbc_repo/` car code, any safety-layer or path-offset/path-angle signal (currently sent as zero; would need a separate safety review).
- Anything justified by lead-following: low-speed is not lead-driven, and weave-vs-lead is co-occurrence, not a trigger.

---

## 8. Final Recommendation

**Do not change driving code yet. Status: ready for an offline model/path-vs-road discrimination analysis plan — NOT ready for a vehicle-control implementation plan.**

Justification: the root-cause *direction* (upstream of the controller) is high-confidence and robust, but the *actionable mechanism* (model artifact vs real road feature vs loop limit-cycle) is undecided, and every candidate fix lives upstream where a wrong guess can degrade centering or curve behavior. Changing controller gains now would be a symptom fix against the evidence.

**Gate to flip to "ready for implementation plan":** the work becomes implementation-ready when **either**
1. the offline overlay/same-corridor replay (Section 7.1) shows a **specific model/path behavior** whose change *removes the oscillation in replay* on the logged scenes (then plan a model/path-side change with a controlled A/B), **or**
2. any analysis shows **first growth at a controller/final-command stage** that contradicts the current upstream evidence (then, and only then, plan a controller change).

Until one of those holds, the next deliverable is analysis, not a driving-code patch.

---

## 9. Commands Run & Verification Results

All commands run from the repo root with `.venv311/bin/python`; all read-only.

```text
# Repo cleanliness (constraints)
git status --short opendbc_repo panda            -> empty (untouched) ✓
git check-ignore retrospective_lateral/results/reports/symptom_catalog.csv -> ignored ✓
git status --short | grep results                -> no results/ files staged ✓

# Artifact row/col counts (27 artifacts) -> all match handoff exactly ✓
symptom_catalog.csv                       1557 x 22  (75 low_speed + 1482 weave) ✓
drilldown_stage_gain_lag_summary.csv       234 x 27 ✓
drilldown_low_speed_model_horizon_audit.csv 75 x 61 ✓
drilldown_low_speed_source_geometry_26row_summary.csv      26 x 152 ✓
drilldown_low_speed_source_geometry_26row_counterexamples.csv 40 x 12 ✓
low_speed_visual_inventory.csv               6 x 11 ✓
cache/manifest.json: 147 routes, 145 NPZ; route_67/route_68 no NPZ; route_8d corrupt-segment-but-NPZ ✓

# Re-derived statistics (all reproduced)
26-row geometry classifier: strong=5 partial=7 counterexample=11 weak=2 near5m=1
26-row controller classifier: downstream/attenuated=24 gain_possible=2
y10 ratio>1=15/26; y20 ratio>1=17/26; rows w/>=1 counterexample bin=18/26; bins=40
75-row root_cause buckets: 32/26/7/6/4  (=58/75 upstream; 4 controller-gain-not-excluded)
cp_final_over_desired (low-speed): median 0.875, attenuated 60/68
stage ladder weave: orientation_rate 2.256e-4, desired 2.089e-4, cp_final 1.655e-4, path 2.195e-4
stage ladder low-speed: desired 10.40e-4 (peak), cp_final 8.41e-4, path 6.29e-4
low-speed lead: 0/75 near-lead (lead_far=72, no_lead=3)
pi_set: low-speed weak=17/unknown=58/golden=0; weave unknown=949/weak=495/golden=38 (route_b8/be/bf, lc_kp=0.0005)

# Adversarial cross-checks (5) — all support upstream-origin / controller-not-source
1 lead-split:      cp_final<desired in all 6 cells; near-lead attenuates less (0.826 vs 0.758), never amplifies
2 speed-spectral:  speed real (r≈-0.8) but peak ~0.20 Hz speed-independent; worst energy mid-speed (30-50 mph)
3 route-6b:        ladder attenuates (0.88); desired/lane lead steering ~1.2-1.5s; @2329.1 is a SOURCE row, @2691.4 is the counterexample
4 lowspeed-metric: degeneracy REFUTED (no 1/v inflation; agrees with steering-deg & model-path-m)
5 config-confound: REFUTED for weave (38 golden rows, command/desired=1.0); real only for low-speed (0 golden)
```

Verification method note: artifact counts and the §3–§4 statistics were re-derived directly with pandas in this session; the 14 prior reports were digested and their claims checked against those re-derived numbers by an independent parallel review pass. No `retrospective_lateral/results/` files were created or modified; `opendbc_repo/` and `panda/` were not touched.
