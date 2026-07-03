# 2026-06-28 Retrospective Lateral — Fresh Full-Analysis Root-Cause Review (v2)

Subject: 2021 Ford Explorer ST sunnypilot lateral driving-performance issues
1. Low-speed steering-wheel swing at 1–10 mph.
2. Straight/gentle-section weave at 10–70 mph.

Method: `superpowers:systematic-debugging` (Phase 1 root-cause investigation → Phase 2 pattern analysis → Phase 3 hypothesis testing). This is an **independent re-review**. No prior conclusion, tuning value, or report verdict was inherited without re-deriving it from the CSV/JSON/NPZ artifacts. The 14 handoff reports plus the prior 2026-06-24 review and the newer Jun-25 artifacts were deep-read by an 11-cluster parallel review pass, each agent **independently reproducing the report's load-bearing numbers from the underlying CSVs**; the load-bearing numbers were *also* re-derived by hand in this session.

Constraints honored: no vehicle-control code touched; `opendbc_repo/` and `panda/` verified clean; no remote comma actions; all `retrospective_lateral/results/` files remain gitignored and unstaged; analysis was read-only Python/pandas over existing outputs.

> **What changed vs the 2026-06-24 review.** That review left the decisive question — is the upstream oscillation a **model artifact (A)**, a **real road feature the model follows (B)**, or a **fixed-period loop limit-cycle (C)** — as *undecidable (~50/50)*, and recommended an offline discrimination analysis. That analysis was then **built and run** (Jun-25 `discrimination_*`, `corridor_repro_*`, `model_vs_loop_signals.csv`, `model_era_weave.csv`). This review folds in that newer evidence. The net effect: the weave's source resolves **toward A (model path-prediction artifact)** and **away from B and C** — but the same evidence shows the obvious fix (swap the driving model) is **not** validated and may not work, because the artifact persists across all three model eras.

---

## 1. Executive Summary

**Both symptoms are real and both originate UPSTREAM of the steering controller.** Across every available stage metric, the band-limited oscillation is already at full size at the model / planner `desiredCurvature` stage and does **not** grow through the controller (EMA → PI → rate-limit → final Ford command); the controller chain **attenuates** it. This reproduces exactly and is the single most robust result in the corpus.

**For the 10–70 mph weave, the newer evidence specifies the upstream source: it is a model PATH-PREDICTION ARTIFACT, not the controller, not a real road feature, and not a closed-loop limit cycle.** The decisive new test is an **open-loop (disengaged) comparison at matched speed**: when the car is being steered by a human and openpilot is *not* in control, the model's *own* predicted `desiredCurvature` / `model_y20` / `model_minus_lane` signals **still weave at ~0.2 Hz, and weave 1.5–1.7× MORE than when engaged**. A closed-loop limit cycle (C) cannot be larger with the loop open → **C refuted**. `model_minus_lane` (the model path's deviation from its *own* perceived lane center) is the largest weaving signal → the model invents motion beyond the lane it perceives → **a faithful follow of a real road (B) is refuted**, and the corpus-level corridor repeat-pass test agrees the weave is **not road-locked** (same-corridor median per-cell corr ≈ 0.10). What remains is **A: the model's predicted path is itself unstable in a slow side-to-side band.**

**The catch (and the reason this is still not an implementation plan): the model artifact persists across every model era.** The open-loop weave is present in CD210, Nevada, **and** OPM7 (dis/eng 1.8–5.3; the `model_minus_lane` residual is *largest* in OPM7). The apparent "CD210 weaves more than OPM7 when engaged" signal is **route-confounded** (different routes per era, not matched corridors). So the historical memory lever "**revert CD210 → OPM7 to kill the weave**" is **not supported** by this matched-speed open-loop evidence and must not be treated as a validated fix.

**For the low-speed (1–10 mph) wheel swing**, the same upstream-of-controller result holds (desiredCurvature is the ladder peak; controller attenuates), and the real growth step is at the **planner** (orientation-rate 7.26 → desired 10.40 ×10⁻⁴ = ×1.43). But the *specific carrier* is **not** a universal near-horizon-geometry signature: ~12/26 of the strongest rows fit "near/short-horizon lane-center + desired geometry," while **11/26 show real 6.5–11.7° swings with the near/short model essentially absent**. Low-speed is also a **different frequency regime** (spectral peak slope vs speed is *not* flat, unlike the weave), consistent with the model's forward horizon collapsing at single-digit speeds (path ≈ speed × time ≈ 12 m at 3 mph). Low-speed has **zero golden-PI coverage** and **2 of its 3 strongest rows are camera-unverified**, so its mechanism is medium-confidence at best.

**Recommendation: DO NOT CHANGE DRIVING CODE YET.** The root-cause *direction* (upstream model path prediction, not the controller) is high-confidence and the A/B/C question is now substantially resolved toward A — a genuine advance — but the actionable *fix* (a model change) is **not** validated and the cross-era evidence actively warns it may not work. Status: **ready for a matched-corridor cross-model offline analysis (and possibly a model-only controlled A/B drive), NOT ready for a vehicle-control implementation plan.** Section 9 gives the gate to flip this.

---

## 2. Confidence Snapshot (this review's calibrated estimates)

| Claim | Confidence | Δ vs 2026-06-24 review | Why |
| --- | ---: | --- | --- |
| Controller (PI / EMA / rate-limit / final Ford command) is **not** the source of either symptom; it attenuates. | **High, ~85%** | ≈ | Stage-attenuation ladder reproduced by ≥8 independent agents + by hand: weave cp_final/desired 0.79–0.91, low-speed 0.86–0.88; only 8% of weave windows and 4–8/75 low-speed rows even leave controller gain "not excluded," none top-severity. |
| 10–70 mph weave originates upstream of the controller. | **High, ~85%** | ≈ | 1482/1482 model_or_desired; ladder + golden-PI natural experiment (38 rows, indistinguishable) + lead-split + band/yaw robustness. |
| The weave is a **model path-prediction artifact (A)**, not a real road feature (B) or loop limit-cycle (C). | **Medium-high, ~70%** | ↑↑ (was "undecidable ~50/50") | Open-loop test: model signals weave 1.5–1.7× MORE disengaged (refutes C); `model_minus_lane` is the largest weaving signal + corridor not road-reproducible (refutes B). Held back by an equivocal episode classifier and a not-formally-null-tested corridor statistic. |
| A model swap (e.g. CD210→OPM7) is a **validated clean fix** for the weave. | **Low, ~20%** | new / **refutes a memory lever** | Open-loop weave persists across CD210/Nevada/OPM7 (dis/eng 1.8–5.3, residual largest in OPM7); engaged era difference is route-confounded, not matched-corridor. |
| Speed is the dominant **severity** modulator (not the generative source). | **High descriptive, ~80%** | ≈ | Spearman(speed, weave) −0.801 reproduced; but weave spectral peak ~0.2 Hz is **speed-independent** (slope −0.0008 Hz/mph) → temporal/model origin, not a spatial road wavelength. |
| Low-speed swing originates upstream of the final command. | **Medium-high, ~75%** | ≈ | desired is ladder peak; controller attenuates 60/68; but the model channel is sparse and actually *lags* steering, support leans on the desired/planner stage. |
| Low-speed immediate carrier is near/short-horizon lane-center + desired geometry. | **Medium, ~50%** | ↓ (slightly) | 12/26 source rows but the **largest single class (11/26) is the desired-high/model-absent counterexample with real swings**; geometry-leads-steering lag is a coin flip (lane 53%, model 15%); within-event speed never controlled. |
| PI config / steerRatio / branch era / dirty-state / lead-follow is a causal lever. | **Low, ~10–15%** | ≈ | Golden experiment; lead is a speed-confounded co-occurrence (dies under route+speed matching, sign-test p=0.885); iter3 smooth_tau tried & reverted. |

---

## 3. Artifact Verification (row counts re-checked before any conclusion)

All **27 core artifacts match the handoff's documented shapes exactly** (verified this session with pandas — see §11). Spot results:

| Artifact | Verified shape | Check |
| --- | ---: | --- |
| `symptom_catalog.csv` | 1557 × 22 | 75 `low_speed_wheel_swing` + 1482 `weave_10_70` ✓ |
| `cache/manifest.json` | 147 routes | catalog/reports built against the 145-NPZ Jun-24 cache ✓ |
| `drilldown_stage_gain_lag_summary.csv` | 234 × 27 | stage ladder reproduced (see §5/§6) ✓ |
| `drilldown_low_speed_model_horizon_audit.csv` | 75 × 61 | root-cause buckets 32/26/7/6/4 ✓ |
| `drilldown_low_speed_source_geometry_26row_summary.csv` | 26 × 152 | geometry classifier 5/7/11/2/1; controller 24/2 ✓ |
| `low_speed_visual_inventory.csv` | 6 × 11 | only `route_8d` has local camera ✓ |

**Cache / catalog drift (new, important).** The working tree has an **in-progress schema-v5 extraction rebuild** (`config.py`: `retrolat-v2`→`retrolat-v5`; `extract.py` adds lead/radar + full model-geometry lookahead y0–y30 + road edges). It rebuilt the cache to **148 NPZ on Jun-25 09:36**, *after* the Jun-24 10:08 catalog/reports. The current cache (e.g. `route_b5.npz`) has the rich 137-channel v5 schema. **Consequence:** the Jun-24 reports describe the *older* leaner cache, and the Jun-25 `discrimination_*` CSVs (07:46) predate even the v5 rebuild (09:36). The discrimination's large "ambiguous/insufficient" fractions partly reflect that leaner cache. This does not invalidate the directional conclusions (the open-loop and corridor tests use channels present before v5), but it means **the discrimination + corridor + open-loop suite should be re-run on the current v5 cache** before any number from them is treated as final.

**Newer artifacts beyond the handoff** (folded into this review): `discrimination_summary.csv`, `discrimination_episode_classification.csv`, `discrimination_repeat_pass.csv`, `discrimination_frequency_speed.csv`, `discrimination_window_records.csv`, `corridor_repro_summary.csv`, `corridor_repro_pairs.csv`, `model_vs_loop_signals.csv`, `model_era_weave.csv`, `route_model_labels.csv`, plus the prior `2026-06-24-...-fresh-full-analysis-root-cause-review.md` and the `2026-06-24-...-model-vs-road-discrimination.md` plan.

---

## 4. Method: how this review was produced

1. Read the required orientation set (AGENTS, EXPLORER_ST, README, both QA notes, design spec, plan, handoff) + the master hypothesis report, the prior fresh review, and the model-vs-road discrimination plan.
2. **Parallel cluster review (11 agents).** Each agent deep-read one report cluster *and re-derived its 2–4 load-bearing numbers from the underlying CSVs*. This is where the corrections in §8 came from. (An adversarial-verifier phase was also launched but was interrupted when the host slept; its function was covered by the reader-level re-derivations plus the by-hand checks below — every cross-cutting hypothesis ended up tested.)
3. **By-hand re-derivation in this session** of the decisive numbers: the full stage-attenuation ladder (both symptoms), Spearman(speed, weave) = −0.801, the 26-row geometry classifier (11 counterexample / 12 source), the discrimination summary, the corridor repeat-pass, the frequency-vs-speed slope, the open-loop `model_vs_loop_signals`, the cross-era `model_era_weave`, and the lane-geometry top-decile reversal.

---

## 5. Ranked Root-Cause: 10–70 mph Straight/Gentle Weave

**Observed symptom:** slow side-to-side path wander on straights/gentle curves (1482 episodes, 104 routes, median 49.4 mph, path slow-band RMS median 1.97×10⁻⁴, p95 4.57, max 13.1; spectral peak ~0.20 Hz). Worst rows are lower/mid-speed constrained corridors (e.g. `route_b5@1038`).

**Inferred stage (reproduced):** the weave is full-size at the model's own output and is attenuated downstream. Verified stage ladder (median band-RMS ×10⁻⁴, "all" subset):

`orientation_rate 2.26 → desired 2.09 → cp_ema 1.66 → cp_pre_rate_limit 1.66 → cp_final_command 1.66 → realized path 2.20` (cx1 variants ~1.9).

The largest stage in the entire chain is `orientation_rate_curvature` (the model's predicted orientation rate). Every controller and final-command stage is **below** desired. The realized path (2.20) slightly exceeds the command, i.e. plant/road adds a little — but **not** the controller.

| Rank | Hypothesis | Confidence | Evidence for | Evidence against / limits |
| ---: | --- | --- | --- | --- |
| 1 | Weave is present at the model/planner output; the controller passes/attenuates it (controller is not the source). | **High (~85%)** | Ladder above; command/desired ≈ 1.0 with cp_final < desired; golden-PI natural experiment (38 rows, indistinguishable); survives lead-split, CP/CX1 availability, CAN/calibrated yaw, narrow/wide band. | `act_curvature` is **aliased to** `desiredCurvature` in the data (not a measured actuator output), so "desired==act ratio=1.000" is a tautology and there is no genuine measured plant stage; the real evidence is `cp_final`/`cx1_command` (0.79–0.91) + realized path, which still hold. |
| 2 | The upstream weave is a **model path-prediction artifact (A)** — not a real road feature (B) or a closed-loop limit-cycle (C). | **Medium-high (~70%)** | **Open-loop test**: model-native signals weave 1.5–1.7× MORE disengaged than engaged at matched speed (desired 1.67, model_y20 1.56, model_minus_lane 1.53) → refutes C. `model_minus_lane` is the largest weaving signal (model deviates from its own perceived lane) → refutes B. Corridor repeat-pass: same-corridor median per-cell corr ≈ 0.095 (2/68 reproducible) → not road-locked. Spectral peak ~0.2 Hz **speed-independent** → temporal/model, not a road wavelength. | The per-episode A/B/C classifier is **equivocal** (weave A=7 ties B=7, ambiguous+insufficient=21 dominate) — no "clear majority" per the plan's own gate. The corridor `corr_corrected` is a raw correlation (sign-flip only), **not** null/permutation-tested. A weaker `discrimination_repeat_pass` test (16/23 reproducible @0.68) superficially contradicts the corridor test (the two are unreconciled; the corridor test is the more principled/higher-n one). Discrimination ran on the pre-v5 cache. |
| 3 | Speed is the dominant **severity** modulator (not the generative root). | High descriptive (~80%) | Spearman −0.801; path RMS 4.19→0.97 ×10⁻⁴ from 10–20 to 60–70 mph. | Speed is partly definitional (eligibility); spectral peak is speed-*independent*, so speed scales the symptom but doesn't generate the oscillation. |
| 4 | The weave is the model **faithfully following a real wandering lane/corridor** (B), implicating lane perception. | **Low–medium (~25%)** | `lane-geometry-audit` reports strong model↔lane-center coupling at 20–30 m. | **Overclaimed (see §8).** The high 20–30 m coupling is a lookahead/geometry artifact present equally in *calm* episodes; at the discriminating 10 m lookahead the **most severe weave is the LEAST lane-coupled** (top-decile model↔lane corr 0.33 vs all-rows 0.58, reproduced by hand), lane-center *lags* steering (+0.65 s), and ~18% of severe rows are geometry-calm. |
| 5 | A model **swap** (CD210→OPM7/Nevada) removes the weave. | **Low (~20%)** | Engaged model_y20 weave is descriptively lower for OPM7 (2.65) and Nevada (2.74) than CD210 (3.74). | The open-loop artifact **persists in all three eras** (dis/eng 1.8–5.3; residual *largest* in OPM7). The engaged era gap is **route-confounded** (CD210=14 routes, Nevada=19, OPM7=8 — different roads, not matched corridors). Not a validated lever. |
| 6 | PI / steerRatio / branch era / dirty-state / lead-follow is primary. | **Low (~12%)** | — | Golden experiment; lead is a speed-confounded co-occurrence (dies under route+speed matching); symptom spans all configs. |

**Bottom line (weave):** High confidence it originates upstream of the controller; **medium-high** confidence the upstream source is a model path-prediction artifact (A) rather than B or C — a real advance over "undecidable" — but **the model-swap fix is unproven and the cross-era data argues against it.** The single best next test is a **matched-corridor cross-model open-loop comparison**.

---

## 6. Ranked Root-Cause: Low-Speed Wheel Swing (1–10 mph)

**Observed symptom:** large left/right wheel motion at single-digit speeds (75 episodes, 37 routes, median 4.55 mph, steering peak-to-peak median 10.2°, p90 14.5°, max 20.5°). Strongest rows: `route_6b@2317/2329` (20.5°), `route_8d@1436` (20.1°), `route_3d@1616` (18.1°).

**Inferred stage (reproduced):** upstream of the final command. Verified ladder (median band-RMS ×10⁻⁴): `orientation_rate 7.26 → desired 10.40 (peak) → cp_ema 8.37 → cp_final 8.41 → realized path 6.29`. The controller attenuates (cp_final/desired ≈ 0.86–0.88, attenuating in 60/68 rows). **The real growth step is at the planner**: orientation-rate 7.26 → desired 10.40 = ×1.43.

| Rank | Hypothesis | Confidence | Evidence for | Evidence against / limits |
| ---: | --- | --- | --- | --- |
| 1 | Swing originates upstream of the controller; controller attenuates. | **High (~75%)** | 58/75 upstream buckets; desired is ladder peak; cp_final/desired attenuating 60/68; metric-degeneracy refuted (real path/wheel motion, no 1/v inflation). | 0 golden-PI low-speed coverage; 7 CP-missing rows; the model channel is *sparse and actually lags steering* (−0.8 s), so the "upstream" support leans on the **desired/planner** stage, not model_y20. |
| 2 | Immediate carrier is near/short-horizon **lane-center + desired** geometry; forward model horizon collapses at low speed (enabling context). | **Medium (~50%)** | 12/26 strong/partial lane-center source rows; route_8d (camera-verified) shows constrained-corridor geometry; at 3 mph path≈12 m so the planner derives desired from noisy near-field geometry. | **The largest single class (11/26) is the desired-high/model-absent counterexample with REAL 6.5–11.7° swings** (5 rows have y10 model finite=0.0); geometry-leads-steering lag is a coin flip (lane 53%, model 15%); within-event speed never controlled; 2/3 strongest rows camera-unverified. |
| 3 | Two physical subcases, not one signature. | Medium-high | `route_8d` = 20–30 m model+lane common-mode (camera-verified constrained corridor); `route_6b` = lane-center/near-field-path driven with a truncated forward horizon (model rarely reaches 20–30 m). | Both are correlational; route_6b's own per-lookahead signals self-contradict in sign and lag at 3 mph. |
| 4 | Low-speed shares the weave's fixed-period model artifact. | **Low–medium (~30%)** | Same upstream-of-controller structure; same model pipeline. | Low-speed spectral peak is **NOT** speed-independent (slope −0.0056 vs weave −0.0008) → a *different* regime, more consistent with the collapsing model horizon than with the weave's fixed ~0.2 Hz wobble. The open-loop/cross-era/corridor tests were run on **weave**, not low-speed; low-speed A/B/C is 7 A, **zero B, zero C, 33/40 undecided**. |
| 5 | PI/final-command or plant/EPAS is the primary low-speed cause. | **Low (~12%)** | — | Controller attenuates; rows are not steering-only; symptom under weak & unknown PI. |

**Bottom line (low-speed):** High confidence it is upstream of the controller, with the growth located at the **planner/desired** stage. **Medium** confidence on the specific carrier (near/short-horizon geometry) because ~40% of focus bins are counterexamples and the lag evidence is mixed. Low-speed is **less resolved than the weave** (no golden coverage, mostly camera-unverified, A/B/C overwhelmingly undecided) and is a distinct frequency regime — it should be treated as a **separate investigation**, not assumed to be the same model artifact.

---

## 7. Cross-Cutting Hypothesis Tests (adversarial)

Each was attacked with the intent to refute, using CSV re-derivation (reader clusters + by-hand). Verdicts:

| # | Hypothesis (attacked) | Verdict | Decisive number |
| --- | --- | --- | --- |
| H1 | Controller/PI/final-command is NOT the source. | **Survives** | cp_final/desired 0.79–0.91 (weave), 0.86–0.88 (low-speed); orientation_rate is the largest stage; only ≤8% of weave windows amplify. |
| H2 | Weave originates upstream in model/path geometry. | **Survives, refined** | Upstream YES (ladder), but **not** as "lane perception the model follows": top-decile severe weave is *less* lane-coupled (0.33 vs 0.58); the upstream source is the model's *predicted path* (open-loop test), not the perceived lane. |
| H3 | Low-speed carrier = near/short-horizon geometry. | **Partial** | Holds for ~12/26 strongest rows; **11/26 swing with the near model absent** → not universal. |
| H4 | Speed is the real lever / the upstream finding is a speed artifact. | **Refuted as "the lever"; speed is a severity modulator** | Spearman −0.80 (real) but spectral peak ~0.2 Hz **speed-independent**; worst *energy* is mid-speed (30–50 mph), not the slowest bin. |
| H5 | Newer discrimination proves a clean model-artifact verdict. | **Partial / over-read if taken alone** | Open-loop + corridor + cross-era align on A and refute B/C; **but** the per-episode classifier is equivocal (no majority) and the corridor statistic isn't null-tested → "medium-high," not "proven." |
| H6 | Near-lead/headway triggers the weave. | **Refuted as trigger** | Pooled lead_near>no_lead gap is a speed confound (no_lead skews ~10 mph faster); dies under route+speed matching (sign-test p=0.885; stratified perm p=0.28). Co-occurrence only. |

---

## 8. Corrections to Prior Reports (review value-add)

1. **`act_curvature` is aliased to `desiredCurvature`** in the catalog/stage data (identical to 6 sig figs in 1557/1557 episodes). Any "desired==act, so plant is ruled out / motion is present at the actuator" framing is a **tautology**. The conclusion survives on the genuinely independent `cp_final_command` / `cx1_command_curvature` stages (which attenuate), but reports should stop citing the act/desired identity as evidence.
2. **`lane-geometry-audit.md` (report #5) is overclaimed.** Its load-bearing top-decile coupling numbers (0.828/0.934 @20/30 m, 87.9% |corr|≥0.8) **do not reproduce** (≈0.67/0.88, 65%); the high 20–30 m coupling is a lookahead-geometry artifact present equally in calm episodes, and **severe weave is *less* lane-coupled** (top-decile model↔lane corr 0.33 vs 0.58 all-rows, reproduced by hand). Its printed top-decile threshold also yields the wrong n. Treat its "high-confidence upstream lane-geometry" claim as **not supported**; it actually points *toward* a model artifact, not a faithful lane-follow.
3. **The prior 2026-06-24 review's "A-vs-C undecidable" is now outdated** by the Jun-25 open-loop (`model_vs_loop_signals`) and corridor (`corridor_repro`) tests, which refute C and B and support A for the weave.
4. **The memory lever "revert CD210→OPM7 to kill the weave" is contradicted** by `model_era_weave.csv`: the open-loop model artifact persists across CD210/Nevada/OPM7 and the engaged era difference is route-confounded. (Memory entry `finding_slow_weave_cd210` dates to 2026-06-07 and predates this matched-speed open-loop evidence.)
5. **The two 2026-06-23 reports describe a pre-lead-extraction snapshot** ("lead not extracted"); radar was later read and low-speed is **0/75 near-lead** (not lead-following).
6. **`stage_first_growth = model_or_desired` (1482/1482) is partly circular** (the `model_y20` amplitude threshold can pull the label early). The load-bearing evidence is the magnitude-attenuation ladder, not the label.
7. **The discrimination ships two contradictory repeat-pass tests** (`corridor_repro` "not road-locked" vs `discrimination_repeat_pass` "16/23 reproducible") without reconciling them; the corridor test is higher-n and more principled, but neither is null/permutation-tested.
8. **Silent data gaps:** `route_67`/`route_68` produced no NPZ and are absent from every downstream table; low-speed `steering_rate_rms_deg_s` is all-zero and `spectral_peak_hz` all-NaN (channels unpopulated), so steering-rate could not corroborate the low-speed metric.

None of these overturn the central conclusion (upstream-of-controller, model-side); items 2–4 *tighten* it and remove an unsupported fix.

---

## 9. Correction Recommendation Matrix

Target-layer key: **A** = analysis-only (no driving code). **M** = model selection / model-side path (`modeld`, model bundle, `LAT_SMOOTH_SECONDS`). **P** = lateral planning input (lane-centering target/offset). **C** = controller (PI / EMA `smooth_tau` / rate limits / final Ford command — vehicle-control, **do not touch**).

| # | Correction idea | Layer | Evidence supporting | Risk | Smallest validation step | Ready for implementation planning? |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | **Matched-corridor cross-model open-loop weave comparison** (CD210 vs OPM7 vs Nevada at the *same* GPS cells + speed bins, disengaged). | **A** | Open-loop artifact is real but era difference is route-confounded; this is the gating question for any model change. | None (read-only) | Re-run `model_era_weave` logic restricted to shared GPS cells + matched speed; report per-model model_y20/model_minus_lane band-RMS. | **Yes — do this first.** |
| 2 | **Re-run the discrimination + corridor + open-loop suite on the current v5 cache** (richer channels, lead-gated, calibrated yaw, road edges). | **A** | Jun-25 outputs predate the v5 rebuild; ambiguous/insufficient fractions may shrink; corridor stat needs a null/permutation reference. | None | `discrimination.py` + corridor + open-loop over v5 cache; add a permutation null to `corr_corrected`. | **Yes — do this first.** |
| 3 | **Model-side path smoothing / alternate model bundle** for the weave. | **M** | Weave originates at the model's predicted path (open-loop). | **Medium-high** — broad behavior change; could worsen curves; **artifact persists across eras so may not help**. | Offline replay of a candidate model/smoothing on the worst logged scenes **before** any drive; then a matched controlled A/B. | **No** — gated on #1 result. |
| 4 | **Filter / re-target the lane-centering input** for the low-speed near-field carrier. | **P** | Low-speed growth at the planner from noisy near-field geometry. | Medium — centering-vs-swing tradeoff. | Offline replay of the input transform on the low-speed rows; resolve the 11/26 model-absent counterexamples first. | **No** — low-speed mechanism only medium-confidence. |
| 5 | Adjust **`smooth_tau` EMA**. | **C** | EMA is the one controller sub-stage that adds a little band energy (×1.08–1.09). | Low-med — adds lag; **iter3 (0.25,0.12) already tested on-device and reverted, no benefit**. | Offline replay only. | **No** — evidence says controller isn't the source; already tried. |
| 6 | Tune **PI gain / int cap / rate limits / final Ford command / steerRatio**. | **C** | — (evidence is *against*). | High — degrades centering/curves; safety-adjacent. | None warranted now. | **No — explicitly do not change.** |

### 9.1 Separated recommendation tiers

**Evidence-backed correction direction:** Aim any future correction at the **driving model's predicted path (model layer, upstream of the controller)** and **away from PI / final-command / `smooth_tau` tuning**. This is supported by the attenuation ladder, the golden-PI natural experiment, the lead-split, *and now* the open-loop test (the model weaves even when not steering the car) and the corridor not-road-locked test.

**Implementation hypotheses (test offline — do NOT implement):** (a) a model path-prediction instability removable by a *specific* model bundle or model-side path smoothing — **but only if a matched-corridor cross-model test shows a model that is materially quieter at matched location** (current cross-era data says the artifact is everywhere); (b) a low-speed near-field lane-centering-input wobble removable by filtering/re-targeting the lane offset. These are mutually distinguishable offline; do not act on either until #1/#2 confirm.

**Smallest offline / log-analysis validation step (do next):** Recommendations #1 and #2 above — a **matched-corridor cross-model open-loop comparison** plus a **v5-cache re-run with a permutation null on the corridor statistic**. Both are analysis-only, write under the ignored `results/`, and require no driving-code change. The decisive output is: *is there a model bundle whose predicted-path weave is materially lower than CD210 at the same physical corridors and matched speed?* If yes → a model change is justified; if no → the artifact is intrinsic to this model family and a model swap is not the fix.

**Smallest on-road validation step (only if offline cannot decide, and only after #1/#2):** one controlled corridor, **PI fixed at golden**, varying exactly **one upstream factor — the driving model bundle** — at a time; hold branch/steerRatio/tire/load/route constant; capture CP/CX1 telemetry, camera, lead state, lane quality; include a **mid-speed straight (30–50 mph, where weave energy peaks)** and a **2–8 mph creep** for low-speed, with explicit no-lead / lead-far / lead-near passes; reset learned params each pass; ≥5 passes/config (8–10 target). Any device action must follow the `AGENTS.md` offroad-safe detached workflow (`IsOffroad=true` gating, `flock`, fast-forward-only target-branch fetch, status file, reboot only on success).

**What should NOT be changed yet (no evidence supports it; risk is real):**
- PI gains (`lc_kp`), integrator cap/decay, P/I terms; rate limits; the EMA `smooth_tau` (already tried heavier and reverted); the final Ford curvature/curvature-rate command path and its sign convention.
- `steerRatio`, `steerActuatorDelay` as a *fix*.
- `panda/ford.h`, `opendbc_repo/` car code, any safety-layer or path-offset/path-angle signal (currently sent as zero — would need a separate safety review).
- **A blind CD210→OPM7 model swap as a "fix"** — the open-loop + cross-era evidence says it is not validated and may not work; it must pass the matched-corridor test (#1) first.
- Anything justified by lead-following: low-speed is not lead-driven; weave-vs-lead is co-occurrence, not a trigger.

---

## 10. Final Recommendation

**Do not change driving code yet. Status: ready for a matched-corridor cross-model offline analysis (and possibly a model-only controlled A/B drive) — NOT ready for a vehicle-control implementation plan.**

Justification: the root-cause *direction* (upstream model path prediction, not the controller) is high-confidence and robust, and the A/B/C question is now substantially resolved toward **A (model artifact)** for the weave — a real advance over the prior "undecidable." But the *actionable fix* is a model-side change, and the only cross-model evidence available (route-confounded, cross-era) says the artifact is present in every model era and may not be removed by a swap. Changing controller gains now would be a symptom fix against the evidence; swapping the model now would be an unvalidated guess against the cross-era evidence.

**Gate to flip to "ready for implementation plan":** when **either**
1. the **matched-corridor cross-model open-loop comparison (§9 #1)** identifies a specific model bundle with materially lower predicted-path weave **at the same physical corridors and matched speed** (then plan a model-side change with a controlled A/B), **or**
2. an analysis (e.g. v5-cache re-run, §9 #2) shows **first growth at a controller/final-command stage** contradicting the current upstream evidence (then, and only then, plan a controller change).

Until one holds, the next deliverable is analysis, not a driving-code patch.

---

## 11. Confounders & Missing Evidence

- **Cache/catalog drift:** Jun-24 reports describe the pre-v5 cache; Jun-25 discrimination predates the v5 rebuild. Re-run on v5 before finalizing any discrimination number.
- **`act_curvature` aliased to `desiredCurvature`** → no genuine measured-actuator stage exists; the "no plant amplification" inference rests on realized path vs final command only.
- **Lead-follow** co-occurs with most weave rows and the upstream signal scales with lead presence, but the effect is a **speed confound** that dies under route+speed matching (so it is context, not cause) — still, low-speed lead gating is powerless (3 no_lead rows).
- **Speed** is both a strong severity modulator and partly definitional in detector eligibility; always control it.
- **Config/era coverage:** 104/147 routes unknown PI; golden-PI has 38 weave rows (3 routes) and **0 low-speed**; model labels recovered for a minority (`route_model_labels.csv` is mostly `unknown`); historical groups are not location-matched.
- **Camera:** only `route_8d` has local frames; 2 of the 3 strongest low-speed rows are camera-unverified and the routes are gone from the device.
- **Discrimination statistics:** episode classifier dominated by ambiguous/insufficient (no majority); `corridor_repro` corr is not null/permutation-tested; the two repeat-pass tests disagree.
- **Open-loop test caveats:** when engaged, openpilot drives a smoother scene, so part of the engaged<disengaged gap could be input-coupling rather than pure artifact; `model_minus_lane` could also reflect the model anticipating curvature the lane-line-only center misses. The conclusion is robust at matched-speed straight/gentle gates but would be stronger re-run on v5 with a within-corridor control.
- **Low-speed mechanism** is a distinct frequency regime, only medium-confidence, and should be a separate track from the weave.

---

## 12. Commands Run & Verification Results (read-only)

All from repo root with `.venv311/bin/python`.

```text
# Constraints
git status --short opendbc_repo panda                 -> empty (untouched) ✓
git status --short retrospective_lateral/results      -> empty (results ignored/unstaged) ✓
git check-ignore retrospective_lateral/results/reports/symptom_catalog.csv -> ignored ✓

# Artifact row/col counts (27 core artifacts) -> ALL match handoff exactly ✓
symptom_catalog.csv 1557x22 (75 low_speed + 1482 weave); manifest.json 147 routes;
drilldown_stage_gain_lag_summary 234x27; low_speed_model_horizon_audit 75x61;
source_geometry_26row_summary 26x152; low_speed_visual_inventory 6x11; (+21 more) ✓

# Stage-attenuation ladder (median band-RMS x1e-4, "all" subset)  [by hand]
weave:     orientation_rate 2.256 > desired 2.089 > cp_ema 1.663 = cp_final 1.655 ; realized path 2.195
low-speed: orientation_rate 7.258 < desired 10.397 (peak) > cp_final 8.413 ; realized path 6.287
=> controller attenuates both; weave energy peaks at the model's own orientation-rate output.

# Speed & geometry  [by hand]
Spearman(speed, weave path-RMS) = -0.801 (n=1482); bins 4.19 (10-20) -> 0.97 (60-70) x1e-4
26-row geometry classifier: counterexample=11, partial=7, strong=5, weak=2, near5m=1; controller 24 downstream / 2 gain-possible
lane-geometry top-decile model<->lane corr 0.331 vs all-rows 0.584 (severe weave LESS coupled)

# Newer Jun-25 discrimination  [by hand]
discrimination_summary: weave A=7/B=7/C=5/ambiguous=10/insufficient=11 ; low-speed A=7/ambiguous=24/insufficient=9 (0 B, 0 C)
discrimination_frequency_speed: weave slope -0.0008 Hz/mph (flat=True) ; low-speed -0.0056 (flat=False)
corridor_repro_summary: same-dir 68 pairs, 2 reproducible, median corr 0.095, p90 0.307, max 0.605 ; opposite 59 pairs, 0 reproducible
model_vs_loop_signals (matched 12 speed bins): dis/eng = desired 1.673, model_y20 1.557, model_minus_lane 1.531, orientation_rate 1.351 (all >1 => model weaves MORE open-loop)
model_era_weave: dis/eng persists CD210 (1.80-2.61) / Nevada (1.80-3.91) / OPM7 (2.48-5.33); engaged model_y20 CD210 3.74 > OPM7 2.65 (route-confounded, not matched)

# Cache schema
route_b5.npz: 137 channels incl road_edge_*_y20, lane_center_y20/y30, model_y20/y30, yaw_rate_calibrated (v5);
cache rebuilt Jun-25 09:36 AFTER Jun-24 catalog and AFTER Jun-25 07:46 discrimination outputs.

# Parallel review: 11 reader clusters each re-derived report headline stats from CSVs
# (adversarial-verifier phase interrupted by host sleep; coverage replaced by reader re-derivations + by-hand checks above)
```

Verification method note: all §3/§5/§6/§8 numbers were re-derived directly with pandas this session; the 14 handoff reports + prior review + newer artifacts were digested and their claims checked against those re-derived numbers. No `retrospective_lateral/results/` files were created or modified; `opendbc_repo/` and `panda/` were not touched.
