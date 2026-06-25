# 2026-06-24 Model/Path-vs-Road Discrimination Findings

## Executive Summary
Classified the top 40 weave (`weave_10_70`) and 40 low-speed (`low_speed_wheel_swing`)
episodes — 80 in total — as model-artifact (A), road-feature (B), loop-limit-cycle (C),
or undecided (ambiguous / insufficient_evidence), using only the existing route NPZ caches.
Result counts per symptom (from `discrimination_summary.csv`):

| symptom | model_artifact_A | road_feature_B | loop_limit_cycle_C | ambiguous | insufficient_evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| weave_10_70 | 7 | 5 | 5 | 12 | 11 |
| low_speed_wheel_swing | 7 | 0 | 0 | 24 | 9 |

**Recommendation: do not change driving code yet.** The discriminator is built, passes its
unit suite, and produces real signal on the corpus, but at the current first-pass thresholds
**most episodes are undecided** (weave 23/40 ambiguous+insufficient; low-speed 33/40), and the
road-feature (B) test is **under-powered** — only 9 repeat-pass pairs exist across all 80
episodes, so the corpus cannot yet decisively separate A vs B vs C. These results are
**preliminary and threshold-sensitive**. Per the decision rule below, the next deliverable is
a **follow-up discrimination run targeting repeat-traversed corridors** (to power the B test),
not a vehicle-control change.

## Method
- Common curvature axis via `offset_to_curvature(y, L = 20 m)` (`DISCRIM_LOOKAHEAD_M`): a lateral
  offset at a forward lookahead becomes an implied path curvature `kappa ~= 2*y/L^2`, so the
  vision-model path, perceived lane-center, road-edge center, planner `desired_curvature`, and
  controller `cp_final_command` all live on the same 1/m axis.
- Independent, model- and EPAS-independent realized-path references: CAN yawRate/v, calibrated
  yawRate/v, and GPS-heading-rate/v.
- Three tests per episode:
  1. **Within-window coherence/residual** — band-filtered cross-correlation of model-vs-lane and
     lane-vs-independent-realized motion, plus a model-minus-lane residual ratio
     (`model_residual_over_lane`).
  2. **Cross-pass spatial reproducibility** — band-filtered curvature binned by GPS cell, compared
     pairwise across episodes on *different* routes that share `>= DISCRIM_REPRO_MIN_SHARED_CELLS`
     (= 4) GPS cells; median pairwise Pearson `>= DISCRIM_REPRO_FRACTION_ROAD` (= 0.5) ⇒
     road-reproducible.
  3. **Spectral-peak-vs-speed slope** — OLS of per-window `spectral_peak_hz` on
     `speed_mph_median`; `|slope| < DISCRIM_FREQ_FLAT_HZ_PER_MPH` (= 0.002) ⇒ speed-independent
     (loop-like).
- Classifier priority (`classify_source`): insufficient → road (reproducible AND coherent) →
  artifact (model adds residual AND not moving-together) → loop (flat frequency AND
  straight-clean-lead-free) → ambiguous. Run config: `top_n = 40` per symptom.

## Results
- Window records: **80** (`discrimination_window_records.csv`, 31 cols);
  repeat-pass pairs: **9** (`discrimination_repeat_pass.csv`, 7 cols).
- Frequency-vs-speed slope (`discrimination_frequency_speed.csv`):
  - weave_10_70: **-0.000791 Hz/mph** (n = 17, **flat = True**).
  - low_speed_wheel_swing: **-0.005628 Hz/mph** (n = 16, **flat = False**).
- Per-episode classification (`discrimination_episode_classification.csv`, 80 rows):

  - **Model-artifact (A) — 14 episodes** (7 weave + 7 low-speed). These show the model path
    carrying band-energy beyond the lane/road reference and not moving coherently with the
    independent realized motion. Highest residual ratios: `route_16` (weave, 33 mph,
    residual/lane = 2.79), `route_10@300.75` (low-speed, 2.9 mph, 1.81), `route_stock`
    (weave, 38 mph, 1.62), `route_b1` (weave, 55 mph, 1.48), `route_92` (weave, 47 mph, 1.45).
  - **Road-feature (B) — 5 episodes, weave only** (`route_10`, `route_6a`, `route_7f`,
    `route_9d`, `route_7b`), spanning 28–50 mph. Each is reproducible by location against at
    least one other route's pass (best repro corr 0.86–1.00) *and* coherent across
    planned/perceived/realized. **0 low-speed episodes reach B** — see the under-powered caveat.
  - **Loop-limit-cycle (C) — 5 episodes, weave only** (`route_1c`, `route_60`, `route_8d`;
    two routes appear at two windows each), at 13–21 mph: fixed-frequency oscillation on a
    straight, clean, lead-free road, consistent with the speed-independent weave slope.
  - **Undecided — 56 episodes**: weave ambiguous 12 + insufficient 11 = 23/40; low-speed
    ambiguous 24 + insufficient 9 = 33/40. The discriminator is **conservative**: most episodes
    do not cross any single test's threshold at this first pass.

### Honest interpretation (do not over-read the raw counts)
- **The B (road) test is under-powered.** Only 9 repeat-pass pairs exist across 80 episodes,
  because the selected top-N episodes rarely share `>= 4` GPS cells across *different* routes.
  Single-pass corridors get NaN reproducibility and **cannot** be classified B — this is a lack
  of repeat coverage, **not** evidence against a road cause. The low B count (and the 0 low-speed
  B) must not be read as "weave is mostly artifact/loop."
- **High undecided share ⇒ preliminary.** With 23/40 weave and 33/40 low-speed episodes
  ambiguous or insufficient at the current `DISCRIM_*` thresholds, the A/B/C breakdown is
  **threshold-sensitive** and not a stable population estimate.
- **The flat weave slope is suggestive, not conclusive.** The weave peak sits near ~0.20 Hz and
  is speed-independent (slope ~ -0.0008 Hz/mph, flat = True), which is *consistent* with the
  loop-limit-cycle (C) hypothesis from the prior review's recommendation #2. But it is one
  signal among several and is partly a band-detection artifact (the inspection band biases the
  peak toward its center). Treat it as corroborating, not deciding. The low-speed slope is not
  flat (-0.0056 Hz/mph), so the low-speed swing does not share the fixed-period signature.

## Decision (gate from the review report)
This run is exactly the analysis the prior review (§7.1 step 1–2, §8 gate) demanded before any
implementation plan. Applying that gate:

- If a clear majority classify **B** (road feature) and reproduce by location → correction layer
  is lane-centering target/filtering; plan a model/path change. **Not met:** only 5 weave B,
  0 low-speed B, and the B test is under-powered.
- If a clear majority classify **A** (model artifact) → correction layer is model selection /
  model-side path smoothing; plan an offline replay of a candidate model. **Not met:** only 14/80
  reach A.
- If **C** (loop limit-cycle) dominates on straight clean roads → investigate the fixed-period
  loop element (delay/filter/update cadence), still upstream of PI gains. **Not met:** only 5
  weave C, though the flat weave slope is a supporting clue.
- If **ambiguous dominates** → the analysis is not yet decisive. **This is the current state**
  (56/80 undecided), driven primarily by missing repeat-pass coverage.

**Verdict: the gate does NOT flip.** Status remains *ready for further offline analysis — NOT
ready for a vehicle-control implementation plan*, fully consistent with §8 of the prior review.
The single most valuable next step is a **follow-up discrimination run whose episode selection is
seeded from repeat-traversed corridors** (GPS cells crossed on ≥2 passes/days; the prior review
notes 164 multi-route weave contexts already exist) so the B test has the pairs to either confirm
or rule out a road cause. The smallest on-road controlled drive (PI fixed at golden, vary one
upstream factor) remains justified only if that powered offline run still cannot decide.

## Confounders / limits
- GPS-heading curvature is coarse and noisy; it corroborates the realized-motion references, it
  does not arbitrate alone.
- `offset_to_curvature` is a small-angle (small-curvature arc) approximation.
- Repeat-pass classification needs `>= DISCRIM_REPRO_MIN_SHARED_CELLS` (= 4) shared GPS cells
  between two *different-route* passes; single-pass corridors get NaN reproducibility and cannot
  be classified B. Only 9 such pairs exist in this corpus — the B test is **under-powered**.
- The 0.20 Hz "flat" weave-frequency signal is partly a band-detection artifact; it is suggestive
  of C, not proof.
- Coverage limit: this run uses `top_n = 40` per symptom (80 episodes total). A wider or
  corridor-seeded selection could shift the distribution.
- `route_67` and `route_68` have no NPZ cache and are absent from the corpus (verified: 145
  `route_*.npz` caches present, neither route_67 nor route_68 among them).
- No vehicle-control code, `opendbc_repo/`, or `panda/` was touched; all five generated CSVs are
  gitignored under `retrospective_lateral/results/`.

## Commands Run
All commands run from the repo root with `.venv311/bin/python`; all read-only except the
expected (gitignored) CSV writes.

1. Full retrospective test suite (no regressions):
   ```
   .venv311/bin/python -m pytest retrospective_lateral/tests -q
   → 116 passed in 3.61s  (includes the 16 test_discrimination.py tests)
   ```

2. Regenerate outputs over the real corpus:
   ```
   .venv311/bin/python -m retrospective_lateral.code.discrimination --top-n 40
   → {"classified": 80, "episodes": 80, "repeat_pass_pairs": 9}
   ```

3. Verify CSV row/col counts and the classification breakdown:
   ```
   discrimination_window_records.csv          rows=80  cols=31
   discrimination_repeat_pass.csv             rows=9   cols=7
   discrimination_frequency_speed.csv         rows=2   cols=4
   discrimination_episode_classification.csv  rows=80  cols=8
   discrimination_summary.csv                 rows=8   cols=3

   groupby(symptom, source_label):
     low_speed_wheel_swing  ambiguous              24
                            insufficient_evidence   9
                            model_artifact_A        7
     weave_10_70            ambiguous              12
                            insufficient_evidence  11
                            loop_limit_cycle_C      5
                            model_artifact_A        7
                            road_feature_B          5
   ```

4. Cleanliness / ignore checks:
   ```
   git check-ignore retrospective_lateral/results/reports/discrimination_summary.csv
   → retrospective_lateral/results/reports/discrimination_summary.csv   (ignored)

   git status --short opendbc_repo panda
   → (empty — no changes)

   git status --short retrospective_lateral/results
   → (empty — results stay ignored/unstaged)
   ```
