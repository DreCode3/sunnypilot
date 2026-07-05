# 2026-06-25 Weave Root-Cause Synthesis — It's a Model/Path-Prediction Artifact

Subject: 2021 Ford Explorer ST sunnypilot, **10–70 mph straight/gentle-section weave**.

This is the capstone synthesis for the weave investigation. It consolidates the retrospective corpus analysis, the fresh Tuesday (2026-06-23) golden-PI repeat-corridor drive, and the two discriminating tests (road-locking and open-loop) into one root-cause conclusion and a correction recommendation matrix.

Analysis-only. No vehicle-control code, `opendbc_repo/`, or `panda/` was modified. The device was touched only for a read-only, offroad log pull. All generated outputs are gitignored under `retrospective_lateral/results/`.

Prior reports in this chain:
- [2026-06-24 fresh full-analysis root-cause review](2026-06-24-retrospective-lateral-fresh-full-analysis-root-cause-review.md) — controller exonerated; upstream origin; A/B/C undecided.
- [2026-06-24 model-vs-road discrimination](2026-06-24-retrospective-lateral-model-vs-road-discrimination.md) — built the discriminator; B test under-powered.
- [2026-06-25 weave not road-locked](2026-06-25-retrospective-lateral-weave-not-road-locked.md) — B refuted with the Tuesday corridors.

## Executive Summary

**The 10–70 mph weave is the driving model's own path-prediction jitter — an intrinsic ~0.20 Hz wobble in `modelV2`'s predicted path and the planner's `desiredCurvature`. The controller is not the source (it attenuates and passes the wobble through), it is not a fixed roadway feature (the weave does not recur by location), and it is not a closed-loop limit-cycle (it persists, and grows, with the loop open).**

The correction lever is therefore the **driving model / model-side path processing**, upstream of the PI controller — not controller tuning, and not avoidable by routing. The next step is an **offline model-replay validation** (re-run candidate model bundles on the logged scenes and measure whether the open-loop model-output weave drops) **before** any on-road change.

**Readiness: ready for an offline model-replay validation plan. NOT yet ready for a driving-code change** — a model swap should be validated by replay first, and controller tuning remains off the table.

## The Evidence Chain (closed)

| Candidate layer | Verdict | Decisive evidence |
| --- | --- | --- |
| Controller (custom PI / EMA / rate-limit / final Ford command) | **Not the source** | Stage band-RMS ladder: `desired` 2.09e-4 → `cp_final` 1.66e-4 (attenuated) → realized path 2.20e-4; golden-PI natural experiment (38 rows) indistinguishable from weak/unknown; survives lead/speed/yaw/band robustness. |
| **B — real roadway feature the model follows** | **Refuted** | On 68 same-direction golden-PI repeat corridors (Tuesday × corpus), the **road shape reproduces by location at 0.76 median (to 0.99)** while the **weave reproduces at 0.095 median, 3% above threshold**. Positive control validates the method; speed-matching does not rescue it. |
| **C — fixed-period closed-loop limit-cycle** | **Refuted** | Open-loop test: the model's path/desired weaves **as much or more disengaged than engaged** (dis/eng 1.56 pooled across model-native signals; 6/8 golden routes show disengaged ≥ engaged; direction robust to a transition-erosion control). A closed-loop resonance must collapse when the loop opens; it does not. |
| **A — model/path-prediction artifact** | **Supported** | By elimination + direct signature: the wobble is present (and larger) in the raw model output open-loop; the engaged controller attenuates it. Fixed ~0.20 Hz peak independent of speed. |

## How Each Verdict Was Reached

**Controller exonerated (prior review).** Band-limited oscillation is full size at the model/planner stage and is not amplified across the EMA/PI/rate-limit/final-command stages — on the median it is attenuated (`cp_final/desired` 0.875 low-speed; final < desired weave). A golden-PI natural experiment (current config) shows the same oscillation as weak/unknown configs, refuting a PI-config cause.

**B refuted (road not the trigger).** Tuesday's out-and-back provided golden-PI coverage of corridors that 68 prior corpus drives also traverse same-direction (sharing up to 1,500 fine cells). Binning the weave-band lane curvature onto fine ~15 m cells (vs the original 80 m, which washed out the ~110 m weave) and correlating per cell across passes: the road's own shape reproduces strongly (median 0.76), but the weave does not (median 0.095). The weave does not happen at the same places. (`corridor_repro.py`; positive control in `scratchpad/posctrl.py`.)

**C refuted, A supported (open-loop test).** When lateral control is disengaged, the model still produces a predicted path and `desiredCurvature`, but openpilot is not actuating — the loop is open. On straight, speed-matched, golden-PI windows, the model-output weave-band RMS is **larger disengaged than engaged** for every model-native signal:

| model-output signal | engaged 1e-4 | disengaged 1e-4 | dis/eng |
| --- | ---: | ---: | ---: |
| desired_curvature | 3.59 | 6.01 | 1.67 |
| model_y20 | 3.49 | 5.44 | 1.56 |
| model_minus_lane | 5.09 | 7.79 | 1.53 |

A closed-loop limit-cycle cannot grow when you open the loop, so C is refuted. The pooled model-native dis/eng median is 1.56; per route, 6 of 8 golden routes show disengaged ≥ engaged (route_46 and route_bf are the exceptions; route_45 is dropped for insufficient disengaged coverage, hence 8 of 9). The *direction* (disengaged ≥ engaged) is robust to a transition-erosion control, though the exact per-route median is convention-dependent (≈1.3 with a symmetric both-state 2 s guard, ≈1.6–1.7 otherwise). The wobble is intrinsic to the model's output; the engaged controller attenuates it (engaged < disengaged), consistent with the controller-exoneration finding. (`model_vs_loop.py`.)

## Correction Recommendation Matrix

Target-layer key: **A** = analysis-only. **M** = driving model / model-side path (`modeld`, model bundle, `LAT_SMOOTH_SECONDS`). **C** = controller (PI / `smooth_tau` / rate limits / final command — do not touch).

| # | Correction idea | Layer | Evidence supporting | Risk | Smallest validation | Ready? |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | **Offline model-replay**: re-run candidate model bundles on logged scenes; measure open-loop model-output weave-band | **A** | Root cause localized to the model's path prediction | None (offline) | This is itself the validation harness; build it next | **Yes — do first** |
| 2 | **Swap the driving model** to one with a quieter path prediction | **M** | Weave is intrinsic to the current model's output | Medium — broad behavior change; could affect curves | Confirm via #1 replay, then one controlled corridor A/B (PI held golden) | No — needs #1 |
| 3 | Heavier **model-side path smoothing** (`LAT_SMOOTH_SECONDS`, currently 0.1) to damp the 0.2 Hz jitter | **M** | The jitter is a temporal ~0.2 Hz oscillation in the model output | Med — adds lag/delay; could blunt curve response | Offline replay / pipeline-sim sweep of the smoothing constant | No — needs offline sweep |
| 4 | Tune PI gain / int cap / rate limits / `smooth_tau` / final command | **C** | — (evidence is against) | High — wrong layer; controller already attenuates; `smooth_tau` heavier already tried & reverted | None warranted | **No — do not change** |

### Separated tiers

**Evidence-backed correction direction:** the **driving model / model-side path**, upstream of the controller. The controller is correctly attenuating an oscillation it did not create; tuning it is the wrong layer.

**Implementation hypotheses (test offline before implementing):** (a) a quieter driving-model bundle removes the intrinsic path wobble; (b) heavier model-side path smoothing damps the 0.2 Hz jitter at acceptable lag cost. Neither is yet validated.

**Smallest offline validation (next deliverable):** a model-replay harness that runs candidate model bundles on the logged camera/scene data and recomputes the **open-loop model-output weave-band** (the metric this investigation used). A candidate that lowers it is a real fix; one that doesn't is rejected without ever touching the car. The original design parked GPU/model reruns as the step to take "if the logs point to model-perception ambiguity" — they now decisively do.

**Smallest on-road validation (only after offline):** model-swap A/B on a repeat corridor, **PI held at golden**, current vs candidate model, with model output + camera + lane quality logged. Any device action follows the AGENTS.md offroad-safe detached workflow.

**What should NOT be changed yet:** PI gains (`lc_kp`), integrator cap/decay, rate limits, `smooth_tau` (heavier already tried/reverted), the final Ford curvature command path, `steerRatio`, `panda/ford.h`, `opendbc_repo/` car code, safety layers.

## Scope and Caveats

- This conclusion is **weave-specific (10–70 mph)**. The low-speed (1–10 mph) wheel swing is a separate mechanism (collapsed forward model horizon feeding near-field lane geometry into desired curvature) and was not retested here.
- The open-loop test's disengaged condition is human-driven; the **direction** of the result (model weaves ≥ open-loop) refutes C regardless of human-driving dynamics, and the per-cell `model_minus_lane` weave (model path vs perceived lane) largely removes the lane-tracking component (though it does not fully strip human-driving dynamics). The direction is robust to a transition-erosion control; the exact ratio is convention-dependent (≈1.3–1.7).
- The road positive control is a low-pass (`<0.035 Hz`) signal, so its high spatial reproducibility (0.76) is partly inherent smoothness, not only road-locking — its role is to prove the GPS-alignment/binning/correlation method *can* detect a location-locked signal. The load-bearing comparison is that the weave does not reproduce **in its own 0.10–0.35 Hz band** (0.095).
- The fixed ~0.20 Hz peak is partly a band-detection artifact; it corroborates but is not load-bearing.
- The 2/68 marginally-reproducible corridors and the per-route exceptions (route_46, route_bf with dis/eng < 1) are within expected spread and do not overturn the central tendencies.

## Artifacts, Commands, Commits

- New analysis modules (committed, pushed to `origin`): `retrospective_lateral/code/corridor_repro.py`, `retrospective_lateral/code/model_vs_loop.py` (+ tests). Suite: 128 passed.
- Commits: `f05f3af1cb` (corridor B test), `6d3fee0890` (B-refuted report), `cb70f724b0` (A-vs-C module). Pushed: `origin/2021_explorer_st-mici @ cb70f724b0`.
- Generated (gitignored): `results/reports/corridor_repro_*.csv`, `model_vs_loop_signals.csv`; Tuesday caches `results/cache/route_c{1,2,3}.npz`; Tuesday rlogs `explorer_st_logs/route_c{1,2,3}/`.
- Key runs: `python -m retrospective_lateral.code.corridor_repro --focus route_c1 --focus route_c2 --focus route_c3`; `python -m retrospective_lateral.code.model_vs_loop --pi-set golden`.
- Hygiene: `opendbc_repo`/`panda` untouched; the user's uncommitted `config.py`/`extract.py` work left untouched; all heavy compute parallelized across cores (`ProcessPoolExecutor`).

## Recommendation

**Build the offline model-replay validation harness next.** It is the smallest step that can confirm a specific model-side fix, it is analysis-only, and the root-cause evidence now squarely justifies it. Only after a candidate model/smoothing change lowers the open-loop model-output weave in replay should an on-road A/B (PI held golden) be planned. Controller tuning stays off the table.
