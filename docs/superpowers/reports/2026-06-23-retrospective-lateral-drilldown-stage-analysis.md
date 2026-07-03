# Retrospective Lateral Drilldown Stage Analysis

Date: 2026-06-23

Scope: follow-up analysis for the 2021 Ford Explorer ST sunnypilot lateral symptoms, covering the requested drilldown steps 1-5:

1. Enrich symptom rows with route/config/context metadata.
2. Run stage-lag and amplitude analysis.
3. Build location + speed matched context comparisons.
4. Split low-speed wheel swing from 10-70 mph weave.
5. Run robustness checks across lane quality, yaw source, and detector bands.

This is analysis-only. No vehicle-control code, `opendbc_repo/`, `panda/`, or comma remote workflow was modified.

## Generated Outputs

Generated CSVs under `retrospective_lateral/results/reports/` remain ignored by git:

| Output | Rows | Purpose |
| --- | ---: | --- |
| `drilldown_enriched_catalog.csv` | 1,557 | Symptom rows joined with manifest, config, GPS cell, heading bin, speed bin, lane quality, telemetry availability, and lead availability status. |
| `drilldown_stage_metrics.csv` | 28,026 | Long-form per-stage RMS, peak-to-peak, correlation, and lag rows for every episode/window. |
| `drilldown_stage_gain_lag_summary.csv` | 234 | Repeatable aggregate stage gain/lag medians, quantiles, correlations, and lag summaries for all rows, top-decile rows, and lead-status groups. |
| `drilldown_lead_event_alignment.csv` | 21,403 | Lead onset/exit event-aligned pre/post stage deltas with speed, heading, lane-quality, and 10-70 mph eligibility flags. |
| `drilldown_episode_stage_summary.csv` | 1,557 | One row per symptom row with earliest supported stage/family. |
| `drilldown_location_speed_context.csv` | 1,299 | Same GPS-cell / heading / speed-bin context summaries. |
| `drilldown_location_speed_pi_comparisons.csv` | 5 | Same-context PI comparisons where both weak and golden PI appear. |
| `drilldown_robustness_summary.csv` | 40 | Lead/lane/stage and detector-variant robustness counts. |

## Executive Findings

The drilldown strengthened the upstream-stage hypothesis:

| Symptom | Rows | Earliest supported family |
| --- | ---: | --- |
| Low-speed wheel swing | 75 | 75 `model_or_desired` |
| 10-70 mph weave | 1,482 | 1,482 `model_or_desired` |

Earliest supported stage split:

| Symptom | `model_y20` | `orientation_rate_curvature` | `desired_curvature` |
| --- | ---: | ---: | ---: |
| Low-speed wheel swing | 48 | 20 | 7 |
| 10-70 mph weave | 1,282 | 200 | 0 |

Interpretation:

- The deeper audit no longer merely says "final command or before." For all 75 low-speed rows, the earliest supported signal is upstream of final command, usually `model_y20` or orientation-rate-derived curvature.
- For all 1,482 10-70 mph weave rows, the earliest supported signal is still upstream, mostly `model_y20`.
- The sign of correlation can be negative in some low-speed rows because the analysis does not normalize every stage's sign convention. The support rule uses absolute correlation plus amplitude threshold; lag signs are therefore diagnostic, not a control-sign conclusion.

Recommendation remains: **do not change driving code yet**. The next root-cause split is model/path input versus environment/perception context, not PI tuning.

## Step 1: Enriched Evidence

Enrichment joined the original symptom catalog to manifest sidecars and NPZ-derived context.

| Dimension | Low-speed | 10-70 mph weave |
| --- | ---: | ---: |
| Rows | 75 | 1,482 |
| Lead status `not_extracted` | 75 | 1,482 |
| Lane quality median | 0.846 | 0.929 |
| Rows with lane quality >= 0.5 | 69 | 1,450 |
| Rows with lane quality >= 0.8 | 53 | 1,208 |
| CP only rows | 37 | 732 |
| CP + CX1 rows | 37 | 712 |
| No CP/CX1 rows | 1 | 38 |

Lead/radar remains the major missing input: 100% of symptom rows are marked `not_extracted`, so no conclusion may claim independence from lead-follow behavior.

## Step 2: Stage-Lag And Amplitude

Stage audit order:

`model_y20 -> orientation_rate_curvature -> desired_curvature -> CP/CX1 desired -> CP/CX1 predicted -> CP/CX1 EMA/pre-rate-limit -> CP/CX1 rate-limited/final -> act_curvature -> path_curvature -> steering_angle_deg`

Top low-speed examples:

| Route | Start s | Speed mph | Steering p2p deg | First supported stage | CP | CX1 | PI |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| route_6b | 2317.3 | 3.24 | 20.5 | `orientation_rate_curvature` | yes | no | unknown |
| route_8d | 1435.8 | 3.13 | 20.1 | `model_y20` | yes | yes | unknown |
| route_3d | 1615.9 | 5.67 | 18.1 | `model_y20` | yes | no | unknown |
| route_8f | 705.5 | 7.41 | 18.0 | `model_y20` | yes | yes | unknown |
| route_b4 | 2355.3 | 5.05 | 15.3 | `model_y20` | yes | yes | weak |

Top weave examples:

| Route | Start s | Speed mph | Path RMS e-4 | First supported stage | First-stage corr | First-stage lag s | PI |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| route_b5 | 1038.5 | 31.64 | 13.10 | `model_y20` | 0.995 | 0.60 | weak |
| route_61 | 1215.4 | 29.41 | 11.49 | `model_y20` | 0.994 | 0.60 | unknown |
| route_a8 | 1430.1 | 29.87 | 9.07 | `model_y20` | 0.968 | 0.60 | weak |
| route_92 | 1085.2 | 46.71 | 8.65 | `model_y20` | 0.960 | 0.65 | unknown |
| route_3d | 2248.4 | 34.67 | 7.50 | `model_y20` | 0.969 | 0.60 | unknown |

The lag numbers are useful for ordering, not a sign-convention claim. A positive lag means the stage signal leads the reference signal by that amount under this cross-correlation convention.

## Step 3: Location + Speed Matched Contexts

Context grouping by GPS cell, heading bin, and 5 mph speed bin produced:

| Symptom | Context rows | Covered symptom rows | Multi-route contexts | Max routes in one context |
| --- | ---: | ---: | ---: | ---: |
| Low-speed wheel swing | 68 | 75 | 3 | 2 |
| 10-70 mph weave | 1,231 | 1,481 | 164 | 9 |

This is enough to build same-road/same-speed summaries for weave, but cross-PI matched evidence is still sparse.

Only five weak-vs-golden PI same-context comparisons were found, all for 10-70 mph weave:

| Comparison | Contexts | Median path effect | Median steering effect | Interpretation |
| --- | ---: | ---: | ---: | --- |
| weak minus golden | 5 | +25.7% | +14.0% | Directionally worse for weak in median, but sample is tiny and mixed. |

Individual weak-minus-golden contexts ranged from -45.4% to +35.0% path effect. This is not enough to identify PI as causal, and it still does not beat the stage evidence pointing upstream.

## Step 4: Split By Symptom

Low-speed:

- All 75 rows are upstream-family after the deeper stage audit.
- Robustness variants keep all 75 in `model_or_desired`.
- Stage identity shifts somewhat by band: `model_y20` ranges 42-52 rows, orientation-rate curvature 13-20 rows, desired curvature 7-13 rows.
- The symptom is therefore not steering-wheel-only and not final-command-only. The remaining split is model/path geometry versus low-speed perception/creep context.

10-70 mph weave:

- All 1,482 rows are upstream-family after the deeper stage audit.
- Default, CAN-yaw, calibrated-yaw, narrow-band, and wide-band variants all keep 1,482/1,482 rows in `model_or_desired`.
- Stage identity shifts between `model_y20` and orientation-rate curvature under band changes, but never downstream to controller/final-command/plant as first supported family.

## Step 5: Robustness Checks

Robustness variant counts:

| Symptom | Variant | `model_or_desired` rows |
| --- | --- | ---: |
| Low-speed | default band | 75/75 |
| Low-speed | slower band 0.08-0.50 Hz | 75/75 |
| Low-speed | wider band 0.05-0.80 Hz | 75/75 |
| Weave | default band, best yaw | 1,482/1,482 |
| Weave | default band, CAN yaw | 1,482/1,482 |
| Weave | default band, calibrated yaw | 1,482/1,482 |
| Weave | narrow band 0.12-0.30 Hz | 1,482/1,482 |
| Weave | wide band 0.08-0.45 Hz | 1,482/1,482 |

Weave first-stage split by variant:

| Variant | `model_y20` | `orientation_rate_curvature` |
| --- | ---: | ---: |
| Default/best yaw | 1,282 | 200 |
| Default/CAN yaw | 1,282 | 200 |
| Default/calibrated yaw | 1,282 | 200 |
| Narrow band | 1,192 | 290 |
| Wide band | 1,327 | 155 |

This robustness pattern is strong evidence that the first supported family is upstream. It does not yet distinguish perception/model artifact from a real road/lane/path feature that the model is faithfully following.

## Updated Root-Cause Ranking

| Rank | Hypothesis | Status after drilldown | Next evidence step |
| ---: | --- | --- | --- |
| 1 | Model/path output contains the slow-band motion before controller/final command. | Strengthened. Survives stage-lag and robustness variants. | Inspect model/lane/path features in the same GPS contexts, especially whether lane lines/road edges/path offset move together. |
| 2 | Low-speed swing is a low-speed model/path geometry issue, not steering-only. | Strengthened. All 75 rows upstream-family. | Add stop/creep context and lead/radar extraction; audit top low-speed rows visually if video is available. |
| 3 | Speed amplifies severity. | Still likely; not re-tested as a causal lever here. | Same-context speed ladder or matched cells across speed bins. |
| 4 | PI config is causal. | Weaker. Matched contexts are sparse and mixed; stage evidence is upstream. | Do not tune PI until a controlled A/B or stronger matched scorecard contradicts upstream-stage evidence. |
| 5 | Plant/EPAS/final-command only. | Further weakened. No robustness variant moves first supported family downstream. | Estimate transfer function later, after upstream path source is characterized. |

## Recommended Next Work

Do not change driving code yet.

The smallest next root-cause drilldown is to extract or derive the missing model/perception context for the exact upstream-supported rows:

1. Add lead/radar/headway extraction to close the largest residual confound.
2. For top same-context weave cells, compare model path, lane center, lane width, lane probabilities, road edges, orientationRate, and desired curvature phase/amplitude.
3. For top low-speed rows, add stop/creep state and inspect whether model path/lane center moves before desired curvature.
4. If video/frame data is available locally, sample the top `model_y20` rows and check whether the model is following real lane geometry, a perception wobble, or a path-planner artifact.
5. Only after that split should a controlled drive choose a lever to vary.

## Commands Run

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py -q
.venv311/bin/python -m retrospective_lateral.code.drilldown
.venv311/bin/python - <<'PY'  # summarized enriched, stage, context, PI comparison, and robustness CSVs
```
