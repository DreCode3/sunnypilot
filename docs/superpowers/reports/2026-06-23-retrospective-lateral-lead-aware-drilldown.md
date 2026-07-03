# 2026-06-23 Lead-Aware Retrospective Lateral Drilldown

## Executive Summary

Lead/headway context was added to the retrospective cache and drilldown tables, then the full local route set was rebuilt with cache schema `retrolat-v3`. The rebuilt catalog contains 1,557 ok symptom rows: 75 low-speed wheel-swing rows and 1,482 10-70 mph weave rows.

The new evidence does not justify vehicle-control code changes yet.

Primary ranking after lead-aware drilldown:

| Rank | Hypothesis | Confidence | Applies To | Current Recommendation |
| --- | --- | --- | --- | --- |
| 1 | The lateral model/path/desired-curvature signal already contains the oscillation before final command/control tuning. | High | Both symptoms | Continue log analysis at perception/planner signal source. Do not tune PI/final command yet. |
| 2 | Near-lead/headway context amplifies or co-occurs with the 10-70 mph weave, but is not sufficient by itself. | Medium | 10-70 mph weave | Run event-aligned lead onset/exit analysis, speed/heading/lane matched. |
| 3 | Low-speed wheel swing is a low-speed model/desired curvature artifact that the actuator follows; near-lead is not implicated. | Medium-high | 1-10 mph swing | Inspect low-speed raw timeseries/video around the top episodes before changing controller code. |
| 4 | Lane/corridor perception quality is a meaningful confounder for weave severity. | Medium | 10-70 mph weave | Add lane-center/lane-width/road-edge visual and numeric audit for top-decile weave. |
| 5 | PI tuning, steer ratio, branch era, and dirty-state differences explain the symptoms. | Low with current evidence | Both symptoms | Keep as confounders, not root-cause targets, until controlled A/B evidence exists. |

## Commands Run

Key verification and analysis commands:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract.py::test_resample_channels_extracts_model_and_radar_lead_context -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract.py -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py::test_stage_gain_lag_summary_groups_overall_top_decile_and_lead_status -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py::test_lead_event_alignment_reports_onset_and_exit_stage_deltas -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py::test_build_drilldown_outputs_writes_expected_artifacts -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py -q
.venv311/bin/python -m pytest retrospective_lateral/tests -q
.venv311/bin/python -m retrospective_lateral.code.run_all
.venv311/bin/python -m retrospective_lateral.code.drilldown
```

`drilldown_stage_gain_lag_summary.csv` now formalizes the prior one-off stage gain/lag pandas checks as repeatable output. Additional local pandas one-off commands were run to inspect and verify the generated CSVs by symptom, lead status, speed bin, first supported stage, PI set, steer ratio, dirty state, and CP/CX1 availability.
`drilldown_lead_event_alignment.csv` adds repeatable lead onset/exit event alignment with pre/post stage amplitudes and 10-70 mph eligibility flags.

## Evidence Updates

### Row Counts

| Symptom | Rows | Lead Near | Lead Far | No Lead |
| --- | ---: | ---: | ---: | ---: |
| low_speed_wheel_swing | 75 | 0 | 72 | 3 |
| weave_10_70 | 1,482 | 815 | 458 | 209 |

### Stage Evidence

The robustness table still places the earliest supported family at `model_or_desired` for every row:

| Symptom | First Supported Family | Rows |
| --- | --- | ---: |
| low_speed_wheel_swing | model_or_desired | 75 |
| weave_10_70 | model_or_desired | 1,482 |

Default first-supported-stage split:

| Symptom | Stage | Rows |
| --- | --- | ---: |
| low_speed_wheel_swing | model_y20 | 48 |
| low_speed_wheel_swing | orientation_rate_curvature | 20 |
| low_speed_wheel_swing | desired_curvature | 7 |
| weave_10_70 | model_y20 | 1,282 |
| weave_10_70 | orientation_rate_curvature | 200 |

Top-decile severity strengthens the model-path signal:

| Symptom | Severity Metric | Top-Decile Stage Result |
| --- | --- | --- |
| low_speed_wheel_swing | steering peak-to-peak >= 14.46 deg | 7 `model_y20`, 1 `orientation_rate_curvature`, 0 `desired_curvature` |
| weave_10_70 | path curvature band RMS >= 3.97e-4 | 149 `model_y20`, 0 `orientation_rate_curvature` |

### Stage Gain/Lag Evidence

Curvature-stage medians show the final command is mostly inheriting or attenuating the upstream oscillation, not introducing a new higher-energy oscillation.

The repeatable aggregate output is:

| Output | Rows | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_stage_gain_lag_summary.csv` | 234 | Stage gain/lag medians, p75/p90/p95 RMS, correlation, lag quantiles, all rows, top-decile rows, overall groups, and lead-status groups. |

For weave rows, all rows median RMS x1e4:

| Stage | Median RMS x1e4 | Median Corr To Path | Median Lag s |
| --- | ---: | ---: | ---: |
| orientation_rate_curvature | 2.256 | 0.995 | 0.00 |
| desired_curvature | 2.089 | 0.972 | 0.65 |
| cp_desired_curvature | 1.713 | 0.949 | 0.65 |
| cp_predicted_curvature | 1.544 | 0.955 | 0.55 |
| cp_final_command | 1.655 | 0.957 | 0.60 |
| act_curvature | 2.089 | 0.972 | 0.65 |
| path_curvature | 2.195 | 1.000 | 0.00 |

For top-decile weave rows:

| Stage | Median RMS x1e4 | Median Corr To Path | Median Lag s |
| --- | ---: | ---: | ---: |
| orientation_rate_curvature | 5.009 | 0.996 | 0.00 |
| desired_curvature | 4.641 | 0.974 | 0.55 |
| cp_desired_curvature | 4.093 | 0.961 | 0.55 |
| cp_predicted_curvature | 3.788 | 0.968 | 0.55 |
| cp_final_command | 3.963 | 0.972 | 0.50 |
| act_curvature | 4.641 | 0.974 | 0.55 |
| path_curvature | 4.784 | 1.000 | 0.00 |

For low-speed top-decile rows:

| Stage | Median RMS x1e4 | Median Corr To Steering | Median Lag s |
| --- | ---: | ---: | ---: |
| orientation_rate_curvature | 11.82 | -0.947 | -0.10 |
| desired_curvature | 16.59 | -0.810 | 0.65 |
| cp_desired_curvature | 13.75 | -0.879 | 0.90 |
| cp_predicted_curvature | 11.52 | -0.791 | 0.70 |
| cp_final_command | 13.62 | -0.811 | 0.475 |
| act_curvature | 16.59 | -0.810 | 0.65 |
| path_curvature | 9.955 | -0.982 | -0.15 |

## Low-Speed Wheel Swing

### Observed Symptom

The low-speed symptom is a steering-wheel swing at 1-10 mph. The rebuilt catalog has 75 rows. Top steering peak-to-peak examples:

| Route | Start s | Speed mph | Steering P2P deg | First Supported Stage | Lead Status | PI Set | Steer Ratio | CP | CX1 |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| route_6b | 2317 | 3.24 | 20.5 | orientation_rate_curvature | lead_far | unknown | 17.2 | yes | no |
| route_8d | 1436 | 3.13 | 20.1 | model_y20 | lead_far | unknown | 17.2 | yes | yes |
| route_3d | 1616 | 5.67 | 18.1 | model_y20 | lead_far | unknown | 17.2 | yes | no |
| route_8f | 705.5 | 7.41 | 18.0 | model_y20 | lead_far | unknown | 17.2 | yes | yes |
| route_55 | 1009 | 5.98 | 16.0 | model_y20 | lead_far | unknown | 17.2 | yes | no |
| route_b4 | 2355 | 5.05 | 15.3 | model_y20 | lead_far | weak | 17.2 | yes | yes |

### Inferred Stage

`model_or_desired` is the first supported family for all 75 low-speed rows. The default first-supported-stage split is 48 `model_y20`, 20 `orientation_rate_curvature`, and 7 `desired_curvature`.

The symptom catalog's broader `stage_first_growth` label is `final_command_or_before`, but the drilldown stage summary resolves the earliest supported family to `model_or_desired`. Use the drilldown stage summary rather than the broad catalog label for root-cause ranking.

### Root-Cause Hypotheses

1. **Low-speed model/path/desired signal oscillation. Confidence: medium-high.**
   Evidence: no `lead_near` rows; first-supported family is always `model_or_desired`; the largest rows are mostly `model_y20`; CP/final command does not show a new larger oscillation after desired curvature.

2. **Low-speed steering/actuator plant makes an upstream path artifact visible as wheel swing. Confidence: medium.**
   Evidence: steering and path/curvature stages are strongly correlated, with low-speed sign conventions producing negative correlations. This explains visible wheel motion but does not move root cause downstream into PI tuning.

3. **Lead/headway behavior causes low-speed swing. Confidence: low.**
   Evidence against: 0 of 75 low-speed rows are `lead_near`; 72 are `lead_far`, 3 are `no_lead`.

4. **PI/final-command tuning causes low-speed swing. Confidence: low.**
   Evidence against: no golden low-speed comparison rows; weak and unknown rows both contain the symptom; first-supported stages are upstream; final-command RMS does not create a new larger signal.

### Smallest Next Experiment

Do a raw low-speed episode audit on the top 8-10 rows above:

1. Plot `model_y20`, `orientation_rate_curvature`, `desired_curvature`, CP/CX1 desired/predicted/final command, steering angle, vEgo, lane probabilities, lane width, and lead fields on the same timeline.
2. Add video/model visual inspection for `route_6b@2317`, `route_8d@1436`, `route_3d@1616`, and `route_8f@705.5`.
3. Confirm whether the path oscillation starts in model path/lane geometry before desired curvature and command.

No vehicle-control change should be made before this audit.

## 10-70 MPH Weave

### Observed Symptom

The weave catalog has 1,482 rows from 10-70 mph. Lead-aware split:

| Lead Status | Rows | Routes | Median Speed mph | Median Path RMS x1e4 | P90 Path RMS x1e4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lead_near | 815 | 85 | 47.45 | 2.149 | 4.105 |
| lead_far | 458 | 83 | 48.95 | 1.983 | 4.012 |
| no_lead | 209 | 61 | 57.28 | 1.251 | 2.948 |

Top path RMS examples:

| Route | Start s | Speed mph | Path RMS x1e4 | First Supported Stage | Lead Status | Min Headway s | PI Set | Lane Prob Min Median |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: |
| route_b5 | 1038 | 31.64 | 13.10 | model_y20 | lead_near | 0.536 | weak | 0.813 |
| route_61 | 1215 | 29.41 | 11.49 | model_y20 | lead_near | 0.927 | unknown | 0.876 |
| route_a8 | 1430 | 29.87 | 9.074 | model_y20 | lead_near | 1.064 | weak | 0.564 |
| route_92 | 1085 | 46.71 | 8.651 | model_y20 | lead_near | 1.036 | unknown | 0.535 |
| route_3d | 2248 | 34.67 | 7.499 | model_y20 | lead_near | 1.253 | unknown | 0.415 |
| route_15 | 310.3 | 50.26 | 6.211 | model_y20 | no_lead | 4.594 | unknown | 0.772 |
| route_stock | 2220 | 38.13 | 5.862 | model_y20 | no_lead | 3.070 | unknown | 0.973 |

### Inferred Stage

Every weave row has earliest supported family `model_or_desired`. Default first-supported-stage split is 1,282 `model_y20` and 200 `orientation_rate_curvature`.

Top-decile weave is entirely `model_y20` first-supported: 149 of 149 rows. That is the strongest current stage evidence.

### Root-Cause Hypotheses

1. **Model/path signal creates the weave before final command. Confidence: high.**
   Evidence: 1,482 of 1,482 rows first-supported family `model_or_desired`; 1,282 `model_y20`; all top-decile rows are `model_y20`; CP/final command has lower median RMS than desired/path while remaining highly correlated.

2. **Near lead/headway amplifies or co-occurs with weave severity. Confidence: medium.**
   Evidence for: 815 of 1,482 weave rows are `lead_near`; 90 of 149 top-decile rows are `lead_near`; the largest route examples are mostly near-lead; median path RMS is higher for `lead_near` than `no_lead`.
   Evidence against: 458 `lead_far` rows and 209 `no_lead` rows also show weave; top-decile has 51 `lead_far` and 8 `no_lead`; `route_stock` has a clear no-lead model-y20 weave row. Event-aligned onset analysis does not show a median upstream amplitude increase after lead onset in the matched 10-70 mph subset.

3. **Lane/corridor perception quality contributes to severity. Confidence: medium.**
   Evidence for: some top rows have low-ish lane probability medians, including 0.415, 0.459, 0.480, 0.535, and 0.564. Evidence against: robustness filters still preserve the model-or-desired result at lane quality >= 0.5 and >= 0.8.

4. **PI/final-command tuning is primary. Confidence: low.**
   Evidence against: only five weak-vs-golden location/speed matched contexts, with mixed direction; golden rows are few and dirty; weak, unknown, and golden rows all show model/path-stage weave; CP/final command attenuates rather than introduces the oscillation.

5. **Steer ratio/branch era/dirty state is primary. Confidence: low with current data.**
   Evidence against: symptoms span many commits and 17.2 ratio runs; 18.0 rows are sparse and confounded. Golden rows are dirty, so they cannot be treated as clean controlled evidence.

### Lead Event Alignment

Implemented output:

| Output | Rows | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_lead_event_alignment.csv` | 21,403 | One row per lead transition per stage, with pre/post stage RMS, percent delta, speed/lane/heading context, `speed_range_ok`, `context_match`, and `analysis_eligible`. |

Detected route-level lead transitions:

| Event Type | Unique Events | Context + 10-70 Eligible | Eligible + Overlaps Weave Window |
| --- | ---: | ---: | ---: |
| onset | 612 | 131 | 103 |
| exit | 647 | 152 | 113 |

Eligible events require similar pre/post speed, same heading bin, similar lane probability, and both pre/post median speeds inside 10-70 mph.

Eligible onset rows that overlap weave windows do **not** show median upstream growth after lead onset:

| Stage | Events | Median Delta % | Positive Fraction |
| --- | ---: | ---: | ---: |
| model_y20 | 103 | -9.45% | 0.456 |
| orientation_rate_curvature | 103 | -6.51% | 0.456 |
| desired_curvature | 103 | -7.08% | 0.447 |
| cp_final_command | 103 | -7.10% | 0.456 |
| path_curvature | 103 | -3.85% | 0.447 |

Eligible exit rows that overlap weave windows also show small median decreases or mixed behavior:

| Stage | Events | Median Delta % | Positive Fraction |
| --- | ---: | ---: | ---: |
| model_y20 | 113 | -3.86% | 0.460 |
| orientation_rate_curvature | 113 | -5.92% | 0.451 |
| desired_curvature | 113 | -3.94% | 0.434 |
| cp_final_command | 113 | -5.62% | 0.407 |
| path_curvature | 113 | -4.03% | 0.469 |

Interpretation: near lead remains a meaningful route-context marker and may still correlate with close-follow scenarios, but the event-aligned evidence does not support a simple immediate lead-onset trigger for weave. Positive outlier onsets exist, so the next useful step is targeted raw timeline/video audit rather than PI or final-command tuning.

### Raw Timeline Audit

Implemented output:

| Output | Rows | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_raw_timeline_audit.csv` | 36 | Pre/episode/post or lead-event pre/post summaries for the Step 3 target windows. |
| `retrospective_lateral/results/reports/drilldown_raw_timeline_bins.csv` | 469 | One-second raw timeline bins for the same targets. |
| `retrospective_lateral/results/reports/raw_timeline_frames/route_8d_*.png` | 4 | Extracted `fcamera` stills around `route_8d@1436`; generated and ignored. |

The focused Step 3 report is `docs/superpowers/reports/2026-06-23-retrospective-lateral-raw-timeline-audit.md`.

Key result: the top 10-70 mph weave episodes have strong model-y20 to lane-center-y20 coupling in the weave band. Episode correlations were 0.879 to 0.961, with lane-center RMS comparable to model-y20 RMS. CP final command remained below desired curvature in the audited top-weave windows, with CP/desired RMS ratios from 0.822 to 0.873.

For positive lead-onset outliers, post-onset growth is real, but it grows upstream too: model, lane center, path, desired, and CP/final command all increase together. Post-onset CP/final remains below desired curvature. This refines the lead hypothesis toward close-lead/corridor co-occurrence rather than an isolated lead-tracking or final-command root cause.

For low speed, `route_8d@1436` had available camera frames and showed a slow construction/traffic corridor with cones/barrier geometry and a truck close to the right. Its numeric window had model/lane-center correlation 0.984. `route_6b@2317` remained a mixed low-speed subcase without local video coverage.

### Lane Geometry Audit

Implemented output:

| Output | Rows | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_lane_geometry_audit.csv` | 7,956 | Episode and eligible lead-event pre/post model, lane-line, lane-center, lane-width, and road-edge metrics at 0, 10, 20, and 30 m lookaheads. |

The focused Step 4 report is `docs/superpowers/reports/2026-06-23-retrospective-lateral-lane-geometry-audit.md`.

Key result: top-decile 10-70 mph weave is strongly lane-center/model-path coupled at forward lookaheads. Median model-to-lane-center correlation is 0.828 at 20 m and 0.934 at 30 m; at 30 m, 87.9% of top-decile weave rows have absolute correlation >= 0.8. CP final command remains below desired curvature in the aggregate.

For low speed, the result is mixed. `route_8d@1436` remains strongly lane-center coupled at 20-30 m, while `route_6b@2317` is still a distinct subcase with weak/negative 20 m coupling and missing 30 m model-path data in the audited window.

Eligible lead-onset events still do not show median immediate upstream growth at 20 m: model y20 RMS -9.45%, lane-center y20 RMS -2.53%, path RMS -3.85%, desired -7.08%, and CP final -7.10%. The positive outliers grow in lane geometry as well as model/path/controller stages.

## Confounders And Residual Risks

| Confounder | Current State |
| --- | --- |
| Location/speed matching | Still sparse for PI A/B: only five weak-vs-golden contexts. |
| Dirty state | Golden rows in the rebuilt set are dirty, so they are not clean controls. |
| Lead status | Now extracted, but only summarized at episode level; needs event-aligned onset/exit analysis. |
| Lane quality | Robustness filters preserve conclusions, but top severity includes some lower lane-prob rows. |
| CP/CX1 availability | Useful but incomplete: low-speed rows split 37 with CX1 and 37 without; weave has 712 with CX1 and 732 CP-only rows. |
| Corrupted logs | `route_8d` includes one corrupted segment warning but still produced a successful cache. `route_67` and `route_68` failed with zero samples. |
| Stock/reference route | `route_stock` has no CP/CX1 telemetry and should be used only as a broad reference, not direct sunnypilot stage evidence. |

## Recommended Next Steps

1. **Completed: formalize the stage gain/lag analysis as a repeatable CSV/report.**
   Implemented as `drilldown_stage_gain_lag_summary.csv`, regenerated from the v3 cache with 234 aggregate rows.

2. **Completed: run lead onset/exit event analysis for weave.**
   Implemented as `drilldown_lead_event_alignment.csv`. Result: no median upstream growth after eligible lead onsets; behavior is mixed with positive outliers.

3. **Completed: run a top-episode raw timeline audit.**
   Implemented as `drilldown_raw_timeline_audit.csv`, `drilldown_raw_timeline_bins.csv`, and a focused report. Result: top weave rows are strongly lane-center/model-path coupled, positive lead outliers grow upstream and downstream together, and final command still attenuates desired curvature.

4. **Completed: add lane geometry/perception audit.**
   Implemented as `drilldown_lane_geometry_audit.csv` after rebuilding the cache as `retrolat-v4`. Result: top-decile weave is strongly lane-center/model-path coupled at 20-30 m; lead-onset medians remain negative; low-speed evidence is mixed.

5. **Run targeted visual/model replay or design a controlled drive.**
   First inspect geometry-coupled cases with model/lane overlays if available: `route_b5@1038`, `route_61@1215`, `route_a8@1430`, `route_92@1085`, and `route_8d@1436`. If a controlled drive is needed after that, hold branch, steer ratio, tire/load state, route, speed, and PI config constant. Collect explicit no-lead, lead-far, and lead-near passes. The offroad-safe detached workflow in `AGENTS.md` remains mandatory for any later remote comma action.

## Recommendation

Do not change driving code yet.

The evidence is ready for a focused lane-geometry/perception analysis plan, not a vehicle-control implementation plan. Vehicle-control changes should wait until those outputs show a downstream controller/final-command root cause or a clean controlled drive isolates one.
