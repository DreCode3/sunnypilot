# 2026-06-24 Low-Speed 26-Row Source-Geometry Validation

## Executive Summary

This pass applied the source-geometry decomposition from the three-row drilldown to all **26 horizon-limited low-speed wheel-swing episodes**.

The evidence partially validates the three-row conclusion, but it also makes the root-cause statement more precise:

1. The strongest low-speed steering swings are best explained by **near/short-horizon lane-center and model-path geometry being present and moving together before steering response**.
2. This is not a universal single-signal y10 model signature. Only **12/26** rows classify as strong or partial lane-center/y10-model source rows, while y10 model availability is sparse in many rows.
3. A broader lane-center relation is still visible: focus-bin lane-center amplitude is higher than nonfocus at y10 in **15/26** rows and y20 in **17/26** rows.
4. Desired curvature and CP final command alone are not sufficient explanations. **18/26** episodes contain at least one desired-high, model-absent, low-steering counterexample bin, totaling **40** counterexample bins.
5. Controller/final-command remains unlikely as the primary first-growth source. **24/26** rows classify as downstream or attenuated, with median focus CP-final/desired RMS ratio **0.877**.

Recommendation: **do not change driving code yet**. The next smallest step is read-only visual/camera scene extraction for the top-ranked geometry-source rows and their desired-high/model-absent counterexample bins, if camera files are available.

## Generated Artifacts

Generated artifacts are under ignored `retrospective_lateral/results/` paths.

| Artifact | Rows | Columns | Scope |
| --- | ---: | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_summary.csv` | 26 | 152 | Episode-level source geometry and controller classifier summary. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_ranked.csv` | 26 | 152 | Same rows ranked by lane/model source score. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_bins.csv` | 265 | 32 | One-second bins with model horizon state, steering, desired, CP final, and y10/y20/y30 lane/model geometry. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_signal_corr.csv` | 468 | 17 | Per-episode geometry-signal correlation and lag rows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_controller_26row_signal_corr.csv` | 182 | 16 | Per-episode controller-stage correlation and lag rows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_counterexamples.csv` | 40 | 12 | Desired-high, model-absent, low-steering bins. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_group_summary.csv` | 7 | 9 | Grouped classifier summary. |

## Method

No vehicle-control code was modified.

Inputs:

- Target set: all 26 rows in `drilldown_low_speed_horizon_transition_episode_summary.csv`.
- Systematic focus bins: bins marked `high_steering_bin` in `drilldown_low_speed_horizon_transition_bins.csv`.
- Signals: existing `retrospective_lateral/results/cache/{route}.npz` files.
- Band-pass: `0.08-0.80 Hz`.
- Lookaheads: 5, 10, 15, 20, and 30 m.

Classifier features:

- y10 model availability in high-steering focus bins.
- y10 model-to-lane-center correlation where the model path exists.
- focus-vs-nonfocus lane-center amplitude ratios at y10 and y20.
- desired-high/model-absent/low-steering counterexample bins.
- CP-final/desired RMS gain and controller flags.

## 26-Row Result Summary

| Metric | Result |
| --- | ---: |
| Episodes | 26 |
| Strong or partial lane-center/y10-model source rows | 12 |
| Focus y10 model finite fraction >= 0.5 | 13 |
| Absolute focus y10 model/lane-center corr >= 0.65 | 14 |
| Focus/nonfocus lane-center y10 amplitude ratio > 1 | 15 |
| Focus/nonfocus lane-center y20 amplitude ratio > 1 | 17 |
| Episodes with desired-high/model-absent/low-steer bins | 18 |
| Controller downstream or attenuated rows | 24 |
| Focus CP-final/desired RMS > 1.15 rows | 2 |

Medians:

| Metric | Median |
| --- | ---: |
| Focus model y10 finite fraction | 0.4625 |
| Focus model/lane-center corr y10 | 0.1326 |
| Focus/nonfocus lane-center y10 amplitude ratio | 1.1471 |
| Focus/nonfocus lane-center y20 amplitude ratio | 1.3676 |
| Focus CP-final/desired RMS | 0.8773 |

Classifier counts:

| Geometry classifier | Rows |
| --- | ---: |
| `desired_high_model_absent_counterexample` | 11 |
| `partial_lane_center_y10_model_source` | 7 |
| `strong_lane_center_y10_model_source` | 5 |
| `weak_or_confounded_geometry_source` | 2 |
| `near_5m_lane_desired_source` | 1 |

Controller classifier counts:

| Controller classifier | Rows |
| --- | ---: |
| `controller_downstream_or_attenuated` | 24 |
| `controller_gain_possible_amplifier` | 2 |

## Top Supporting Episodes

These are the highest-ranked rows for near/short-horizon lane-center/model source evidence.

| Rank | Episode | P2P deg | Geometry classifier | Controller classifier | y10 finite | y10 model/lane corr | y10 lane ratio | y20 lane ratio | Counter bins | CP/desired |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `route_3d@1616.8` | 18.1 | strong lane-center/y10 model | downstream/attenuated | 1.000 | 0.882 | 1.798 | 3.049 | 4 | 0.741 |
| 2 | `route_9b@1682.9` | 12.4 | strong lane-center/y10 model | downstream/attenuated | 1.000 | 0.916 | 2.268 | 1.437 | 0 | 0.970 |
| 3 | `route_6b@2329.1` | 20.5 | strong lane-center/y10 model | downstream/attenuated | 0.800 | 0.990 | 1.740 | 1.701 | 1 | 0.910 |
| 4 | `route_8d@1135.9` | 11.1 | strong lane-center/y10 model | downstream/attenuated | 1.000 | 0.956 | 1.200 | 1.202 | 1 | 0.903 |
| 5 | `route_0d@815.7` | 11.7 | partial lane-center/y10 model | downstream/attenuated | 0.625 | 0.837 | 1.949 | 1.380 | 1 | 0.766 |
| 6 | `route_8f@1194.3` | 14.1 | partial lane-center/y10 model | possible controller amplifier | 1.000 | -0.530 | 0.892 | 0.792 | 2 | 1.442 |
| 7 | `route_5c@2193.1` | 6.2 | partial lane-center/y10 model | downstream/attenuated | 0.667 | 0.968 | 1.253 | 1.402 | 2 | 1.006 |
| 8 | `route_8d@1115.8` | 6.0 | partial lane-center/y10 model | downstream/attenuated | 0.683 | -0.801 | 1.340 | 1.667 | 3 | 0.732 |
| 9 | `route_8d@1286.6` | 15.1 | strong lane-center/y10 model | downstream/attenuated | 1.000 | 0.818 | 0.984 | 1.155 | 0 | 1.027 |
| 10 | `route_60@1773.3` | 13.6 | partial lane-center/y10 model | downstream/attenuated | 0.917 | 0.097 | 2.132 | 1.751 | 0 | 0.865 |

## Observed Symptom, Inferred Stage, Hypothesis, Next Experiment

| Observed symptom | Inferred stage | Root-cause hypothesis | Evidence status | Smallest next experiment |
| --- | --- | --- | --- | --- |
| Low-speed wheel swing at 1-10 mph in horizon-limited rows. | Upstream near/short-horizon geometry is often already moving before steering response. | Lane-center and short model path geometry are immediate sources in the strongest rows. | Medium-high: 12/26 strong or partial rows; y10/y20 lane-center amplitude elevated in 15/26 and 17/26 rows. | Extract camera frames and model/lane overlays for ranks 1-5, especially high-steering focus bins. |
| Desired curvature and CP final sometimes grow without steering growth. | Desired/CP is not the sole sufficient stage. | Desired command inherits geometry but does not alone explain steering swing. | Strong against desired-only: 18/26 episodes and 40 bins meet desired-high/model-absent/low-steer counterexample criteria. | Compare high-steering focus bins against model-absent desired-high bins in the same episode. |
| Model y20/y30 is absent or sparse in many rows. | Horizon loss is context, not the direct amplitude trigger. | Short-horizon availability and near-field lane/model geometry matter more than pure long-horizon dropout. | Medium-high from prior horizon audit plus current y10 availability split. | Add visual scenario labels by horizon state: driveway, turn, lane split, curb, edge, occlusion, or lead/traffic. |
| CP final tracks desired with limited gain in most rows. | Controller command is mostly downstream or attenuated. | Controller/final-command is unlikely to be the primary root cause. | Strong against primary controller cause: 24/26 downstream/attenuated; median CP/desired 0.877. | Inspect the 2 gain rows separately for amplification, but do not generalize them to all rows. |

## Counterexamples

The counterexample set is important because it separates "desired curvature got large" from "the wheel visibly swung." A bin entered this file when the model path was absent, desired curvature was at least as high as the focus-bin median, and steering amplitude was lower than the focus-bin median.

Summary:

| Counterexample metric | Result |
| --- | ---: |
| Counterexample bins | 40 |
| Episodes with counterexample bins | 18 |
| Episodes with 4 counterexample bins | `route_3d@1616.8`, `route_5c@2303.8`, `route_77@244.1` |

Canonical counterexamples:

| Episode | Bin | rel to peak s | desired abs x1e4 | focus median desired x1e4 | steering abs deg | focus median steering deg | lane y10 abs m | lane y20 abs m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_3d@1616.8` | 6 | 5.55 | 21.79 | 7.86 | 1.33 | 1.68 | 0.012 | 0.030 |
| `route_3d@1616.8` | 7 | 6.55 | 11.97 | 7.86 | 1.37 | 1.68 | 0.024 | 0.052 |
| `route_3d@1616.8` | 8 | 7.55 | 12.72 | 7.86 | 0.63 | 1.68 | 0.018 | 0.018 |
| `route_3d@1616.8` | 9 | 8.30 | 19.94 | 7.86 | 0.20 | 1.68 | 0.002 | 0.027 |
| `route_6b@2329.1` | 17 | 5.70 | 14.64 | 13.18 | 2.77 | 4.63 | 0.001 | 0.023 |

Interpretation:

- `route_3d@1616.8` remains the strongest discriminator. Late model-absent bins have much larger desired curvature than the early high-steering focus bins, but steering amplitude is lower.
- The same structure appears in other episodes, though not always as cleanly.
- This strongly argues against a desired-curvature-only or pure final-command root cause.

## Ranked Root-Cause Hypotheses

| Rank | Hypothesis | Confidence | Evidence supporting | Evidence against / missing |
| ---: | --- | --- | --- | --- |
| 1 | Near/short-horizon lane-center and model-path geometry is the immediate upstream source for the strongest low-speed swings. | Medium-high | 5 strong + 7 partial source rows; top rows have y10 model present and high y10 model/lane correlation; lane-center y10/y20 focus ratios are > 1 in most rows. | Not universal; focus y10 finite median is only 0.4625; several rows classify as counterexample or weak/confounded. |
| 2 | Desired curvature and CP final command inherit the upstream geometry but are not sufficient root causes by themselves. | High | 18/26 episodes have desired-high/model-absent/low-steer bins; `route_3d@1616.8` has late desired/CP growth with lower steering. | Desired and CP are still mechanically close to steering in many rows, so they remain useful stage markers. |
| 3 | Model horizon loss is a context/enabler, not the direct trigger. | Medium-high | Prior audit showed high steering is more common with short model present than model absent; current classifier ranks short/near model source rows highest. | Long-horizon absence could still change planner state or path selection; this pass does not test interventions. |
| 4 | Controller gain or final-command behavior amplifies a minority of episodes. | Low-medium for minority rows, low as global cause | `route_8f@1194.3` and `route_5c@2303.8` have focus CP-final/desired > 1.15. | 24/26 rows are downstream/attenuated; median focus CP/desired is 0.877; rate/anti-windup/override evidence has not identified first growth at controller final. |
| 5 | Lane width, road-edge width, or road-edge center is the primary source. | Low-medium | Width/edge signals move in some rows and may explain the visual source behind lane-center movement. | Current classifier separates steering better with lane-center/model presence than width-only or edge-only metrics; camera frames are still required. |

## Known Confounders and Residual Risks

- Numeric lane/model signals do not prove the visual scene cause. Camera frames are needed to distinguish lane split, curb, road edge, driveway, occlusion, parked car, lead/traffic, or mapless path behavior.
- y10 model-path availability is sparse in many rows, so a pure y10 model classifier would create false negatives.
- Some rows are mixed-mode: high steering can occur with near-5m or lane/desired evidence even when y10 is weak.
- CP/CX1 telemetry coverage varies by branch/era, and CX1 is not available for all routes.
- PI config, branch era, commit cleanliness, and route context remain historical confounders from the retrospective corpus.
- One-second bins can smooth rapid geometry transitions. The report should be used for ranking and scene selection, not as a final causal intervention result.

## Recommendation

Do **not** change driving code yet.

The root-cause space is now narrower: low-speed swing is most likely rooted in near/short-horizon perception geometry rather than PI/final-command tuning. But the evidence is still numeric and retrospective. The smallest useful next step is:

1. Read-only inventory of available camera files for the top-ranked rows.
2. Extract frames around focus and counterexample bins for:
   - `route_3d@1616.8`
   - `route_9b@1682.9`
   - `route_6b@2329.1`
   - `route_8d@1135.9`
   - `route_0d@815.7`
   - `route_8d@1286.6`
3. Compare focus high-steering frames against same-episode desired-high/model-absent/low-steer frames.
4. Assign scene labels and rerun the classifier by visual scene type.

Only after that visual split should an implementation plan be considered.

## Commands Run

```bash
pwd
git status --short
ls -l retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_*.csv retrospective_lateral/results/reports/drilldown_low_speed_source_controller_26row_signal_corr.csv
python - <<'PY'  # failed: python unavailable in shell
python3 - <<'PY'  # failed: pandas unavailable in system python3
rg --files -g 'pyproject.toml' -g 'requirements*.txt' -g 'Pipfile' -g 'poetry.lock' -g 'uv.lock' -g 'environment*.yml' -g 'setup.py'
rg -n "pandas|pytest|uv run|source .*venv|python3" AGENTS.md EXPLORER_ST.md retrospective_lateral/README.md docs/superpowers/plans/2026-06-14-retrospective-lateral-weave-analysis.md docs/superpowers/specs/2026-06-14-retrospective-lateral-weave-analysis-design.md retrospective_lateral -g '*.md' -g '*.py'
ls -la
find .. -maxdepth 2 -type d \( -name '.venv' -o -name 'venv' -o -name 'env' \) -print
.venv311/bin/python - <<'PY'  # printed artifact row counts, classifier counts, medians, transition summary, and top 12 rows
.venv311/bin/python - <<'PY'  # printed counterexample counts and canonical route_3d/route_6b counterexample bins
.venv311/bin/python - <<'PY'  # printed full ranked 26-row classifier table
sed -n '1,260p' docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-source-geometry-decomposition.md
sed -n '1,260p' docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-horizon-transition-audit.md
sed -n '1,220p' docs/superpowers/reports/2026-06-24-retrospective-lateral-route-6b-low-speed-counterexample.md
.venv311/bin/python - <<'PY'  # verified generated CSV row counts and classifier assertions
git check-ignore -v retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_summary.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_ranked.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_bins.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_signal_corr.csv retrospective_lateral/results/reports/drilldown_low_speed_source_controller_26row_signal_corr.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_counterexamples.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_group_summary.csv
git diff --check -- docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-26row-source-geometry-validation.md
git status --short opendbc_repo panda
.venv311/bin/python -m pytest retrospective_lateral/tests -q
git status --short
```
