# 2026-06-24 Route 6b Low-Speed Counterexample Drilldown

## Executive Summary

This drilldown inspected `route_6b@2317`, the strongest low-speed counterexample from the lane/model replay audit. The finding is narrower than "route_6b disproves the upstream lane/corridor hypothesis." It shows a different low-speed subcase:

`route_6b@2317` is **not** a clean 20-30 m model-path common-mode case. Instead, the swing is already present in near-field model/lane/desired/path signals while model path coverage at 20-30 m is sparse or absent.

The route_6b evidence continues to argue against PI/final-command as the first source. Desired curvature and CP final command both contain the low-speed motion; CP final remains lower than desired. The new discriminator is model horizon/near-field behavior, not vehicle-control tuning.

Recommendation: **do not change driving code yet**. The next smallest analysis step is to add model-path horizon metrics across all low-speed rows, then pair that with camera or a controlled low-speed creep drive.

## Generated Artifacts

| Artifact | Rows/Files | Scope |
| --- | ---: | --- |
| `drilldown_route_6b_low_speed_counterexample_signal_summary.csv` | 40 rows | Signal-level comparison for `route_6b@2317` and `route_8d@1436`. |
| `drilldown_route_6b_low_speed_counterexample_bins.csv` | 20 rows | One-second bins across the `route_6b@2317` event. |
| `drilldown_route_6b_low_speed_counterexample_finite_regions.csv` | 10 rows | Finite model-y10/y20/y30 regions for route_6b and route_8d target windows. |
| `drilldown_route_6b_low_speed_all_episodes_summary.csv` | 3 rows | All route_6b low-speed wheel-swing episodes. |
| `drilldown_route_6b_low_speed_model_horizon_summary.csv` | 4 rows | Direct local rlog scan of modelV2 path horizon for route_6b low-speed rows and route_8d reference. |
| `route_6b_counterexample/route_6b_2317_low_speed_counterexample_timeline.svg` | 1 file | Focused numeric replay timeline for route_6b. |

## Observed Symptom

`route_6b@2317`:

| Field | Value |
| --- | ---: |
| Window | 2317.30-2336.95 s |
| Speed | 3.24 mph |
| Steering peak-to-peak | 20.5 deg |
| Low-speed band steering RMS | 3.41 deg |
| Lane probability min median | 0.863 |
| Lead status | `lead_far` |
| Lead-near fraction | 0.000 |
| PI set | `unknown` |
| CX1 telemetry | absent |

## Inferred Stage

The low-speed swing is already visible upstream of steering response:

| Signal | Raw finite frac | Band RMS | PTP | Corr to steering | Lag s, positive signal leads steering |
| --- | ---: | ---: | ---: | ---: | ---: |
| model_y10 | 0.579 | 0.0356 m | 0.147 m | -0.832 | +1.50 |
| model_y20 | 0.160 | 0.0210 m | 0.072 m | +0.988 | +2.50 |
| model_y30 | 0.000 | NaN | NaN | NaN | NaN |
| lane_center_y20 | 1.000 | 0.0787 m | 0.362 m | -0.792 | +1.30 |
| lane_center_y30 | 1.000 | 0.1291 m | 0.577 m | -0.759 | +1.40 |
| orientation_rate_curvature | 0.860 | 13.11 x1e-4 | 54.50 x1e-4 | -0.921 | -0.05 |
| path_curvature_best | 0.761 | 12.13 x1e-4 | 47.70 x1e-4 | -0.964 | -0.15 |
| desired_curvature | 1.000 | 14.28 x1e-4 | 60.75 x1e-4 | -0.832 | +1.50 |
| cp_final_command | 1.000 | 12.57 x1e-4 | 50.84 x1e-4 | -0.876 | +1.20 |

Interpretation: the event is not steering-only, plant-only, or final-command-only. Desired curvature and final command both contain the low-speed swing, and CP final is attenuated relative to desired (`CP/desired = 0.880`).

The important difference from `route_8d` is the forward model-path horizon. `route_6b` does not have usable model-y30 in the event, and model-y20 is available only for 16% of raw samples. Lane-center y20/y30 remains available for the whole window.

## Model Horizon Evidence

Direct local rlog inspection of `modelV2.position.x` shows the route_6b model path often does not extend to the 20-30 m lookaheads during low-speed windows, while lane lines still extend to 192 m.

| Episode | modelV2 msgs | Median max position.x | Frac >= 10 m | Frac >= 20 m | Frac >= 30 m | Lane-line max x median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| route_6b 1966.70-1976.35 | 242 | 5.26 m | 0.264 | 0.112 | 0.021 | 192 m |
| route_6b 2317.30-2336.95 | 431 | 12.53 m | 0.596 | 0.193 | 0.063 | 192 m |
| route_6b 2688.75-2695.75 | 203 | 4.50 m | 0.281 | 0.108 | 0.005 | 192 m |

This explains why the earlier 20-30 m common-mode audit treated `route_6b` as a counterexample. The audit was asking a forward-model-path question at lookaheads where the model path was frequently not defined. The lane-line geometry was still defined, and the near-field model/desired command still moved.

## All Route 6b Low-Speed Rows

| Start s | Speed mph | Steering P2P deg | Lane prob | model_y10 RMS m | model_y20 RMS m | model_y30 RMS m | lane_y20 RMS m | lane_y30 RMS m | Desired RMS x1e-4 | CP RMS x1e-4 | model_y20 finite | model_y30 finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1966.70 | 2.46 | 12.2 | 0.488 | 0.0022 | NaN | NaN | 0.0733 | 0.1136 | 7.52 | 6.63 | 0.000 | 0.000 |
| 2317.30 | 3.24 | 20.5 | 0.863 | 0.0356 | 0.0210 | NaN | 0.0787 | 0.1291 | 14.28 | 12.57 | 0.160 | 0.000 |
| 2688.75 | 4.23 | 10.4 | 0.817 | 0.0392 | NaN | NaN | 0.0487 | 0.0819 | 22.06 | 21.33 | 0.000 | 0.000 |

Route_6b therefore has a repeatable low-speed pattern: lane-center and desired/final command are present, but 20-30 m model path is mostly absent.

## Hypothesis Impact

| Hypothesis | Impact |
| --- | --- |
| Low-speed swing is upstream of final command. | Strengthened. Desired, CP final, path, and steering carry the motion; CP final does not amplify desired. |
| Low-speed swing can be a corridor/lane geometry issue. | Refined. For route_6b, the evidence points to lane-center and near-field path/desired behavior rather than 20-30 m model path. |
| 20-30 m model/lane common-mode explains all low-speed rows. | Weakened. Route_6b shows why x20/x30 model-y alone is not a sufficient low-speed discriminator. |
| Road-edge width instability is primary. | Weak. Road-edge width has large RMS, but low phase coupling to steering/desired in the target row. |
| PI/final-command tuning is primary. | Further weakened. CP final RMS is below desired, and final-command stages inherit the motion rather than introducing it. |

## Known Confounders

- No camera is available for `route_6b`, locally or on the connected device.
- CX1 telemetry is absent for this route.
- The PI set is `unknown`.
- Low-speed model path may have valid near-field points but short or intermittent forward horizon; treating 20-30 m model-y NaN as "no model involvement" would be a false negative.
- CAN yaw and calibrated yaw have opposite correlation signs in this window; the existing drilldown uses calibrated/best yaw. Use yaw-source comparisons only as a sign/phase robustness check, not as sole root-cause evidence.

## Next Smallest Step

The next analysis-only step should be to add model-path horizon coverage to the retrospective cache/report for all low-speed rows:

- max `modelV2.position.x` per model packet
- fractions of model packets reaching 10, 20, and 30 m
- corresponding lane-line and road-edge horizon coverage
- low-speed grouping by horizon coverage, lane probability, lane-center RMS, desired curvature RMS, and steering P2P

That would tell us whether route_6b is a rare counterexample or a common low-speed mode masked by the prior x20/x30 audit. Still no vehicle-control implementation plan is justified until that split is resolved.

## Commands Run

```bash
.venv311/bin/python - <<'PY'  # route_6b exact-window stage, geometry, yaw-source, and finite-fraction audit
.venv311/bin/python - <<'PY'  # route_8d versus route_6b low-speed comparison
.venv311/bin/python - <<'PY'  # route_6b finite model-path regions and one-second geometry bins
.venv311/bin/python - <<'PY'  # generated signal summary, bins, finite-region CSVs, and SVG timeline
.venv311/bin/python - <<'PY'  # generated all-route_6b-low-speed episode summary
.venv311/bin/python - <<'PY'  # direct local rlog scan of modelV2.position.x horizon
```
