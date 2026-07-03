# 2026-06-24 Low-Speed Model-Horizon Audit

## Executive Summary

This step extended the `route_6b` counterexample finding across all 75 low-speed steering-wheel-swing rows. It directly scanned local `modelV2.position.x` horizons around each low-speed episode and joined those horizon metrics to the existing lane geometry, desired curvature, CP final command, steering, lead, PI, and lane-quality metrics.

The result is a cleaner split of the low-speed root-cause picture:

- **41/75 rows** have forward model horizon available. Most of these behave like the earlier `route_8d` common-mode corridor case.
- **34/75 rows** are horizon-limited, partial-horizon, or mostly missing forward model path at 20-30 m. These behave more like `route_6b`: low-speed swing is still upstream of final command, but the useful model/path signal is near-field or lane/desired rather than 20-30 m model-y.
- **Top-decile low-speed severity is mixed:** 4 rows are forward-horizon common-mode, 3 rows are horizon-limited upstream near-field/desired, and 1 row is low-lane-quality confounded.

The evidence does **not** move us toward a vehicle-control implementation plan. CP final command remains below desired in the dominant buckets, and the few rows where CP final exceeds desired are not top-severity cases.

Recommendation remains: **do not change driving code yet**.

## Generated Artifacts

| Artifact | Rows | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_audit.csv` | 75 | One row per low-speed symptom window with direct model-horizon scan plus joined signal metrics. |
| `retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_group_summary.csv` | 19 | Group summaries by horizon bucket, root-cause bucket, and their cross-product. |
| `retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_route_summary.csv` | 37 | Route-level summary of low-speed horizon/root-cause bucket patterns. |

## Scan Verification

| Check | Result |
| --- | ---: |
| Low-speed rows scanned | 75 |
| Routes scanned | 37 |
| Rows with zero modelV2 messages | 0 |
| Rows with scan errors | 0 |
| Median modelV2 messages per row | 305 |
| Median scanned files per row | 1 |

One known `LogReader` corrupted-events warning appeared while scanning historical logs, but every low-speed row still received direct model-horizon coverage and `scan_error_count = 0`.

## Horizon Buckets

| Horizon bucket | Rows | Routes | Median steering P2P deg | P90 steering P2P deg | Median speed mph | Median position >=20 m | Median position >=30 m | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `forward_horizon_available` | 41 | 24 | 10.0 | 14.7 | 5.15 | 0.789 | 0.627 | 20-30 m model path is available enough for the prior common-mode model/lane audit. |
| `model_path_mostly_missing` | 16 | 15 | 9.7 | 11.6 | 3.48 | 0.156 | 0.042 | Model path is mostly absent even near 20 m, while lane lines are generally still available. |
| `near_field_horizon_limited` | 15 | 11 | 11.7 | 16.5 | 5.08 | 0.195 | 0.078 | Near-field path/lane/desired evidence is usable; 20-30 m model-y is a poor discriminator. |
| `partial_forward_horizon` | 3 | 2 | 11.1 | 14.3 | 3.18 | 0.460 | 0.179 | Some x20 coverage, weak x30 coverage. |

Interpretation: the prior low-speed model/lane audit was too dependent on x20/x30 model-y. That is useful for forward-horizon rows, but it under-reads horizon-limited low-speed rows.

## Root-Cause Buckets

| Root-cause bucket | Rows | Routes | Median steering P2P deg | Median CP/desired | Median position >=20 m | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `forward_model_or_corridor_common_mode` | 32 | 20 | 9.78 | 0.873 | 0.741 | Forward model/lane corridor signal is available and upstream; CP final attenuates desired. |
| `horizon_limited_upstream_lane_or_near_field_desired` | 26 | 18 | 10.85 | 0.860 | 0.170 | Low-speed swing is upstream, but the useful evidence is near-field/lane/desired, not x20/x30 model-y. |
| `cp_final_missing_controller_stage_unresolved` | 7 | 2 | 10.00 | NaN | 0.945 | CP final is missing in-window, so final-command attenuation cannot be judged. |
| `low_lane_quality_confounded` | 6 | 6 | 11.50 | 0.845 | 0.177 | Lane probability is too low for a clean lane/model conclusion. |
| `controller_gain_not_excluded` | 4 | 4 | 7.45 | 1.126 | 0.996 | CP final exceeds desired, but these are not severe rows. |

The important ranking impact: **58/75 rows** are now cleanly upstream geometry/near-field/desired buckets with CP final generally attenuating desired. Only **4/75 rows** have CP final greater than desired enough that controller gain cannot be excluded, and none are top-decile severity.

## Top-Severity Rows

Top-decile threshold: steering peak-to-peak >= 14.46 deg.

| Route | Start s | Steering P2P deg | Speed mph | Lane prob | Position >=20 m | Position >=30 m | model_y20 finite | model_y30 finite | Horizon bucket | Root-cause bucket | CP/desired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| `route_6b` | 2317.30 | 20.5 | 3.24 | 0.863 | 0.193 | 0.078 | 0.160 | 0.000 | `near_field_horizon_limited` | `horizon_limited_upstream_lane_or_near_field_desired` | 0.880 |
| `route_8d` | 1435.85 | 20.1 | 3.13 | 0.854 | 0.420 | 0.275 | 0.465 | 0.384 | `forward_horizon_available` | `forward_model_or_corridor_common_mode` | 0.894 |
| `route_3d` | 1615.90 | 18.1 | 5.67 | 0.791 | 0.171 | 0.010 | 0.052 | 0.000 | `near_field_horizon_limited` | `horizon_limited_upstream_lane_or_near_field_desired` | 0.752 |
| `route_8f` | 705.50 | 18.0 | 7.41 | 0.773 | 0.988 | 0.758 | 1.000 | 0.668 | `forward_horizon_available` | `forward_model_or_corridor_common_mode` | 0.932 |
| `route_55` | 1009.40 | 16.0 | 5.98 | 0.781 | 0.844 | 0.827 | 0.791 | 0.780 | `forward_horizon_available` | `forward_model_or_corridor_common_mode` | 0.730 |
| `route_b4` | 2355.30 | 15.3 | 5.05 | 0.947 | 0.825 | 0.627 | 0.921 | 0.614 | `forward_horizon_available` | `forward_model_or_corridor_common_mode` | 0.912 |
| `route_8d` | 1277.75 | 15.1 | 5.03 | 0.981 | 0.516 | 0.179 | 0.539 | 0.106 | `partial_forward_horizon` | `horizon_limited_upstream_lane_or_near_field_desired` | 0.981 |
| `route_91` | 1569.30 | 14.7 | 6.28 | 0.329 | 0.943 | 0.923 | 1.000 | 1.000 | `forward_horizon_available` | `low_lane_quality_confounded` | 0.903 |

Top-severity interpretation: the largest low-speed swings are not one mechanism. There are at least two upstream subcases:

1. **Forward-horizon common-mode cases** where 20-30 m model/lane geometry is available and useful.
2. **Horizon-limited near-field cases** where the prior 20-30 m audit produces false negatives, but desired/path/steering still show upstream motion.

## What This Argues Against

- **Against PI/final-command as primary:** dominant buckets have CP final below desired. The four `controller_gain_not_excluded` rows have median steering P2P 7.45 deg and do not include the top-severity cases.
- **Against a single 20-30 m model-y low-speed explanation:** 34/75 rows are horizon-limited, partial, or mostly missing at 20-30 m. Low-speed analysis needs shorter lookaheads.
- **Against treating route_6b as a downstream counterexample:** route_6b now fits the horizon-limited upstream bucket, not a controller/plant-only bucket.

## Remaining Confounders

- No camera is available for many horizon-limited rows, including `route_6b`.
- CX1 telemetry is absent in many historical rows.
- CP final is missing or unresolved in 7 rows, mostly `route_1c`/`route_1b`; those remain a controller-stage evidence gap.
- Lane probability confounds 6 rows; those should not drive a lane/model conclusion.
- The bucket thresholds are descriptive, not a formal causal estimator. They are meant to rank the next evidence step, not select a driving-code change.

## Next Recommendation

Do **not** change driving code yet.

The next logical analysis-only step is to extend the low-speed geometry audit to shorter lookaheads: 5, 10, 15, 20, and 30 m, plus per-packet model horizon. The current cache has 0/10/20/30 m; `route_6b` shows that 20-30 m can be the wrong lens at 1-5 mph.

Smallest controlled evidence step after that, if needed: one low-speed creep route with two scenes, held constant for branch/model/PI/tire/load:

- open, clear lane with no close edge/corridor clutter
- curb/construction/close-edge corridor

Collect camera, model horizon, lane lines, desired curvature, CP/CX1 final command, yaw, and steering. Keep PI fixed.

## Commands Run

```bash
.venv311/bin/python - <<'PY'  # counted low-speed rows/routes and local rlog availability
.venv311/bin/python - <<'PY'  # checked route_8d direct-horizon consistency against cache finite coverage
.venv311/bin/python - <<'PY'  # generated all-row low-speed horizon audit, group summary, and route summary
.venv311/bin/python - <<'PY'  # relabeled CP-final-missing rows separately from controller-gain-not-excluded rows
.venv311/bin/python - <<'PY'  # printed horizon/root-cause summaries, top-decile rows, and scan quality
```
