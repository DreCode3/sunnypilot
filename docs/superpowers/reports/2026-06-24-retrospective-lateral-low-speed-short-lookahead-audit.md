# 2026-06-24 Low-Speed Short-Lookahead Geometry Audit

## Executive Summary

This step extended the low-speed geometry audit from 0/10/20/30 m to **0/5/10/15/20/30 m** and rebuilt the retrospective cache as `retrolat-v5`.

The added short lookaheads sharpen the low-speed root-cause split:

- **Forward/common-mode subcase remains real:** 32/75 low-speed rows have usable forward horizon and behave like the earlier `route_8d` corridor case.
- **Horizon-limited upstream subcase is also real:** 26/75 rows lose model-path availability rapidly with lookahead; these behave like `route_6b`, where 20-30 m model-y is often the wrong lens.
- **Top-decile severity is mixed:** 4 forward/common-mode rows, 3 horizon-limited upstream rows, and 1 low-lane-quality-confounded row.
- **Final-command/PI remains a weak primary-cause hypothesis:** median CP-final/desired is 0.876 across all low-speed rows and 0.900 in top-decile rows, so CP final generally attenuates rather than amplifies desired curvature where CP exists.

Recommendation: **do not change driving code yet**. The next best step is a within-window transition audit for horizon-limited rows: determine whether the swing grows during model-horizon dropout/shortening, lane-center motion, desired-curvature motion, or their phase relationship.

## Generated Artifacts

| Artifact | Rows | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_lane_geometry_audit.csv` | 11,934 | Full regenerated geometry audit with 0/5/10/15/20/30 m lookaheads. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_enriched.csv` | 450 | 75 low-speed rows x 6 lookaheads, joined to raw finite fractions and horizon buckets. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_summary.csv` | 12 | Overall and top-decile summaries by lookahead. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_bucket_summary.csv` | 72 | Horizon/root-cause bucket summaries by lookahead. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_top_episodes.csv` | 72 | Top 12 low-speed episodes x 6 lookaheads. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_targets.csv` | 96 | Focus rows for route_6b, route_8d, and top-severity discriminators. |

## Verification Counts

| Check | Result |
| --- | ---: |
| Manifest routes | 147 |
| Manifest schema | 147 `retrolat-v5` |
| Successful route caches | 145 |
| Failed route caches | 2 (`route_67`, `route_68`) |
| Symptom catalog rows | 1,557 |
| Lane-geometry audit rows | 11,934 |
| Low-speed geometry rows | 450 |
| Low-speed episodes | 75 |
| Low-speed routes | 37 |
| Low-speed lookaheads | 0, 5, 10, 15, 20, 30 |

## Low-Speed Population

| Bucket | Episodes | Interpretation |
| --- | ---: | --- |
| `forward_model_or_corridor_common_mode` | 32 | Forward model/lane/corridor geometry is available enough for common-mode evidence. |
| `horizon_limited_upstream_lane_or_near_field_desired` | 26 | Useful evidence is near-field/lane/desired; 20-30 m model-y frequently under-reads the event. |
| `cp_final_missing_controller_stage_unresolved` | 7 | CP final is missing in-window, so final-command attenuation cannot be judged. |
| `low_lane_quality_confounded` | 6 | Lane probability is too low for a clean lane/model conclusion. |
| `controller_gain_not_excluded` | 4 | CP final can exceed desired, but these are not severe rows. |

Confounding is lower in the severe subset: all 8 top-decile rows are `dirty=False`, all have steer ratio 17.2, and PI is `unknown` for 7 rows and `weak` for 1 row.

## Lookahead Summary

Top-decile threshold: steering peak-to-peak >= 14.46 deg.

| Scope | Lookahead m | Episodes | Median model finite | Fraction model finite >=80% | Median model RMS m | Median lane-center RMS m | Median abs model/steer corr | Median abs lane/steer corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 5 | 75 | 0.941 | 0.587 | 0.0165 | 0.0337 | 0.901 | 0.875 |
| All | 10 | 75 | 0.764 | 0.467 | 0.0288 | 0.0403 | 0.944 | 0.879 |
| All | 15 | 75 | 0.571 | 0.347 | 0.0416 | 0.0500 | 0.927 | 0.896 |
| All | 20 | 75 | 0.541 | 0.280 | 0.0573 | 0.0639 | 0.921 | 0.883 |
| All | 30 | 75 | 0.340 | 0.173 | 0.1118 | 0.1046 | 0.964 | 0.874 |
| Top decile | 5 | 8 | 0.973 | 0.750 | 0.0237 | 0.0407 | 0.890 | 0.881 |
| Top decile | 10 | 8 | 0.847 | 0.750 | 0.0405 | 0.0572 | 0.929 | 0.935 |
| Top decile | 15 | 8 | 0.726 | 0.500 | 0.0619 | 0.0784 | 0.889 | 0.913 |
| Top decile | 20 | 8 | 0.665 | 0.375 | 0.0841 | 0.1047 | 0.954 | 0.905 |
| Top decile | 30 | 8 | 0.496 | 0.125 | 0.1763 | 0.1723 | 0.857 | 0.908 |

Interpretation: short lookaheads recover useful low-speed model signal, especially at 5-15 m. But availability still drops materially as lookahead increases, even in top-decile rows. Lane-center remains available and steering-correlated across the same windows.

## Bucket Comparison

| Bucket | Horizon | Lookahead m | Episodes | Median model finite | Fraction finite >=80% | Median model RMS m | Median lane RMS m | Median abs model/steer corr | Median abs lane/steer corr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Forward/common-mode | forward available | 5 | 32 | 1.000 | 0.906 | 0.0174 | 0.0349 | 0.810 | 0.834 |
| Forward/common-mode | forward available | 10 | 32 | 0.939 | 0.750 | 0.0365 | 0.0442 | 0.885 | 0.863 |
| Forward/common-mode | forward available | 15 | 32 | 0.810 | 0.531 | 0.0502 | 0.0569 | 0.899 | 0.865 |
| Forward/common-mode | forward available | 20 | 32 | 0.747 | 0.375 | 0.0652 | 0.0731 | 0.887 | 0.859 |
| Forward/common-mode | forward available | 30 | 32 | 0.566 | 0.156 | 0.1127 | 0.1065 | 0.957 | 0.882 |
| Horizon-limited | model mostly missing | 5 | 10 | 0.398 | 0.000 | 0.0124 | 0.0281 | 0.995 | 0.946 |
| Horizon-limited | model mostly missing | 10 | 10 | 0.129 | 0.000 | 0.0205 | 0.0330 | 0.997 | 0.932 |
| Horizon-limited | model mostly missing | 15 | 10 | 0.014 | 0.000 | 0.0263 | 0.0408 | 0.998 | 0.925 |
| Horizon-limited | model mostly missing | 20 | 10 | 0.000 | 0.000 | NaN | 0.0545 | NaN | 0.919 |
| Horizon-limited | near-field limited | 5 | 14 | 0.579 | 0.143 | 0.0125 | 0.0277 | 0.911 | 0.804 |
| Horizon-limited | near-field limited | 10 | 14 | 0.407 | 0.000 | 0.0234 | 0.0284 | 0.971 | 0.770 |
| Horizon-limited | near-field limited | 15 | 14 | 0.177 | 0.000 | 0.0312 | 0.0368 | 0.993 | 0.794 |
| Horizon-limited | near-field limited | 20 | 14 | 0.026 | 0.000 | 0.0279 | 0.0554 | 0.985 | 0.806 |

The model-mostly-missing and near-field-limited buckets are not downstream-only cases. They still show strong model/steering correlation when model samples exist, while lane-center remains present and steering-correlated when model-y disappears at 20-30 m.

## Top Supporting Episodes

| Episode | Speed mph | Steering P2P deg | Bucket | Key short-lookahead evidence |
| --- | ---: | ---: | --- | --- |
| `route_6b@2329.1` | 3.24 | 20.5 | Horizon-limited near-field | Model finite falls 0.674 -> 0.580 -> 0.333 -> 0.160 -> 0.000 from 5/10/15/20/30 m; lane-center remains present. |
| `route_8d@1445.2` | 3.13 | 20.1 | Forward/common-mode | Model/lane coupling is strong from 5-30 m; abs model/steer correlation is >=0.919 at 5-30 m. |
| `route_3d@1616.8` | 5.68 | 18.1 | Horizon-limited near-field | Model finite falls to 0.053 at 20 m and 0 at 30 m; lane-center/steering remains about 0.80 at 20-30 m. |
| `route_8f@711.9` | 7.41 | 18.0 | Forward/common-mode | Model finite is 1.0 through 20 m and 0.667 at 30 m; lane-center/steering correlation is >=0.970 at 20-30 m. |
| `route_8d@1286.6` | 5.04 | 15.1 | Partial forward horizon | Model finite is good through 20 m but drops to 0.106 at 30 m; lane-center/steering stays >=0.846 at 30 m. |

`route_8d` remains a useful forward/common-mode reference, but it has one corrupted `rlog.zst` note in the rebuilt cache. The route still succeeded and has many samples, but that note remains a QA confounder.

## Evidence By Claim

| Observed symptom | Inferred stage | Root-cause hypothesis | Evidence strength | Next experiment |
| --- | --- | --- | --- | --- |
| Low-speed wheel swing with forward horizon available | Upstream model/lane/desired, followed by attenuated CP final | Common-mode corridor/lane/model geometry creates desired motion before final command | Strong for 32/75 rows; top examples include `route_8d@1445.2`, `route_8f@711.9`, `route_55@1010.6`, `route_b4@2358.9` | Compare camera/model replay against lane-center and desired curvature for the forward/common-mode top rows. |
| Low-speed wheel swing where model path shortens or disappears by 20-30 m | Upstream near-field/lane/desired, not PI-only | Short-horizon model/lane/desired behavior drives the swing; 20-30 m audit produces false negatives | Stronger after adding 5/15 m; 26/75 rows, including `route_6b@2329.1` and `route_3d@1616.8` | One-second transition audit conditioned on model-y finite vs missing at 5/10/15/20/30 m. |
| Low-speed swing with CP final unavailable | Stage unresolved after desired/CP boundary | Controller-stage attribution cannot be closed from current telemetry | Weak/limited; 7/75 rows | Add or recover CP final/CX1 command telemetry for those exact rows before classifying. |
| Low-speed swing with low lane probability | Lane/model conclusion confounded | Perception quality may be part of the cause, but not cleanly separable | Weak; 6/75 rows and only 1 top-decile row | Visual/model replay or exclude from causal ranking until lane-quality is resolved. |
| Low-speed swing in controller-gain-not-excluded rows | CP final may amplify desired | Possible controller gain contribution in a minority bucket | Weak as primary; 4/75 rows and none top decile | Defer any control tuning until upstream buckets are explained and controller rows are replicated under controlled conditions. |

## What This Argues Against

- **Against a single 20-30 m model-path explanation:** 26/75 rows are horizon-limited upstream cases; their key evidence often exists at 5-15 m or in lane/desired, not at 20-30 m.
- **Against a pure controller/PI primary cause:** CP final is usually lower than desired, and top-decile rows are not concentrated in the controller-gain-not-excluded bucket.
- **Against dismissing `route_6b` as an outlier:** `route_6b@2329.1` is the strongest low-speed row and is consistent with a broader horizon-limited bucket, not a one-off downstream exception.
- **Against treating all `route_8d` low-speed rows the same:** `route_8d` contains both forward/common-mode and partial/horizon-limited episodes.

## Residual Risks

- The previous direct model-horizon audit is joined into this report; it is descriptive and threshold-based, not a formal causal estimator.
- `route_8d` has one corrupted-log note, even though the rebuilt route cache succeeded.
- CX1 telemetry remains sparse or absent in many historical rows.
- `stage_first_growth` is coarse for low-speed rows: all 75 are cataloged as `final_command_or_before`, so the finer split depends on the drilldown geometry/desired/CP evidence.
- No local or device camera was recovered for `route_6b`, so the strongest horizon-limited example remains numeric-only.

## Next Recommendation

Do **not** change driving code yet.

The smallest next analysis step is a **low-speed horizon-transition audit** over the 26 horizon-limited rows:

1. Bin each event into 0.5-1.0 s slices.
2. For each bin, compute model-y finite fraction at 5/10/15/20/30 m.
3. Compare lane-center y, desired curvature, CP final command, yaw/path curvature, and steering phase in bins where model-y is present versus missing.
4. Rank whether steering growth follows model-horizon dropout, lane-center motion, desired-curvature motion, or a CP-final change.

Only after that split is resolved should we consider a controlled drive or an implementation plan.

## Commands Run

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract.py::test_resample_channels_extracts_multi_lookahead_lane_geometry_and_road_edges -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py::test_lane_geometry_audit_reports_per_lookahead_model_lane_coupling -q
.venv311/bin/python -m pytest retrospective_lateral/tests -q
.venv311/bin/python -m retrospective_lateral.code.run_all
.venv311/bin/python -m retrospective_lateral.code.drilldown
.venv311/bin/python - <<'PY'  # verified manifest/catalog/audit row counts and lookahead coverage
.venv311/bin/python - <<'PY'  # generated low-speed short-lookahead enriched/summary/top/target CSVs
.venv311/bin/python - <<'PY'  # printed bucket counts, top-decile rows, dirty/PI/steer-ratio confounders
```
