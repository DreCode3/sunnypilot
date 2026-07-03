# 2026-06-24 Low-Speed Horizon-Transition Audit

## Executive Summary

This audit binned the **26 horizon-limited low-speed wheel-swing rows** into 1-second slices and compared model-path availability at 5/10/15/20/30 m against lane-center, desired curvature, CP final command, path/yaw, and steering phase.

The result refines the prior hypothesis:

- The horizon-limited rows are **not primarily model-dropout-triggered**. Only **2/26** rows show steering amplitude higher when model horizon is shorter.
- The dominant pattern is **short-horizon upstream motion**: **18/26** rows have short model signal present while 20-30 m model-y is missing.
- High-steering bins are more common when short/near model signal is present, not absent: `short_10m_only` bins have 39.5% high-steering rate, `near_5m_only` 35.2%, `model_absent` only 14.9%.
- CP final still looks downstream/follower, not primary: median event CP-final/desired is **0.860** across these 26 rows.

Recommendation remains: **do not change driving code yet**. The next smallest evidence step is raw-shape review for the top horizon-limited episodes, especially `route_6b@2329.1` and `route_3d@1616.8`, focusing on whether short model path, lane-center geometry, and desired curvature are moving together before steering.

## Generated Artifacts

| Artifact | Rows | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_bins.csv` | 265 | 1-second bins across the 26 horizon-limited low-speed rows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_episode_summary.csv` | 26 | One row per horizon-limited episode with finite-fraction, lag/correlation, and high/low-bin deltas. |
| `retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_signal_ranking.csv` | 364 | Per-episode signal-to-steering lag/correlation rows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_group_summary.csv` | 10 | Grouped transition-pattern summaries. |
| `retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_family_counts.csv` | 5 | Descriptive best-correlation signal-family counts. |

## Target Set

| Check | Result |
| --- | ---: |
| Horizon-limited low-speed rows | 26 |
| Routes | 18 |
| 1-second bins | 265 |
| Median model-y5 finite fraction | 0.534 |
| Median model-y10 finite fraction | 0.317 |
| Median model-y20 finite fraction | 0.000 |
| Median model-y30 finite fraction | 0.000 |
| Median CP-final/desired | 0.860 |

## Transition Patterns

| Pattern | Rows | Interpretation |
| --- | ---: | --- |
| `short_model_present_long_horizon_missing` | 18 | 5-15 m model signal exists, but 20-30 m model-y is missing or sparse. |
| `nearfield_sparse_model_lane_desired_present` | 6 | Model path is sparse even near-field, but lane/desired evidence remains steering-correlated. |
| `partial_forward_horizon_lane_desired` | 1 | Some 20 m coverage, weak 30 m coverage, lane/desired still strong. |
| `model_mostly_absent_lane_desired_present` | 1 | Model mostly absent; lane/desired still carry the event. |

This converts the previous "horizon-limited" label into a more specific claim: most of these are **short-horizon model/lane/desired events**, not pure model absence.

## Dropout Test

| Horizon/steering relationship | Rows | Meaning |
| --- | ---: | --- |
| `steering_higher_when_horizon_longer` | 17 | High steering occurs when short model horizon is more available. |
| `no_clear_horizon_amplitude_relation` | 7 | Model horizon varies, but not monotonically with steering amplitude. |
| `steering_higher_when_horizon_shorter` | 2 | Only these rows weakly support horizon dropout as a direct amplitude trigger. |

Bin-level horizon states:

| Model horizon state | Bins | High-steering rate |
| --- | ---: | ---: |
| `model_absent` | 94 | 0.149 |
| `short_10m_only` | 76 | 0.395 |
| `near_5m_only` | 54 | 0.352 |
| `forward_20m_present` | 27 | 0.407 |
| `sparse_model` | 14 | 0.000 |

Interpretation: steering swing is usually not largest when the model path disappears. It is largest when some near/short model signal is still present and upstream lane/desired signals are also moving.

## Signal Evidence

Median event-level absolute correlations to steering:

| Signal group | Median abs corr | Caution |
| --- | ---: | --- |
| Best model lookahead | 0.996 | Strong but often sparse; best lookahead varies by row. |
| Best lane-center lookahead | 0.926 | Strong and usually more continuously available than model-y20/y30. |
| Desired curvature | 0.896 | Strong upstream/final-command-boundary evidence. |
| CP final command | 0.916 | Strong follower evidence; magnitude usually attenuated vs desired. |
| Path curvature | 0.962 | Mechanically coupled to steering; descriptive, not root-cause by itself. |
| Orientation-rate curvature | 0.939 | Also response/vehicle-motion coupled; useful for phase, not standalone cause. |

Median lags using the existing convention "positive signal leads steering":

| Signal | Median lag s |
| --- | ---: |
| Best model lookahead | -0.575 |
| Best lane-center lookahead | 0.550 |
| Desired curvature | 0.750 |
| CP final command | 0.525 |

The model lag is less stable because model samples are sparse and the "best" model lookahead can shift by episode. The lane/desired/CP lags are more interpretable: desired and CP generally move before the steering response in these low-speed windows.

## High-Steering Vs Low-Steering Bins

Across the 26 rows, median high-bin minus low-bin deltas:

| Metric | High-bin median | Low-bin median | Median delta |
| --- | ---: | ---: | ---: |
| model-y5 finite fraction | 1.000 | 0.475 | +0.288 |
| model-y10 finite fraction | 0.463 | 0.000 | +0.350 |
| model-y20 finite fraction | 0.000 | 0.000 | 0.000 |
| model-y30 finite fraction | 0.000 | 0.000 | 0.000 |
| desired abs mean, 1e4 | 8.217 | 7.476 | +1.604 |
| CP final abs mean, 1e4 | 7.996 | 6.729 | +0.571 |
| lane-center y10 abs mean, m | 0.0274 | 0.0234 | +0.0081 |
| lane-center y20 abs mean, m | 0.0479 | 0.0368 | +0.0056 |
| CP-final/desired | 0.929 | 0.869 | +0.146 |

This argues that high steering is a short-horizon/near-field event with larger lane/desired/CP motion, not a simple "long-horizon path disappears, controller swings" event.

## Top Episodes

| Episode | Steering P2P deg | Pattern | Dropout association | Model finite 5/10/20/30 | Key evidence |
| --- | ---: | --- | --- | --- | --- |
| `route_6b@2329.1` | 20.5 | nearfield sparse model/lane/desired | steering higher when horizon longer | 0.673 / 0.579 / 0.158 / 0.000 | Desired corr 0.834, CP corr 0.877, CP/desired 0.880; high bins occur with 5-10 m model present. |
| `route_3d@1616.8` | 18.1 | short model present, long missing | steering higher when horizon longer | 0.542 / 0.474 / 0.053 / 0.000 | Best model corr 1.000, lane y20 corr 0.803, CP/desired 0.752. |
| `route_8d@1286.6` | 15.1 | partial forward horizon | no clear relation | 1.000 / 0.855 / 0.541 / 0.106 | Lane y5 corr 0.958, desired corr 0.923, CP corr 0.964, CP/desired 0.981. |
| `route_8f@1194.3` | 14.1 | short model present, long missing | steering higher when horizon longer | 0.552 / 0.315 / 0.007 / 0.000 | Best model corr 0.997, lane y15 corr 0.916, CP/desired 1.044. |
| `route_60@1773.3` | 13.6 | short model present, long missing | steering higher when horizon longer | 0.720 / 0.383 / 0.000 / 0.000 | Desired corr 0.957, CP corr 0.956, CP/desired 0.931. |

`route_6b@2329.1` is no longer a downstream/controller counterexample. It is the strongest short-horizon upstream case.

## Evidence By Claim

| Observed symptom | Inferred stage | Root-cause hypothesis impact | Evidence status | Next experiment |
| --- | --- | --- | --- | --- |
| Low-speed swing in horizon-limited rows | Upstream short model/lane/desired before steering | Strengthens short-horizon upstream hypothesis | Strong for 18/26 short-model-present rows and 6/26 sparse near-field rows | Inspect raw path/lane shape during high bins for `route_6b@2329.1` and `route_3d@1616.8`. |
| High steering during model-y20/y30 absence | Not enough to imply dropout cause | Weakens "dropout directly triggers swing" hypothesis | Strong: only 2/26 rows have steering higher when horizon is shorter | Treat dropout as a context/visibility limit unless raw shape shows discontinuity at onset. |
| Desired and CP final move with steering | Desired/CP boundary is already active before steering response | Supports upstream command generation; CP final usually follows/attenuates | Strong: median desired corr 0.896, CP corr 0.916, CP/desired 0.860 | Compare desired-vs-CP phase in top rows; do not tune gains from this alone. |
| Path/yaw strongly correlate with steering | Vehicle response is present | Not a root-cause discriminator by itself | Strong correlation but confounded by steering mechanics | Use only as timing/response evidence. |

## What This Argues Against

- **Against model dropout as the main low-speed trigger:** high-steering bins are less common when model is fully absent.
- **Against PI/final-command as the primary low-speed root cause:** CP final remains mostly below desired and tracks desired/steering rather than standing out as the first growth stage.
- **Against using 20-30 m model-y as the discriminator for 1-10 mph:** the median model-y20 and model-y30 finite fractions are both zero in this target set.

## Residual Risks

- The bin audit is descriptive. It tests timing and association, not a controlled causal intervention.
- 1-second bins may smooth very fast transitions; the next raw-shape audit should inspect individual samples around high bins.
- Route camera evidence remains unavailable for `route_6b`, so its scene-level cause is still numeric-only.
- `path_curvature` and `orientation_rate_curvature` are response-coupled and should not be used as primary cause labels.
- CP/CX1 telemetry remains incomplete historically, especially outside CP final.

## Next Recommendation

Do **not** change driving code yet.

The next smallest analysis-only step is a **raw-shape audit** for the top horizon-limited rows:

1. Plot and tabulate raw `model_y5/10/15/20`, lane-center y5/y10/y20, desired curvature, CP final, steering, and finite masks around high-steering bins.
2. Start with `route_6b@2329.1` bins 9-13 and 16, where steering is high and short model signal is present while 20-30 m is absent.
3. Repeat for `route_3d@1616.8` and `route_8d@1286.6` to see whether the shape is route-specific or common.
4. If camera is unavailable for `route_6b`, use the raw-shape result to define a controlled low-speed creep route that captures the same near-field geometry with full camera/model telemetry.

Only after the raw-shape audit shows a stable source pattern should an implementation plan be considered.

## Commands Run

```bash
.venv311/bin/python - <<'PY'  # listed 26 horizon-limited rows and available NPZ channels
.venv311/bin/python - <<'PY'  # generated transition bins, episode summary, signal ranking, group summary, and family counts
.venv311/bin/python - <<'PY'  # printed transition-pattern, dropout-association, high-vs-low-bin, and top-episode summaries
```
