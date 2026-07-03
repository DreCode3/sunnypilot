# Retrospective Lateral Root-Cause Hypotheses

Date: 2026-06-23

Subject: 2021 Ford Explorer ST sunnypilot lateral driving performance issues:

1. Low-speed steering-wheel swing at 1-10 mph.
2. Straight/gentle-section weave at 10-70 mph.

## Executive Summary

The completed retrospective outputs support a conservative conclusion: both symptom classes are real, command-correlated, and usually visible upstream of the vehicle response, but the current artifacts do not yet satisfy the design gate for a validated vehicle-control implementation plan.

Verified corpus shape:

| Artifact | Count |
| --- | ---: |
| Manifest routes | 147 |
| Successful route caches / NPZ files | 145 |
| Manifest extraction failures | 2 |
| Symptom catalog rows | 1,557 |
| Low-speed wheel-swing rows | 75 |
| 10-70 mph weave rows | 1,482 |

Strongest evidence:

- Low-speed: 75/75 wheel-swing rows are `command_correlated` and `final_command_or_before`. Supplemental CP/CX1 stage auditing shows `desired_curvature` peak-to-peak is effectively identical to `act_curvature` in all 75 rows; where CX1 is available, CX1 desired curvature is also nearly identical to the high-level command. This argues against a steering-wheel-only or plant-only first source.
- 10-70 mph: 1,482/1,482 weave windows localize to `model_or_desired`. Slow-band command and desired curvature are essentially identical at the median, and both track path/steering strongly. Spearman correlations: path vs command = 0.983, path vs desired = 0.983, path vs model_y20 = 0.980.
- Speed is a dominant descriptive factor for 10-70 mph weave: Spearman(speed, path-weave metric) = -0.801. Median path-weave metric falls from 4.19 in the 10-20 mph bin to 0.97 in the 60-70 mph bin.
- PI config, steer ratio, branch/commit era, and dirty state are not validated root causes from these outputs. Their apparent differences are observational and confounded by route, speed, telemetry availability, and unmatched historical context.

Recommendation: **do not change driving code yet**. The next work should be a smallest-possible evidence step: lead-gated, location/speed-matched stage-lag analysis on existing logs, followed only if needed by a controlled corridor drive with PI fixed and model/config varied one factor at a time.

## Ranked Hypotheses

| Rank | Root-cause hypothesis | Applies to | Confidence | Evidence supporting | Evidence against / limits | Smallest next step |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Slow-band path motion is already present in model/path-planner desired curvature, and the controller mostly passes it through. | 10-70 mph weave | Moderate-high as descriptive stage evidence; not yet root-cause-likely | 1,482/1,482 weave rows are `model_or_desired`; command/desired median ratio = 1.000; path vs command/desired Spearman = 0.983; holds with CP-only, CP+CX1, and no controller telemetry. | Lead/radar not gated; no location-matched scorecard; stage label can be pulled early by model_y20 threshold; catalog does not expose full provenance. | Lead-gated same-corridor stage-lag analysis: model_y20/orientationRate/desired/CP/CX1/final command/yaw/steer, with alternate yaw source check. |
| 2 | Low-speed wheel swing starts before or at desired curvature command, then becomes visible as large wheel motion at 1-10 mph. | Low-speed swing | Moderate | 75/75 rows are command-correlated; desired/act curvature peak-to-peak ratio median = 1.000; CX1 desired/act median = 0.993 where present; top episodes show large commanded curvature and path response. | Low-speed built-in detector has coarse stage labels; CP exists for 68/75 and CX1 for 37/75; no stop/lead gating; no golden-PI low-speed rows to compare. | Per-episode lag/cross-correlation for top low-speed episodes, then one repeated stop/creep approach with CP/CX1, lead state, and lane quality captured. |
| 3 | Vehicle speed is the main severity amplifier, not an independent root by itself. | Both, strongest for 10-70 mph | Moderate | 10-70 speed correlation = -0.801; lower speed bins have much worse path-weave metric; all low-speed rows are single-digit mph by detector definition. | Speed is also part of detector eligibility; road/route and lead context can covary with speed; this does not identify whether model, planner, controller, or plant creates the motion. | Same-road speed ladder or matched historical road cells at 20/30/45/60 mph with config held constant. |
| 4 | PI config, steer ratio, and branch era are not proven causal levers in this corpus. | Both | Moderate negative evidence | Weave remains `model_or_desired` across unknown, weak, and golden PI groups; low-speed exists under unknown and weak PI; steerRatio 17.2 dominates the corpus; golden has only 38 weave rows from 3 dirty routes and no low-speed rows. | Config labels are unknown for 104/147 routes; on-device dirty edits can diverge from commits; historical groups are not speed/location matched. | Build a location/speed-matched scorecard only after provenance is joined into catalog; do not use PI toggles as fixes until stage evidence says PI is causal. |
| 5 | EPAS/plant/road-only behavior is not the first source, though it may amplify visible motion. | Both | Moderate-low negative evidence | Command/desired/model-stage signals contain the same slow-band motion before actual path and steering; low-speed rows are not steering-only. | Path and steering gain still vary with speed; yaw-source and road/lead confounds remain; plant amplification is not ruled out. | Estimate command-to-yaw/steer transfer by speed on repeated road cells using calibrated yaw and CAN yaw. |

## Low-Speed Wheel Swing

### Observed Symptom

The catalog contains 75 low-speed wheel-swing rows from 37 routes. Median speed is 4.55 mph; steering peak-to-peak median is 10.2 deg, p90 is 14.46 deg, and max is 20.5 deg.

### Inferred Stage

Built-in detector result:

| Stage / confidence | Rows |
| --- | ---: |
| `final_command_or_before` / `command_correlated` | 75 |

Supplemental stage audit over the same low-speed windows:

| Stage metric | Availability | Median peak-to-peak |
| --- | ---: | ---: |
| `desired_curvature` | 75/75 | 60.32e-4 |
| `act_curvature` | 75/75 | 60.32e-4 |
| `cp_desired_curvature` | 68/75 | 50.00e-4 |
| `cp_final_command` | 68/75 | 48.78e-4 |
| `cx1_desired_curvature` | 37/75 | 58.05e-4 |
| `cx1_command_curvature` | 37/75 | 50.19e-4 |
| `model_y20` | 54/75 | 0.36 m |

Key ratios:

| Ratio | Median | p10 | p90 |
| --- | ---: | ---: | ---: |
| desired / act curvature peak-to-peak | 1.000 | 1.000 | 1.000 |
| CP desired / act peak-to-peak | 0.868 | 0.701 | 0.954 |
| CP final / act peak-to-peak | 0.792 | 0.621 | 0.946 |
| CX1 desired / act peak-to-peak | 0.993 | 0.962 | 1.002 |
| CX1 command / act peak-to-peak | 0.859 | 0.686 | 0.962 |

Interpretation: the low-speed wheel swing is not first appearing only at steering response. The high-level desired/act curvature command already contains the motion. CP/CX1 final command is often slightly lower than desired/act, which argues against final rate limiting or final Ford command generation as the sole amplifier. The exact earliest stage remains unresolved because low-speed model/path curvature is less reliable and CP/CX1 coverage is uneven.

### Top Supporting Episodes

| Route | Start s | Speed mph | Steering p2p deg | Path RMS e-4 | Act p2p e-4 | Desired p2p e-4 | PI | Commit | CP | CX1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| route_6b | 2317.3 | 3.24 | 20.5 | 12.13 | 90.46 | 90.44 | unknown | 046780c5 | yes | no |
| route_8d | 1435.8 | 3.13 | 20.1 | 11.71 | 173.33 | 173.33 | unknown | 17cb595f | yes | yes |
| route_3d | 1615.9 | 5.67 | 18.1 | 6.03 | 92.18 | 92.18 | unknown | 748d6c5b | yes | no |
| route_8f | 705.5 | 7.41 | 18.0 | 7.72 | 77.00 | 77.00 | unknown | 17cb595f | yes | yes |
| route_55 | 1009.4 | 5.98 | 16.0 | 7.90 | 102.00 | 102.00 | unknown | c594321b | yes | no |
| route_b4 | 2355.3 | 5.05 | 15.3 | 5.62 | 94.02 | 94.02 | weak | 5e0785b9 | yes | yes |

### Group Comparisons

| Group | Rows | Routes | Median steering p2p deg | p90 steering p2p deg |
| --- | ---: | ---: | ---: | ---: |
| PI unknown | 58 | 27 | 10.2 | 14.82 |
| Weak PI proven | 17 | 10 | 9.4 | 12.67 |
| Golden PI proven | 0 | 0 | n/a | n/a |
| steerRatio 17.2 | 73 | 35 | 10.2 | 14.58 |
| steerRatio 16.8 | 1 | 1 | 8.3 | 8.3 |
| steerRatio 18.0 | 1 | 1 | 7.8 | 7.8 |
| Dirty build true | 0 | 0 | n/a | n/a |
| Dirty build false | 75 | 37 | 10.2 | 14.46 |
| CP only | 37 | 18 | 10.1 | 12.88 |
| CP + CX1 | 37 | 18 | 10.3 | 14.86 |
| No CP/CX1 | 1 | 1 | 8.3 | 8.3 |

Evidence argues against:

- A steering-only comfort issue: 75/75 rows have correlated command movement.
- A plant-only first source: desired/act curvature already moves in the same windows.
- A dirty-build explanation: all low-speed rows came from `dirty=false` routes.
- A validated PI explanation: low-speed rows appear under unknown and weak PI; golden PI has no comparable low-speed evidence, so absence is not proof of a fix.

Smallest next experiment:

1. Re-run a low-speed stage-lag audit for the top 20 rows using model_y20, orientationRate-derived curvature, desired curvature, CP/CX1 desired, CP/CX1 final, yaw, and steering.
2. Add lead/stop-approach context if available from existing logs; if unavailable, run one fixed stop/creep corridor at 2-4 mph and 6-8 mph with the same model/PI and CP/CX1 telemetry enabled.

## 10-70 mph Weave

### Observed Symptom

The catalog contains 1,482 weave windows from 104 routes. Median speed is 49.36 mph. Path curvature slow-band RMS median is 1.97e-4; p95 is 4.57e-4; max is 13.10e-4.

### Inferred Stage

| Stage | Rows |
| --- | ---: |
| `model_or_desired` | 1,482 |

Stage localization is consistent across controller-telemetry availability:

| CP available | CX1 available | Rows | Stage |
| --- | --- | ---: | --- |
| yes | no | 732 | all `model_or_desired` |
| yes | yes | 712 | all `model_or_desired` |
| no | no | 38 | all `model_or_desired` |

Band-ratio summary:

| Metric | Median |
| --- | ---: |
| command/path RMS | 0.939 |
| desired/path RMS | 0.940 |
| command/desired RMS | 1.000 |
| steer/path | 0.401 |
| model_y20 slow-band RMS | 0.034 m |
| spectral peak | 0.200 Hz |

Interpretation: the 10-70 mph weave appears upstream of final vehicle response and usually upstream of controller-only stages. The controller is not cleanly exonerated, because it can pass or shape the motion, but the current evidence does not point to PI/final-command generation as the first source.

### Top Supporting Windows

| Route | Start s | Speed mph | Path RMS e-4 | Steering RMS deg | Command RMS e-4 | Desired RMS e-4 | model_y20 RMS m | PI | Commit | CP | CX1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| route_b5 | 1038.5 | 31.64 | 13.10 | 4.34 | 16.27 | 16.27 | 0.288 | weak | 5e0785b9 | yes | yes |
| route_61 | 1215.4 | 29.41 | 11.49 | 3.75 | 11.28 | 11.28 | 0.234 | unknown | c594321b | yes | no |
| route_a8 | 1430.1 | 29.87 | 9.07 | 3.05 | 11.13 | 11.13 | 0.196 | weak | 5e0785b9 | yes | yes |
| route_92 | 1085.2 | 46.71 | 8.65 | 3.20 | 10.65 | 10.65 | 0.157 | unknown | 17cb595f | yes | yes |
| route_3d | 2248.4 | 34.67 | 7.50 | 2.52 | 6.55 | 6.55 | 0.137 | unknown | 748d6c5b | yes | no |
| route_10 | 312.8 | 38.87 | 7.26 | 2.53 | 6.21 | 6.21 | 0.135 | unknown | 645af4ba | yes | no |

### Speed Pattern

| Speed bin mph | Rows | Routes | Median path RMS e-4 | p90 path RMS e-4 |
| --- | ---: | ---: | ---: | ---: |
| 10-20 | 30 | 14 | 4.19 | 5.99 |
| 20-30 | 58 | 28 | 3.71 | 5.36 |
| 30-40 | 233 | 63 | 3.27 | 4.66 |
| 40-50 | 451 | 77 | 2.27 | 3.63 |
| 50-60 | 456 | 81 | 1.53 | 2.49 |
| 60-70 | 254 | 73 | 0.97 | 1.64 |

### Config / Era Comparisons

These are descriptive only. They are not speed/location matched and cannot prove a lever.

| Group | Rows | Routes | Median path RMS e-4 | p90 path RMS e-4 |
| --- | ---: | ---: | ---: | ---: |
| PI unknown | 949 | 71 | 1.98 | 3.98 |
| Weak PI proven | 495 | 30 | 1.95 | 3.95 |
| Golden PI proven | 38 | 3 | 1.83 | 3.68 |
| steerRatio 17.2 | 1,418 | 99 | 1.98 | 4.00 |
| steerRatio 18.0 | 26 | 3 | 1.28 | 2.73 |
| steerRatio 16.8, delay 0.25 | 23 | 1 | 1.72 | 2.86 |
| steerRatio 16.8, delay 0.20 | 15 | 1 | 2.07 | 2.86 |
| Dirty build false | 1,437 | 100 | 1.97 | 3.97 |
| Dirty build true | 45 | 4 | 1.91 | 3.65 |

Most common commit/dirty groups in weave rows:

| Branch / commit | Dirty | Rows | Routes | Median path RMS e-4 | p90 path RMS e-4 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2021_explorer_st-mici / 5e0785b9 | false | 304 | 12 | 2.11 | 4.03 |
| 2021_explorer_st-mici / c594321b | false | 223 | 15 | 2.02 | 3.73 |
| 2021_explorer_st-mici / 17cb595f | false | 188 | 15 | 2.47 | 4.25 |
| 2021_explorer_st-mici / 046780c5 | false | 115 | 6 | 1.79 | 3.93 |
| 2021_explorer_st-mici / 748d6c5b | false | 114 | 9 | 1.65 | 3.33 |
| 2021_explorer_st-mici / 5e0785b9 | true | 45 | 4 | 1.91 | 3.65 |

Evidence argues against:

- PI as first source: every weak/golden/unknown PI weave row still localizes to `model_or_desired`.
- Final command as first source: command and desired slow-band RMS are nearly identical; final-command-specific stages do not emerge as the first-growth label in the current report.
- Dirty state as a first-order explanation: dirty and clean rows have similar descriptive distributions, and the worst rows include clean builds.

Smallest next experiment:

1. Build the missing location/speed-matched stage report from the existing route caches, not a driving-code patch.
2. Gate or label lead-follow context. The current QA explicitly leaves lead/radar contamination unresolved.
3. For the highest-overlap route cells, compare model_y20/orientationRate/desired/CP/CX1/final command/yaw/steer phase and amplitude at the same speed band.
4. Only if existing logs cannot isolate model/path versus controller pass-through, run a controlled corridor A/B with PI fixed and model/path input varied.

## Known Confounders And Residual Risks

- Lead/radar extraction and headway gating are absent. Results must not claim independence from lead-follow behavior.
- `symptom_catalog.csv` does not surface full provenance, dirty-state caveats, or all sidecar metadata; this report joined sidecars manually.
- Route-scoped smoke runs are not fully route-isolated on a populated cache.
- JSON-only extraction failures are not promoted into `symptom_catalog.csv`; manifest shows two failed routes.
- The design requested separate stage-localization, historical-scorecard, April 6/8 case-study, and next-experiment artifacts; the current implementation provides the core catalog and a descriptive Markdown report, not those full artifacts.
- CP/CX1 availability is era-dependent: 75 routes have CP only, 45 have CP+CX1, and 27 have neither.
- Config recovery is incomplete: 104/147 routes have unknown PI config.
- Dirty/on-device edits remain a provenance risk, especially for the small golden-PI sample.
- Historical comparisons are not location matched; route counts are not independent randomized samples.

## Recommendation

**Do not change driving code yet.**

The current evidence is strong enough to prioritize an upstream model/path/desired-curvature investigation and to deprioritize blind PI/steer-ratio tuning. It is not strong enough to select a vehicle-control fix. The next deliverable should be a lead-aware, location/speed-matched stage-localization report. A controlled drive should follow only for factors that remain undecidable after that report.

## Commands Run

Primary evidence and verification commands used before report verification:

```bash
wc -l AGENTS.md EXPLORER_ST.md retrospective_lateral/README.md retrospective_lateral/qa/qa_cooperative.md retrospective_lateral/qa/qa_adversarial.md docs/superpowers/specs/2026-06-14-retrospective-lateral-weave-analysis-design.md docs/superpowers/plans/2026-06-14-retrospective-lateral-weave-analysis.md retrospective_lateral/results/reports/symptom_catalog.csv retrospective_lateral/results/reports/retrospective_lateral_report.md retrospective_lateral/results/cache/manifest.json
git status --short
find retrospective_lateral/results -maxdepth 3 -type f | sort
sed -n '1,220p' AGENTS.md
sed -n '1,180p' EXPLORER_ST.md
sed -n '1,120p' retrospective_lateral/README.md
sed -n '1,120p' retrospective_lateral/qa/qa_cooperative.md
sed -n '1,140p' retrospective_lateral/qa/qa_adversarial.md
sed -n '1,420p' docs/superpowers/specs/2026-06-14-retrospective-lateral-weave-analysis-design.md
sed -n '1,2600p' docs/superpowers/plans/2026-06-14-retrospective-lateral-weave-analysis.md
sed -n '1,120p' retrospective_lateral/results/reports/retrospective_lateral_report.md
.venv311/bin/python - <<'PY'  # pandas summary of symptom_catalog.csv row counts, columns, symptom/stage counts
.venv311/bin/python - <<'PY'  # JSON/NPZ metadata audit for manifest sidecars and CP/CX1 availability
.venv311/bin/python - <<'PY'  # joined catalog+manifest grouping by route, speed, PI, steer ratio, commit, dirty, telemetry
.venv311/bin/python - <<'PY'  # supplemental low-speed CP/CX1/desired/act stage peak-to-peak audit
git show -s --format='%h %ci %s' <selected_manifest_commits>
```

Post-write verification:

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests -q
# 84 passed in 3.71s

.venv311/bin/python - <<'PY'  # verified report exists, route/cache/catalog counts, stage counts, required phrases
# report_exists True
# manifest_routes 147
# manifest_success_true 145
# manifest_success_false 2
# npz_files 145
# catalog_rows 1557
# low_stage_counts: final_command_or_before 75
# weave_stage_counts: model_or_desired 1482

wc -l docs/superpowers/reports/2026-06-23-retrospective-lateral-root-cause-hypotheses.md retrospective_lateral/results/reports/symptom_catalog.csv retrospective_lateral/results/cache/manifest.json
# 255 report lines, 1558 catalog CSV lines, 3521 manifest JSON lines

find retrospective_lateral/results/cache -name 'route_*.npz' | wc -l
# 145

git check-ignore -v retrospective_lateral/results/cache/route_0c.npz retrospective_lateral/results/reports/symptom_catalog.csv retrospective_lateral/results/cache/manifest.json
# all three paths ignored by retrospective_lateral/results/.gitignore
```
