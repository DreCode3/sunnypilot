# 2026-06-24 Retrospective Lateral Full-Analysis Handoff

## Purpose

This file saves the current investigation state for a fresh context session. It indexes the generated retrospective lateral reports, CSV/JSON/PNG artifacts, current evidence summary, known gaps, and a ready-to-paste prompt for a new analysis session.

The goal for the next session is to independently review all generated evidence and produce the most logical root-cause conclusion plus correction recommendations. It should not rely on conversation memory.

## Repository

Repository:

```text
/Users/dregilley/Documents/GitHub/sunnypilot
```

Important constraints:

- Use `superpowers:systematic-debugging`.
- Do not modify vehicle-control code.
- Do not modify `opendbc_repo/` or `panda/`.
- Do not perform remote comma update/install/reboot actions.
- If any future remote comma action becomes necessary, follow `AGENTS.md` offroad-safe detached workflow.
- Generated files under `retrospective_lateral/results/` should remain ignored and unstaged.
- Local Python/pandas analysis over existing CSV/JSON/NPZ outputs is allowed.

## Required Reading

Read these first:

- `AGENTS.md`
- `EXPLORER_ST.md`
- `retrospective_lateral/README.md`
- `retrospective_lateral/qa/qa_cooperative.md`
- `retrospective_lateral/qa/qa_adversarial.md`
- `docs/superpowers/specs/2026-06-14-retrospective-lateral-weave-analysis-design.md`
- `docs/superpowers/plans/2026-06-14-retrospective-lateral-weave-analysis.md`

Then read these reports in order:

1. `docs/superpowers/reports/2026-06-23-retrospective-lateral-root-cause-hypotheses.md`
2. `docs/superpowers/reports/2026-06-23-retrospective-lateral-drilldown-stage-analysis.md`
3. `docs/superpowers/reports/2026-06-23-retrospective-lateral-lead-aware-drilldown.md`
4. `docs/superpowers/reports/2026-06-23-retrospective-lateral-raw-timeline-audit.md`
5. `docs/superpowers/reports/2026-06-23-retrospective-lateral-lane-geometry-audit.md`
6. `docs/superpowers/reports/2026-06-24-retrospective-lateral-visual-model-replay.md`
7. `docs/superpowers/reports/2026-06-24-retrospective-lateral-route-6b-low-speed-counterexample.md`
8. `docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-model-horizon-audit.md`
9. `docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-short-lookahead-audit.md`
10. `docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-horizon-transition-audit.md`
11. `docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-raw-shape-audit.md`
12. `docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-source-geometry-decomposition.md`
13. `docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-26row-source-geometry-validation.md`
14. `docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-visual-camera-inventory.md`

## Core Data Artifacts

Main corpus:

| Artifact | Shape / size | SHA-256 prefix | Purpose |
| --- | ---: | --- | --- |
| `retrospective_lateral/results/reports/symptom_catalog.csv` | 1557 rows x 22 cols | `bc373ab397f50c06` | Full symptom catalog: 75 low-speed wheel-swing rows and 1482 10-70 mph weave rows in current rebuilt output. |
| `retrospective_lateral/results/reports/retrospective_lateral_report.md` | 4108 bytes | `aa269ab7e40dd40c` | Main generated retrospective report. |
| `retrospective_lateral/results/cache/manifest.json` | 102469 bytes | `4e5ea6c9df12a35d` | Cache manifest/provenance for route NPZ files. |
| `retrospective_lateral/results/reports/drilldown_stage_gain_lag_summary.csv` | 234 rows x 27 cols | `5948e18d60661f7d` | Stage gain/lag summary used to separate upstream vs final-command behavior. |

Low-speed hierarchy:

| Artifact | Shape | SHA-256 prefix | Purpose |
| --- | ---: | --- | --- |
| `retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_audit.csv` | 75 rows x 61 cols | `1e9ede5fb9d2217c` | All low-speed rows split by model-horizon availability and root-cause bucket. |
| `retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_group_summary.csv` | 21 rows x 18 cols | `8159c5e1ad48d466` | Grouped horizon/root-cause summary. |
| `retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_route_summary.csv` | 37 rows x 10 cols | `c23fd890508bbf75` | Low-speed route-level summary. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_enriched.csv` | 450 rows x 90 cols | `77155bb3821116c8` | Short-lookahead enriched evidence. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_summary.csv` | 12 rows x 22 cols | `5594bc6d76dc91a1` | Short-lookahead grouped summary. |
| `retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_top_episodes.csv` | 72 rows x 28 cols | `89eb9cfd47050438` | Top low-speed episode/lookahead rows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_episode_summary.csv` | 26 rows x 68 cols | `62dd3278dd2604c0` | 26 horizon-limited low-speed target rows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_bins.csv` | 265 rows x 75 cols | `e6d2b96c817efb7e` | One-second bins for 26 horizon-limited rows. |

Low-speed source geometry:

| Artifact | Shape | SHA-256 prefix | Purpose |
| --- | ---: | --- | --- |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_summary.csv` | 3 rows x 107 cols | `c1a7b2d38990b785` | Three-row source-geometry decomposition. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_bins.csv` | 440 rows x 66 cols | `86d7ee836f2aee75` | Three-row source-geometry bins. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_slices.csv` | 115 rows x 117 cols | `aa14bcd524208b8e` | Three-row source/controller slices. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_summary.csv` | 26 rows x 152 cols | `59573dd9666ff035` | 26-row source-geometry classifier summary. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_ranked.csv` | 26 rows x 152 cols | `59573dd9666ff035` | Same 26 rows ranked by lane/model source score. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_bins.csv` | 265 rows x 32 cols | `369cccce24a734b5` | Source-geometry one-second bins for 26 rows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_counterexamples.csv` | 40 rows x 12 cols | `f968f91d581ae877` | Desired-high/model-absent/low-steering counterexample bins. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_signal_corr.csv` | 468 rows x 17 cols | `cd5662709aeb5cda` | 26-row signal correlation/lag table. |

Route 6b counterexample:

| Artifact | Shape | SHA-256 prefix | Purpose |
| --- | ---: | --- | --- |
| `retrospective_lateral/results/reports/drilldown_route_6b_low_speed_counterexample_signal_summary.csv` | 40 rows x 25 cols | `cbc5a71000e6edfc` | Route 6b signal-stage drilldown. |
| `retrospective_lateral/results/reports/drilldown_route_6b_low_speed_counterexample_bins.csv` | 20 rows x 44 cols | `f7622206d0f942b0` | Route 6b one-second bins. |
| `retrospective_lateral/results/reports/drilldown_route_6b_low_speed_counterexample_finite_regions.csv` | 10 rows x 6 cols | `e60a5ffc298e4334` | Finite model-path regions. |
| `retrospective_lateral/results/reports/drilldown_route_6b_low_speed_all_episodes_summary.csv` | 3 rows x 28 cols | `a8cd8a39fff8cfe5` | All route 6b low-speed rows. |
| `retrospective_lateral/results/reports/drilldown_route_6b_low_speed_model_horizon_summary.csv` | 4 rows x 12 cols | `529c449f446fe0a9` | Direct route 6b modelV2 horizon scan. |

Visual and replay evidence:

| Artifact | Shape / files | SHA-256 prefix | Purpose |
| --- | ---: | --- | --- |
| `retrospective_lateral/results/reports/drilldown_raw_timeline_audit.csv` | 36 rows x 103 cols | `5d6a4ad64430d6a9` | Raw timeline audit for top weave and low-speed examples. |
| `retrospective_lateral/results/reports/drilldown_visual_model_replay_summary.csv` | 24 rows x 32 cols | `ed83bca08937d5a9` | Visual/model replay summary for top weave and low-speed rows. |
| `retrospective_lateral/results/reports/low_speed_visual_inventory.csv` | 6 rows x 11 cols | `d7fece931bd17ca0` | Low-speed camera availability inventory. |
| `retrospective_lateral/results/reports/low_speed_visual_frame_plan.csv` | 30 rows x 9 cols | `7dcd399fd5323eb6` | Planned/extracted frame rows. |
| `retrospective_lateral/results/reports/low_speed_visual_inventory/*.png` | 11 PNG files | not listed here | 9 route_8d `fcamera` stills and 2 contact sheets. |

## Current Evidence Summary

Separate observed symptom, inferred stage, root-cause hypothesis, and missing evidence.

### Low-Speed Steering-Wheel Swing, 1-10 mph

Observed symptom:

- Low-speed steering-wheel swing / peak-to-peak wheel motion, strongest in top rows like `route_6b@2329.1`, `route_3d@1616.8`, `route_8d@1286.6`, and related horizon-limited rows.

Inferred stage:

- The symptom is usually present before or at desired-curvature generation, not introduced by PI/final command.
- CP final/final-command generally follows or attenuates desired curvature.
- The 26-row source-geometry classifier finds:
  - 12/26 rows as strong or partial lane-center/y10-model source rows.
  - 15/26 rows with focus/nonfocus lane-center y10 amplitude ratio > 1.
  - 17/26 rows with focus/nonfocus lane-center y20 amplitude ratio > 1.
  - 24/26 rows as controller downstream/attenuated.
  - 18/26 rows with at least one desired-high/model-absent/low-steering counterexample bin.
  - 40 total desired-high/model-absent/low-steering counterexample bins.

Root-cause hypothesis:

- Most likely immediate source: near/short-horizon lane-center/model-path perception geometry, often in constrained-corridor visual contexts.
- This is not one universal y10 model signature. Several rows are mixed, horizon-limited, or counterexample-classified.
- Model horizon loss is likely a context/enabler, not the direct amplitude trigger.
- Desired curvature and CP final command are stage markers/followers, not sufficient causes by themselves.

Missing or weak evidence:

- Camera exists locally only for `route_8d` target rows.
- `route_3d`, `route_9b`, `route_6b`, and `route_0d` target camera files were not present locally or on the device at `192.168.98.237`.
- Strongest numeric low-speed rows remain camera-unverified.

### Straight/Gentle-Section Weave, 10-70 mph

Observed symptom:

- Slow straight/gentle-section weave in the 10-70 mph range, with 1482 current catalog rows.

Inferred stage:

- Stage drilldowns repeatedly point to model/path/desired before final command.
- Top weave rows show strong model-y20/y30 and lane-center common-mode motion.
- CP/final command generally attenuates or follows desired, rather than introducing a new oscillation.

Root-cause hypothesis:

- Most likely source: upstream lane/corridor/model-path geometry.
- Near lead/headway appears to be a context/co-occurrence factor, not a sufficient standalone trigger.
- Visual `route_b5@1038` evidence shows a close lead, curb/right-edge constraints, and lane geometry changes, consistent with constrained corridor complexity rather than a clean controller-only oscillation.

Missing or weak evidence:

- Several top 10-70 mph routes still lack camera frames.
- Near-lead context is confounded with route and visual corridor complexity.

## Current Confidence Snapshot

These are starting-point estimates for the next session to challenge:

| Claim | Current confidence | Why |
| --- | ---: | --- |
| PI/final-command tuning is not the primary global root cause. | High, about 75-85% | CP final usually attenuates desired; controller flags/stages do not show first growth; many counterexamples have desired/CP growth without high steering. |
| 10-70 mph weave originates upstream in model/lane/path geometry. | High, about 80-85% | Strong top-row model/lane common-mode evidence, stage drilldown, and visual `route_b5` context. |
| Low-speed swing originates upstream of final command. | Medium-high, about 70-80% | Strong route_6b/route_3d/route_8d stage evidence and 24/26 downstream/attenuated controller classification. |
| Low-speed immediate source is near/short-horizon lane-center/model geometry. | Medium-high, about 65-75% | 12/26 strong/partial source rows and lane-center amplitude ratios in most rows; not universal. |
| The visual cause of low-speed swing is constrained corridor geometry. | Medium, about 45-60% | Supported by `route_8d` frames; strongest numeric rows remain camera-unverified. |

## New Session Prompt

Copy and paste this into a fresh context session:

```text
Use Superpowers systematic-debugging to perform a fresh full-analysis review of the retrospective lateral root-cause evidence for the 2021 Ford Explorer ST sunnypilot driving performance issues.

Repository: /Users/dregilley/Documents/GitHub/sunnypilot

Goal:
Review all generated retrospective lateral analysis reports and data artifacts from the prior sessions. Provide the most logical root-cause conclusion for:
1. Low-speed steering-wheel swing at 1-10 mph.
2. Straight/gentle-section weave at 10-70 mph.

Also provide recommendations for correcting the issue, but separate:
- evidence-backed correction direction,
- implementation hypotheses,
- smallest offline/log-analysis validation step,
- smallest on-road validation step if unavoidable,
- and what should NOT be changed yet.

Constraints:
- Do not modify vehicle-control code.
- Do not modify opendbc_repo/ or panda/.
- Do not perform remote comma update/install/reboot actions.
- Generated files under retrospective_lateral/results/ should remain ignored and unstaged.
- Use local Python/pandas commands over existing CSV/JSON/NPZ outputs as needed.
- If any new analysis code is needed, keep it separate from driving-code changes and explain why.

Read first:
- AGENTS.md
- EXPLORER_ST.md
- retrospective_lateral/README.md
- retrospective_lateral/qa/qa_cooperative.md
- retrospective_lateral/qa/qa_adversarial.md
- docs/superpowers/specs/2026-06-14-retrospective-lateral-weave-analysis-design.md
- docs/superpowers/plans/2026-06-14-retrospective-lateral-weave-analysis.md
- docs/superpowers/reports/2026-06-24-retrospective-lateral-full-analysis-handoff.md

Then read the reports listed in the handoff in order.

Primary data artifacts to verify with local commands:
- retrospective_lateral/results/reports/symptom_catalog.csv
- retrospective_lateral/results/cache/manifest.json
- retrospective_lateral/results/reports/drilldown_stage_gain_lag_summary.csv
- retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_audit.csv
- retrospective_lateral/results/reports/drilldown_low_speed_short_lookahead_enriched.csv
- retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_episode_summary.csv
- retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_bins.csv
- retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_summary.csv
- retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_bins.csv
- retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_counterexamples.csv
- retrospective_lateral/results/reports/drilldown_route_6b_low_speed_counterexample_signal_summary.csv
- retrospective_lateral/results/reports/drilldown_route_6b_low_speed_model_horizon_summary.csv
- retrospective_lateral/results/reports/drilldown_raw_timeline_audit.csv
- retrospective_lateral/results/reports/drilldown_visual_model_replay_summary.csv
- retrospective_lateral/results/reports/low_speed_visual_inventory.csv
- retrospective_lateral/results/reports/low_speed_visual_frame_plan.csv

Required analysis discipline:
- Do not trust prior conclusions without re-checking row counts and key group statistics.
- Explicitly separate observed symptom, inferred stage, root-cause hypothesis, and correction recommendation.
- Rank hypotheses for each symptom with confidence and evidence.
- Identify evidence that argues against each hypothesis.
- Identify confounders and missing evidence.
- For recommendations, do not jump directly to PI/final-command tuning unless evidence supports controller-stage first growth.
- If recommending model/path/perception-side correction, explain what observable evidence would confirm the recommendation before implementation.

Expected output:
- A Markdown report under docs/superpowers/reports/, dated 2026-06-24 or later.
- Executive summary.
- Ranked root-cause conclusion for low-speed swing.
- Ranked root-cause conclusion for 10-70 mph weave.
- A correction recommendation matrix:
  - correction idea,
  - target layer,
  - evidence supporting it,
  - risk,
  - smallest validation step,
  - whether ready for implementation planning.
- A clear recommendation: "do not change driving code yet" or "ready for implementation plan", with justification.
- Commands run and verification results.

Before finalizing:
- Run local commands to verify row counts for the core artifacts.
- Check generated results remain ignored if any new result files are created.
- Check opendbc_repo/ and panda/ are untouched.
- Report all commands run.
```

## Suggested Starting Checks For New Session

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot

.venv311/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
files = [
  'retrospective_lateral/results/reports/symptom_catalog.csv',
  'retrospective_lateral/results/reports/drilldown_stage_gain_lag_summary.csv',
  'retrospective_lateral/results/reports/drilldown_low_speed_model_horizon_audit.csv',
  'retrospective_lateral/results/reports/drilldown_low_speed_horizon_transition_episode_summary.csv',
  'retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_summary.csv',
  'retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_26row_counterexamples.csv',
  'retrospective_lateral/results/reports/drilldown_visual_model_replay_summary.csv',
  'retrospective_lateral/results/reports/low_speed_visual_inventory.csv',
]
for f in files:
    df = pd.read_csv(f)
    print(f'{f}: rows={len(df)} cols={len(df.columns)}')
PY

git status --short opendbc_repo panda
```

## Commands Run To Create This Handoff

```bash
find docs/superpowers/reports -maxdepth 1 -type f -name '2026-06-2*.md' -print | sort
.venv311/bin/python - <<'PY'  # printed core artifact shapes and SHA-256 prefixes
find retrospective_lateral/results/reports/low_speed_visual_inventory -maxdepth 1 -type f -name '*.png' -print | sort
find retrospective_lateral/results/reports -maxdepth 1 -type f | rg 'short_lookahead|low_speed_model_horizon|route_6b|source_geometry|horizon_transition|raw_shape|visual' | sort
.venv311/bin/python - <<'PY'  # printed low-speed drilldown artifact shapes and SHA-256 prefixes
```
