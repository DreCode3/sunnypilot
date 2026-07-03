# 2026-06-23 Lane Geometry Audit

## Executive Summary

Step 4 rebuilt the retrospective cache as `retrolat-v4` with multi-lookahead model, lane-line, lane-center, lane-width, and road-edge channels at 0, 10, 20, and 30 m. The drilldown now writes:

| Output | Rows | Scope |
| --- | ---: | --- |
| `drilldown_lane_geometry_audit.csv` | 7,956 | Episode and eligible lead-event pre/post lane-geometry metrics across four lookaheads. |

The evidence strengthens the upstream lane/corridor geometry hypothesis for the 10-70 mph weave. Top-decile weave has median model-to-lane-center correlation of 0.828 at 20 m and 0.934 at 30 m. At 30 m, 87.9% of top-decile weave rows have absolute model/lane-center correlation >= 0.8.

Recommendation remains: **do not change driving code yet.**

## Cache And Output Verification

| Check | Result |
| --- | ---: |
| Manifest rows | 147 |
| `retrolat-v4` rows | 147 |
| Successful route caches | 145 |
| Zero-sample route caches | 2 |
| Symptom rows | 1,557 |
| Lane geometry audit rows | 7,956 |
| Episode audit rows | 6,228 |
| Lead-event pre/post rows | 1,728 |

The zero-sample routes remain `route_67` and `route_68`, consistent with prior residual risks.

## Weave Evidence

Top-decile weave threshold: path curvature band RMS >= 3.972 x1e-4.

| Subset | Lookahead m | Episodes | Median Model/Lane Corr | Abs Corr >= 0.8 | Lane Center / Model RMS | CP Final / Desired RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All weave | 10 | 1,482 | 0.534 | 18.5% | 5.62 | 0.838 |
| All weave | 20 | 1,482 | 0.749 | 41.6% | 2.09 | 0.838 |
| All weave | 30 | 1,482 | 0.877 | 66.4% | 1.41 | 0.838 |
| Top-decile weave | 10 | 149 | 0.494 | 27.5% | 2.73 | 0.866 |
| Top-decile weave | 20 | 149 | 0.828 | 54.4% | 1.24 | 0.866 |
| Top-decile weave | 30 | 149 | 0.934 | 87.9% | 1.09 | 0.866 |

The 0 m lookahead is not useful for lane/model RMS ratios because model lateral offset near the ego origin is often near zero; the meaningful signal is at 20-30 m.

Interpretation: the most severe weave is strongly tied to lane-center/model-path movement at forward lookaheads. CP final command remains below desired curvature, so the controller/final-command path is still not the best root-cause target.

## Top Episode Evidence

| Target | Lookahead m | Lead Near | Lane Prob Min | Model RMS m | Lane Center RMS m | Model/Lane Corr | Lane/Model RMS | CP/Desired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_b5@1038` | 20 | 0.927 | 0.812 | 0.303 | 0.348 | 0.952 | 1.15 | 0.873 |
| `route_b5@1038` | 30 | 0.927 | 0.812 | 0.545 | 0.564 | 0.982 | 1.04 | 0.873 |
| `route_61@1215` | 20 | 1.000 | 0.875 | 0.238 | 0.151 | 0.961 | 0.636 | 0.833 |
| `route_61@1215` | 30 | 1.000 | 0.875 | 0.455 | 0.367 | 0.989 | 0.807 | 0.833 |
| `route_a8@1430` | 20 | 0.516 | 0.565 | 0.193 | 0.192 | 0.879 | 0.997 | 0.853 |
| `route_a8@1430` | 30 | 0.516 | 0.565 | 0.354 | 0.348 | 0.964 | 0.983 | 0.853 |
| `route_92@1085` | 20 | 0.429 | 0.536 | 0.158 | 0.200 | 0.949 | 1.27 | 0.822 |
| `route_92@1085` | 30 | 0.429 | 0.536 | 0.275 | 0.306 | 0.985 | 1.11 | 0.822 |

The four top weave targets are consistently lane-center coupled at 20-30 m. Two rows have only moderate lane-probability medians, but the coupling is also strong in higher-lane-probability rows, so this is not just a low-probability artifact.

## Low-Speed Evidence

Top-decile low-speed threshold: steering peak-to-peak >= 14.46 deg.

| Subset | Lookahead m | Episodes | Median Model/Lane Corr | Abs Corr >= 0.8 | Lane Center / Model RMS | CP Final / Desired RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All low-speed | 20 | 75 | 0.726 | 28.0% | 1.22 | 0.876 |
| All low-speed | 30 | 75 | 0.778 | 28.0% | 1.16 | 0.876 |
| Top-decile low-speed | 20 | 8 | 0.658 | 37.5% | 1.26 | 0.901 |
| Top-decile low-speed | 30 | 8 | 0.795 | 25.0% | 1.18 | 0.901 |

Low-speed evidence is more mixed than weave. `route_8d@1436` is strongly lane-center coupled at 20-30 m, while `route_6b@2317` remains a distinct subcase with weak/negative 20 m model/lane coupling and missing 30 m model-path data in the audited window.

| Target | Lookahead m | Lead Near | Lane Prob Min | Model RMS m | Lane Center RMS m | Model/Lane Corr | Lane/Model RMS | CP/Desired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_8d@1436` | 20 | 0.000 | 0.854 | 0.120 | 0.128 | 0.984 | 1.07 | 0.893 |
| `route_8d@1436` | 30 | 0.000 | 0.854 | 0.156 | 0.209 | 0.975 | 1.34 | 0.893 |
| `route_6b@2317` | 20 | 0.000 | 0.864 | 0.021 | 0.079 | -0.242 | 3.74 | 0.880 |
| `route_6b@2317` | 30 | 0.000 | 0.864 | NaN | 0.129 | NaN | NaN | 0.880 |

Interpretation: low-speed wheel swing likely has at least two subcases. `route_8d` fits the corridor-geometry hypothesis; `route_6b` still needs visual confirmation or deeper orientation/path-source inspection.

## Lead Event Evidence

At 20 m, eligible lead-onset events overlapping weave still do not show a median immediate upstream increase:

| Metric | Median Post-vs-Pre Delta |
| --- | ---: |
| Model y20 RMS | -9.45% |
| Lane-center y20 RMS | -2.53% |
| Lane-width y20 RMS | -15.48% |
| Road-edge-width y20 RMS | -13.73% |
| Path curvature RMS | -3.85% |
| Desired curvature RMS | -7.08% |
| CP final RMS | -7.10% |

Positive fractions remain mixed: model 45.6%, lane center 43.7%, path 44.7%, desired 44.7%, CP final 45.6%.

The positive outliers are real, but they grow in lane geometry as well as model/path/controller stages:

| Target | Model Delta | Lane-Center Delta | Lane-Width Delta | Road-Width Delta | Path Delta | Desired Delta | CP Delta | Post Lane Corr | Post CP/Desired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_59@2273` | +282% | +334% | +92% | +34% | +298% | +363% | +322% | 0.743 | 0.848 |
| `route_ad@2648` | +200% | +206% | -22% | +6% | +127% | +234% | +236% | 0.944 | 0.760 |
| `route_4d@2930` | +238% | +207% | +12% | +41% | +252% | +225% | +210% | 0.620 | 0.971 |
| `route_39@187.9` | +166% | +605% | -32% | -16% | +183% | +151% | +152% | -0.291 | 0.954 |

Interpretation: near-lead remains a context/co-occurrence factor. The positive outliers do not isolate lead tracking or final command; they point to upstream geometry/path changes during close-follow contexts.

## Root-Cause Ranking Impact

| Rank | Hypothesis | Step 4 Impact |
| ---: | --- | --- |
| 1 | Model/path/corridor geometry is upstream of the visible weave. | Strengthened to high confidence for 10-70 mph weave. |
| 2 | Lane/corridor perception quality or geometry contributes to severity. | Strengthened. Top-decile weave is strongly coupled at 20-30 m. |
| 3 | Near lead/headway co-occurs with weave but is not sufficient. | Refined. Median lead-onset event deltas remain negative, but positive outliers are lane-geometry coupled. |
| 4 | Low-speed swing is a low-speed model/corridor/desired artifact followed by the actuator. | Partly strengthened. Fits `route_8d`; mixed for `route_6b`. |
| 5 | PI/final-command tuning is primary. | Further weakened. CP final remains below desired in the audited aggregates and targets. |

## Next Step

The smallest useful next step is targeted visual/model replay for the geometry-coupled cases, not a vehicle-control implementation plan.

Prioritize:

1. `route_b5@1038`, `route_61@1215`, `route_a8@1430`, and `route_92@1085` for 20-30 m lane/model overlay inspection.
2. `route_8d@1436` for low-speed construction-corridor visual confirmation.
3. `route_6b@2317` as the main low-speed counterexample requiring orientation/path-source inspection.
4. The positive lead-onset outliers only after geometry overlays are available, because their numeric signature already shows upstream lane/model growth.

Do not change driving code until this distinguishes perception/corridor motion from planner/controller motion in visual/model replay or a controlled drive.

## Commands Run

```bash
.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract.py::test_resample_channels_extracts_multi_lookahead_lane_geometry_and_road_edges -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py::test_lane_geometry_audit_reports_per_lookahead_model_lane_coupling -q
.venv311/bin/python -m pytest retrospective_lateral/tests/test_drilldown.py::test_build_drilldown_outputs_writes_expected_artifacts -q
.venv311/bin/python -m pytest retrospective_lateral/tests -q
.venv311/bin/python -m retrospective_lateral.code.run_all
.venv311/bin/python -m retrospective_lateral.code.drilldown
.venv311/bin/python - <<'PY'  # manifest/cache field checks and lane-audit summaries
git check-ignore -v retrospective_lateral/results/reports/drilldown_lane_geometry_audit.csv retrospective_lateral/results/cache/manifest.json retrospective_lateral/results/cache/route_b5.npz
```
