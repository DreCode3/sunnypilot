# 2026-06-24 Low-Speed Source-Geometry Decomposition

## Executive Summary

This pass decomposed the same three low-speed target rows from the raw-shape audit into geometry-source signals and controller-stage signals.

The evidence further narrows the likely low-speed root cause:

1. The steering swing is now best described as **short/near model path and lane-center geometry moving together, then desired curvature and CP final following that geometry**.
2. The model path is probably **not inventing a separate shape** in the strongest rows. At y10, model-minus-lane-center motion is small in focus bins, and model/lane-center correlation is high where model is present.
3. Lane width and road-edge width are less likely as primary sources. They move, but the steering/desired discriminator is lane-center/model alignment, not width alone.
4. Controller evidence still looks downstream. CP pre-rate-limit, rate-limited, and final are identical in these bins; rate-limit flags, anti-windup, override, and steering-pressed are all zero in focus slices.
5. `route_3d@1616.8` remains the strongest counterexample: late model-absent bins have larger desired/CP than the early focus bins, but lower steering amplitude.

Recommendation: **do not change driving code yet**. The next smallest step is to run this same source-geometry classifier over all 26 horizon-limited low-speed rows, then inspect camera frames for the top lane-center/model source cases if the files are available.

## Generated Artifacts

| Artifact | Rows/files | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_bins.csv` | 440 rows | 0.5-second bins x 5 lookaheads with model, lane, road-edge, width, and derived geometry deltas. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_slices.csv` | 115 rows | Event/focus/nonfocus/high plus route-specific comparison slices x lookahead. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_signal_corr.csv` | 225 rows | Geometry-source lag/correlation to desired and steering. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_controller_slices.csv` | 759 rows | Controller-stage RMS/ratio/flag metrics by slice. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_controller_signal_corr.csv` | 63 rows | Controller-stage lag/correlation to desired and steering. |
| `retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_summary.csv` | 3 rows | Episode-level summary for focus/nonfocus discriminators. |
| `retrospective_lateral/results/reports/low_speed_source_geometry/*_source_geometry.svg` | 3 files | Timelines for steering/desired, y10 geometry, y20 geometry, width/quality, and controller stages. |

Generated artifacts are under ignored `retrospective_lateral/results/` paths.

## Method

No vehicle-control code was modified. Signals came from the existing NPZ cache.

- Band-pass: `0.08-0.80 Hz`
- Lookaheads: `5, 10, 15, 20, 30 m`
- Geometry sources: model path, lane left/right/center, lane width, road-edge left/right/center, road-edge width, model-minus-lane-center, model-minus-road-center, and lane-center-minus-road-center.
- Controller sources: desired, CP desired/predicted/EMA/pre-rate-limit/rate-limited/final, act/current/controls curvature, and CX1 equivalents where present.
- Flag checks: CP rate limit, anti-windup, override, reset, steering pressed, lateral active.

## Episode Summary

| Episode | Focus steer abs deg | Nonfocus steer abs deg | Focus desired RMS 1e4 | Focus CP/desired | Focus model y10 finite | Focus model y20 finite | y10 model-lane corr | y20 model-lane corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_6b@2329.1` | 5.12 | 1.32 | 19.94 | 0.871 | 0.833 | 0.083 | 0.984 | 0.796 |
| `route_3d@1616.8` | 1.83 | 0.92 | 9.51 | 0.741 | 1.000 | 0.167 | 0.882 | -0.987 |
| `route_8d@1286.6` | 4.69 | 1.85 | 13.38 | 1.027 | 1.000 | 0.550 | 0.818 | 0.711 |

Interpretation: focus bins consistently have higher steering amplitude and short model availability. Where y10 model is present, it closely follows lane-center motion.

## Source Findings

### `route_6b@2329.1`

Observed symptom:

- Focus steering-band abs mean is `5.12 deg` vs `1.32 deg` outside focus bins.
- The strongest source slice is short-only bins `10-13`: desired RMS `22.85e-4`, CP final RMS `20.07e-4`, steering abs `5.44 deg`.

Inferred stage:

- y10 model is fully present in short-only focus bins and tracks lane center strongly: model/lane-center correlation `0.987`.
- y10 model-minus-lane-center band abs mean is small: `0.018 m`.
- y20 model is absent in short-only bins, but lane-center y20 remains large: band abs mean `0.111 m`.
- Lane-center raw slope is much larger in focus short-only bins than outside focus: y20 `0.130 m/s` vs `0.001 m/s`.

Root-cause hypothesis:

- Strong support for **lane-center/near-model geometry source**. The y10 model path appears to follow lane-center motion rather than create an independent path.

Evidence against alternatives:

- Lane width moves less than lane center at y20: focus y20 lane width band abs `0.037 m` vs lane-center `0.093 m`.
- Road-edge center is not a consistent driver here: focus y10 lane-center/road-center correlation is only `0.071`, y20 only `0.164`.
- CP final is attenuated and unchanged across pre/rate/final stages; rate-limit flag, anti-windup, override, and steering pressed are all zero.

Next experiment:

- Run the same y10/y20 model-lane-center comparison across all 26 horizon-limited rows and rank rows by focus lane-center motion plus y10 model-lane correlation.

### `route_3d@1616.8`

Observed symptom:

- Early focus bins `0-2`: steering abs `1.83 deg`, desired RMS `9.51e-4`, CP RMS `7.05e-4`.
- Late model-absent bins `6-9`: steering abs `0.98 deg`, desired RMS `17.37e-4`, CP RMS `13.02e-4`.

Inferred stage:

- Early focus bins have model y10 fully present and y20 briefly present; late bins have no model signal.
- Early y10 lane-center band abs is `0.043 m`; late y10 lane-center band abs is `0.016 m`.
- Early y20 lane-center band abs is `0.090 m`; late y20 lane-center band abs is `0.033 m`.

Root-cause hypothesis:

- This is the clearest discriminator: high steering needs the short-model/lane-center condition, not just large desired/CP.

Evidence against alternatives:

- The larger late desired/CP burst does not produce the early high steering amplitude.
- CP final remains attenuated in both early and late slices: early CP/desired `0.741`, late `0.749`.
- Rate-limit flag, anti-windup, override, and steering pressed are zero.

Next experiment:

- Use `route_3d` as the template for a classifier: "high steering when model y10 is present and lane-center band is elevated" vs "desired high with model absent."

### `route_8d@1286.6`

Observed symptom:

- Focus steering abs is `4.69 deg` vs `1.85 deg` outside focus bins.
- Forward-focus bins `5-6`: steering abs `3.66 deg`, desired RMS `6.15e-4`, CP/desired `1.535`.
- Short-focus bins `8-9`: steering abs `5.72 deg`, desired RMS `17.89e-4`, CP/desired `0.949`.

Inferred stage:

- The short-only portion is the stronger steering/desired event.
- y10 model remains present in both forward and short focus slices.
- Short-focus y10 model/lane-center correlation is `0.950`; y20 model is mostly absent in short-focus bins.
- y20 lane-center grows sharply in the short-focus slice: `0.138 m` band abs vs `0.019 m` in forward-focus bins.

Root-cause hypothesis:

- Strong support for lane-center/model geometry becoming severe during the transition to short-only horizon.
- y20/y30 dropout is context; y10 model and lane-center geometry are the usable source boundary.

Evidence against alternatives:

- The only obvious CP amplification occurs in forward-focus bins, but steering and desired are larger in the later short-focus bins where CP/desired is below 1.0.
- CX1 command is available and tracks CP/desired rather than revealing a separate command source.
- Rate-limit flag, anti-windup, override, and steering pressed are zero.

Next experiment:

- Use `route_8d` to validate the controller-stage conclusion because CX1 telemetry is present. If all 26-row analysis finds similar geometry but missing CX1 telemetry, `route_8d` is the best controller cross-check.

## Controller-Stage Result

| Episode | Focus desired RMS 1e4 | Focus CP desired/desired | Focus CP final/desired | CP pre/rate/final equality | Rate-limit flag | Anti-windup | Override | Steering pressed | CX1 command/desired |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `route_6b@2329.1` | 19.94 | 0.877 | 0.871 | yes | 0.0 | 0.0 | 0.0 | 0.0 | unavailable |
| `route_3d@1616.8` | 9.51 | 0.672 | 0.741 | yes | 0.0 | 0.0 | 0.0 | 0.0 | unavailable |
| `route_8d@1286.6` | 13.38 | 0.943 | 1.027 | yes | 0.0 | 0.0 | 0.0 | 0.0 | 1.016 |

The controller boundary does not show clipping, rate limiting, override, anti-windup, reset, or driver input as the first-growth source. CP final usually attenuates desired. `route_8d` has a small focus CP-final gain above desired, but the route-specific split shows the largest CP gain in forward-focus bins while the largest steering/desired occurs later in short-focus bins.

## Ranked Hypotheses After This Step

| Rank | Hypothesis | Confidence | Evidence now | Evidence against / missing |
| ---: | --- | --- | --- | --- |
| 1 | Lane-center/near-model geometry is the immediate upstream source. | High for these three rows. | Focus bins have elevated lane-center motion and short-model presence. y10 model follows lane center strongly where present. `route_3d` late desired/CP-only burst is a counterexample that lowers steering. | Needs validation over all 26 horizon-limited rows. Numeric geometry does not prove visual scene cause. |
| 2 | Model horizon loss is a marker/enabler, not the direct cause. | Medium-high. | y20/y30 often disappear, but high bins retain y5/y10 model. `route_8d` high bins include forward and short-only states. | The transition may still change planner behavior; this pass did not run an intervention. |
| 3 | Road-edge/scene geometry is the visual source behind lane-center motion. | Medium-low. | Road-edge widths and centers move in some rows, especially `route_8d` at longer lookahead. | Road-edge center correlation is weak/inconsistent in `route_6b` and `route_3d`; road-edge widths are large and likely noisy. Camera frames are needed. |
| 4 | Controller/final-command behavior is the primary source. | Low. | CP/CX1 are correlated with steering, and `route_8d` has a small focus gain. | Pre/rate/final are identical, rate-limit and anti-windup flags are zero, and `route_3d` has larger late desired/CP with lower steering. |
| 5 | Lane-width-only instability is the source. | Low. | Width changes exist. | Width is not the best discriminator; late `route_3d` has width motion and larger desired/CP but lower steering, while lane-center/model presence separates the symptom better. |

## Recommendation

Do not change driving code yet.

The next logical workflow step is analysis-only:

1. Run the source-geometry classifier over all 26 horizon-limited low-speed rows.
2. Rank rows by focus-vs-nonfocus lane-center amplitude, y10 model availability, y10 model-lane correlation, and `route_3d`-style desired-high/model-absent counterexamples.
3. If camera files are available on-device, extract frames for the top-ranked source-geometry rows to identify the visual scene pattern behind lane-center movement.

Only after the 26-row validation confirms this pattern should an implementation plan be considered.

## Commands Run

```bash
sed -n '1,240p' docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-raw-shape-audit.md
.venv311/bin/python - <<'PY'  # generated source-geometry bins, slices, correlations, controller slices, controller correlations, summary, and SVGs
.venv311/bin/python - <<'PY'  # printed episode-level summary and focus source metrics
.venv311/bin/python - <<'PY'  # printed geometry-source correlation rankings to desired and steering
.venv311/bin/python - <<'PY'  # printed key route-specific y10/y20 geometry slices
.venv311/bin/python - <<'PY'  # printed controller event and slice decomposition
.venv311/bin/python - <<'PY'  # printed route_3d early-vs-late counterexample contrast
find retrospective_lateral/results/reports/low_speed_source_geometry -maxdepth 1 -type f -name '*source_geometry.svg' -print -exec wc -c {} \\;
.venv311/bin/python - <<'PY'  # verified generated CSV row counts, SVG XML parsing, and report existence
git check-ignore -v retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_bins.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_slices.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_signal_corr.csv retrospective_lateral/results/reports/drilldown_low_speed_source_controller_slices.csv retrospective_lateral/results/reports/drilldown_low_speed_source_controller_signal_corr.csv retrospective_lateral/results/reports/drilldown_low_speed_source_geometry_summary.csv retrospective_lateral/results/reports/low_speed_source_geometry/route_6b_2329_source_geometry.svg retrospective_lateral/results/reports/low_speed_source_geometry/route_3d_1617_source_geometry.svg retrospective_lateral/results/reports/low_speed_source_geometry/route_8d_1287_source_geometry.svg
git diff --check -- docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-source-geometry-decomposition.md
git status --short opendbc_repo panda
.venv311/bin/python -m pytest retrospective_lateral/tests -q
```
