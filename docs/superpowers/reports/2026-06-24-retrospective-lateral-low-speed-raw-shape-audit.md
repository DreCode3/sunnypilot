# 2026-06-24 Low-Speed Raw-Shape Audit

## Executive Summary

This audit inspected per-sample raw signal shape for the three highest-value low-speed horizon-transition rows:

- `route_6b@2329.1`, focus bins `9,10,11,12,13,16`
- `route_3d@1616.8`, focus bins `0,1,2`
- `route_8d@1286.6`, focus bins `5,6,8,9`

The raw-shape evidence strengthens the current low-speed hypothesis:

1. The low-speed steering-wheel swing is still best explained as **upstream short-horizon geometry/lane-center motion propagating through desired curvature and final command**, not as a pure controller/final-command problem.
2. Long model horizon loss is a **context and transition marker**, not the primary trigger. The high-steering bins usually retain 5-10 m model signal and lane-center signal while 20-30 m model path is sparse or absent.
3. `route_3d@1616.8` is an important guardrail: its biggest desired/CP burst occurs later, after model disappears, but does not produce the same high steering. That argues against "desired/CP magnitude alone" as the root cause.
4. No driving-code change is ready. The next smallest step is a source-geometry decomposition of the same bins: lane-left/right, lane width, road edges, model path sign/shape, and controller-stage decomposition around the same timestamps.

## Generated Artifacts

| Artifact | Rows/files | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_low_speed_raw_shape_samples.csv` | 985 rows | Per-sample raw and band-pass values for the three target episodes, including focus/high-bin labels. |
| `retrospective_lateral/results/reports/drilldown_low_speed_raw_shape_bins.csv` | 88 rows | 0.5-second bin summaries over the event windows. |
| `retrospective_lateral/results/reports/drilldown_low_speed_raw_shape_event_summary.csv` | 3 rows | One row per target episode with finite fractions, focus/nonfocus deltas, peak timing, and RMS ratios. |
| `retrospective_lateral/results/reports/drilldown_low_speed_raw_shape_signal_corr.csv` | 66 rows | Per-signal lag/correlation to steering and desired curvature. |
| `retrospective_lateral/results/reports/low_speed_raw_shape/*_raw_shape.svg` | 3 files | Four-panel raw-shape timelines for steering, model-y, lane-center-y, and curvature/control signals. |

Generated files are under ignored `retrospective_lateral/results/` paths.

## Method

Signals were extracted from the existing NPZ cache only. No vehicle-control code was modified.

- Band-pass: `LOW_SPEED_INSPECT_BAND_HZ = 0.08-0.80 Hz`
- Sample rate: `20 Hz`
- Timing convention: positive lag means the signal leads steering or desired curvature.
- Path curvature uses the existing audit convention: calibrated yaw rate when available, divided by `v_ego`, only where `v_ego > 1.0 m/s`.
- Focus bins are the bins previously identified by the horizon-transition audit, not newly selected after seeing this result.

## Episode-Level Evidence

| Episode | Speed mph | Raw steer p2p deg | Band steer p2p deg | Focus steer abs deg | Nonfocus steer abs deg | Focus desired RMS 1e4 | Focus CP RMS 1e4 | Focus CP/desired | Focus model finite 5/10/20/30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `route_6b@2329.1` | 3.25 | 20.5 | 13.04 | 5.12 | 1.32 | 19.94 | 17.37 | 0.871 | 0.917 / 0.833 / 0.083 / 0.000 |
| `route_3d@1616.8` | 5.68 | 18.1 | 5.47 | 1.83 | 0.92 | 9.51 | 7.05 | 0.741 | 1.000 / 1.000 / 0.167 / 0.000 |
| `route_8d@1286.6` | 5.04 | 15.1 | 10.44 | 4.69 | 1.85 | 13.38 | 13.73 | 1.027 | 1.000 / 1.000 / 0.550 / 0.375 |

Focus bins have much higher steering-band amplitude than nonfocus bins in all three episodes. They also retain strong short-lookahead model signal. That is the main raw-shape result.

## Focus-Bin Raw Shape

### `route_6b@2329.1`

Observed symptom:

- High steering bins: `9,11,12,13,16`
- Focus steering-band abs mean: `5.12 deg`, compared with `1.32 deg` outside focus bins.
- Bins `10-13` are `short_10m_only`; bin `16` is `near_5m_only`.
- In focus bins, model-y5 finite fraction is `0.917`, model-y10 is `0.833`, model-y20 is `0.083`, model-y30 is `0.000`.

Inferred stage:

- The raw sign snapshots show lane-center y10/y20 and desired curvature moving through a large sign-changing lobe around the steering peak.
- Desired and CP final absolute peaks occur `0.75 s` before the steering absolute peak.
- CP final remains attenuated versus desired: event CP/desired `0.880`, focus CP/desired `0.871`.

Root-cause hypothesis:

- Strong support for upstream short-horizon geometry/lane-center excitation feeding desired curvature, then CP final, then steering/vehicle response.
- Weak support for pure model dropout: the largest steering bins are not model-absent bins.

Next experiment:

- Decompose the same focus bins into model-y, lane-left/right, lane-center, lane width, road-edge width, and lane probabilities. The key question is whether the short-horizon model path is following lane-center/edge geometry or introducing its own near-field path shape.

### `route_3d@1616.8`

Observed symptom:

- High steering bins: `0,1,2`
- Focus steering-band abs mean: `1.83 deg`, compared with `0.92 deg` outside focus bins.
- Focus bins retain model-y5/y10 completely; model-y20 is present only in the first part of bin `0`.
- After bin `5`, the model becomes sparse/absent while desired/CP can still grow.

Inferred stage:

- Early high steering coincides with forward-to-short model horizon and lane-center movement.
- The largest desired/CP peak occurs much later: desired peak is `5.50 s` after the steering absolute peak, CP peak is `5.75 s` after it.
- Later desired/CP growth with model absent does not reproduce the early high steering amplitude.

Root-cause hypothesis:

- This episode weakens any root cause framed as "desired curvature magnitude alone" or "CP final alone."
- It supports a more constrained hypothesis: steering swing is strongest when near/short model and lane geometry move together near the steering response, not merely when controller commands are large.

Next experiment:

- Compare early focus bins `0-2` against later model-absent bins `6-9` using the same source-geometry decomposition. This is the best within-route counterexample for separating short-model/lane geometry from desired/CP magnitude.

### `route_8d@1286.6`

Observed symptom:

- High/focus bins: `5,6,8,9`
- Focus steering-band abs mean: `4.69 deg`, compared with `1.85 deg` outside focus bins.
- Focus bins `5-6` have forward 20-30 m model coverage; bins `8-9` are `short_10m_only`.
- Focus model-y20 finite fraction is `0.550`; model-y30 is `0.375`.

Inferred stage:

- This is a partial-forward-horizon transition, not a pure dropout event.
- Desired peak occurs `0.80 s` before steering; CP final peak occurs `0.50 s` before steering.
- CX1 command curvature is available here and tracks desired/CP strongly, so this route gives the best controller-stage cross-check among the three.

Root-cause hypothesis:

- Supports the common pattern: geometry/desired/CP motion leads the steering response.
- Argues against 20-30 m dropout being necessary: high steering occurs both with forward horizon present and after transition to short-only.

Next experiment:

- Use `route_8d` as the controller-telemetry reference case because CX1 command telemetry is present. Compare CP desired, CP final, CX1 command, rate limiting, and pre-rate-limit channels around bins `5,6,8,9`.

## Ranked Root-Cause Hypotheses

| Rank | Hypothesis | Confidence | Supporting evidence | Evidence against / limits | Smallest next step |
| ---: | --- | --- | --- | --- | --- |
| 1 | Short-horizon upstream geometry/lane-center motion drives desired curvature, CP final follows, steering responds. | High for `route_6b` and `route_8d`; medium-high overall. | Focus bins have much higher steering and retain model-y5/y10. Desired/CP peaks lead steering in `route_6b` and `route_8d`. Lane-center y10/y20 amplitudes are larger in focus bins. | `route_3d` has later desired/CP peaks after the initial high-steering event, so desired magnitude alone is insufficient. Camera scene context is still missing. | Source-geometry decomposition of focus bins: lane-left/right, lane center, lane width, road-edge width, model-y sign/shape. |
| 2 | Model-horizon transition/dropout is an enabling condition, not the direct trigger. | Medium-high. | High bins usually retain short model signal while long horizon is missing. `route_8d` high bins include both forward-horizon and short-only states. | Does not by itself explain why lane/desired shape moves. A transition can mark the scene rather than cause the swing. | Align all 26 horizon-limited rows at first y20/y30 loss and test pre/post desired, lane-center, and steering amplitude. |
| 3 | Controller/final-command behavior is the primary low-speed cause. | Low-medium, mostly as a possible amplifier. | CP final is strongly correlated with steering and leads it in these rows. `route_8d` focus CP/desired is slightly above 1.0. | CP final generally tracks desired and is attenuated: event CP/desired is `0.880`, `0.752`, `0.981`. `route_3d` later desired/CP burst does not produce the highest steering. | Controller-stage audit around the same bins: CP desired, pre-rate-limit, rate-limited, final, anti-windup, reset, and CX1 channels where available. |
| 4 | Vehicle response/steering mechanics are the root cause. | Low as primary cause; useful as response evidence. | Path curvature is highly correlated with steering in all three episodes. | Path/yaw are mechanically response-coupled and do not identify the upstream source. Low speed also makes curvature division sensitive near `v_ego <= 1.0 m/s`. | Use path/yaw only for timing validation after upstream/controller source is identified. |
| 5 | Route-specific visual scene/road geometry causes these episodes. | Plausible but unproven. | The common short-horizon/lane-center signature is compatible with tight low-speed road/driveway geometry or lane/edge perception artifacts. | `route_6b` camera evidence is still unavailable locally, and numeric geometry cannot prove the visual scene cause. | If camera files are available, extract frames around `route_6b@2329.1`, `route_3d@1616.8`, and `route_8d@1286.6`; otherwise design a controlled low-speed creep capture. |

## What This Argues Against

- Against **pure model dropout**: focus bins are not model-absent. In `route_6b`, focus model-y5/y10 finite fractions are `0.917/0.833`; in `route_3d`, `1.000/1.000`; in `route_8d`, `1.000/1.000`.
- Against **long-horizon loss as necessary**: `route_8d` high bins include forward 20-30 m coverage before shifting to short-only.
- Against **CP final as standalone root cause**: CP final mostly follows desired and is usually attenuated. The strongest counterexample is `route_3d`, where later desired/CP growth does not produce the initial high steering.
- Against **path/yaw as source**: path curvature is too close to vehicle response to diagnose the upstream cause by itself.

## Known Confounders And QA Risks

- The band-pass filter can shift apparent peak timing near short event boundaries. The CSV includes raw values so conclusions do not rely only on filtered peaks.
- Model-y signals are sparse in these low-speed windows. Correlations involving sparse model-y20/y30 can look artificially strong.
- `route_6b` and `route_3d` do not have CX1 command telemetry in this cache; `route_8d` does.
- Path curvature is unavailable or fragile below `v_ego > 1.0 m/s`, which affects the slow tail of `route_3d`.
- Camera/scene evidence was not part of this step, so visual-root-cause claims remain hypotheses.
- Focus bins were selected from prior analysis, which is appropriate for drilldown but not a replacement for a fresh blind validation set.

## Recommendation

Do not change driving code yet.

The evidence has narrowed the likely root cause from generic "low-speed steering swing" to **short-horizon upstream geometry/lane-center excitation that reaches desired curvature before steering response**. The next logical workflow step is still analysis-only:

1. Run a source-geometry decomposition on the same focus bins.
2. In the same pass, add controller-stage decomposition for CP/CX1 channels around those bins.
3. Use `route_3d` early-vs-late bins as the counterexample test: early short-model/lane focus bins should differ from later model-absent desired/CP-only bins.

Only after that source decomposition identifies a stable source stage should an implementation plan be written.

## Commands Run

```bash
sed -n '1,220p' docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-horizon-transition-audit.md
.venv311/bin/python - <<'PY'  # inspected NPZ channel availability for route_6b, route_3d, route_8d
.venv311/bin/python - <<'PY'  # generated raw-shape sample, bin, event-summary, signal-correlation CSVs and SVGs
.venv311/bin/python - <<'PY'  # printed event summary row counts and focus/nonfocus metrics
.venv311/bin/python - <<'PY'  # printed signal correlation rankings and controller-stage summaries
.venv311/bin/python - <<'PY'  # printed route_3d full 0.5-second bins as the counterexample check
ls -lh retrospective_lateral/results/reports/low_speed_raw_shape
```
