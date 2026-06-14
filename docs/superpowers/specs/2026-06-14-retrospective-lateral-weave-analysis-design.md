# Retrospective Lateral Weave Analysis Design

Date: 2026-06-14

## Objective

Design an existing-log analysis for the 2021 Ford Explorer ST lateral-control comfort problem, starting from the full local log corpus rather than from prior conclusions.

Primary target: straight and gentle-section weave, because it has the highest comfort impact. The analysis must translate subjective feedback into objective metrics, correlate those metrics to actual data, and identify which historical factors or pipeline stages are most likely responsible.

The design intentionally treats all prior findings, tuning values, controller choices, and documentation verdicts as hypotheses to re-examine. Prior analysis can guide candidate signals and pitfalls, but no conclusion is inherited without independent evidence from this analysis.

## Scope

In scope:

- Existing local rlogs/qlogs and derived route data, including the multi-month corpus over 100 GB.
- Two subjective symptom classes:
  - Low-speed wheel swing: large steering-wheel left/right motion at single-digit speeds, such as stoplight approaches.
  - 10 to 70 mph visible/comfort weave: smaller slow wander on straights and gentle curves, visible externally and uncomfortable over longer drives.
- Corpus-wide metric and episode discovery before any route-specific case study.
- Pipeline-stage localization: model output, planner desired curvature, predicted/orientation-rate curvature, blended and filtered command, PI trim, final Ford curvature command, actual yaw/path, and steering wheel response.
- Historical model/controller/config comparisons where the logs can support them.
- April 6 and April 8, 2026 Powder Springs GA to Madison AL road-trip tuning drives as a later case study, not the initial metric calibration target.
- Sharp-curve behavior only as a regression guardrail.

Out of scope for this spec:

- Code changes to vehicle control.
- On-device updates or installs.
- Safety-layer changes.
- A new controlled drive protocol, except as a final recommendation for questions existing logs cannot answer.

## Safety Constraints

This is an analysis-only design. It must not propose or perform control changes.

The Ford safety architecture is a hard boundary for any future implementation:

- The 2021 Explorer ST is Ford Q3/CAN and is curvature-commanded, not torque-commanded.
- `panda` `ford.h` bounds lateral command ranges and rate changes.
- Current app code uses an internal curvature convention and negates curvature/rate before CAN send. Any future design involving curvature, path angle, or path offset must state its sign convention explicitly.
- Path offset and path angle are available in the Ford polynomial API but are currently sent as zero. Any future use of those signals would require a separate safety review and is not part of this analysis design.

If later work touches a comma device, it must follow the remote update policy: detached offroad-safe jobs, `IsOffroad=true` gating, locking, fast-forward-only target-branch fetches, submodule sync, full logs, explicit status files, and reboot only after successful offroad update.

## Recommended Approach

Use an episode-first, stage-localized retrospective workflow.

The sequence is:

1. Build trusted objective symptom detectors across the full corpus.
2. Use those detectors to create ranked low-speed and 10-70 mph episode catalogs.
3. Localize each episode through the lateral pipeline to determine where oscillation first appears or grows.
4. Compare historical eras/configurations/models only after the metrics are trusted.
5. Recommend controlled drives only for factors the logs cannot isolate.

Rejected alternatives:

- Era/config scorecard first: faster, but too likely to repeat prior confounding errors if the metric is wrong.
- Full causal model first: useful later, but premature before the symptom detectors and eligible windows are validated.

## Data Model

Use a new retrospective dataset rather than extending older tables whose eligibility and metrics may encode stale assumptions.

Hierarchy:

```text
raw route segments
  -> resampled route timelines
  -> candidate windows and episodes
  -> matched road cells
  -> era/config summaries
  -> report artifacts
```

Core row types:

- `route`: one local route folder or route-id group.
- `segment`: one rlog/qlog segment inside a route.
- `timeline_sample`: a fixed-rate resampled sample used for signal alignment.
- `window`: a bounded interval eligible for a symptom metric.
- `episode`: a detected symptom event, with start/end, peak time, and context.
- `road_cell`: GPS/heading/speed-matched physical road unit.
- `comparison`: a config/model/era contrast under a defined evidence tier.

## Extraction

Extract broadly enough that downstream analyses can be rerun without rereading all rlogs.

Required signal families:

- Route metadata: route id, segment id, wall date, initData commit, branch, dirty flag, CarParams, fingerprint, process build metadata where available.
- Engagement: `carControl.latActive`, MADS/SP engagement state where available, cruise state, controls allowed if present.
- Vehicle motion: speed, yaw rate from CAN, calibrated yaw/angular velocity from locationd where available, GPS position, GPS heading, heading validity, roll/pitch where available.
- Steering: steering angle, steering rate, steering torque, steeringPressed/override, steering faults.
- Control chain: `modelV2` path, lane lines, road edges, lane probabilities, lane width, orientationRate, desired curvature, carControl actuator curvature.
- Ford controller telemetry: parsed `CP:`, `CX1:`, and `LC:` logMessage records where present.
- PI internals: lane offset, integral, P/I terms, lc_kp if logged or inferable, cap behavior if inferable.
- Model metadata: active model bundle or model label if logged or recoverable from params/logs.
- Context: blinkers, lane-change state, lead/radar state if available, brake/gas, standstill/stop approach indicators, canValid/calibration validity.
- Learned/calibration params: liveParameters, liveCalibration, angle offsets, steer ratio, stiffness where available.

Config recovery must be evidence-based:

- `proven`: direct telemetry, logged params, or exact commit/config reconstruction.
- `inferred`: strong indirect evidence such as commit/date plus matching telemetry shape.
- `unknown`: insufficient evidence.

Route date or route counter alone is not enough for a high-confidence config label because route counters reset and on-device edits can diverge from the repo.

The extractor must support:

- Old and new on-disk route layouts.
- Missing CX1/LC telemetry.
- Corrupt or partial segments.
- GPS warmup gaps.
- Dirty builds.
- Route-counter resets.
- Resumable per-route caches with schema versions.

## Performance

The main workload is CPU-bound rlog parsing, alignment, filtering, and NumPy/SciPy metric computation. Use the M4 Max aggressively:

- Process routes in parallel across available CPU cores.
- Use per-route caches to avoid rereading compressed logs.
- Keep worker functions top-level and arguments picklable for macOS spawn safety.
- Use vectorized NumPy/SciPy operations for metrics.
- Prefer chunked writes and compact columnar/cache formats for large arrays.
- Track extraction timing and cache hit rates.

GPU use is not required for the first design because the analysis reads log outputs rather than rerunning driving models. GPU/model reruns can be a later optional extension if the logs point to model-perception ambiguity that cannot be resolved from recorded outputs.

## Symptom Detectors

### Low-Speed Wheel Swing

Target: driver-visible 10-12 degree steering-wheel left/right movement near stoplight or creeping approaches.

Eligibility:

- Speed roughly 1-10 mph.
- Lateral active.
- No steering override, with buffer.
- No lane change or blinker, with buffer.
- Engagement transitions excluded.
- Stop/approach state recorded but not required.

Primary metrics:

- Steering angle peak-to-peak over short windows.
- Steering angle band/RMS in a low-speed-specific band chosen after PSD inspection.
- Steering-rate RMS and peak.
- Steering zero-crossing/reversal rate.
- Commanded curvature peak-to-peak.
- Final command vs steering lead/lag.
- Actual yaw/path response, where speed is high enough for curvature to be meaningful.

Outputs:

- Ranked low-speed wheel-swing episodes.
- Whether the motion begins in final command, steering response, or another stage.
- Speed/standstill/engagement context.

### 10-70 mph Visible/Comfort Weave

Target: slow visible side-to-side path wander on straights and gentle curves.

Eligibility:

- Speed approximately 10-70 mph.
- Lateral active.
- No steering override, with buffer.
- No lane change or blinker, with buffer.
- Straight or gentle road gate based on low-passed path curvature and/or GPS heading curvature.
- Lead-follow contamination tracked and preferably gated.
- Lane quality retained as a covariate; do not blindly discard all low-confidence cases before understanding them.

Primary metrics:

- Band-passed model-independent path curvature `yawRate / vEgo`.
- Steering angle band-RMS in the same slow band.
- Estimated lateral displacement or peak-to-peak path wander.
- Episode rate per engaged mile/minute.
- Worst-window p90/p95 metrics.
- Steer-per-path gain: steering weave divided by path weave.
- Spectral peak frequency and peakedness.

`aLat` is not primary because it weights by speed/frequency and can understate slow weave. It can remain a secondary comfort/body-motion channel.

Initial slow-weave band should include the historically observed 0.10-0.35 Hz region, but the detector must inspect PSDs and robustness bands rather than assuming that band is final.

## Stage Localization

For each eligible window or episode, compute comparable oscillation metrics at every available stage:

- Model path/lane center.
- Model orientationRate-derived predicted curvature.
- Planner/model desiredCurvature.
- Predicted/desired blend.
- Carcontroller EMA/smooth stage, if reconstructable from CX1 or simulator replay.
- PI trim contribution.
- Final commanded curvature sent to Ford EPAS, after rate limits and sign handling.
- Actual path curvature from yaw/speed.
- Steering angle and steering rate.

Interpretation rules:

- If oscillation is already present in model path or desiredCurvature, model/perception is implicated.
- If oscillation grows between desiredCurvature and final command, controller/blend/filter/PI is implicated.
- If final command is relatively clean but actual yaw or steering oscillates, EPAS/plant/road interaction is implicated.
- If steering is busy but path is calm, the issue may be wheel comfort rather than externally visible weave.
- If path weaves with little steering motion, the issue may be plant, road, or low-speed curvature geometry.

Each localization result should include confidence, missing-signal caveats, and the earliest stage where the symptom appears.

## Historical Comparisons

Use evidence tiers rather than a single pass/fail result:

- `descriptive`: broad corpus trends. Hypothesis generation only.
- `speed_matched`: same speed bins and similar road curvature.
- `location_matched`: same GPS cell, heading, and speed band. Preferred retrospective tier.
- `same_corridor_transition`: repeated road section before/after a known config/model change. Strongest retrospective tier.
- `controlled_drive_needed`: historical data cannot isolate the factor.

Candidate comparison factors:

- Driving model: CD210, OPM7, and any other model labels recoverable from logs.
- Controller era: PI settings, smooth_tau, blend/lookahead, rate limits, steerRatio, steerActuatorDelay, Path 4 enable state.
- Backend/localization changes: camera odometry delay compensation, livePose timestamp fixes, filter-time changes.
- Learned/cumulative params: liveParameters, angleOffset, steerRatio, calibration, LaneBiasIntegral.
- Operating context: speed, road curvature, lead-follow, lane quality, lighting/weather proxies if inferable, GPS heading, road direction.

No historical comparison may claim a win from:

- A single drive alone.
- Pooled variance without pass/route-level independence.
- Unmatched speed.
- Unmatched road/location when location matching is available.
- A metric that mathematically discards the symptom axis being claimed.

## April 6/8 Case Study

After corpus-wide metrics exist, analyze the April 6 and April 8, 2026 Powder Springs GA to Madison AL trip as a high-value case study.

Known candidate local routes from the existing summary include:

- April 6: `route_17` through roughly `route_27`.
- April 8: `route_2e` through roughly `route_37`.

The case study should:

- Reconstruct route sequence, timing, commits, dirty state, model labels, and controller changes.
- Re-run the new symptom detectors on those drives.
- Identify same-road repeats within the trip if GPS overlap exists.
- Compare early vs late tuning states under speed/location matching where possible.
- Treat the trip as explanatory evidence, not as the metric-definition source.

## Outputs

The first analysis should produce:

1. `symptom_catalog`: ranked low-speed wheel-swing and 10-70 mph weave episodes with route, segment, time, GPS, speed, engagement, and context.
2. `stage_localization_report`: where each symptom first appears or grows in the pipeline.
3. `historical_scorecard`: model/controller/config-era comparisons by evidence tier.
4. `case_study_apr_6_8`: focused analysis of the April 6/8 road trip after corpus-wide metrics are defined.
5. `next_experiment_recommendation`: controlled tests only for unresolved questions, with exact speed/corridor/config/pass requirements.
6. Reproducible cache and manifest files with schema versions and provenance.

## Decision Gates

A historical config/model is considered better only if:

- The primary path metric improves under location+speed matching.
- At least one driver-facing corroborator improves in the same direction.
- Episode-rate or worst-window tails do not hide a regression.
- Sharp-curve guardrails do not worsen.
- The evidence survives reasonable band/window/gate robustness checks.

A lever is considered root-cause likely only if:

- Stage-localization evidence and matched config/factor evidence agree.
- The effect is not explained by speed, location, lead-follow, lane quality, or missing telemetry.
- The result survives a self-refutation attempt using stricter gates and alternate yaw sources.

Otherwise, classify the result as `suggestive`, `confounded`, `refuted`, or `controlled_drive_needed`.

## Regression Guardrails

Although sharp curves are not the primary target, reports must track:

- Override clusters in moderate/sharp curves.
- Curve entry under-command or delayed response.
- Curve apex/exit overshoot or lane crossing, where detectable.
- Command clipping or safety/rate-limit interactions.
- Driver-takeover events following any historically "better" weave era/config.

Any future recommendation that improves weave but worsens these guardrails should be flagged as unsafe for adoption without a separate curve-focused design.

## Validation And Self-Checks

The analysis implementation should include:

- Extractor smoke tests on representative old/new route layouts.
- Cache schema version checks.
- Synthetic signal tests for filter bands, RMS, peak-to-peak, and episode detection.
- Sign-convention assertions for curvature/yaw/steering where possible.
- Cross-yaw-source comparison: CAN yaw vs calibrated angular velocity where available.
- Reproducibility checks: rerun a subset and confirm stable outputs.
- Null/permutation or block bootstrap calibration before trusting significance.
- Sensitivity sweeps for speed bins, road-curvature gates, bands, window length, and GPS cell size.
- Audit flags for missing telemetry, dirty builds, config uncertainty, and poor location overlap.

## Open Questions For Planning

These are planning questions, not blockers for this design:

- Which exact cache format best balances size, read speed, and schema evolution?
- Should low-speed analysis use separate resampling/window lengths from 10-70 mph analysis?
- How much of carcontroller can be replayed faithfully from historical logs without CX1?
- Which model labels can be proven from route logs versus inferred from dates/commits?
- Are WD-40 or other BluePilot-recommended models present in any existing logs, or only candidates for future controlled tests?

## Acceptance Criteria

The design is successful if the eventual implementation can:

- Reprocess the full local corpus without manual per-route intervention.
- Produce objective episode catalogs that correspond to the user's described low-speed and 10-70 mph symptoms.
- Localize episodes to pipeline stages with explicit uncertainty.
- Compare historical models/configs using evidence tiers that distinguish descriptive trends from matched evidence.
- Identify which questions remain undecidable from existing logs and specify the smallest controlled drive needed to answer them.

