# Clean-Room Lateral Weave Analysis

## Scope

I used only the contents of `cleanroom_lateral_analysis/` plus the local openpilot/sunnypilot Cap'n Proto schema needed to decode the provided rlogs. I did not inspect prior-analysis files elsewhere in the workspace.

Generated code and outputs:

- `analysis/extract_signals.py` - minimal zstd/Cap'n Proto rlog extractor.
- `analysis/analyze_weave.py` - alignment, gating, metrics, statistics, robustness checks, figures.
- `analysis/requirements.txt` - Python package versions used.
- `analysis/results/*.csv` - all derived tables.
- `analysis/figures/*.png` - summary plots.

## Method

I interpreted the symptom as a slow, low-amplitude left-right oscillation during stable lateral engagement on straight or very gentle road geometry. I aligned relevant streams to a 20 Hz grid and computed metrics in non-overlapping 30 s windows. A window was retained only if at least 20 s passed all gates:

- `carControl.latActive == true`, eroded by +/-2 s to avoid engagement transitions.
- No steering override within +/-1 s.
- No blinkers and no model lane-change state.
- Speed 20-35 m/s for the primary analysis.
- Slowly varying road curvature `abs(yawRate / vEgo)` low-passed at 0.035 Hz was below `0.0015 1/m`.

Primary weave band: `0.10-0.35 Hz`, corresponding to roughly 3-10 s oscillations. Robustness variants changed the curvature gate, speed gate, and band.

## Metrics

Primary metrics:

- `steer_rms_deg`: RMS of band-passed measured steering angle. This best captures what the driver sees/feels at the wheel.
- `actual_curv_rms_1pm`: RMS of band-passed actual vehicle curvature estimated as `carState.yawRate / vEgo`, reported in `1e-4 1/m`. This is the best log-only proxy for visible path weave.
- `lat_accel_rms_mps2`: band-passed lateral acceleration proxy, `vEgo^2 * actual curvature`, capturing physical lateral motion.
- `cmd_curv_rms_1pm` / `desired_curv_rms_1pm`: command/reference oscillation, useful for determining whether the motion is already present in controller output.
- `model_path_y20_rms_m` and `lane_center_y20_rms_m`: model/lane reference lateral motion at 20 m lookahead, used as a secondary source check.

Uncertainty is based on drive-level medians, not per-sample statistics. The independent sample count is very small: A has 3 drives, B has 1, C has 2. I report drive-bootstrap intervals and exact drive-label permutation p-values where possible, but B has no between-drive variance estimate.

## Comparability

Primary eligible data after gating:

| Config | Model | Controller | Drives | Windows | Eligible min | Median speed |
|---|---|---:|---:|---:|---:|---:|
| A | CD210 | Set1 | 3 | 42 | 18.3 | 52.3 mph |
| B | CD210 | Set2 | 1 | 11 | 5.1 | 61.9 mph |
| C | OPM7 | Set1 | 2 | 31 | 13.9 | 58.2 mph |

Confounds are substantial:

- B has only one drive and is much faster than A.
- C is also faster than A and covers different route portions.
- GPS/speed/curvature matched overlap is small: A-B has 5 windows, A-C has 8 windows, B-C has 6 windows.
- Steering and actual-curvature weave metrics decrease with speed inside all configs, so speed imbalance can make a faster config look better on steering/curvature RMS.

The logs also show differing build commit strings across some drives. I treated the package documentation as authoritative that the intended controlled variables are model and controller parameters, but any undocumented build difference remains a residual confound.

## Primary Results

Drive-median aggregate results over all primary eligible windows:

| Config | Steering RMS | Actual curvature RMS | Lat accel RMS | Command curvature RMS | Peak freq |
|---|---:|---:|---:|---:|---:|
| A | 0.841 deg | 2.089 | 0.113 m/s^2 | 1.875 | 0.195 Hz |
| B | 0.740 deg | 1.713 | 0.132 m/s^2 | 1.586 | 0.156 Hz |
| C | 0.614 deg | 1.423 | 0.089 m/s^2 | 1.417 | 0.186 Hz |

Curvature values are `1e-4 1/m`. The symptom is indeed in the slow band, roughly 0.16-0.20 Hz, or a 5-6 s period.

All-eligible descriptive differences:

- B vs A: steering -12%, actual curvature -18%, but lateral acceleration +17%.
- C vs A: steering -27%, actual curvature -32%, lateral acceleration -21%.
- C vs B: steering -17%, actual curvature -17%, lateral acceleration -33%.

These all-eligible differences are descriptive, not causal, because speed/location overlap is weak.

## Confound-Control Checks

Speed/curvature matching, ignoring GPS:

- A-B reverses: B is worse than A by about +27% steering RMS and +25% actual-curvature RMS.
- A-C mostly remains lower for C: about -17% steering RMS and -27% actual-curvature RMS.
- B-C remains lower for C.

GPS + speed + curvature matching:

- A-B collapses to near-zero/sign-unstable differences across only 2 strata.
- A-C collapses to near-zero/sign-unstable differences across only 4 strata.
- B-C remains lower for C across 3 strata, but that comparison changes both variables at once and mostly compares one B drive against one C drive.

The matched-stratum table is `analysis/results/matched_strata_summary.csv`; the per-stratum rows are in `analysis/results/matched_strata_detail.csv`.

## Attribution

The design partially separates the two variables:

- Controller effect: A vs B uses CD210 in both, but changes Set1 -> Set2. Result: insufficient. All-eligible suggests B has less steering/curvature weave, speed/curvature matching suggests B has more, and GPS matching shows no stable difference. B has only one drive.
- Model effect: A vs C uses Set1 in both, but changes CD210 -> OPM7. Result: suggestive but not secure. Aggregate and speed-matched analyses show C lower, but the GPS-matched subset is tiny and does not preserve a stable C advantage.
- Combined B vs C: C is lower in the small GPS-matched subset, but this comparison changes both model and controller, so it cannot attribute cause.

## Robustness

The broad pattern over all eligible windows survives several metric choices: C is generally lowest and A highest on steering/actual-curvature weave. But the key causal finding does not survive the strongest confound control because route-overlap data are too sparse and matched results become sign-sensitive.

This is the decisive self-refutation: if a claimed model/controller benefit depends on unmatched road portions and weakens or reverses on GPS-matched windows, it should not be treated as a confident configuration effect.

## Verdict

The data clearly contain measurable slow weave in the stated band. Descriptively, config C shows the lowest aggregate weave: about 27% lower steering RMS and 32% lower actual-curvature RMS than A in all eligible windows. However, I do not consider that a confident causal finding because speed and location confounds are large, independent drive counts are tiny, and GPS-matched overlap is too small.

Conclusion: insufficient/confounded for a rigorous claim that any configuration measurably reduces the symptom. The most defensible statement is that C is a promising aggregate signal, not a proven better configuration; A-B controller attribution is especially unsupported.

## Experiment To Settle It

Run a counterbalanced factorial test on the same route and day:

- Four configs if possible: `CD210/Set1`, `CD210/Set2`, `OPM7/Set1`, `OPM7/Set2`.
- At least 3 independent repeats per config, preferably alternating configs across the same highway segment.
- Fixed speed setpoint bands, no lane changes, minimal traffic, and same tire/load/weather conditions.
- Record or reset the PI integrator initial state, since it is persisted across drives.
- Target at least 20-30 minutes of primary-eligible straight/gentle data per config on the same GPS cells.

That design would separate model, controller, and interaction effects while making the 30 s windows genuinely pairable by road segment and speed.

## Reproduction

From the repository root:

```bash
python3.11 -m venv cleanroom_lateral_analysis/.venv
cleanroom_lateral_analysis/.venv/bin/python -m pip install -r cleanroom_lateral_analysis/analysis/requirements.txt
cleanroom_lateral_analysis/.venv/bin/python cleanroom_lateral_analysis/analysis/extract_signals.py --force
cleanroom_lateral_analysis/.venv/bin/python cleanroom_lateral_analysis/analysis/analyze_weave.py
```

