# 2026-06-23 Raw Timeline Audit

## Executive Summary

Step 3 inspected the recommended top episodes and positive lead-onset outliers using the v3 retrospective cache. This pass added generated numeric outputs under `retrospective_lateral/results/reports/`:

| Output | Rows | Scope |
| --- | ---: | --- |
| `drilldown_raw_timeline_audit.csv` | 36 | Pre/episode/post or pre/post lead-event summary rows for 10 target windows. |
| `drilldown_raw_timeline_bins.csv` | 469 | One-second binned raw timeline summaries for the same windows. |
| `raw_timeline_frames/route_8d_*.png` | 4 | Extracted `fcamera` stills around `route_8d@1436`; generated and ignored. |

The strongest new evidence is that the top 10-70 mph weave episodes are tightly coupled to lane-center/model-path movement in the same band. The controller/final-command stage mostly attenuates desired curvature rather than amplifying it. This moves the next investigation toward lane/corridor perception and model-path geometry, not PI/final-command tuning.

Recommendation remains: **do not change driving code yet.**

## Top Episode Numeric Audit

| Target | Speed mph | Lead Near Frac | Lane Prob Min | Model Y20 RMS m | Lane Center RMS m | Model-Lane Corr | Lane/Model RMS | Path RMS x1e4 | Desired RMS x1e4 | CP Final RMS x1e4 | CP/Desired | Steering P2P deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_b5@1038` | 31.65 | 0.927 | 0.812 | 0.303 | 0.348 | 0.952 | 1.148 | 13.38 | 16.78 | 14.64 | 0.873 | 29.3 |
| `route_61@1215` | 29.42 | 1.000 | 0.875 | 0.238 | 0.151 | 0.961 | 0.636 | 11.63 | 11.54 | 9.61 | 0.833 | 26.7 |
| `route_a8@1430` | 29.87 | 0.516 | 0.565 | 0.193 | 0.192 | 0.879 | 0.997 | 9.14 | 10.44 | 8.91 | 0.853 | 28.8 |
| `route_92@1085` | 46.73 | 0.429 | 0.536 | 0.158 | 0.200 | 0.949 | 1.270 | 8.71 | 10.77 | 8.85 | 0.822 | 23.0 |
| `route_6b@2317` | 3.24 | 0.000 | 0.864 | 0.021 | 0.079 | -0.242 | 3.741 | 12.13 | 14.30 | 12.58 | 0.880 | 20.5 |
| `route_8d@1436` | 3.13 | 0.000 | 0.854 | 0.120 | 0.128 | 0.984 | 1.066 | 13.85 | 28.00 | 25.01 | 0.893 | 20.1 |

## Interpretation

### 10-70 MPH Weave

The four top weave episodes have strong model-to-lane-center correlation in the weave band: 0.879 to 0.961. Lane-center RMS is also comparable to model-y20 RMS, with lane/model ratios from 0.636 to 1.270. That means the model path motion is not isolated from the lane/corridor geometry channels.

All four top weave rows still show final-command attenuation relative to desired curvature: CP/desired ratios from 0.822 to 0.873. Steering was not pressed and the CP rate-limit flag fraction was 0.0 in the episode windows. This argues against PI/final-command amplification as the primary root cause for these top rows.

Lead remains a context marker, not a sufficient trigger. The top rows range from 0.429 to 1.000 lead-near fraction, while the earlier event-aligned analysis showed no median upstream growth after eligible lead onsets.

### Low-Speed Wheel Swing

`route_8d@1436` has strong model-to-lane-center coupling: correlation 0.984 and lane/model RMS ratio 1.066. Local `fcamera` stills around 1433-1442 s show slow traffic in a construction corridor, with cones/barrier geometry on the left and a large truck close on the right. This visual context supports a lane/corridor geometry confounder for that episode.

`route_6b@2317` is different: model-y20 RMS is small, lane-center RMS is larger, and model/lane-center correlation is weak and negative. Prior drilldown had `orientation_rate_curvature` as the first supported stage for this row. Video was not available locally for this target, so it remains a weaker low-speed subcase that needs visual confirmation or more raw-channel extraction.

Both low-speed targets have lead-near fraction 0.0, no steering-pressed fraction, no CP rate-limit fraction, and final command below desired curvature. That keeps lead and PI/final-command amplification low in the root-cause ranking for the low-speed swing.

## Positive Lead-Onset Outliers

| Target | Speed Pre | Speed Post | Pre Near | Post Near | Model Delta | Lane-Center Delta | Path Delta | Desired Delta | CP Delta | Post Lane Corr | Post CP/Desired |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `route_39@187.9` | 59.83 | 63.07 | 0.020 | 1.000 | +166% | +605% | +183% | +151% | +152% | -0.291 | 0.954 |
| `route_4d@2930` | 64.53 | 66.78 | 0.035 | 1.000 | +238% | +207% | +252% | +225% | +210% | 0.620 | 0.971 |
| `route_59@2273` | 51.56 | 47.80 | 0.000 | 1.000 | +282% | +334% | +298% | +363% | +322% | 0.743 | 0.848 |
| `route_ad@2648` | 50.84 | 46.41 | 0.005 | 1.000 | +200% | +206% | +127% | +234% | +236% | 0.944 | 0.760 |

These are real positive outliers: the post-onset 10 s windows grow across model, lane center, path, desired, and CP/final command. But the growth is already present upstream, and final command is not larger than desired in the post-onset windows. Three of the four have positive post-onset model/lane-center coupling; `route_39` is the odd case, with very large lane-center RMS and negative model/lane correlation.

This supports a more specific lead hypothesis: close-lead episodes may co-occur with a lane/corridor geometry transition, but the evidence still does not isolate lead tracking or controller behavior as the causal source.

## Video Availability

Only `route_8d@1436` had local target camera coverage in this Step 3 set. The extracted frames are generated under:

`retrospective_lateral/results/reports/raw_timeline_frames/`

The other target routes did not have local `fcamera`, `ecamera`, `qcamera`, or HEVC files in their route directories, so their visual state remains a residual gap. The numeric cache does not include road-edge positions or full left/right lane-line coordinates, only lane center and lane width at x=0 and x=20 m.

## Root-Cause Ranking Impact

| Rank | Hypothesis | Step 3 Impact |
| ---: | --- | --- |
| 1 | Model/path/corridor geometry is upstream of the visible weave. | Strengthened. Top weave rows have high model-to-lane-center band coupling. |
| 2 | Lane/corridor perception quality contributes to severity. | Strengthened. Top weave rows have large lane-center and lane-width motion; two have only moderate lane-probability medians. |
| 3 | Near lead/headway co-occurs with weave but is not sufficient. | Refined. Positive lead-onset outliers exist, but their growth appears upstream and lane-coupled. |
| 4 | Low-speed swing is a low-speed model/corridor/desired artifact followed by the actuator. | Strengthened for `route_8d`; still mixed for `route_6b`. |
| 5 | PI/final-command tuning is primary. | Further weakened. CP final is below desired in the audited windows and no rate-limit/override flags explain the episodes. |

## Next Step

Proceed to Step 4: add a lane geometry/perception audit.

Smallest useful implementation:

1. Extend extraction or a dedicated analysis pass to include model path y at multiple lookaheads, left/right lane-line y at multiple lookaheads, and road-edge y if available.
2. Re-run the top weave, positive lead-onset, and low-speed windows with per-lookahead lane/model correlations and lead/lag.
3. For routes with camera coverage, extract matching stills or short clips and compare visual lane geometry to numeric lane-center/model movement.
4. Do not design a driving-code change until this distinguishes perception/corridor motion from planner/controller motion.

## Commands Run

```bash
.venv311/bin/python - <<'PY'  # generated drilldown_raw_timeline_audit.csv and drilldown_raw_timeline_bins.csv
.venv311/bin/python - <<'PY'  # printed compact report tables from the generated CSV
find explorer_st_logs/<route> -maxdepth 2 -type f ...
ffprobe -hide_banner -loglevel error -show_entries format=duration,size -show_streams explorer_st_logs/route_8d/.../fcamera.hevc
ffmpeg -hide_banner -loglevel error -i explorer_st_logs/route_8d/.../fcamera.hevc -ss <offset> -frames:v 1 -update 1 -y retrospective_lateral/results/reports/raw_timeline_frames/<frame>.png
git check-ignore -v retrospective_lateral/results/reports/drilldown_raw_timeline_audit.csv retrospective_lateral/results/reports/drilldown_raw_timeline_bins.csv
```

One initial `ffmpeg` attempt to write JPEG stills failed on MJPEG color-range handling. The PNG extraction with `-update 1` succeeded.
