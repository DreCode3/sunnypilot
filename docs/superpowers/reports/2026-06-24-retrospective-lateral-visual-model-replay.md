# 2026-06-24 Visual/Model Replay Drilldown

## Executive Summary

This step used the device only for read-only log/video availability checks. The device was reachable at `192.168.98.237`, reported `IsOffroad=1`, and no update/install/reboot/code-edit action was performed.

The device still had camera coverage for `route_b5`, one of the top 10-70 mph weave episodes. It did not have the other requested target route IDs (`route_61`, `route_a8`, `route_92`, `route_6b`, or `route_8d`) in `/data/media/0/realdata`. `route_8d` already had local camera frames from the prior audit.

The newly copied `route_b5@1038` camera frames are important: the episode is not a clean open-lane straight. The forward camera shows a close lead vehicle, curb/right-edge constraints, and lane geometry changing through a median/side-lane split area. That visual context fits the numeric signature: strong common-mode lane-center/model-path motion at 20-30 m, with final command attenuating desired curvature rather than amplifying it.

Recommendation remains: **do not change driving code yet**. The evidence is now stronger for lane/corridor/model-path source investigation than for PI or final-command tuning.

## Generated Artifacts

| Artifact | Rows/Files | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/drilldown_visual_model_replay_summary.csv` | 24 rows | Per-target/per-lookahead visual availability, lane/model, width/edge, lag, and controller attenuation summary. |
| `retrospective_lateral/results/reports/visual_model_replay/*_timeline.svg` | 6 files | Numeric replay timelines for all target episodes. |
| `retrospective_lateral/results/reports/visual_model_replay/route_b5_1038_fcamera_*.png` | 4 files | New `route_b5@1038` forward-camera stills from the copied device camera segment. |
| `explorer_st_logs/route_b5/000000b5--5c90a7b95c--{16,17,18}/{fcamera,ecamera,qcamera}` | 9 files | Read-only copied camera files for the route_b5 target and neighbor segments; ignored raw log corpus. |

## Camera Availability

| Target | Device/local camera status | Result |
| --- | --- | --- |
| `route_b5@1038` | Found on device; segments 16-18 copied locally. | Visual confirmation available. |
| `route_61@1215` | Not present on device; local rlog-only. | Numeric replay only. |
| `route_a8@1430` | Not present on device; local rlog-only. | Numeric replay only. |
| `route_92@1085` | Not present on device; local rlog-only. | Numeric replay only. |
| `route_8d@1436` | Local camera frames already available. | Visual confirmation available from prior audit. |
| `route_6b@2317` | Not present on device; no local camera. | Numeric replay only; remains the low-speed counterexample. |

## Top Weave Replay Evidence

All four top weave targets retain the same core signature at 20-30 m: model path and lane-center movement are strongly correlated, and final command is lower than desired curvature.

| Target | Lookahead m | Speed mph | Lead Near | Lane Prob Min | Model RMS m | Lane Center RMS m | Model/Lane Corr | Lane/Model | CP/Desired | Signature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `route_b5@1038` | 20 | 31.64 | 0.925 | 0.812 | 0.303 | 0.348 | 0.952 | 1.15 | 0.873 | common-mode lane center/model path |
| `route_b5@1038` | 30 | 31.64 | 0.925 | 0.812 | 0.545 | 0.564 | 0.982 | 1.04 | 0.873 | common-mode lane center/model path |
| `route_61@1215` | 20 | 29.41 | 1.000 | 0.876 | 0.237 | 0.151 | 0.961 | 0.64 | 0.832 | common-mode lane center/model path |
| `route_61@1215` | 30 | 29.41 | 1.000 | 0.876 | 0.455 | 0.367 | 0.989 | 0.81 | 0.832 | common-mode lane center/model path |
| `route_a8@1430` | 20 | 29.87 | 0.517 | 0.564 | 0.193 | 0.192 | 0.879 | 1.00 | 0.853 | common-mode lane center/model path |
| `route_a8@1430` | 30 | 29.87 | 0.517 | 0.564 | 0.354 | 0.348 | 0.964 | 0.98 | 0.853 | common-mode lane center/model path |
| `route_92@1085` | 20 | 46.71 | 0.431 | 0.535 | 0.158 | 0.200 | 0.949 | 1.27 | 0.822 | common-mode lane center/model path |
| `route_92@1085` | 30 | 46.71 | 0.431 | 0.535 | 0.275 | 0.306 | 0.985 | 1.11 | 0.822 | common-mode lane center/model path |

Interpretation: for the most severe 10-70 mph weave examples, the motion is already present in model/lane corridor geometry. The controller/final-command path is still mostly passing through or attenuating the upstream signal.

The `route_b5` camera context strengthens this: the episode has visual lane/corridor complexity and a close lead, not a simple straight-road-only controller oscillation.

## Low-Speed Replay Evidence

| Target | Lookahead m | Speed mph | Lead Near | Lane Prob Min | Model RMS m | Lane Center RMS m | Model/Lane Corr | Lane/Model | CP/Desired | Signature |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `route_8d@1436` | 20 | 3.13 | 0.000 | 0.854 | 0.120 | 0.128 | 0.984 | 1.07 | 0.893 | low-speed common-mode corridor |
| `route_8d@1436` | 30 | 3.13 | 0.000 | 0.854 | 0.156 | 0.209 | 0.975 | 1.34 | 0.893 | low-speed common-mode corridor |
| `route_6b@2317` | 20 | 3.24 | 0.000 | 0.863 | 0.021 | 0.079 | -0.242 | 3.74 | 0.880 | low-speed counterexample |
| `route_6b@2317` | 30 | 3.24 | 0.000 | 0.863 | NaN | 0.129 | NaN | NaN | 0.880 | low-speed counterexample |

Interpretation: low-speed remains split into at least two subcases. `route_8d` fits the common-mode corridor hypothesis and has visual support from the construction/traffic corridor. `route_6b` does not: model-y at 20 m is small, lane center is larger, correlation is weak/negative, and 30 m model-y is missing in the audited window. That keeps `route_6b` as the best next low-speed root-cause drilldown target.

## Root-Cause Ranking Impact

| Rank | Hypothesis | Impact From This Step |
| ---: | --- | --- |
| 1 | 10-70 mph weave originates in upstream lane/corridor/model-path geometry. | Strengthened. Four top weave episodes are common-mode lane/model coupled at 20-30 m; `route_b5` now has visual context consistent with corridor complexity. |
| 2 | Near lead/headway is a context/co-occurrence factor, not a sufficient direct trigger. | Refined. `route_b5` has a close lead and complex lane context; numeric evidence still shows the motion upstream of final command. |
| 3 | Low-speed swing includes a corridor/common-mode subcase. | Strengthened for `route_8d`; visual and numeric evidence agree. |
| 4 | Low-speed swing also has a separate orientation/path-source subcase. | Strengthened by `route_6b`, which remains inconsistent with simple model-y/lane-center coupling. |
| 5 | PI/final-command tuning is the primary root cause. | Further weakened. CP final remains below desired for all target rows in this replay summary. |

## Next Smallest Step

Do not move to a vehicle-control implementation plan yet.

The next best drilldown is `route_6b@2317`: inspect orientation-rate curvature, yaw-source choice, desired curvature, path curvature, lane/road-edge geometry, and command/steering phase over that exact low-speed window. It is the best available counterexample to the common-mode lane/model pattern and therefore the highest-value discriminator.

If further device/video recovery becomes possible, look for camera archives outside `/data/media/0/realdata` for `route_61`, `route_a8`, `route_92`, and `route_6b`. Otherwise, the smallest controlled drive later would be a repeatable low-speed creep/stop corridor that captures both an open-lane case and a curb/construction/close-edge case with the same model, PI config, tire/load state, and branch.

## Commands Run

```bash
ssh-add --apple-load-keychain ~/.ssh/id_ed25519
ssh -o ControlMaster=yes -o ControlPersist=10m -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 -fN comma@192.168.98.237
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 comma@192.168.98.237 '... read-only host/IsOffroad/realdata inventory ...'
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 comma@192.168.98.237 '... read-only target route inventory ...'
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 comma@192.168.98.237 '... read-only route_b5 segment file listing ...'
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 comma@192.168.98.237 'cd /data/media/0/realdata && tar cf - ... route_b5 camera files ...' | tar xf - -C explorer_st_logs/route_b5
ffprobe -hide_banner -loglevel error -show_entries format=duration,size -of default=nw=1 explorer_st_logs/route_b5/000000b5--5c90a7b95c--17/fcamera.hevc
ffmpeg -hide_banner -loglevel error -i explorer_st_logs/route_b5/000000b5--5c90a7b95c--17/fcamera.hevc -ss <offset> -frames:v 1 -update 1 -y retrospective_lateral/results/reports/visual_model_replay/<frame>.png
.venv311/bin/python - <<'PY'  # generated drilldown_visual_model_replay_summary.csv and SVG timelines
```
