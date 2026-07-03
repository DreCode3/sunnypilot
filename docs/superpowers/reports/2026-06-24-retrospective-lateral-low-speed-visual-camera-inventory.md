# 2026-06-24 Low-Speed Visual Camera Inventory

## Executive Summary

This step performed the recommended read-only camera availability drilldown for the top-ranked low-speed source-geometry rows.

The device was reachable at `192.168.98.237`, reported `IsOffroad=1`, and no update, install, reboot, or code-edit action was performed. Device inventory was read-only.

Result:

1. The device currently has newer `000000b*`/`000000c*` route folders in `/data/media/0/realdata`, but none of the target route prefixes `0000000d`, `0000003d`, `0000006b`, `0000008d`, or `0000009b`.
2. Local camera coverage exists only for the two `route_8d` target rows: `route_8d@1135.9` and `route_8d@1286.6`.
3. Nine local `fcamera` stills were extracted for those `route_8d` rows: focus/high-steering frames, peaks, and the `route_8d@1135.9` desired-high/model-absent/low-steer counterexample frame.
4. The available `route_8d` frames show constrained visual context: dense traffic, concrete barrier/left edge, lane split/merge or construction geometry, and adjacent vehicles. These are not clean open-lane low-speed examples.
5. The strongest numeric rows remain camera-unverified: `route_3d@1616.8`, `route_9b@1682.9`, `route_6b@2329.1`, and `route_0d@815.7`.

Recommendation remains: **do not change driving code yet**. The visual evidence strengthens the constrained-corridor/perception-geometry hypothesis for the available `route_8d` subcase, but it does not close the visual cause for the strongest 26-row source-geometry examples.

## Generated Artifacts

Generated artifacts are under ignored `retrospective_lateral/results/` paths.

| Artifact | Rows/files | Scope |
| --- | ---: | --- |
| `retrospective_lateral/results/reports/low_speed_visual_inventory.csv` | 6 rows | Per-target local/device camera availability and candidate segments. |
| `retrospective_lateral/results/reports/low_speed_visual_frame_plan.csv` | 30 rows | Planned and extracted focus/peak/counterexample frame rows. |
| `retrospective_lateral/results/reports/low_speed_visual_inventory/*.png` | 11 files | 9 extracted `route_8d` frame PNGs and 2 contact sheets. |

## Target Segment Mapping

| Target | Candidate segment(s) | Peak offset | Availability |
| --- | --- | ---: | --- |
| `route_3d@1616.8` | 26, 27 | 56.85 s | No local camera; not present on device. |
| `route_9b@1682.9` | 28 | 2.90 s | No local camera; not present on device. |
| `route_6b@2329.1` | 38 | 49.10 s | No local camera; not present on device. |
| `route_8d@1135.9` | 18, 19 | 55.95 s | Local `fcamera/ecamera/qcamera` available. |
| `route_0d@815.7` | 13 | 35.70 s | No local camera; not present on device. |
| `route_8d@1286.6` | 21 | 26.55 s | Local `fcamera/ecamera/qcamera` available. |

## Extracted Frame Set

| Target | Extracted frames | Scene interpretation |
| --- | ---: | --- |
| `route_8d@1135.9` | 5 | Slow dense traffic with a concrete barrier/left edge, adjacent vehicles, overhead lane/exits, and a lane split/merge context. The counterexample frame remains in a constrained traffic corridor rather than an open-lane scene. |
| `route_8d@1286.6` | 4 | Slow traffic near construction barrels and a left-side concrete barrier, with an orange lane/merge sign and vehicles in adjacent lanes. |

Contact sheets:

- `retrospective_lateral/results/reports/low_speed_visual_inventory/route_8d_1135p9_contact_sheet.png`
- `retrospective_lateral/results/reports/low_speed_visual_inventory/route_8d_1286p6_contact_sheet.png`

## Evidence Impact

Observed symptom:

- Low-speed wheel swing in the `route_8d` source-geometry rows.

Inferred stage:

- Numeric evidence had already put these rows upstream of final command, with short/near lane-center and model-path geometry moving before steering response.
- The frames add scene context: available `route_8d` rows occur in constrained, visually complex corridor conditions, not clean straight/open-road conditions.

Root-cause hypothesis impact:

- Strengthens the hypothesis that at least one low-speed subcase is perception/corridor geometry driven.
- Does not prove that all low-speed source-geometry rows share the same visual cause, because the top-ranked `route_3d`, `route_9b`, `route_6b`, and `route_0d` rows remain unvisualized.

Evidence against premature implementation:

- The strongest numeric discriminator, `route_3d@1616.8`, still has no camera confirmation.
- `route_6b@2329.1`, the key low-speed counterexample family, also remains camera-unverified.
- Available visual evidence is route-specific and retrospective.

## Recommendation

Do **not** change driving code yet.

The next smallest evidence step is one of:

1. Recover off-device camera archives for `route_3d`, `route_9b`, `route_6b`, and `route_0d` if they exist outside the device's current `/data/media/0/realdata`.
2. If archives are unavailable, run a controlled low-speed capture with fixed branch/model/PI/tire/load state and two scene classes:
   - open-lane low-speed creep/stop
   - constrained corridor with barrier/curb/construction/merge/adjacent-vehicle geometry
3. Collect camera, model horizon, lane lines, road edges, desired curvature, CP/CX1 command, yaw, and steering for both scene classes.

Only after that visual split should an implementation plan be considered.

## Commands Run

```bash
sed -n '1,220p' AGENTS.md
sed -n '1,180p' EXPLORER_ST.md
sed -n '1,260p' docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-26row-source-geometry-validation.md
rg -n "camera|dcamera|fcamera|ecamera|qcamera|hevc|frames|ffmpeg|ssh|comma|192|segment|route" AGENTS.md EXPLORER_ST.md retrospective_lateral docs/superpowers/reports -g '*.md' -g '*.py'
sed -n '1,240p' docs/superpowers/reports/2026-06-24-retrospective-lateral-visual-model-replay.md
find explorer_st_logs -maxdepth 3 -type f \( -name 'fcamera' -o -name 'ecamera' -o -name 'dcamera' -o -name 'qcamera' -o -name '*camera*' -o -name '*.hevc' -o -name '*.mp4' \)
find retrospective_lateral/results/reports -maxdepth 3 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.mp4' -o -name '*.hevc' \)
.venv311/bin/python - <<'PY'  # derived target segment offsets and frame plan
find explorer_st_logs/route_{0d,3d,6b,8d,9b} -maxdepth 2 -type f
ffmpeg -version
ssh-add -l
ssh -o ControlMaster=auto -o ControlPersist=10m -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 -fN comma@192.168.98.237
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 comma@192.168.98.237 '... read-only host/IsOffroad/realdata check ...'
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 comma@192.168.98.237 '... read-only target prefix inventory ...'
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 comma@192.168.98.237 '... read-only realdata sample/archive directory inventory ...'
.venv311/bin/python - <<'PY'  # generated low_speed_visual_inventory.csv and initial frame plan
ffprobe -hide_banner -loglevel error -show_entries format=duration,size -show_streams -of default=nw=1 explorer_st_logs/route_8d/0000008d--c6482d0967--18/fcamera.hevc
ffmpeg -f hevc -i explorer_st_logs/route_8d/0000008d--c6482d0967--18/fcamera.hevc ...  # validated raw HEVC demuxing
.venv311/bin/python - <<'PY'  # extracted 9 route_8d fcamera PNGs
.venv311/bin/python - <<'PY'  # generated two contact sheets with ffmpeg tile
.venv311/bin/python - <<'PY'  # verified visual artifact row counts and frame/contact-sheet existence
git check-ignore -v retrospective_lateral/results/reports/low_speed_visual_inventory.csv retrospective_lateral/results/reports/low_speed_visual_frame_plan.csv retrospective_lateral/results/reports/low_speed_visual_inventory/route_8d_1135p9_contact_sheet.png retrospective_lateral/results/reports/low_speed_visual_inventory/route_8d_1286p6_contact_sheet.png retrospective_lateral/results/reports/low_speed_visual_inventory/route_8d_1135p9_peak_seg18_55.95s_fcamera.png retrospective_lateral/results/reports/low_speed_visual_inventory/route_8d_1286p6_peak_seg21_26.55s_fcamera.png
git diff --check -- docs/superpowers/reports/2026-06-24-retrospective-lateral-low-speed-visual-camera-inventory.md
git status --short opendbc_repo panda
.venv311/bin/python -m pytest retrospective_lateral/tests -q
ssh -S /Users/dregilley/.ssh/cm/comma-192.168.98.237 -O exit comma@192.168.98.237
```
