# AI Vehicle Tuning Playbook

Use this guide when starting a fresh AI/Codex session to tune a different vehicle on sunnypilot (example: 2021 Ford Explorer ST on comma 4).

## Copy/Paste Prompt
```text
You are helping tune sunnypilot lateral behavior for a specific vehicle.

Vehicle under test:
- 2021 Ford Explorer ST
- Device: comma 4 (mici)
- Software base: my fork branch under sunnypilot

Primary objective:
- Reduce lane-center hunting (ping-pong / constant center correction)
- Maintain or improve turn tracking (avoid under/over-steer in curves)
- Keep behavior safe and stable

Work style requirements:
- Do all log/data analysis locally in the repo for speed.
- Prefer objective metrics over subjective feel, but correlate with driver notes.
- If data is insufficient for a claim, state that clearly.
- Do not change code unless explicitly asked; start with analysis and recommendations.

Repository context to use:
- Core lateral controller:
  - selfdrive/controls/lib/latcontrol_torque.py
  - sunnypilot/selfdrive/controls/lib/latcontrol_torque_v0.py
  - sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py
  - sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext_override.py
- Live torque parameter pipeline:
  - selfdrive/locationd/torqued.py
  - sunnypilot/selfdrive/locationd/torqued_ext.py
  - selfdrive/controls/controlsd.py
- Planplus parameter usage:
  - sunnypilot/modeld_v2/modeld.py
- Sunnylink params metadata:
  - sunnypilot/sunnylink/params_metadata.json
- Ford platform references:
  - opendbc_repo/opendbc/car/ford/values.py
  - opendbc_repo/opendbc/car/ford/interface.py
  - opendbc_repo/opendbc/car/ford/carstate.py
  - opendbc_repo/opendbc/car/ford/carcontroller.py
  - selfdrive/ui/sunnypilot/layouts/settings/vehicle/brands/ford.py
- Car-porting docs:
  - docs/car-porting/what-is-a-car-port.md
  - docs/car-porting/car-state-signals.md
  - docs/car-porting/reverse-engineering.md

Local analysis scripts to use/adapt:
- tools/scripts/g70_route_compare.py
- tools/scripts/g70_hunting_score.py
- tools/scripts/g70_override_event_analysis.py

Data acquisition:
- I will provide onebox URLs from comma useradmin.
- Pull route files via comma API endpoint:
  - https://api.comma.ai/v1/route/<dongle>|<route>/files
- Download qlogs and rlogs to a local folder and analyze there.

Required analysis output for each route:
1) Route metadata:
- branch/commit in initData
- whether openpilot was active and sample counts

2) Runtime tuning timeline:
- detect torque override/setting changes from logs
- list each change with timestamp and values (lat accel factor, offset, friction, related toggles)

3) Objective metrics:
- straight hunting metric(s): torque sign flip rate
- turn tracking metric(s): turn_over10%, turn_ratio_p90, turn_abs_err
- center behavior: center_mean, abs_center_p90
- speed-bin breakdown where useful

4) Correlation:
- for each setting window, report metrics and sample size
- identify best and worst windows with enough data
- explicitly call out low-sample windows to avoid overfitting

5) Recommendation:
- propose next A/B or A/B/C tuning matrix with small step sizes
- include pass/fail gates and sample targets for next drive

Testing protocol constraints:
- Tune one axis at a time where possible:
  1) offset for center bias
  2) factor for turn tracking
  3) friction for hunt damping
- Avoid extreme values unless running short bounded experiments.
- Use crossover design (AB then BA) across comparable route segments.

Deliverables:
- Save a markdown report under a logs/analysis folder.
- Include a concise summary in chat:
  - what changed
  - what objectively improved/regressed
  - exact next settings to test
```

## Session Setup Checklist
1. Confirm branch and commit under test.
2. Confirm device type (`mici`) and whether Enforce/Custom/Override toggles are enabled.
3. Pull both `qlog` and `rlog` if available.
4. Verify route commit from `initData` before comparing anything.
5. Require minimum sample gates before drawing conclusions:
- `samples >= 1500`
- `straight_n >= 1000`
- `turn_n >= 250`
6. If runtime override events are present, segment by event windows; otherwise segment by initData snapshots.

## Notes for Comma 4 Remote Workflow
- Remote SSH via `ssh.comma.ai` can be flaky.
- Prefer detached update jobs on-device for resilience (tmux/nohup, lockfile, offroad check, narrow fetch, logs, status file).
- Verify resulting branch/commit after reconnect.

