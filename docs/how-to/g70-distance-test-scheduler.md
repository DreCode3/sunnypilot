# G70 Distance Test Scheduler

This guide runs an automated A/B commute test by distance while driving.

## What it does

- Applies a sequence of test profiles from JSON.
- Counts distance under qualifying conditions (`active_lat` by default).
- Switches profile at each target distance.
- Switches only at a basic safety gate (no blinker, no steering override, near-straight curvature).
- Logs all switch events and persists state across restarts.
- On each switch:
  - writes a cloudlog event
  - sends a bookmark toast
  - plays a short tone (if `tinyplay` and sound file are available)

## Default plan for tomorrow's commute

Plan file:
- `tools/tuning_plans/g70_commute_plan_v1.json`

Profiles (ABBA, 15 qualifying miles each):
- `A`: `laf=4.5`, `lao=0.01`, `fr=0.01`, `PlanplusControl=0.8`
- `B`: `laf=4.5`, `lao=0.00`, `fr=0.01`, `PlanplusControl=0.8`

## Run on device (detached)

```bash
cd /data/openpilot
chmod +x tools/scripts/run_tune_distance_scheduler.sh
tools/scripts/run_tune_distance_scheduler.sh tools/tuning_plans/g70_commute_plan_v1.json
```

This launcher uses a lock and keeps running even if SSH disconnects.

## Auto-start on each drive

The branch now starts `tools/scripts/tune_distance_scheduler.py` automatically on every onroad transition via `manager`.

- Process name: `distance_tune_scheduler`
- Plan: `tools/tuning_plans/g70_commute_plan_v1.json`
- No SSH is required once the device is updated to this commit.
- State resume still applies, so progress continues across drives.

## Monitor status

```bash
cat /data/media/0/tune_scheduler_state.json
tail -n 50 /data/media/0/tune_scheduler_events.log
tail -n 100 /data/media/0/tune_scheduler/run_*.log
```

## Stop

```bash
kill $(cat /data/media/0/tune_scheduler.pid)
```

## Notes

- `count_mode=active_lat` means only distance while lateral control is active and stable counts toward each step.
- If you want fast cycling regardless of control state, set `count_mode` to `total` in the plan.
- Resume is enabled by default: if the process restarts, it continues from saved state when plan ID matches.
