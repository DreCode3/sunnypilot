#!/usr/bin/env bash
set -euo pipefail

cd /data/openpilot

PLAN="${1:-tools/tuning_plans/g70_commute_plan_v1.json}"
LOG_DIR="/data/media/0/tune_scheduler"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/run_${TS}.log"
STATE_FILE="/data/media/0/tune_scheduler_state.json"
EVENTS_FILE="/data/media/0/tune_scheduler_events.log"
PID_FILE="/data/media/0/tune_scheduler.pid"
LOCK_FILE="/tmp/tune_distance_scheduler.lock"

if ! /usr/bin/flock -n "$LOCK_FILE" true; then
  echo "Another scheduler launcher is already running (lock: $LOCK_FILE)"
  exit 1
fi

if [[ -f "$PID_FILE" ]]; then
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Scheduler already running with pid $(cat "$PID_FILE")"
    exit 1
  else
    rm -f "$PID_FILE"
  fi
fi

echo "Starting scheduler with plan: $PLAN"
echo "Run log: $LOG_FILE"

nohup /usr/bin/flock -n "$LOCK_FILE" bash -lc "
  cd /data/openpilot
  PYTHONPATH=. python3 tools/scripts/tune_distance_scheduler.py \
    --plan '$PLAN' \
    --state-path '$STATE_FILE' \
    --events-log '$EVENTS_FILE'
" > "$LOG_FILE" 2>&1 &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "Scheduler started (pid=$PID)"
echo "PID file: $PID_FILE"
echo "State file: $STATE_FILE"
echo "Events log: $EVENTS_FILE"
