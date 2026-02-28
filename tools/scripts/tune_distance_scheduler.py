#!/usr/bin/env python3
"""
Distance-based onroad tuning scheduler for repeatable A/B test runs.

What it does:
- Applies a sequence of parameter profiles (A/B/...) from a JSON plan.
- Advances profiles after a target distance is accumulated under qualifying
  driving conditions.
- Switches profiles only when a basic safety gate passes.
- Persists state to resume after process restarts.
- Emits change notifications (tone + bookmark toast + cloudlog).

Usage (comma device):
  PYTHONPATH=. python3 tools/scripts/tune_distance_scheduler.py \
    --plan tools/tuning_plans/g70_commute_plan_v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cereal.messaging as messaging
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

MILES_TO_METERS = 1609.344


@dataclass
class Step:
  label: str
  distance_miles: float
  params: dict[str, Any]


@dataclass
class RuntimeState:
  current_step_idx: int | None
  pending_step_idx: int | None
  distance_on_current_m: float
  total_distance_m: float
  qualifying_distance_m: float
  cycle: int
  done: bool


class Scheduler:
  def __init__(self, args: argparse.Namespace):
    self.args = args
    self.params = Params()
    self.pm = messaging.PubMaster(["userBookmark"])
    self.sm = messaging.SubMaster(["carState", "selfdriveState", "carControl"], poll="carState")

    self.plan_doc = self._load_plan(Path(args.plan))
    self.plan_id = self._compute_plan_id(self.plan_doc)
    self.steps = self._parse_steps(self.plan_doc)

    self.min_speed_mps = float(self.plan_doc.get("min_speed_mps", args.min_speed_mps))
    self.safe_curvature = float(self.plan_doc.get("safe_curvature", args.safe_curvature))
    self.count_mode = str(self.plan_doc.get("count_mode", args.count_mode))
    self.loop = bool(self.plan_doc.get("loop", args.loop))

    self.state_path = Path(args.state_path)
    self.events_log_path = Path(args.events_log)
    self.errors_log_path = Path(args.errors_log)
    self.tone_path = Path(args.tone_file) if args.tone_file else self._default_tone_path()

    self.state = self._load_or_init_state()
    if self.state.done and self.args.reset_when_done:
      self._reset_runtime_state_for_new_run()
    self.last_t = None
    self.last_state_write_t = 0.0
    self.last_tone_t = 0.0
    self.running = True

  @staticmethod
  def _compute_plan_id(plan_doc: dict[str, Any]) -> str:
    blob = json.dumps(plan_doc, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]

  @staticmethod
  def _load_plan(path: Path) -> dict[str, Any]:
    with path.open() as f:
      data = json.load(f)
    if "steps" not in data or not isinstance(data["steps"], list) or len(data["steps"]) == 0:
      raise ValueError(f"Plan '{path}' is missing a non-empty 'steps' list")
    return data

  @staticmethod
  def _parse_steps(plan_doc: dict[str, Any]) -> list[Step]:
    steps: list[Step] = []
    for i, raw in enumerate(plan_doc["steps"]):
      if not isinstance(raw, dict):
        raise ValueError(f"steps[{i}] must be an object")
      label = str(raw.get("label", f"step_{i}"))
      distance = float(raw.get("distance_miles", 0.0))
      if distance <= 0.0:
        raise ValueError(f"steps[{i}] distance_miles must be > 0")
      params = raw.get("params", {})
      if not isinstance(params, dict) or len(params) == 0:
        raise ValueError(f"steps[{i}] params must be a non-empty object")
      steps.append(Step(label=label, distance_miles=distance, params=params))
    return steps

  def _default_tone_path(self) -> Path:
    # tools/scripts -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "selfdrive" / "ui" / "assets" / "sounds" / "prompt_single_high.wav"

  def _load_or_init_state(self) -> RuntimeState:
    if self.args.resume and self.state_path.exists():
      try:
        with self.state_path.open() as f:
          blob = json.load(f)
        if blob.get("plan_id") == self.plan_id:
          return RuntimeState(
            current_step_idx=blob.get("current_step_idx"),
            pending_step_idx=blob.get("pending_step_idx"),
            distance_on_current_m=float(blob.get("distance_on_current_m", 0.0)),
            total_distance_m=float(blob.get("total_distance_m", 0.0)),
            qualifying_distance_m=float(blob.get("qualifying_distance_m", 0.0)),
            cycle=int(blob.get("cycle", 0)),
            done=bool(blob.get("done", False)),
          )
      except Exception:
        cloudlog.exception("tune_scheduler_failed_to_load_state")

    return RuntimeState(
      current_step_idx=None,
      pending_step_idx=0,
      distance_on_current_m=0.0,
      total_distance_m=0.0,
      qualifying_distance_m=0.0,
      cycle=0,
      done=False,
    )

  def _write_state(self, force: bool = False) -> None:
    now = time.monotonic()
    if not force and (now - self.last_state_write_t) < 1.0:
      return
    self.last_state_write_t = now

    self.state_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
      "plan_id": self.plan_id,
      "plan_name": self.plan_doc.get("name", ""),
      "updated_unix": time.time(),
      "current_step_idx": self.state.current_step_idx,
      "pending_step_idx": self.state.pending_step_idx,
      "distance_on_current_m": self.state.distance_on_current_m,
      "total_distance_m": self.state.total_distance_m,
      "qualifying_distance_m": self.state.qualifying_distance_m,
      "cycle": self.state.cycle,
      "done": self.state.done,
      "current_step_label": self.steps[self.state.current_step_idx].label if self.state.current_step_idx is not None else None,
      "pending_step_label": self.steps[self.state.pending_step_idx].label if self.state.pending_step_idx is not None else None,
    }
    tmp = self.state_path.with_suffix(".tmp")
    with tmp.open("w") as f:
      json.dump(blob, f, indent=2)
    tmp.replace(self.state_path)

  def _reset_runtime_state_for_new_run(self) -> None:
    self.state.current_step_idx = None
    self.state.pending_step_idx = 0
    self.state.distance_on_current_m = 0.0
    self.state.done = False
    self._append_event_log("Resume state was done; resetting scheduler to step 1.")

  def _append_event_log(self, text: str) -> None:
    self.events_log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with self.events_log_path.open("a") as f:
      f.write(f"[{ts}] {text}\n")

  def _append_error_log(self, text: str) -> None:
    self.errors_log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with self.errors_log_path.open("a") as f:
      f.write(f"[{ts}] {text}\n")

  def _play_tone(self) -> None:
    if not self.args.enable_tone:
      return
    now = time.monotonic()
    if (now - self.last_tone_t) < 2.0:
      return
    self.last_tone_t = now

    tinyplay = shutil.which("tinyplay")
    if tinyplay is None or not self.tone_path.exists():
      return

    try:
      subprocess.Popen(
        [tinyplay, str(self.tone_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
      )
    except Exception:
      cloudlog.exception("tune_scheduler_tone_failed")

  def _emit_notification(self, text: str, step_idx: int | None = None) -> None:
    self._append_event_log(text)
    cloudlog.event(
      "tune_scheduler_event",
      text=text,
      step_idx=-1 if step_idx is None else step_idx,
      plan_id=self.plan_id,
      cycle=self.state.cycle,
    )
    if self.args.enable_bookmark_toast:
      try:
        self.pm.send("userBookmark", messaging.new_message("userBookmark"))
      except Exception:
        cloudlog.exception("tune_scheduler_bookmark_notify_failed")
    self._play_tone()

  @staticmethod
  def _params_put(params: Params, key: str, value: Any) -> None:
    # Keep behavior explicit and consistent across value types.
    if isinstance(value, bool):
      params.put_bool(key, value)
    else:
      params.put(key, str(value))

  def _apply_step(self, idx: int) -> None:
    step = self.steps[idx]
    failed_keys: list[str] = []
    if not self.args.dry_run:
      for k, v in step.params.items():
        try:
          self._params_put(self.params, str(k), v)
        except Exception:
          failed_keys.append(str(k))
          cloudlog.exception("tune_scheduler_param_write_failed")
          self._append_error_log(
            f"Failed writing param '{k}' on step '{step.label}':\n{traceback.format_exc()}"
          )

    self.state.current_step_idx = idx
    self.state.pending_step_idx = None
    self.state.distance_on_current_m = 0.0

    msg = f"Applied step {idx+1}/{len(self.steps)} '{step.label}' @ {self.state.qualifying_distance_m/MILES_TO_METERS:.2f} qualifying miles"
    self._emit_notification(msg, idx)
    if failed_keys:
      self._emit_notification(
        f"Step '{step.label}' skipped {len(failed_keys)} params: {', '.join(failed_keys)}",
        idx,
      )

  def _qualifies_for_distance(self) -> bool:
    cs = self.sm["carState"]
    ss = self.sm["selfdriveState"]
    cc = self.sm["carControl"]

    active = bool(getattr(ss, "active", False) or getattr(ss, "enabled", False))
    lat_active = bool(getattr(cc, "latActive", False))
    no_override = not bool(cs.steeringPressed)
    no_blinker = not (bool(cs.leftBlinker) or bool(cs.rightBlinker))
    fast_enough = float(cs.vEgo) >= self.min_speed_mps

    if self.count_mode == "total":
      return fast_enough
    return active and lat_active and no_override and no_blinker and fast_enough

  def _safe_to_switch_now(self) -> bool:
    cs = self.sm["carState"]
    cc = self.sm["carControl"]

    if bool(cs.steeringPressed):
      return False
    if bool(cs.leftBlinker) or bool(cs.rightBlinker):
      return False

    curvature = abs(float(getattr(cc, "currentCurvature", 0.0)))
    return curvature <= self.safe_curvature

  def _advance_or_complete(self) -> None:
    if self.state.current_step_idx is None:
      return
    next_idx = self.state.current_step_idx + 1
    if next_idx < len(self.steps):
      self.state.pending_step_idx = next_idx
      self.state.distance_on_current_m = 0.0
      label = self.steps[next_idx].label
      self._emit_notification(f"Step distance met; waiting for safe moment to apply next step '{label}'", next_idx)
      return

    # End of plan
    if self.loop:
      self.state.cycle += 1
      self.state.pending_step_idx = 0
      self.state.distance_on_current_m = 0.0
      self._emit_notification(f"Completed plan cycle {self.state.cycle}; looping to step 1", 0)
    else:
      self.state.done = True
      self._emit_notification("Completed final step distance; scheduler done")

  def _target_distance_m(self) -> float:
    if self.state.current_step_idx is None:
      return math.inf
    return self.steps[self.state.current_step_idx].distance_miles * MILES_TO_METERS

  def on_signal(self, signum, _frame) -> None:
    self.running = False
    cloudlog.warning(f"tune_scheduler_signal_{signum}")

  def run(self) -> int:
    signal.signal(signal.SIGINT, self.on_signal)
    signal.signal(signal.SIGTERM, self.on_signal)

    self._emit_notification(
      f"Scheduler start (plan={self.plan_doc.get('name', 'unnamed')}, steps={len(self.steps)}, mode={self.count_mode})"
    )

    while self.running:
      if self.state.done:
        self._write_state()
        time.sleep(0.2)
        continue

      try:
        self.sm.update(100)
        if not self.sm.updated["carState"]:
          continue

        t = self.sm.logMonoTime["carState"] * 1e-9
        if self.last_t is None:
          self.last_t = t
          # Try initial apply as soon as we have messages.
          if self.state.pending_step_idx is not None and self._safe_to_switch_now():
            self._apply_step(self.state.pending_step_idx)
            self._write_state(force=True)
          continue

        dt = t - self.last_t
        self.last_t = t
        if dt <= 0.0 or dt > 0.5:
          self._write_state()
          continue

        v_ego = float(self.sm["carState"].vEgo)
        d = v_ego * dt
        self.state.total_distance_m += d

        if self._qualifies_for_distance():
          self.state.qualifying_distance_m += d
          if self.state.pending_step_idx is None and self.state.current_step_idx is not None:
            self.state.distance_on_current_m += d

        # Pending step apply
        if self.state.pending_step_idx is not None and self._safe_to_switch_now():
          self._apply_step(self.state.pending_step_idx)

        # Check step completion
        if self.state.pending_step_idx is None and self.state.current_step_idx is not None:
          if self.state.distance_on_current_m >= self._target_distance_m():
            self._advance_or_complete()

        self._write_state()
      except Exception:
        cloudlog.exception("tune_scheduler_loop_exception")
        self._append_error_log(traceback.format_exc())
        time.sleep(0.2)

    self._write_state(force=True)
    if self.state.done:
      cloudlog.warning("tune_scheduler_done")
      self._append_event_log("Scheduler finished all plan steps.")
      return 0
    cloudlog.warning("tune_scheduler_stopped")
    self._append_event_log("Scheduler stopped by signal.")
    return 0


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Distance-based runtime tuning scheduler")
  p.add_argument(
    "--plan",
    default="tools/tuning_plans/g70_commute_plan_v1.json",
    help="Path to JSON plan file",
  )
  p.add_argument(
    "--state-path",
    default="/data/media/0/tune_scheduler_state.json",
    help="Persistent scheduler state path",
  )
  p.add_argument(
    "--events-log",
    default="/data/media/0/tune_scheduler_events.log",
    help="Event log output path",
  )
  p.add_argument(
    "--errors-log",
    default="/data/media/0/tune_scheduler_errors.log",
    help="Error log output path",
  )
  p.add_argument(
    "--tone-file",
    default="",
    help="Optional WAV file for switch notification tone (default: prompt_single_high.wav)",
  )
  p.add_argument(
    "--enable-tone",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable tone when profile changes",
  )
  p.add_argument(
    "--enable-bookmark-toast",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Send userBookmark message on profile change to show a brief on-screen toast",
  )
  p.add_argument(
    "--resume",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Resume from saved state if plan_id matches",
  )
  p.add_argument(
    "--reset-when-done",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="If resumed state is done, reset to step 1 for a new drive",
  )
  p.add_argument(
    "--dry-run",
    action="store_true",
    help="Do not write params; only log what would happen",
  )
  p.add_argument(
    "--min-speed-mps",
    type=float,
    default=15.0,
    help="Fallback qualifying speed threshold if not specified by plan",
  )
  p.add_argument(
    "--safe-curvature",
    type=float,
    default=0.0012,
    help="Max |currentCurvature| to allow a step switch",
  )
  p.add_argument(
    "--count-mode",
    choices=["active_lat", "total"],
    default="active_lat",
    help="Fallback distance counting mode if not specified by plan",
  )
  p.add_argument(
    "--loop",
    action="store_true",
    help="Fallback: loop plan when completed (overridden by plan.loop if present)",
  )
  return p.parse_args()


def main() -> int:
  args = parse_args()
  sched = Scheduler(args)
  return sched.run()


if __name__ == "__main__":
  sys.exit(main())
