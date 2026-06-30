#!/usr/bin/env python3
"""Analyze runtime events and correlate with lane/steering metrics for Explorer ST.

Adapted from g70_override_event_analysis.py for Ford angle-based lateral control.
Since Ford uses open-loop angle control (not torque), the torque override params
(latAccelFactor, latAccelOffset, friction) are NOT active. Instead this script
tracks PlanplusControl and any other angle-relevant parameter changes.

For now, this script provides the same interval-based metric correlation but
using angle-based signals (curvature commands, steering angle error).

Example:
  PYTHONPATH=. .venv311/bin/python3 tools/scripts/explorer_st_override_event_analysis.py \
    "explorer_st_logs/962fb90eaa5b23f1_00000018--024bf527b5--*--qlog.zst"
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np

from openpilot.tools.lib.logreader import LogReader

MAX_DT_S = 0.2


@dataclass
class IntervalMetrics:
  t0: float
  t1: float
  duration_s: float
  samples: int
  straight_samples: int
  turn_samples: int
  straight_curv_flip: float          # curvature command sign-flip rate on straights
  straight_angle_err_p90: float      # |angle error| p90 on straights
  straight_angle_err_mean: float     # mean angle error on straights (bias indicator)
  turn_over10_pct: float             # % turns with |actual|/|desired| > 1.10
  turn_angle_ratio_p90: float
  turn_angle_err_mean: float         # mean |angle error| in turns
  center_mean: float
  abs_center_p90: float


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Correlate runtime events to driving metrics (Explorer ST angle control).")
  p.add_argument("identifier", help="Route glob/path/onebox URL")
  p.add_argument("--logs-dir", default="explorer_st_logs", help="Default logs dir")
  p.add_argument("--min-speed", type=float, default=20.0)
  p.add_argument("--straight-curvature", type=float, default=5e-4)
  p.add_argument("--curv-deadband", type=float, default=1e-4)
  p.add_argument("--min-interval-samples", type=int, default=80)
  return p.parse_args()


def sort_key(path: str) -> tuple[int, int, str]:
  base = os.path.basename(path)
  m = re.search(r"--(\d+)--(rlog|qlog)\.zst$", base)
  if not m:
    return (1_000_000, 1, base)
  seg = int(m.group(1))
  typ = 0 if m.group(2) == "rlog" else 1
  return (seg, typ, base)


def parse_onebox(identifier: str) -> str:
  if "useradmin.comma.ai" not in identifier:
    return identifier.strip()
  m = re.search(r"onebox=([^&]+)", identifier)
  if not m:
    return identifier.strip()
  token = m.group(1).replace("%7C", "|")
  if "|" in token:
    d, r = token.split("|", 1)
    return f"{d}_{r}"
  return token


def resolve_files(identifier: str, logs_dir: str) -> list[str]:
  identifier = identifier.strip()
  if os.path.exists(identifier):
    if os.path.isdir(identifier):
      files = sorted(glob.glob(os.path.join(identifier, "*.zst")), key=sort_key)
      if files:
        return files
    return [identifier]

  globbed = sorted(glob.glob(identifier), key=sort_key)
  if globbed:
    return globbed

  token = parse_onebox(identifier)
  if "|" in token:
    d, r = token.split("|", 1)
    token = f"{d}_{r}"
  pats = [
    os.path.join(logs_dir, f"{token}--*--rlog.zst"),
    os.path.join(logs_dir, f"{token}--*--qlog.zst"),
  ]
  files: list[str] = []
  for pat in pats:
    files.extend(glob.glob(pat))
  files = sorted(set(files), key=sort_key)
  if not files:
    raise FileNotFoundError(f"No files found for '{identifier}'")
  return files


def decode_bool(v: bytes | str | None) -> bool:
  if v is None:
    return False
  if isinstance(v, bytes):
    v = v.decode("utf-8", errors="ignore")
  return str(v).strip().lower() in ("1", "true", "yes", "on")


def decode_float(v: bytes | str | None) -> float:
  if v is None:
    return math.nan
  if isinstance(v, bytes):
    v = v.decode("utf-8", errors="ignore")
  try:
    return float(v)
  except Exception:
    return math.nan


def sign_flip_rate(times: np.ndarray, vals: np.ndarray, deadband: float) -> float:
  if len(times) < 2:
    return math.nan
  transitions = 0
  duration_s = 0.0
  prev_t = None
  prev_sign = 0
  for t, v in zip(times, vals):
    contiguous = True
    if prev_t is not None:
      dt = t - prev_t
      if 0 < dt <= MAX_DT_S:
        duration_s += dt
      else:
        contiguous = False
        prev_sign = 0
    sign = 1 if v > deadband else (-1 if v < -deadband else 0)
    if sign != 0:
      if contiguous and prev_sign != 0 and sign != prev_sign:
        transitions += 1
      prev_sign = sign
    prev_t = t
  return float(transitions / duration_s) if duration_s > 0 else math.nan


def safe_p(arr: np.ndarray, p: int) -> float:
  if len(arr) == 0:
    return math.nan
  return float(np.percentile(arr, p))


def parse_event_payload(txt: str) -> tuple[str, dict[str, float]] | None:
  try:
    obj = json.loads(txt)
  except Exception:
    return None
  msg = obj.get("msg")
  if not isinstance(msg, dict):
    return None
  ev = msg.get("event")
  if not isinstance(ev, str):
    return None
  # Accept any logged event (not just torque_override_ for angle control)
  vals: dict[str, float] = {}
  for k in ("lat_accel_factor", "lat_accel_offset", "friction", "frame",
            "planplus_control", "steer_actuator_delay"):
    if k in msg:
      try:
        vals[k] = float(msg[k])
      except Exception:
        pass
  if "enabled" in msg:
    vals["enabled"] = 1.0 if bool(msg["enabled"]) else 0.0
  return ev, vals


def analyze(files: list[str], min_speed: float, straight_curvature: float, deadband: float) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray, np.ndarray]:
  active = False
  cs = {
    "vEgo": 0.0,
    "steeringPressed": False,
    "leftBlinker": False,
    "rightBlinker": False,
    "cruiseEnabled": False,
  }
  angle_st = {"t": -1e9, "actual": math.nan, "desired": math.nan}

  init: dict[str, float] = {
    "planplusControl": math.nan,
  }
  got_init = False

  events: list[dict[str, float]] = []
  # rows: t, speed, curv_cmd, angle_cmd, angle_ratio, angle_err
  rows: list[tuple[float, float, float, float, float, float]] = []
  # mrows: t, active, cruise, speed, lane_width, lane_center, left_prob, right_prob
  mrows: list[tuple[float, float, float, float, float, float, float, float]] = []

  for msg in LogReader(files):
    which = msg.which()
    t_s = msg.logMonoTime * 1e-9

    if which == "initData" and not got_init:
      entries = {e.key: e.value for e in msg.initData.params.entries}
      init["planplusControl"] = decode_float(entries.get("PlanplusControl"))
      got_init = True

    elif which in ("logMessage", "errorLogMessage"):
      payload = parse_event_payload(str(getattr(msg, which)))
      if payload is not None:
        ev, vals = payload
        vals["t"] = t_s
        vals["event"] = ev  # type: ignore[assignment]
        events.append(vals)

    elif which == "selfdriveState":
      active = bool(getattr(msg.selfdriveState, "active", False) or getattr(msg.selfdriveState, "enabled", False))

    elif which == "carState":
      c = msg.carState
      cs["vEgo"] = float(c.vEgo)
      cs["steeringPressed"] = bool(c.steeringPressed)
      cs["leftBlinker"] = bool(c.leftBlinker)
      cs["rightBlinker"] = bool(c.rightBlinker)
      cs["cruiseEnabled"] = bool(c.cruiseState.enabled)

    elif which == "controlsState":
      lcs = msg.controlsState.lateralControlState
      if lcs.which() == "angleState":
        st = lcs.angleState
        angle_st["t"] = t_s
        angle_st["actual"] = float(getattr(st, "steeringAngleDeg", math.nan))
        angle_st["desired"] = float(getattr(st, "steeringAngleDesiredDeg", math.nan))

    elif which == "carControl":
      cc = msg.carControl
      common = (
        active
        and bool(getattr(cc, "latActive", False))
        and cs["vEgo"] >= min_speed
        and not cs["steeringPressed"]
        and not cs["leftBlinker"]
        and not cs["rightBlinker"]
      )
      if not common:
        continue
      curv_cmd = float(cc.actuators.curvature)
      angle_cmd = float(cc.actuators.steeringAngleDeg)
      angle_ratio = math.nan
      angle_err = math.nan
      if abs(t_s - angle_st["t"]) <= 0.05 and np.isfinite(angle_st["actual"]) and np.isfinite(angle_st["desired"]):
        angle_err = angle_st["desired"] - angle_st["actual"]
        if abs(angle_st["desired"]) > 0.5:
          angle_ratio = abs(angle_st["actual"]) / (abs(angle_st["desired"]) + 1e-9)
      rows.append((t_s, cs["vEgo"], curv_cmd, angle_cmd, angle_ratio, angle_err))

    elif which == "drivingModelData":
      if not (cs["vEgo"] >= min_speed and not cs["steeringPressed"] and not cs["leftBlinker"] and not cs["rightBlinker"]):
        continue
      d = msg.drivingModelData
      ly = float(getattr(d.laneLineMeta, "leftY", math.nan))
      ry = float(getattr(d.laneLineMeta, "rightY", math.nan))
      lp = float(getattr(d.laneLineMeta, "leftProb", math.nan))
      rp = float(getattr(d.laneLineMeta, "rightProb", math.nan))
      if not (np.isfinite(ly) and np.isfinite(ry)):
        continue
      lw = ry - ly
      lc = 0.5 * (ly + ry)
      mrows.append((t_s, 1.0 if active else 0.0, 1.0 if cs["cruiseEnabled"] else 0.0, cs["vEgo"], lw, lc, lp, rp))

  arr = np.asarray(rows, dtype=float) if rows else np.empty((0, 6), dtype=float)
  m_arr = np.asarray(mrows, dtype=float) if mrows else np.empty((0, 8), dtype=float)
  return init, events, arr, m_arr


def metrics_for_window(arr: np.ndarray, m_arr: np.ndarray, t0: float, t1: float,
                       straight_curvature: float, deadband: float, min_samples: int) -> IntervalMetrics | None:
  w = (arr[:, 0] >= t0) & (arr[:, 0] < t1)
  a = arr[w]
  if len(a) < min_samples:
    return None

  straight = np.abs(a[:, 2]) < straight_curvature
  turn = np.abs(a[:, 2]) >= straight_curvature

  # Straight angle error
  straight_err = a[straight, 5]
  straight_err = straight_err[np.isfinite(straight_err)]

  # Turn angle ratio
  ratio_turn = a[turn, 4]
  ratio_turn = ratio_turn[np.isfinite(ratio_turn)]
  err_turn = a[turn, 5]
  err_turn = err_turn[np.isfinite(err_turn)]

  mw = (m_arr[:, 0] >= t0) & (m_arr[:, 0] < t1)
  mm = m_arr[mw]
  center_mean = math.nan
  abs_center_p90 = math.nan
  if len(mm):
    good = (
      (mm[:, 1] > 0.5)
      & (mm[:, 3] >= 20.0)
      & (mm[:, 6] >= 0.6)
      & (mm[:, 7] >= 0.6)
      & (mm[:, 4] > 2.8)
      & (mm[:, 4] < 4.2)
    )
    if np.any(good):
      c = mm[good, 5]
      center_mean = float(np.mean(c))
      abs_center_p90 = safe_p(np.abs(c), 90)

  return IntervalMetrics(
    t0=t0,
    t1=t1,
    duration_s=(t1 - t0),
    samples=len(a),
    straight_samples=int(np.sum(straight)),
    turn_samples=int(np.sum(turn)),
    straight_curv_flip=sign_flip_rate(a[straight, 0], a[straight, 2], deadband) if np.any(straight) else math.nan,
    straight_angle_err_p90=safe_p(np.abs(straight_err), 90) if len(straight_err) else math.nan,
    straight_angle_err_mean=float(np.mean(straight_err)) if len(straight_err) else math.nan,
    turn_over10_pct=(100.0 * float(np.mean(ratio_turn > 1.10))) if len(ratio_turn) else math.nan,
    turn_angle_ratio_p90=safe_p(ratio_turn, 90),
    turn_angle_err_mean=float(np.mean(np.abs(err_turn))) if len(err_turn) else math.nan,
    center_mean=center_mean,
    abs_center_p90=abs_center_p90,
  )


def fmt(v: float, d: int = 3) -> str:
  if not np.isfinite(v):
    return "nan"
  return f"{v:.{d}f}"


def main() -> None:
  args = parse_args()
  files = resolve_files(args.identifier, args.logs_dir)
  files = [f for f in files if f.endswith(".zst")]

  init, events, arr, m_arr = analyze(files, args.min_speed, args.straight_curvature, args.curv_deadband)
  if len(arr) == 0:
    raise SystemExit("No active lateral control samples found.")

  t_start = float(arr[:, 0].min())
  t_end = float(arr[:, 0].max())

  events = [e for e in events if (t_start - 1.0) <= float(e["t"]) <= (t_end + 1.0)]
  events = sorted(events, key=lambda e: float(e["t"]))

  print(f"Route files: {len(files)}")
  print(f"Active samples: {len(arr)}")
  print(f"Window range: {fmt(t_start,1)} -> {fmt(t_end,1)} ({fmt(t_end-t_start,1)}s)")
  print(f"Control type: angle (Ford open-loop)")
  print("Initial settings from initData:")
  print(f"  planplusControl={fmt(init['planplusControl'],4)}")

  if not events:
    print("\nRuntime events: none found")
  else:
    print(f"\nRuntime events: {len(events)}")
    for i, e in enumerate(events, 1):
      ev = str(e.get("event"))
      print(f"  {i:02d}. t={fmt(float(e['t']),1)} ev={ev} {dict((k,v) for k,v in e.items() if k not in ('t','event'))}")

  # Full-route metrics (single interval)
  m = metrics_for_window(arr, m_arr, t_start, t_end, args.straight_curvature, args.curv_deadband, 1)
  if m:
    print("\nFull-route metrics")
    print(f"  duration: {fmt(m.duration_s,1)}s, samples: {m.samples}")
    print(f"  straight: n={m.straight_samples}, curv_flip/s={fmt(m.straight_curv_flip)}, "
          f"|err|p90={fmt(m.straight_angle_err_p90)}, err_mean={fmt(m.straight_angle_err_mean)}")
    print(f"  turns: n={m.turn_samples}, over10%={fmt(m.turn_over10_pct,1)}, "
          f"ratio_p90={fmt(m.turn_angle_ratio_p90)}, |err|_mean={fmt(m.turn_angle_err_mean)}")
    print(f"  center: mean={fmt(m.center_mean)}, |center|p90={fmt(m.abs_center_p90)}")


if __name__ == "__main__":
  main()
