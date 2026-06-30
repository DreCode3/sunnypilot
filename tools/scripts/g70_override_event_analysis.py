#!/usr/bin/env python3
"""Analyze runtime torque override events and correlate with lane/turn metrics.

Example:
  PYTHONPATH=. python tools/scripts/g70_override_event_analysis.py \
    "G70 logs/f1d7ca35cfefb3af_0000000d--1b7a6a5bb3--*--rlog.zst"
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
  straight_flip_db03: float
  straight_torque_p90: float
  turn_over10_pct: float
  turn_ratio_p90: float
  turn_abs_err_mean: float
  center_mean: float
  abs_center_p90: float


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Correlate runtime torque override events to driving metrics.")
  p.add_argument("identifier", help="Route glob/path/onebox URL")
  p.add_argument("--logs-dir", default="G70 logs", help="Default logs dir when identifier is a route token")
  p.add_argument("--min-speed", type=float, default=20.0)
  p.add_argument("--straight-curvature", type=float, default=8e-4)
  p.add_argument("--torque-deadband", type=float, default=0.03)
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
  if not isinstance(ev, str) or not ev.startswith("torque_override_"):
    return None
  vals: dict[str, float] = {}
  for k in ("lat_accel_factor", "lat_accel_offset", "friction", "frame"):
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
  tq = {"t": -1e9, "actual": math.nan, "desired": math.nan}

  init = {
    "laf": math.nan,
    "lao": math.nan,
    "fr": math.nan,
    "overrideEnabled": 0.0,
  }
  got_init = False

  events: list[dict[str, float]] = []
  rows: list[tuple[float, float, float, float, float, float]] = []
  mrows: list[tuple[float, float, float, float, float, float, float, float]] = []

  for msg in LogReader(files):
    which = msg.which()
    t_s = msg.logMonoTime * 1e-9

    if which == "initData" and not got_init:
      entries = {e.key: e.value for e in msg.initData.params.entries}
      init["laf"] = decode_float(entries.get("TorqueParamsOverrideLatAccelFactor"))
      init["lao"] = decode_float(entries.get("TorqueParamsOverrideLatAccelOffset"))
      init["fr"] = decode_float(entries.get("TorqueParamsOverrideFriction"))
      init["overrideEnabled"] = 1.0 if decode_bool(entries.get("TorqueParamsOverrideEnabled")) else 0.0
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
      st = getattr(msg.controlsState.lateralControlState, "torqueState", None)
      if st is not None:
        tq["t"] = t_s
        tq["actual"] = float(getattr(st, "actualLateralAccel", math.nan))
        tq["desired"] = float(getattr(st, "desiredLateralAccel", math.nan))

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
      curv = float(getattr(cc, "currentCurvature", math.nan))
      torque = float(cc.actuators.torque)
      ratio = math.nan
      abs_err = math.nan
      if abs(t_s - tq["t"]) <= 0.05 and np.isfinite(tq["actual"]) and np.isfinite(tq["desired"]) and abs(tq["desired"]) > 0.3:
        ratio = abs(tq["actual"]) / (abs(tq["desired"]) + 1e-9)
        abs_err = abs(tq["actual"]) - abs(tq["desired"])
      rows.append((t_s, cs["vEgo"], torque, curv, ratio, abs_err))

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


def metrics_for_window(arr: np.ndarray, m_arr: np.ndarray, t0: float, t1: float, straight_curvature: float, deadband: float, min_samples: int) -> IntervalMetrics | None:
  w = (arr[:, 0] >= t0) & (arr[:, 0] < t1)
  a = arr[w]
  if len(a) < min_samples:
    return None

  straight = np.abs(a[:, 3]) < straight_curvature
  turn = np.abs(a[:, 3]) >= straight_curvature
  ratio_turn = a[turn, 4]
  ratio_turn = ratio_turn[np.isfinite(ratio_turn)]
  abs_err_turn = a[turn, 5]
  abs_err_turn = abs_err_turn[np.isfinite(abs_err_turn)]

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
    straight_flip_db03=sign_flip_rate(a[straight, 0], a[straight, 2], deadband) if np.any(straight) else math.nan,
    straight_torque_p90=safe_p(np.abs(a[straight, 2]), 90) if np.any(straight) else math.nan,
    turn_over10_pct=(100.0 * float(np.mean(ratio_turn > 1.10))) if len(ratio_turn) else math.nan,
    turn_ratio_p90=safe_p(ratio_turn, 90),
    turn_abs_err_mean=float(np.mean(abs_err_turn)) if len(abs_err_turn) else math.nan,
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

  init, events, arr, m_arr = analyze(files, args.min_speed, args.straight_curvature, args.torque_deadband)
  if len(arr) == 0:
    raise SystemExit("No active lateral control samples found.")

  t_start = float(arr[:, 0].min())
  t_end = float(arr[:, 0].max())

  events = [e for e in events if (t_start - 1.0) <= float(e["t"]) <= (t_end + 1.0)]
  events = sorted(events, key=lambda e: float(e["t"]))

  boundaries = [t_start] + [float(e["t"]) for e in events if t_start < float(e["t"]) < t_end] + [t_end]
  boundaries = sorted(set(boundaries))
  if len(boundaries) < 2:
    boundaries = [t_start, t_end]

  print(f"Route files: {len(files)}")
  print(f"Window range: {fmt(t_start,1)} -> {fmt(t_end,1)} (s mono)")
  print("Initial settings from initData:")
  print(f"  laf={fmt(init['laf'],4)} lao={fmt(init['lao'],4)} fr={fmt(init['fr'],4)} overrideEnabled={int(init['overrideEnabled'])}")

  if not events:
    print("\nRuntime override events: none found")
  else:
    print(f"\nRuntime override events: {len(events)}")
    for i, e in enumerate(events, 1):
      ev = str(e.get("event"))
      laf = e.get("lat_accel_factor", math.nan)
      lao = e.get("lat_accel_offset", math.nan)
      fr = e.get("friction", math.nan)
      en = e.get("enabled", math.nan)
      frame = e.get("frame", math.nan)
      print(f"  {i:02d}. t={fmt(float(e['t']),1)} ev={ev} laf={fmt(float(laf),4)} lao={fmt(float(lao),4)} fr={fmt(float(fr),4)} enabled={fmt(float(en),0)} frame={fmt(float(frame),0)}")

  print("\nInterval metrics")
  print("idx,t0,t1,dur_s,samples,straight_n,turn_n,laf,lao,fr,straight_flip_db03,straight_torque_p90,turn_over10_pct,turn_ratio_p90,turn_abs_err_mean,center_mean,abs_center_p90")

  current = {
    "laf": init["laf"],
    "lao": init["lao"],
    "fr": init["fr"],
    "overrideEnabled": init["overrideEnabled"],
  }
  ev_idx = 0
  for i in range(len(boundaries) - 1):
    b0 = boundaries[i]
    b1 = boundaries[i + 1]

    while ev_idx < len(events) and float(events[ev_idx]["t"]) <= b0 + 1e-9:
      e = events[ev_idx]
      if str(e.get("event")) == "torque_override_params_applied":
        if "lat_accel_factor" in e:
          current["laf"] = float(e["lat_accel_factor"])
        if "lat_accel_offset" in e:
          current["lao"] = float(e["lat_accel_offset"])
        if "friction" in e:
          current["fr"] = float(e["friction"])
      elif str(e.get("event")) == "torque_override_enabled_changed" and "enabled" in e:
        current["overrideEnabled"] = float(e["enabled"])
      ev_idx += 1

    m = metrics_for_window(arr, m_arr, b0, b1, args.straight_curvature, args.torque_deadband, args.min_interval_samples)
    if m is None:
      continue

    print(
      f"{i:02d},{fmt(m.t0,1)},{fmt(m.t1,1)},{fmt(m.duration_s,1)},{m.samples},{m.straight_samples},{m.turn_samples},"
      f"{fmt(float(current['laf']),4)},{fmt(float(current['lao']),4)},{fmt(float(current['fr']),4)},"
      f"{fmt(m.straight_flip_db03)},{fmt(m.straight_torque_p90)},{fmt(m.turn_over10_pct,1)},{fmt(m.turn_ratio_p90)},"
      f"{fmt(m.turn_abs_err_mean)},{fmt(m.center_mean)},{fmt(m.abs_center_p90)}"
    )


if __name__ == "__main__":
  main()

