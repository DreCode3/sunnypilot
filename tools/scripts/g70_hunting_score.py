#!/usr/bin/env python3
import argparse
import math
import re
from dataclasses import dataclass

import numpy as np

from openpilot.tools.lib.logreader import LogReader, ReadMode


MAX_DT_S = 0.2


@dataclass
class RouteStats:
  selfdrive_samples: int
  selfdrive_active_samples: int
  op_state_t: list[float]
  op_state_angle: list[float]
  op_state_rate: list[float]
  op_state_speed: list[float]
  stock_state_t: list[float]
  stock_state_angle: list[float]
  stock_state_rate: list[float]
  stock_state_speed: list[float]
  op_torque_t: list[float]
  op_torque: list[float]
  op_torque_curv: list[float]
  op_torque_straight_t: list[float]
  op_torque_straight: list[float]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
    description="Score lane-centering hunting behavior for G70 logs."
  )
  p.add_argument("identifier", help="Route id, onebox URL, connect URL, or direct log path")
  p.add_argument(
    "--mode",
    choices=("auto", "rlog", "qlog"),
    default="auto",
    help="Preferred log type (default: auto)",
  )
  p.add_argument("--min-speed", type=float, default=20.0, help="Minimum speed in m/s (default: 20.0)")
  p.add_argument("--max-speed", type=float, default=None, help="Optional maximum speed in m/s")
  p.add_argument(
    "--torque-deadband",
    type=float,
    default=0.03,
    help="Deadband for torque sign-flip rate (default: 0.03)",
  )
  p.add_argument(
    "--straight-curvature",
    type=float,
    default=8e-4,
    help="Near-straight threshold abs(curvature) (default: 8e-4)",
  )
  p.add_argument(
    "--top-segments",
    type=int,
    default=0,
    help="If >0, print top N segments by hunt metrics (slower)",
  )
  p.add_argument(
    "--min-segment-samples",
    type=int,
    default=80,
    help="Minimum samples for segment ranking (default: 80)",
  )
  return p.parse_args()


def mode_from_arg(mode: str) -> ReadMode:
  if mode == "rlog":
    return ReadMode.RLOG
  if mode == "qlog":
    return ReadMode.QLOG
  return ReadMode.AUTO


def estimate_hz(times: list[float]) -> float:
  if len(times) < 2:
    return math.nan
  dt = np.diff(np.asarray(times))
  dt = dt[(dt > 0) & (dt <= MAX_DT_S)]
  if len(dt) == 0:
    return math.nan
  return float(1.0 / np.median(dt))


def pctl(vals: list[float] | np.ndarray, p: int) -> float:
  arr = np.asarray(vals, dtype=float)
  if len(arr) == 0:
    return math.nan
  if p == 100:
    return float(np.max(arr))
  return float(np.percentile(arr, p))


def moving_average(vals: np.ndarray, win: int) -> np.ndarray:
  if len(vals) == 0:
    return vals
  if win <= 1:
    return vals.copy()
  if win >= len(vals):
    return np.full_like(vals, float(np.mean(vals)))
  kernel = np.ones(win, dtype=float) / float(win)
  return np.convolve(vals, kernel, mode="same")


def detrended_stats(vals: list[float], sample_hz: float, win_seconds: float) -> dict[str, float]:
  arr = np.asarray(vals, dtype=float)
  if len(arr) == 0 or not np.isfinite(sample_hz):
    return {"std": math.nan, "p90": math.nan, "p99": math.nan, "max_abs": math.nan}
  win = max(3, int(round(sample_hz * win_seconds)))
  detrended = arr - moving_average(arr, win)
  return {
    "std": float(np.std(detrended)),
    "p90": float(np.percentile(detrended, 90)),
    "p99": float(np.percentile(detrended, 99)),
    "max_abs": float(np.max(np.abs(detrended))),
  }


def sign_flip_stats(times: list[float], vals: list[float], deadband: float) -> dict[str, float]:
  if len(times) < 2 or len(vals) < 2:
    return {"transitions": 0.0, "duration_s": 0.0, "flips_per_s": math.nan, "nonzero_samples": 0.0}

  transitions = 0
  duration_s = 0.0
  nonzero_samples = 0
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
      nonzero_samples += 1
      if contiguous and prev_sign != 0 and sign != prev_sign:
        transitions += 1
      prev_sign = sign

    prev_t = t

  flips_per_s = float(transitions / duration_s) if duration_s > 0 else math.nan
  return {
    "transitions": float(transitions),
    "duration_s": float(duration_s),
    "flips_per_s": flips_per_s,
    "nonzero_samples": float(nonzero_samples),
  }


def in_speed_window(v_ego: float, min_speed: float, max_speed: float | None) -> bool:
  if v_ego < min_speed:
    return False
  if max_speed is not None and v_ego > max_speed:
    return False
  return True


def analyze_identifier(identifier: str, mode: ReadMode, args: argparse.Namespace) -> RouteStats:
  lr = LogReader(identifier, default_mode=mode)

  active = False
  cs = {
    "vEgo": 0.0,
    "steeringPressed": False,
    "leftBlinker": False,
    "rightBlinker": False,
    "steeringAngleDeg": 0.0,
    "steeringRateDeg": 0.0,
    "cruiseEnabled": False,
  }

  selfdrive_samples = 0
  selfdrive_active_samples = 0

  op_state_t: list[float] = []
  op_state_angle: list[float] = []
  op_state_rate: list[float] = []
  op_state_speed: list[float] = []

  stock_state_t: list[float] = []
  stock_state_angle: list[float] = []
  stock_state_rate: list[float] = []
  stock_state_speed: list[float] = []

  op_torque_t: list[float] = []
  op_torque: list[float] = []
  op_torque_curv: list[float] = []
  op_torque_straight_t: list[float] = []
  op_torque_straight: list[float] = []

  for msg in lr:
    which = msg.which()

    if which == "selfdriveState":
      state_msg = msg.selfdriveState
      current_active = bool(getattr(state_msg, "active", False) or getattr(state_msg, "enabled", False))
      active = current_active
      selfdrive_samples += 1
      selfdrive_active_samples += int(current_active)

    elif which == "carState":
      car_state = msg.carState
      cs["vEgo"] = float(car_state.vEgo)
      cs["steeringPressed"] = bool(car_state.steeringPressed)
      cs["leftBlinker"] = bool(car_state.leftBlinker)
      cs["rightBlinker"] = bool(car_state.rightBlinker)
      cs["steeringAngleDeg"] = float(car_state.steeringAngleDeg)
      cs["steeringRateDeg"] = float(car_state.steeringRateDeg)
      cs["cruiseEnabled"] = bool(car_state.cruiseState.enabled)

      common = (
        in_speed_window(cs["vEgo"], args.min_speed, args.max_speed)
        and not cs["steeringPressed"]
        and not cs["leftBlinker"]
        and not cs["rightBlinker"]
      )
      t_s = msg.logMonoTime * 1e-9
      if common and active:
        op_state_t.append(t_s)
        op_state_angle.append(cs["steeringAngleDeg"])
        op_state_rate.append(cs["steeringRateDeg"])
        op_state_speed.append(cs["vEgo"])
      if common and (not active) and cs["cruiseEnabled"]:
        stock_state_t.append(t_s)
        stock_state_angle.append(cs["steeringAngleDeg"])
        stock_state_rate.append(cs["steeringRateDeg"])
        stock_state_speed.append(cs["vEgo"])

    elif which == "carControl":
      common = (
        active
        and in_speed_window(cs["vEgo"], args.min_speed, args.max_speed)
        and not cs["steeringPressed"]
        and not cs["leftBlinker"]
        and not cs["rightBlinker"]
      )
      if not common:
        continue

      cc = msg.carControl
      if not bool(getattr(cc, "latActive", False)):
        continue

      torque = float(cc.actuators.torque)
      curvature = float(getattr(cc, "currentCurvature", 0.0))
      t_s = msg.logMonoTime * 1e-9

      op_torque_t.append(t_s)
      op_torque.append(torque)
      op_torque_curv.append(curvature)

      if abs(curvature) < args.straight_curvature:
        op_torque_straight_t.append(t_s)
        op_torque_straight.append(torque)

  return RouteStats(
    selfdrive_samples=selfdrive_samples,
    selfdrive_active_samples=selfdrive_active_samples,
    op_state_t=op_state_t,
    op_state_angle=op_state_angle,
    op_state_rate=op_state_rate,
    op_state_speed=op_state_speed,
    stock_state_t=stock_state_t,
    stock_state_angle=stock_state_angle,
    stock_state_rate=stock_state_rate,
    stock_state_speed=stock_state_speed,
    op_torque_t=op_torque_t,
    op_torque=op_torque,
    op_torque_curv=op_torque_curv,
    op_torque_straight_t=op_torque_straight_t,
    op_torque_straight=op_torque_straight,
  )


def fmt(value: float, digits: int = 3) -> str:
  if not np.isfinite(value):
    return "nan"
  return f"{value:.{digits}f}"


def print_state_metrics(label: str, times: list[float], speed: list[float], steering_rate: list[float], steering_angle: list[float]) -> None:
  print(f"\n{label}")
  print(f"  samples: {len(times)}")
  if len(times) < 2:
    return

  fs = estimate_hz(times)
  d5 = detrended_stats(steering_angle, fs, 5.0)
  d2 = detrended_stats(steering_angle, fs, 2.0)

  print(f"  sample_hz: {fmt(fs, 2)}")
  print(
    "  speed_mps mean/std/p90/p99: "
    f"{fmt(float(np.mean(speed)), 2)} / {fmt(float(np.std(speed)), 2)} / "
    f"{fmt(pctl(speed, 90), 2)} / {fmt(pctl(speed, 99), 2)}"
  )
  abs_rate = np.abs(np.asarray(steering_rate, dtype=float))
  print(
    "  abs_steeringRateDeg mean/std/p90/p99/max: "
    f"{fmt(float(np.mean(abs_rate)), 3)} / {fmt(float(np.std(abs_rate)), 3)} / "
    f"{fmt(pctl(abs_rate, 90), 3)} / {fmt(pctl(abs_rate, 99), 3)} / {fmt(pctl(abs_rate, 100), 3)}"
  )
  print(
    "  detrended_steeringAngleDeg_5s std/p90/p99/max_abs: "
    f"{fmt(d5['std'])} / {fmt(d5['p90'])} / {fmt(d5['p99'])} / {fmt(d5['max_abs'])}"
  )
  print(
    "  detrended_steeringAngleDeg_2s std/p90/p99/max_abs: "
    f"{fmt(d2['std'])} / {fmt(d2['p90'])} / {fmt(d2['p99'])} / {fmt(d2['max_abs'])}"
  )


def print_torque_metrics(stats: RouteStats, torque_deadband: float) -> None:
  print("\nOP torque metrics (carControl.latActive)")
  print(f"  samples_all: {len(stats.op_torque_t)}")
  if len(stats.op_torque_t) < 2:
    return

  abs_torque = np.abs(np.asarray(stats.op_torque, dtype=float))
  flip_all = sign_flip_stats(stats.op_torque_t, stats.op_torque, torque_deadband)
  flip_straight = sign_flip_stats(stats.op_torque_straight_t, stats.op_torque_straight, torque_deadband)

  print(
    "  abs_torque p50/p90/p95/p99/max: "
    f"{fmt(pctl(abs_torque, 50))} / {fmt(pctl(abs_torque, 90))} / "
    f"{fmt(pctl(abs_torque, 95))} / {fmt(pctl(abs_torque, 99))} / {fmt(pctl(abs_torque, 100))}"
  )
  print(
    f"  sign_flip_rate_all deadband={torque_deadband:.3f}: "
    f"{fmt(flip_all['flips_per_s'])}/s "
    f"(transitions={int(flip_all['transitions'])}, duration_s={fmt(flip_all['duration_s'], 1)})"
  )
  print(f"  samples_straight: {len(stats.op_torque_straight_t)}")
  if len(stats.op_torque_straight_t) >= 2:
    abs_torque_st = np.abs(np.asarray(stats.op_torque_straight, dtype=float))
    print(
      "  abs_torque_straight p50/p90/p95/p99/max: "
      f"{fmt(pctl(abs_torque_st, 50))} / {fmt(pctl(abs_torque_st, 90))} / "
      f"{fmt(pctl(abs_torque_st, 95))} / {fmt(pctl(abs_torque_st, 99))} / {fmt(pctl(abs_torque_st, 100))}"
    )
    print(
      f"  sign_flip_rate_straight deadband={torque_deadband:.3f}: "
      f"{fmt(flip_straight['flips_per_s'])}/s "
      f"(transitions={int(flip_straight['transitions'])}, duration_s={fmt(flip_straight['duration_s'], 1)})"
    )


def extract_segment_index(identifier: str) -> int:
  patterns = (
    r"/(\d+)/(?:qlog|rlog)\.zst",
    r"--(\d+)--(?:qlog|rlog)\.zst",
    r"/(\d+)/(?:qlog|rlog)\.",
  )
  for pat in patterns:
    m = re.search(pat, identifier)
    if m:
      return int(m.group(1))
  return -1


def analyze_segment(identifier: str, mode: ReadMode, args: argparse.Namespace) -> dict[str, float] | None:
  active = False
  cs = {
    "vEgo": 0.0,
    "steeringPressed": False,
    "leftBlinker": False,
    "rightBlinker": False,
    "steeringAngleDeg": 0.0,
    "cruiseEnabled": False,
  }

  torque_t: list[float] = []
  torque_v: list[float] = []
  angle_t: list[float] = []
  angle_v: list[float] = []

  for msg in LogReader(identifier, default_mode=mode):
    which = msg.which()
    if which == "selfdriveState":
      active = bool(getattr(msg.selfdriveState, "active", False) or getattr(msg.selfdriveState, "enabled", False))
    elif which == "carState":
      cs_msg = msg.carState
      cs["vEgo"] = float(cs_msg.vEgo)
      cs["steeringPressed"] = bool(cs_msg.steeringPressed)
      cs["leftBlinker"] = bool(cs_msg.leftBlinker)
      cs["rightBlinker"] = bool(cs_msg.rightBlinker)
      cs["steeringAngleDeg"] = float(cs_msg.steeringAngleDeg)
      cs["cruiseEnabled"] = bool(cs_msg.cruiseState.enabled)
      common = (
        active
        and in_speed_window(cs["vEgo"], args.min_speed, args.max_speed)
        and not cs["steeringPressed"]
        and not cs["leftBlinker"]
        and not cs["rightBlinker"]
      )
      if common:
        angle_t.append(msg.logMonoTime * 1e-9)
        angle_v.append(cs["steeringAngleDeg"])
    elif which == "carControl":
      common = (
        active
        and in_speed_window(cs["vEgo"], args.min_speed, args.max_speed)
        and not cs["steeringPressed"]
        and not cs["leftBlinker"]
        and not cs["rightBlinker"]
      )
      if not common:
        continue
      cc = msg.carControl
      if not bool(getattr(cc, "latActive", False)):
        continue
      curvature = float(getattr(cc, "currentCurvature", 0.0))
      if abs(curvature) >= args.straight_curvature:
        continue
      torque_t.append(msg.logMonoTime * 1e-9)
      torque_v.append(float(cc.actuators.torque))

  if len(torque_t) < args.min_segment_samples:
    return None

  flip = sign_flip_stats(torque_t, torque_v, args.torque_deadband)
  abs_torque = np.abs(np.asarray(torque_v, dtype=float))

  fs = estimate_hz(angle_t)
  d5 = detrended_stats(angle_v, fs, 5.0)

  return {
    "segment": float(extract_segment_index(identifier)),
    "samples": float(len(torque_t)),
    "flip_rate": float(flip["flips_per_s"]),
    "flip_transitions": float(flip["transitions"]),
    "duration_s": float(flip["duration_s"]),
    "torque_p90": float(pctl(abs_torque, 90)),
    "detrended5_std": float(d5["std"]),
  }


def print_top_segments(identifier: str, mode: ReadMode, args: argparse.Namespace) -> None:
  if args.top_segments <= 0:
    return

  print("\nAnalyzing per-segment breakdown...")
  route_reader = LogReader(identifier, default_mode=mode)
  rows = []
  for seg_identifier in route_reader.logreader_identifiers:
    row = analyze_segment(seg_identifier, mode, args)
    if row is not None:
      rows.append(row)

  if len(rows) == 0:
    print("No segment-level rows met the minimum sample threshold.")
    return

  top_n = min(args.top_segments, len(rows))
  by_flip = sorted(rows, key=lambda r: (-(r["flip_rate"] if np.isfinite(r["flip_rate"]) else -1e9), -r["samples"]))[:top_n]
  by_osc = sorted(rows, key=lambda r: (-(r["detrended5_std"] if np.isfinite(r["detrended5_std"]) else -1e9), -r["samples"]))[:top_n]

  print(f"\nTop {top_n} segments by torque sign-flip rate (straight, db={args.torque_deadband:.3f})")
  for row in by_flip:
    print(
      f"  seg={int(row['segment']) if row['segment'] >= 0 else -1} "
      f"samples={int(row['samples'])} "
      f"flip/s={fmt(row['flip_rate'])} "
      f"torque_p90={fmt(row['torque_p90'])} "
      f"detrended5_std={fmt(row['detrended5_std'])}"
    )

  print(f"\nTop {top_n} segments by detrended steering-angle std (5s)")
  for row in by_osc:
    print(
      f"  seg={int(row['segment']) if row['segment'] >= 0 else -1} "
      f"samples={int(row['samples'])} "
      f"detrended5_std={fmt(row['detrended5_std'])} "
      f"flip/s={fmt(row['flip_rate'])} "
      f"torque_p90={fmt(row['torque_p90'])}"
    )


def main() -> None:
  args = parse_args()
  mode = mode_from_arg(args.mode)

  print("G70 Hunting Score")
  print(f"identifier: {args.identifier}")
  print(
    f"filters: min_speed={args.min_speed} m/s, "
    f"max_speed={args.max_speed if args.max_speed is not None else 'none'} m/s, "
    f"torque_deadband={args.torque_deadband}, straight_curvature={args.straight_curvature}"
  )

  stats = analyze_identifier(args.identifier, mode, args)
  active_pct = (100.0 * stats.selfdrive_active_samples / stats.selfdrive_samples) if stats.selfdrive_samples else math.nan
  print(
    f"selfdrive active: {stats.selfdrive_active_samples}/{stats.selfdrive_samples} "
    f"({fmt(active_pct, 2)}%)"
  )

  print_state_metrics(
    "OP-active steering metrics (carState)",
    stats.op_state_t,
    stats.op_state_speed,
    stats.op_state_rate,
    stats.op_state_angle,
  )
  print_state_metrics(
    "Stock-like steering metrics (carState inactive + cruise)",
    stats.stock_state_t,
    stats.stock_state_speed,
    stats.stock_state_rate,
    stats.stock_state_angle,
  )
  print_torque_metrics(stats, args.torque_deadband)
  print_top_segments(args.identifier, mode, args)


if __name__ == "__main__":
  main()
