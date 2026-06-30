#!/usr/bin/env python3
"""Compare local Explorer ST route logs and quantify lane-centering/steering behavior.

Adapted from g70_route_compare.py for Ford angle-based lateral control.
Ford uses open-loop angle control (not torque), so metrics use steering angle
error, curvature commands, and model lane-center data.

Usage examples:
  PYTHONPATH=. .venv311/bin/python3 tools/scripts/explorer_st_route_compare.py \
    962fb90eaa5b23f1_00000018--024bf527b5

  PYTHONPATH=. .venv311/bin/python3 tools/scripts/explorer_st_route_compare.py \
    "https://useradmin.comma.ai/?onebox=962fb90eaa5b23f1%7C00000018--024bf527b5"
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import numpy as np

from openpilot.tools.lib.logreader import LogReader

MAX_DT_S = 0.2


@dataclass
class RouteMetrics:
  name: str
  files: int
  selfdrive_active_pct: float
  overriding_pct_of_active: float
  op_samples: int
  speed_mean: float
  speed_p90: float
  # Straight-road metrics (curvature-based, not torque)
  straight_samples: int
  straight_curv_flip_db: float          # curvature command sign-flip rate
  straight_angle_cmd_p90: float         # |commanded steering angle| p90
  straight_angle_cmd_mean: float        # mean commanded steering angle (bias)
  straight_angle_err_mean: float        # mean angle error (desired - actual)
  straight_angle_err_p90: float         # |angle error| p90
  straight_angle_err_bias: float        # mean error / abs mean error
  # Turn metrics
  turn_samples: int
  turn_angle_cmd_p90: float
  turn_angle_ratio_p50: float           # |actual| / |desired| angle ratio
  turn_angle_ratio_p90: float
  turn_over10_pct: float                # % of turns where ratio > 1.10
  turn_angle_err_mean: float            # mean absolute angle error in turns
  left_turn_angle_err_mean: float
  right_turn_angle_err_mean: float
  left_turn_over10_pct: float
  right_turn_over10_pct: float


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Compare local Explorer ST route behavior (hunting/oversteer/centering).")
  p.add_argument("route_a", help="Route prefix, onebox URL, or local path/glob")
  p.add_argument("route_b", nargs="?", default=None, help="Optional second route to compare")
  p.add_argument("--logs-dir", default="explorer_st_logs", help="Local logs directory (default: explorer_st_logs)")
  p.add_argument("--min-speed", type=float, default=20.0, help="Minimum speed m/s (default: 20)")
  p.add_argument("--straight-curvature", type=float, default=5e-4,
                 help="|curvature| threshold for straight (default: 5e-4, tighter than G70 due to angle control)")
  p.add_argument("--curv-deadband", type=float, default=1e-4,
                 help="Curvature deadband for sign-flip rate (default: 1e-4)")
  p.add_argument("--show-segment-groups", action="store_true", help="Print per-segment override groups")
  return p.parse_args()


def _parse_boolish(v: bytes | str | None) -> bool:
  if v is None:
    return False
  if isinstance(v, bytes):
    try:
      v = v.decode("utf-8")
    except UnicodeDecodeError:
      return False
  return str(v).strip().lower() in ("1", "true", "yes", "on")


def _decode(v: bytes | str | None) -> str:
  if v is None:
    return "None"
  if isinstance(v, bytes):
    try:
      return v.decode("utf-8")
    except UnicodeDecodeError:
      return repr(v)
  return str(v)


def _parse_route_token(route: str) -> str:
  route = route.strip()

  if "useradmin.comma.ai" in route:
    query = parse_qs(urlparse(route).query)
    if "onebox" in query and len(query["onebox"]) > 0:
      route = query["onebox"][0]

  if "|" in route:
    dongle, route_id = route.split("|", 1)
    return f"{dongle}_{route_id}"

  return route


def _sort_key(path: str) -> tuple[int, int, str]:
  base = os.path.basename(path)
  m = re.search(r"--(\d+)--(rlog|qlog)\.zst$", base)
  if not m:
    return (1_000_000, 1, base)
  seg = int(m.group(1))
  typ = 0 if m.group(2) == "rlog" else 1
  return (seg, typ, base)


def resolve_local_files(identifier: str, logs_dir: str) -> list[str]:
  if os.path.exists(identifier):
    return [identifier]

  globbed = sorted(glob.glob(identifier), key=_sort_key)
  if len(globbed) > 0:
    return globbed

  token = _parse_route_token(identifier)

  if any(ch in token for ch in "*?[]"):
    patterns = [
      os.path.join(logs_dir, token),
      os.path.join(logs_dir, f"{token}--*--rlog.zst"),
      os.path.join(logs_dir, f"{token}--*--qlog.zst"),
    ]
  else:
    patterns = [
      os.path.join(logs_dir, f"{token}--*--rlog.zst"),
      os.path.join(logs_dir, f"{token}--*--qlog.zst"),
      os.path.join(logs_dir, token),
    ]

  files: list[str] = []
  for pat in patterns:
    files.extend(glob.glob(pat))

  files = sorted(set(files), key=_sort_key)
  if len(files) == 0:
    raise FileNotFoundError(f"No local logs matched '{identifier}' (resolved token '{token}') in '{logs_dir}'")
  return files


def estimate_hz(times: np.ndarray) -> float:
  if len(times) < 2:
    return math.nan
  dt = np.diff(times)
  dt = dt[(dt > 0) & (dt <= MAX_DT_S)]
  if len(dt) == 0:
    return math.nan
  return float(1.0 / np.median(dt))


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


def parse_init_settings(path: str) -> dict[str, str | bool | int]:
  seg_m = re.search(r"--(\d+)--", os.path.basename(path))
  seg = int(seg_m.group(1)) if seg_m else -1

  out: dict[str, str | bool | int] = {
    "seg": seg,
    "gitCommit": "None",
    "gitBranch": "None",
    "planplusControl": "None",
  }

  for msg in LogReader(path):
    if msg.which() != "initData":
      continue

    init = msg.initData
    entries = {e.key: e.value for e in init.params.entries}

    out["gitCommit"] = str(getattr(init, "gitCommit", ""))
    out["gitBranch"] = str(getattr(init, "gitBranch", ""))
    out["planplusControl"] = _decode(entries.get("PlanplusControl"))
    break

  return out


def analyze_files(files: list[str], min_speed: float, straight_curvature: float,
                  curv_deadband: float, label: str) -> tuple[RouteMetrics, np.ndarray, np.ndarray]:
  active = False
  cs = {
    "vEgo": 0.0,
    "steeringPressed": False,
    "leftBlinker": False,
    "rightBlinker": False,
    "cruiseEnabled": False,
  }
  angle_state = {
    "t": -1e9,
    "actual": math.nan,
    "desired": math.nan,
  }

  selfdrive_total = 0
  selfdrive_active = 0
  selfdrive_overriding = 0

  # rows: t, speed, curvature_cmd, angle_cmd, angle_ratio, angle_err
  rows: list[tuple[float, float, float, float, float, float]] = []
  model_rows: list[tuple[float, float, float, float, float, float, float]] = []

  for msg in LogReader(files):
    which = msg.which()
    t_s = msg.logMonoTime * 1e-9

    if which == "selfdriveState":
      selfdrive_total += 1
      state = str(getattr(msg.selfdriveState, "state", ""))
      active = bool(getattr(msg.selfdriveState, "active", False) or getattr(msg.selfdriveState, "enabled", False))
      selfdrive_active += int(active)
      selfdrive_overriding += int(state == "overriding")

    elif which == "carState":
      c = msg.carState
      cs["vEgo"] = float(c.vEgo)
      cs["steeringPressed"] = bool(c.steeringPressed)
      cs["leftBlinker"] = bool(c.leftBlinker)
      cs["rightBlinker"] = bool(c.rightBlinker)
      cs["cruiseEnabled"] = bool(c.cruiseState.enabled)

    elif which == "controlsState":
      lcs = msg.controlsState.lateralControlState
      w = lcs.which()
      if w == "angleState":
        st = lcs.angleState
        angle_state["t"] = t_s
        angle_state["actual"] = float(getattr(st, "steeringAngleDeg", math.nan))
        angle_state["desired"] = float(getattr(st, "steeringAngleDesiredDeg", math.nan))

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

      curv_cmd = float(getattr(cc, "actuators", cc).curvature if hasattr(cc, "actuators") else math.nan)
      angle_cmd = float(cc.actuators.steeringAngleDeg)
      angle_ratio = math.nan
      angle_err = math.nan

      if abs(t_s - angle_state["t"]) <= 0.05 and np.isfinite(angle_state["actual"]) and np.isfinite(angle_state["desired"]):
        angle_err = angle_state["desired"] - angle_state["actual"]
        if abs(angle_state["desired"]) > 0.5:  # threshold for meaningful angle
          angle_ratio = abs(angle_state["actual"]) / (abs(angle_state["desired"]) + 1e-9)

      curv_cmd = float(cc.actuators.curvature)
      rows.append((t_s, cs["vEgo"], curv_cmd, angle_cmd, angle_ratio, angle_err))

    elif which == "drivingModelData":
      if not (cs["vEgo"] >= min_speed and not cs["steeringPressed"] and not cs["leftBlinker"] and not cs["rightBlinker"]):
        continue

      d = msg.drivingModelData
      left_y = float(getattr(d.laneLineMeta, "leftY", math.nan))
      right_y = float(getattr(d.laneLineMeta, "rightY", math.nan))
      left_prob = float(getattr(d.laneLineMeta, "leftProb", math.nan))
      right_prob = float(getattr(d.laneLineMeta, "rightProb", math.nan))

      if not (np.isfinite(left_y) and np.isfinite(right_y)):
        continue

      lane_width = right_y - left_y
      lane_center = 0.5 * (left_y + right_y)
      model_rows.append((
        1.0 if active else 0.0,
        1.0 if cs["cruiseEnabled"] else 0.0,
        cs["vEgo"],
        lane_width,
        lane_center,
        left_prob,
        right_prob,
      ))

  arr = np.asarray(rows, dtype=float) if len(rows) else np.empty((0, 6), dtype=float)
  m_arr = np.asarray(model_rows, dtype=float) if len(model_rows) else np.empty((0, 7), dtype=float)

  if len(arr) == 0:
    metrics = RouteMetrics(
      name=label, files=len(files),
      selfdrive_active_pct=(100.0 * selfdrive_active / selfdrive_total) if selfdrive_total else math.nan,
      overriding_pct_of_active=(100.0 * selfdrive_overriding / selfdrive_active) if selfdrive_active else math.nan,
      op_samples=0, speed_mean=math.nan, speed_p90=math.nan,
      straight_samples=0, straight_curv_flip_db=math.nan,
      straight_angle_cmd_p90=math.nan, straight_angle_cmd_mean=math.nan,
      straight_angle_err_mean=math.nan, straight_angle_err_p90=math.nan, straight_angle_err_bias=math.nan,
      turn_samples=0, turn_angle_cmd_p90=math.nan,
      turn_angle_ratio_p50=math.nan, turn_angle_ratio_p90=math.nan, turn_over10_pct=math.nan,
      turn_angle_err_mean=math.nan,
      left_turn_angle_err_mean=math.nan, right_turn_angle_err_mean=math.nan,
      left_turn_over10_pct=math.nan, right_turn_over10_pct=math.nan,
    )
    return metrics, arr, m_arr

  t = arr[:, 0]
  speed = arr[:, 1]
  curv_cmd = arr[:, 2]
  angle_cmd = arr[:, 3]
  angle_ratio = arr[:, 4]
  angle_err = arr[:, 5]

  straight = np.abs(curv_cmd) < straight_curvature
  turn = np.abs(curv_cmd) >= straight_curvature

  # Straight-road metrics
  straight_angle_cmd = angle_cmd[straight]
  straight_err = angle_err[straight & np.isfinite(angle_err)]

  abs_err_mean = float(np.mean(np.abs(straight_err))) if len(straight_err) else math.nan
  err_bias = (float(np.mean(straight_err) / (abs_err_mean + 1e-9)) if np.isfinite(abs_err_mean) else math.nan)

  # Turn metrics
  ratio_turn = angle_ratio[turn & np.isfinite(angle_ratio)]
  err_turn = angle_err[turn & np.isfinite(angle_err)]

  left_turn = curv_cmd > straight_curvature
  right_turn = curv_cmd < -straight_curvature

  left_ratio = angle_ratio[left_turn & np.isfinite(angle_ratio)]
  right_ratio = angle_ratio[right_turn & np.isfinite(angle_ratio)]
  left_err = angle_err[left_turn & np.isfinite(angle_err)]
  right_err = angle_err[right_turn & np.isfinite(angle_err)]

  metrics = RouteMetrics(
    name=label,
    files=len(files),
    selfdrive_active_pct=(100.0 * selfdrive_active / selfdrive_total) if selfdrive_total else math.nan,
    overriding_pct_of_active=(100.0 * selfdrive_overriding / selfdrive_active) if selfdrive_active else math.nan,
    op_samples=len(arr),
    speed_mean=float(np.mean(speed)),
    speed_p90=safe_p(speed, 90),
    straight_samples=int(np.sum(straight)),
    straight_curv_flip_db=sign_flip_rate(t[straight], curv_cmd[straight], curv_deadband) if np.any(straight) else math.nan,
    straight_angle_cmd_p90=safe_p(np.abs(straight_angle_cmd), 90),
    straight_angle_cmd_mean=float(np.mean(straight_angle_cmd)) if len(straight_angle_cmd) else math.nan,
    straight_angle_err_mean=float(np.mean(straight_err)) if len(straight_err) else math.nan,
    straight_angle_err_p90=safe_p(np.abs(straight_err), 90),
    straight_angle_err_bias=err_bias,
    turn_samples=int(np.sum(turn)),
    turn_angle_cmd_p90=safe_p(np.abs(angle_cmd[turn]), 90),
    turn_angle_ratio_p50=safe_p(ratio_turn, 50),
    turn_angle_ratio_p90=safe_p(ratio_turn, 90),
    turn_over10_pct=(100.0 * float(np.mean(ratio_turn > 1.10))) if len(ratio_turn) else math.nan,
    turn_angle_err_mean=float(np.mean(np.abs(err_turn))) if len(err_turn) else math.nan,
    left_turn_angle_err_mean=float(np.mean(np.abs(left_err))) if len(left_err) else math.nan,
    right_turn_angle_err_mean=float(np.mean(np.abs(right_err))) if len(right_err) else math.nan,
    left_turn_over10_pct=(100.0 * float(np.mean(left_ratio > 1.10))) if len(left_ratio) else math.nan,
    right_turn_over10_pct=(100.0 * float(np.mean(right_ratio > 1.10))) if len(right_ratio) else math.nan,
  )
  return metrics, arr, m_arr


def fmt(v: float, digits: int = 3) -> str:
  if not np.isfinite(v):
    return "nan"
  return f"{v:.{digits}f}"


def print_metrics(metrics: RouteMetrics) -> None:
  print(f"\n{metrics.name}")
  print(f"  files: {metrics.files}")
  print(f"  selfdrive_active_pct: {fmt(metrics.selfdrive_active_pct, 2)}")
  print(f"  overriding_pct_of_active: {fmt(metrics.overriding_pct_of_active, 2)}")
  print(f"  op_samples: {metrics.op_samples}")
  print(f"  speed mean/p90: {fmt(metrics.speed_mean, 2)} / {fmt(metrics.speed_p90, 2)}")
  print(f"  straight samples: {metrics.straight_samples}")
  print(f"  straight curv_flip/s: {fmt(metrics.straight_curv_flip_db)}")
  print(f"  straight |angle_cmd| p90: {fmt(metrics.straight_angle_cmd_p90)}")
  print(f"  straight angle_cmd mean (bias): {fmt(metrics.straight_angle_cmd_mean)}")
  print(f"  straight angle_err mean/|err|p90: {fmt(metrics.straight_angle_err_mean)} / {fmt(metrics.straight_angle_err_p90)}")
  print(f"  straight angle_err bias: {fmt(metrics.straight_angle_err_bias)}")
  print(f"  turn samples: {metrics.turn_samples}")
  print(f"  turn |angle_cmd| p90: {fmt(metrics.turn_angle_cmd_p90)}")
  print(f"  turn angle_ratio p50/p90: {fmt(metrics.turn_angle_ratio_p50)} / {fmt(metrics.turn_angle_ratio_p90)}")
  print(f"  turn over10%%: {fmt(metrics.turn_over10_pct, 1)}")
  print(f"  turn |angle_err| mean: {fmt(metrics.turn_angle_err_mean)}")
  print(f"  left/right turn |angle_err| mean: {fmt(metrics.left_turn_angle_err_mean)} / {fmt(metrics.right_turn_angle_err_mean)}")
  print(f"  left/right turn over10%%: {fmt(metrics.left_turn_over10_pct, 1)} / {fmt(metrics.right_turn_over10_pct, 1)}")


def print_speed_bins(rows: np.ndarray, min_speed: float, straight_curvature: float, curv_deadband: float) -> None:
  if len(rows) == 0:
    return

  print("\n  Speed-Bin Breakdown")
  speed_bins = [(20, 25), (25, 30), (30, 35)]
  for lo, hi in speed_bins:
    if hi <= min_speed:
      continue

    idx = (rows[:, 1] >= lo) & (rows[:, 1] < hi)
    if not np.any(idx):
      print(f"  - {lo}-{hi} m/s: no samples")
      continue

    straight = idx & (np.abs(rows[:, 2]) < straight_curvature)
    turn = idx & (np.abs(rows[:, 2]) >= straight_curvature)

    sf = sign_flip_rate(rows[straight, 0], rows[straight, 2], curv_deadband) if np.any(straight) else math.nan
    st_err = rows[straight, 5]
    st_err = st_err[np.isfinite(st_err)]
    st_err_p90 = safe_p(np.abs(st_err), 90) if len(st_err) else math.nan

    turn_ratio = rows[turn, 4]
    turn_ratio = turn_ratio[np.isfinite(turn_ratio)]
    turn_err = rows[turn, 5]
    turn_err = turn_err[np.isfinite(turn_err)]

    over10 = (100.0 * float(np.mean(turn_ratio > 1.10))) if len(turn_ratio) else math.nan
    abs_err_mean = float(np.mean(np.abs(turn_err))) if len(turn_err) else math.nan

    print(
      f"  - {lo}-{hi} m/s: n={int(np.sum(idx))}, straight_n={int(np.sum(straight))}, "
      f"curv_flip/s={fmt(sf)}, |angle_err|p90={fmt(st_err_p90)}, "
      f"turn_n={int(np.sum(turn))}, turn_over10%={fmt(over10,1)}, turn_|err|_mean={fmt(abs_err_mean)}"
    )


def print_model_center(m_rows: np.ndarray, min_speed: float) -> None:
  if len(m_rows) == 0:
    return

  # cols: active, cruise, vEgo, lane_width, lane_center, left_prob, right_prob
  good = (
    (m_rows[:, 2] >= min_speed)
    & (m_rows[:, 5] >= 0.6)
    & (m_rows[:, 6] >= 0.6)
    & (m_rows[:, 3] > 2.8)
    & (m_rows[:, 3] < 4.2)
  )

  if not np.any(good):
    return

  print("\n  Model Lane-Center")
  for label, idx in (
    ("all", good),
    ("active", good & (m_rows[:, 0] > 0.5)),
    ("inactive+cruise", good & (m_rows[:, 0] < 0.5) & (m_rows[:, 1] > 0.5)),
  ):
    if not np.any(idx):
      continue

    center = m_rows[idx, 4]
    print(
      f"  - {label}: n={len(center)}, center_mean={fmt(float(np.mean(center)))}, "
      f"center_p50={fmt(safe_p(center,50))}, abs_center_p90={fmt(safe_p(np.abs(center),90))}, "
      f"pos%={fmt(100.0 * float(np.mean(center > 0)),1)}"
    )


def summarize_route(identifier: str, logs_dir: str, min_speed: float, straight_curvature: float,
                    curv_deadband: float, show_segment_groups: bool) -> tuple[RouteMetrics, np.ndarray, np.ndarray]:
  files = resolve_local_files(identifier, logs_dir)

  seg_settings = [parse_init_settings(path) for path in files]

  print(f"\n{'=' * 100}")
  print(f"Route: {identifier}")
  print(f"Local files: {len(files)}")

  commits = Counter(str(s["gitCommit"]) for s in seg_settings)
  branches = Counter(str(s["gitBranch"]) for s in seg_settings)
  ppc = Counter(str(s["planplusControl"]) for s in seg_settings)
  print(f"gitCommit groups: {dict(commits)}")
  print(f"gitBranch groups: {dict(branches)}")
  print(f"PlanplusControl groups: {dict(ppc)}")
  print(f"Control type: angle (Ford)")

  if show_segment_groups:
    print("\nPer-segment settings")
    for s in sorted(seg_settings, key=lambda x: int(x["seg"])):
      print(
        f"  seg={s['seg']:>2} planplusControl={s['planplusControl']}"
      )

  all_metrics, all_rows, all_mrows = analyze_files(files, min_speed, straight_curvature, curv_deadband, "ALL")
  print_metrics(all_metrics)
  print_speed_bins(all_rows, min_speed, straight_curvature, curv_deadband)
  print_model_center(all_mrows, min_speed)

  return all_metrics, all_rows, all_mrows


def print_cross_route_delta(a: RouteMetrics, b: RouteMetrics, a_name: str, b_name: str) -> None:
  print(f"\n{'=' * 100}")
  print(f"Cross-Route Delta ({b_name} - {a_name})")
  fields = [f for f in RouteMetrics.__dataclass_fields__.keys() if f not in ("name", "files")]
  for field in fields:
    va = getattr(a, field)
    vb = getattr(b, field)
    if np.isfinite(va) and np.isfinite(vb):
      print(f"  - {field}: {fmt(vb - va)}")


def main() -> None:
  args = parse_args()

  a_metrics, _, _ = summarize_route(
    args.route_a, args.logs_dir, args.min_speed,
    args.straight_curvature, args.curv_deadband,
    args.show_segment_groups,
  )

  if args.route_b is not None:
    b_metrics, _, _ = summarize_route(
      args.route_b, args.logs_dir, args.min_speed,
      args.straight_curvature, args.curv_deadband,
      args.show_segment_groups,
    )
    print_cross_route_delta(a_metrics, b_metrics, args.route_a, args.route_b)


if __name__ == "__main__":
  main()
