from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class LCTelemetry:
  t: float
  offset_m: float
  lane_line_offset_m: float
  path_position_m: float
  lane_scale: float
  confidence: float
  lane_width_m: float
  integral: float
  p_term: float
  i_term: float
  curvature: float
  speed_mps: float


@dataclass(frozen=True)
class CPTelemetry:
  t: float
  desired_curvature: float
  predicted_curvature: float
  ema_curvature: float
  pre_rate_limit: float
  rate_limited: float
  final_command: float
  measured_curvature: float
  override: int
  reset: int
  ramp: int
  rate_limited_flag: int
  anti_windup: int
  steering_angle_deg: float
  steering_torque: float


@dataclass(frozen=True)
class CX1Telemetry:
  frame: int
  speed_mps: float
  yaw_rate: float
  lateral_accel: float
  command_curvature: float
  command_rate: float
  measured_curvature: float
  desired_curvature: float
  predicted_curvature: float
  ema_curvature: float
  pre_rate_limit: float
  rate_limited: float
  command_int: int
  rate_int: int
  steering_angle_deg: float
  steering_rate_deg_s: float
  steering_torque: float
  override: int
  lane_change: int
  lookup_time_s: float
  blend: float
  curvature_factor: float
  lane_offset_m: float
  integral: float
  pred_minus_des: float
  burst: int
  path4_release: int
  smooth_tau_s: float
  path4_enabled: int


@dataclass(frozen=True)
class ConfigEvidence:
  pi_set: str
  confidence: str
  lc_kp: float | None
  lc_ki: float | None
  n_samples: int
  reason: str


LC_RE = re.compile(
  r"LC: off=(?P<off>[-+\d.]+) ll=(?P<ll>[-+\d.]+) pos=(?P<pos>[-+\d.]+) "
  r"scl=(?P<scl>[-+\d.]+) conf=(?P<conf>[-+\d.]+) wid=(?P<wid>[-+\d.]+) "
  r"int=(?P<int>[-+\d.]+) P=(?P<P>[-+\deE.]+) I=(?P<I>[-+\deE.]+) "
  r"curv=(?P<curv>[-+\d.]+) spd=(?P<spd>[-+\d.]+)"
)

CP_RE = re.compile(
  r"CP: des=(?P<des>[-+\d.]+) pred=(?P<pred>[-+\d.]+) ema=(?P<ema>[-+\d.]+) "
  r"preRL=(?P<pre>[-+\d.]+) RL=(?P<rl>[-+\d.]+) send=(?P<send>[-+\d.]+) meas=(?P<meas>[-+\d.]+) "
  r"\| ovr=(?P<ovr>\d+) rst=(?P<rst>\d+) ramp=(?P<ramp>\d+) rlClip=(?P<clip>\d+) aw=(?P<aw>\d+) "
  r"\| ang=(?P<ang>[-+\d.]+) tq=(?P<tq>[-+\d.]+)"
)


def _float_group(match: re.Match[str], name: str) -> float:
  return float(match.group(name))


def parse_lc_line(text: str, t: float) -> LCTelemetry | None:
  match = LC_RE.search(text)
  if match is None:
    return None
  return LCTelemetry(
    t=t,
    offset_m=_float_group(match, "off"),
    lane_line_offset_m=_float_group(match, "ll"),
    path_position_m=_float_group(match, "pos"),
    lane_scale=_float_group(match, "scl"),
    confidence=_float_group(match, "conf"),
    lane_width_m=_float_group(match, "wid"),
    integral=_float_group(match, "int"),
    p_term=_float_group(match, "P"),
    i_term=_float_group(match, "I"),
    curvature=_float_group(match, "curv"),
    speed_mps=_float_group(match, "spd"),
  )


def parse_cp_line(text: str, t: float) -> CPTelemetry | None:
  match = CP_RE.search(text)
  if match is None:
    return None
  return CPTelemetry(
    t=t,
    desired_curvature=_float_group(match, "des"),
    predicted_curvature=_float_group(match, "pred"),
    ema_curvature=_float_group(match, "ema"),
    pre_rate_limit=_float_group(match, "pre"),
    rate_limited=_float_group(match, "rl"),
    final_command=_float_group(match, "send"),
    measured_curvature=_float_group(match, "meas"),
    override=int(match.group("ovr")),
    reset=int(match.group("rst")),
    ramp=int(match.group("ramp")),
    rate_limited_flag=int(match.group("clip")),
    anti_windup=int(match.group("aw")),
    steering_angle_deg=_float_group(match, "ang"),
    steering_torque=_float_group(match, "tq"),
  )


def parse_cx1_line(text: str) -> CX1Telemetry | None:
  if not text.startswith("CX1: ") or "SCHEMA=" in text:
    return None
  parts = text.split()[1:]
  if len(parts) != 29:
    return None
  return CX1Telemetry(
    frame=int(parts[0]),
    speed_mps=float(parts[1]),
    yaw_rate=float(parts[2]),
    lateral_accel=float(parts[3]),
    command_curvature=float(parts[4]),
    command_rate=float(parts[5]),
    measured_curvature=float(parts[6]),
    desired_curvature=float(parts[7]),
    predicted_curvature=float(parts[8]),
    ema_curvature=float(parts[9]),
    pre_rate_limit=float(parts[10]),
    rate_limited=float(parts[11]),
    command_int=int(parts[12]),
    rate_int=int(parts[13]),
    steering_angle_deg=float(parts[14]),
    steering_rate_deg_s=float(parts[15]),
    steering_torque=float(parts[16]),
    override=int(parts[17]),
    lane_change=int(parts[18]),
    lookup_time_s=float(parts[19]),
    blend=float(parts[20]),
    curvature_factor=float(parts[21]),
    lane_offset_m=float(parts[22]),
    integral=float(parts[23]),
    pred_minus_des=float(parts[24]),
    burst=int(parts[25]),
    path4_release=int(parts[26]),
    smooth_tau_s=float(parts[27]),
    path4_enabled=int(parts[28]),
  )


def _ratios(num: Sequence[float], den: Sequence[float], min_abs_den: float) -> list[float]:
  out: list[float] = []
  for n, d in zip(num, den, strict=False):
    if math.isfinite(n) and math.isfinite(d) and abs(d) >= min_abs_den:
      out.append(float(n) / float(d))
  return out


def recover_pi_config(offsets: Sequence[float], p_terms: Sequence[float],
                      integrals: Sequence[float], i_terms: Sequence[float]) -> ConfigEvidence:
  kp_ratios = _ratios(p_terms, offsets, min_abs_den=0.02)
  ki_ratios = _ratios(i_terms, integrals, min_abs_den=0.02)
  lc_kp = median(kp_ratios) if len(kp_ratios) >= 3 else None
  lc_ki = median(ki_ratios) if len(ki_ratios) >= 3 else None
  n = min(len(kp_ratios), len(ki_ratios))
  if lc_kp is None:
    return ConfigEvidence("unknown", "unknown", None, lc_ki, n, "insufficient LC P/off samples")
  if lc_kp >= 0.0003:
    return ConfigEvidence("golden", "proven", float(lc_kp), float(lc_ki) if lc_ki is not None else None, n, "LC P/off median indicates strong Kp")
  return ConfigEvidence("weak", "proven", float(lc_kp), float(lc_ki) if lc_ki is not None else None, n, "LC P/off median indicates weak Kp")
