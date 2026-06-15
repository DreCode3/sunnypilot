import math

from retrospective_lateral.code.telemetry import (
  ConfigEvidence,
  parse_cp_line,
  parse_cx1_line,
  parse_lc_line,
  recover_pi_config,
)


def test_parse_lc_line_extracts_controller_fields():
  line = "LC: off=0.120 ll=0.100 pos=0.0200 scl=1.00 conf=0.90 wid=3.60 int=0.4000 P=0.000060 I=0.000080 curv=0.001234 spd=24.0"
  row = parse_lc_line(line, t=123.0)
  assert row is not None
  assert row.t == 123.0
  assert row.offset_m == 0.120
  assert row.integral == 0.4000
  assert row.p_term == 0.000060
  assert row.i_term == 0.000080
  assert row.speed_mps == 24.0


def test_parse_cp_line_extracts_pipeline_fields():
  line = "CP: des=0.001000 pred=0.002000 ema=0.001200 preRL=0.001250 RL=0.001240 send=0.001240 meas=0.001100 | ovr=0 rst=0 ramp=0 rlClip=0 aw=0 | ang=1.0 tq=0.02"
  row = parse_cp_line(line, t=3.0)
  assert row is not None
  assert row.desired_curvature == 0.001
  assert row.predicted_curvature == 0.002
  assert row.final_command == 0.00124
  assert row.override == 0


def test_parse_cx1_line_extracts_schema_v1_fields():
  line = "CX1: 100 12.00 +0.01000 +0.120 +0.0010000 +0.0001000 +0.001100 +0.000900 +0.001200 +0.001000 +0.001050 +0.001000 950 4090 +3.00 +1.00 +0.20 0 0 0.500 0.300 1.000 +0.020 +0.3000 +0.000300 0 0 0.0400 0"
  row = parse_cx1_line(line)
  assert row is not None
  assert row.frame == 100
  assert row.speed_mps == 12.0
  assert row.command_curvature == 0.001
  assert row.path4_enabled == 0


def test_recover_pi_config_from_lc_terms():
  evidence = recover_pi_config(
    offsets=[0.10, -0.20, 0.12],
    p_terms=[0.00005, -0.00010, 0.00006],
    integrals=[0.50, -0.30, 0.40],
    i_terms=[0.00010, -0.00006, 0.00008],
  )
  assert isinstance(evidence, ConfigEvidence)
  assert evidence.pi_set == "golden"
  assert evidence.confidence == "proven"
  assert math.isclose(evidence.lc_kp, 0.0005, rel_tol=1e-6)
  assert math.isclose(evidence.lc_ki, 0.0002, rel_tol=1e-6)
