import csv
import subprocess
from pathlib import Path

import numpy as np


def test_metrics_cli_writes_episode_catalog(tmp_path):
  cache = tmp_path / "cache"
  out = tmp_path / "out"
  cache.mkdir()
  fs = 20.0
  t = np.arange(0.0, 20.0, 1.0 / fs)
  steer = 6.0 * np.sin(2 * np.pi * 0.25 * t)
  np.savez_compressed(
    cache / "route_synth.npz",
    t=t.astype(np.float32),
    v_ego=np.full_like(t, 3.0, dtype=np.float32),
    steering_angle_deg=steer.astype(np.float32),
    steering_rate_deg=np.gradient(steer, 1.0 / fs).astype(np.float32),
    steering_pressed=np.zeros_like(t, dtype=np.float32),
    lat_active=np.ones_like(t, dtype=np.float32),
    blinker=np.zeros_like(t, dtype=np.float32),
    lane_change_state=np.zeros_like(t, dtype=np.float32),
    act_curvature=(0.001 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32),
    yaw_rate=np.zeros_like(t, dtype=np.float32),
  )
  cmd = [
    ".venv311/bin/python",
    "-m",
    "retrospective_lateral.code.metrics",
    "--cache-root",
    str(cache),
    "--out",
    str(out),
  ]
  subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], check=True)
  catalog = out / "symptom_catalog.csv"
  assert catalog.exists()
  rows = list(csv.DictReader(catalog.open()))
  assert rows[0]["symptom"] == "low_speed_wheel_swing"
