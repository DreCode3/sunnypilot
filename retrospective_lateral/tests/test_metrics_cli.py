import csv
import subprocess
from pathlib import Path

import numpy as np

from retrospective_lateral.code.metrics import write_symptom_catalog


def _write_synthetic_route(path, *, include_lat_active=True):
  fs = 20.0
  t = np.arange(0.0, 20.0, 1.0 / fs)
  steer = 6.0 * np.sin(2 * np.pi * 0.25 * t)
  arrays = dict(
    t=t.astype(np.float32),
    v_ego=np.full_like(t, 3.0, dtype=np.float32),
    steering_angle_deg=steer.astype(np.float32),
    steering_rate_deg=np.gradient(steer, 1.0 / fs).astype(np.float32),
    steering_pressed=np.zeros_like(t, dtype=np.float32),
    blinker=np.zeros_like(t, dtype=np.float32),
    lane_change_state=np.zeros_like(t, dtype=np.float32),
    act_curvature=(0.001 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32),
    yaw_rate=np.zeros_like(t, dtype=np.float32),
  )
  if include_lat_active:
    arrays["lat_active"] = np.ones_like(t, dtype=np.float32)
  np.savez_compressed(path, **arrays)


def test_metrics_cli_writes_episode_catalog(tmp_path):
  cache = tmp_path / "cache"
  out = tmp_path / "out"
  cache.mkdir()
  _write_synthetic_route(cache / "route_synth.npz")
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


def test_write_symptom_catalog_writes_stable_header_for_empty_cache(tmp_path):
  cache = tmp_path / "cache"
  out = tmp_path / "out"
  cache.mkdir()

  rows = write_symptom_catalog(str(cache), str(out))

  assert rows == []
  with (out / "symptom_catalog.csv").open() as fh:
    header = fh.readline().strip().split(",")
  for field in (
    "route_id",
    "symptom",
    "status",
    "error_type",
    "error_message",
    "steering_peak_to_peak_deg",
    "path_curvature_band_rms_1e4",
    "spectral_peak_hz",
  ):
    assert field in header


def test_write_symptom_catalog_records_corrupt_route_and_continues(tmp_path):
  cache = tmp_path / "cache"
  out = tmp_path / "out"
  cache.mkdir()
  (cache / "route_bad.npz").write_bytes(b"not an npz")
  _write_synthetic_route(cache / "route_synth.npz")

  rows = write_symptom_catalog(str(cache), str(out))
  written = list(csv.DictReader((out / "symptom_catalog.csv").open()))

  assert any(row["route_id"] == "route_bad" and row["status"] == "failed" for row in rows)
  assert any(row["route_id"] == "route_synth" and row["symptom"] == "low_speed_wheel_swing" for row in rows)
  bad = next(row for row in written if row["route_id"] == "route_bad")
  assert bad["symptom"] == "route_error"
  assert bad["status"] == "failed"
  assert bad["error_type"]
  assert bad["error_message"]


def test_write_symptom_catalog_records_missing_lat_active_and_continues(tmp_path):
  cache = tmp_path / "cache"
  out = tmp_path / "out"
  cache.mkdir()
  _write_synthetic_route(cache / "route_missing_lat_active.npz", include_lat_active=False)
  _write_synthetic_route(cache / "route_synth.npz")

  rows = write_symptom_catalog(str(cache), str(out))
  written = list(csv.DictReader((out / "symptom_catalog.csv").open()))

  failed = next(row for row in rows if row["route_id"] == "route_missing_lat_active")
  assert failed["symptom"] == "route_error"
  assert failed["status"] == "failed"
  assert "ValueError" in failed["error_type"]
  assert any(row["route_id"] == "route_synth" and row["status"] == "ok" for row in rows)
  written_failed = next(row for row in written if row["route_id"] == "route_missing_lat_active")
  assert written_failed["status"] == "failed"
  assert "ValueError" in written_failed["error_type"]
