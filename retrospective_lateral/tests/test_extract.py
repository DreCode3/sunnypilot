import json
import numpy as np

from retrospective_lateral.code.extract import RawChannels, resample_channels, route_cache_metadata


def test_resample_channels_outputs_fixed_grid_and_masks_large_gaps():
  raw = RawChannels()
  raw.add("carState", 0.0, {"v_ego": 10.0, "steering_angle_deg": 0.0, "yaw_rate": 0.0})
  raw.add("carState", 0.05, {"v_ego": 10.0, "steering_angle_deg": 1.0, "yaw_rate": 0.01})
  raw.add("carState", 1.50, {"v_ego": 10.0, "steering_angle_deg": 2.0, "yaw_rate": 0.02})
  raw.add("carControl", 0.0, {"lat_active": 1.0, "act_curvature": 0.001})
  raw.add("carControl", 1.50, {"lat_active": 1.0, "act_curvature": 0.002})
  out = resample_channels(raw, fs_hz=20.0)
  assert out["t"].shape[0] == 31
  assert out["t"][0] == 0.0
  assert out["t"][-1] == 1.5
  assert np.isfinite(out["v_ego"][0])
  assert np.isnan(out["v_ego"][16])
  assert out["lat_active"][0] == 1.0


def test_resample_channels_handles_sparse_rows_with_per_signal_timestamps():
  raw = RawChannels()
  raw.add("carState", 0.0, {"v_ego": 10.0})
  raw.add("carState", 0.25, {"steering_angle_deg": 2.0})
  raw.add("carState", 0.75, {"steering_angle_deg": 4.0})
  raw.add("carState", 1.0, {"v_ego": 20.0})

  out = resample_channels(raw, fs_hz=4.0)

  assert out["t"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
  assert out["v_ego"].tolist() == [10.0, 12.5, 15.0, 17.5, 20.0]
  assert np.isnan(out["steering_angle_deg"][0])
  assert out["steering_angle_deg"][1:4].tolist() == [2.0, 3.0, 4.0]
  assert np.isnan(out["steering_angle_deg"][4])


def test_nearest_flags_choose_closest_sample():
  raw = RawChannels()
  raw.add("carControl", 0.0, {"lat_active": 0.0})
  raw.add("carControl", 1.0, {"lat_active": 1.0})

  out = resample_channels(raw, fs_hz=10.0)

  assert out["t"][1] == 0.1
  assert out["t"][9] == 0.9
  assert out["lat_active"][1] == 0.0
  assert out["lat_active"][9] == 1.0


def test_route_cache_metadata_is_json_serializable(tmp_path):
  meta = route_cache_metadata(
    route_id="route_b8",
    schema_version="retrolat-v1",
    segments=3,
    config_confidence="proven",
    notes=["LC telemetry recovered"],
  )
  encoded = json.dumps(meta, sort_keys=True)
  assert "route_b8" in encoded
  assert meta["segments"] == 3
  assert set(meta) == {"route_id", "schema_version", "segments", "config_confidence", "notes"}
