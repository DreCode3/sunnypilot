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
  assert np.isfinite(out["v_ego"][0])
  assert np.isnan(out["v_ego"][16])
  assert out["lat_active"][0] == 1.0


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
