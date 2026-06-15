import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from retrospective_lateral.code import extract
from retrospective_lateral.code.extract import RawChannels, resample_channels, route_cache_metadata
from retrospective_lateral.code.routes import RouteRef, SegmentRef


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


def test_resample_channels_masks_numeric_samples_across_large_bracket_gaps():
  raw = RawChannels()
  raw.add("carState", 0.0, {"v_ego": 10.0})
  raw.add("carState", 1.5, {"v_ego": 20.0})

  out = resample_channels(raw, fs_hz=2.0)

  assert out["t"].tolist() == [0.0, 0.5, 1.0, 1.5]
  assert out["v_ego"][0] == 10.0
  assert np.isnan(out["v_ego"][1])
  assert np.isnan(out["v_ego"][2])
  assert out["v_ego"][3] == 20.0


def test_resample_channels_handles_sparse_rows_with_per_signal_timestamps():
  raw = RawChannels()
  raw.add("carState", 0.0, {"v_ego": 10.0})
  raw.add("carState", 0.25, {"steering_angle_deg": 2.0})
  raw.add("carState", 0.5, {"v_ego": 20.0})
  raw.add("carState", 0.75, {"steering_angle_deg": 4.0})
  raw.add("carControl", 1.0, {"lat_active": 0.0})

  out = resample_channels(raw, fs_hz=4.0)

  assert out["t"].tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
  assert out["v_ego"].tolist()[:3] == [10.0, 15.0, 20.0]
  assert np.isnan(out["v_ego"][3:]).all()
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


def test_nearest_flags_default_false_when_nearest_sample_is_stale():
  raw = RawChannels()
  raw.add("carControl", 0.0, {"lat_active": 1.0})
  raw.add("carControl", 2.0, {"lat_active": 1.0})

  out = resample_channels(raw, fs_hz=2.0)

  assert out["t"].tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
  assert out["lat_active"][0] == 1.0
  assert out["lat_active"][1] == 1.0
  assert out["lat_active"][2] == 0.0
  assert out["lat_active"][3] == 1.0
  assert out["lat_active"][4] == 1.0


def test_resample_channels_filters_non_finite_numeric_sources_and_keeps_last_duplicate():
  raw = RawChannels()
  raw.add("carState", 0.0, {"v_ego": 5.0, "a_ego": None})
  raw.add("carState", 0.0, {"v_ego": 7.0, "a_ego": None})
  raw.add("carState", 0.25, {"v_ego": np.nan, "a_ego": None})
  raw.add("carState", 0.5, {"v_ego": 12.0, "a_ego": None})

  out = resample_channels(raw, fs_hz=4.0)

  assert out["v_ego"].tolist() == [7.0, 9.5, 12.0]
  assert np.isnan(out["a_ego"]).all()


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


def test_extract_route_raw_continues_after_bad_message(monkeypatch, tmp_path):
  class BadMessage:
    logMonoTime = 1_000_000_000

    def which(self):
      raise RuntimeError("bad optional field")

  good_message = SimpleNamespace(
    logMonoTime=2_000_000_000,
    which=lambda: "carState",
    carState=SimpleNamespace(
      vEgo=11.0,
      vEgoRaw=10.5,
      aEgo=0.1,
      steeringAngleDeg=1.0,
      steeringRateDeg=0.2,
      steeringTorque=0.3,
      yawRate=0.01,
      steeringPressed=False,
      leftBlinker=False,
      rightBlinker=False,
      canValid=True,
    ),
  )

  monkeypatch.setattr(extract, "LogReader", lambda _: [BadMessage(), good_message])
  segment = SegmentRef(route_id="route_unit", segment_index=0, rlog_path=tmp_path / "rlog.zst")
  route = RouteRef(route_id="route_unit", route_dir=Path(tmp_path), layout="flat", segments=(segment,))

  raw, notes = extract.extract_route_raw(route)

  assert notes
  assert raw.values["carState"]["v_ego"] == [11.0]


def test_write_route_cache_enriches_old_cache_metadata(tmp_path):
  cache_root = tmp_path / "cache"
  cache_root.mkdir()
  t = np.asarray([0.0, 0.05, 0.10], dtype=np.float32)
  np.savez_compressed(cache_root / "route_old.npz", t=t)
  old_meta = {
    "route_id": "route_old",
    "schema_version": "retrolat-v1",
    "segments": 1,
    "config_confidence": "unknown",
    "notes": [],
  }
  (cache_root / "route_old.json").write_text(json.dumps(old_meta))
  segment = SegmentRef(route_id="route_old", segment_index=0, rlog_path=tmp_path / "rlog.zst")
  route = RouteRef(route_id="route_old", route_dir=tmp_path, layout="flat", segments=(segment,))

  meta = extract._write_route_cache(route, cache_root, force=False)

  assert meta["success"] is True
  assert meta["sample_count"] == len(t)
  assert meta["npz_path"] == str(cache_root / "route_old.npz")
  assert json.loads((cache_root / "route_old.json").read_text()) == meta
