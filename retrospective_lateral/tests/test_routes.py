from pathlib import Path

import pytest

from retrospective_lateral.code.routes import discover_routes, segment_index_from_path


def touch(path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(b"")


def test_segment_index_from_modern_segment_dir():
  path = Path("explorer_st_logs/route_b8/000000b8--56af909567--31/rlog.zst")
  assert segment_index_from_path(path) == 31


def test_segment_index_from_flat_segment_file():
  path = Path("explorer_st_logs/route_49/rlog_3.zst")
  assert segment_index_from_path(path) == 3


def test_segment_index_from_legacy_flat_segment_file():
  path = Path("explorer_st_logs/route_1b/rlog_seg11.zst")
  assert segment_index_from_path(path) == 11


def test_segment_index_from_unknown_rlog_name_raises():
  with pytest.raises(ValueError):
    segment_index_from_path(Path("bad/rlog_unknown.zst"))


def test_discover_routes_finds_modern_and_flat_layouts(tmp_path):
  touch(tmp_path / "route_b8" / "000000b8--56af909567--0" / "rlog.zst")
  touch(tmp_path / "route_b8" / "000000b8--56af909567--1" / "rlog.zst")
  touch(tmp_path / "route_49" / "rlog_0.zst")
  touch(tmp_path / "route_49" / "rlog_1.zst")
  routes = discover_routes(tmp_path)
  by_id = {r.route_id: r for r in routes}
  assert sorted(by_id) == ["route_49", "route_b8"]
  assert [s.segment_index for s in by_id["route_b8"].segments] == [0, 1]
  assert [s.segment_index for s in by_id["route_49"].segments] == [0, 1]
  assert by_id["route_b8"].layout == "modern"
  assert by_id["route_49"].layout == "flat"


def test_discover_routes_finds_legacy_flat_segment_names(tmp_path):
  touch(tmp_path / "route_1b" / "rlog_seg0.zst")
  touch(tmp_path / "route_1b" / "rlog_seg1.zst")
  touch(tmp_path / "route_1b" / "rlog_seg10.zst")
  routes = discover_routes(tmp_path)
  by_id = {r.route_id: r for r in routes}
  assert [s.segment_index for s in by_id["route_1b"].segments] == [0, 1, 10]


def test_discover_routes_deduplicates_mixed_flat_segment_names(tmp_path):
  touch(tmp_path / "route_1c" / "rlog_0.zst")
  touch(tmp_path / "route_1c" / "rlog_seg0.zst")
  touch(tmp_path / "route_1c" / "rlog_seg1.zst")
  routes = discover_routes(tmp_path)
  by_id = {r.route_id: r for r in routes}
  assert [s.segment_index for s in by_id["route_1c"].segments] == [0, 1]
