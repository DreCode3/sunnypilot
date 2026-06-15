from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentRef:
  route_id: str
  segment_index: int
  rlog_path: Path
  qlog_path: Path | None = None


@dataclass(frozen=True)
class RouteRef:
  route_id: str
  route_dir: Path
  layout: str
  segments: tuple[SegmentRef, ...]


def segment_index_from_path(path: Path) -> int:
  text = str(path)
  modern = re.search(r"--(\d+)/rlog\.zst$", text)
  if modern:
    return int(modern.group(1))
  flat = re.search(r"rlog_(\d+)\.zst$", text)
  if flat:
    return int(flat.group(1))
  plain = re.search(r"/(\d+)/rlog\.zst$", text)
  if plain:
    return int(plain.group(1))
  return 0


def _route_id_from_dir(route_dir: Path) -> str:
  return route_dir.name


def _modern_segments(route_dir: Path, route_id: str) -> list[SegmentRef]:
  out: list[SegmentRef] = []
  for rlog in route_dir.glob("000000*--*/rlog.zst"):
    out.append(SegmentRef(route_id=route_id, segment_index=segment_index_from_path(rlog), rlog_path=rlog))
  return sorted(out, key=lambda s: s.segment_index)


def _flat_segments(route_dir: Path, route_id: str) -> list[SegmentRef]:
  out: list[SegmentRef] = []
  for rlog in route_dir.glob("rlog_*.zst"):
    out.append(SegmentRef(route_id=route_id, segment_index=segment_index_from_path(rlog), rlog_path=rlog))
  if (route_dir / "rlog.zst").exists():
    out.append(SegmentRef(route_id=route_id, segment_index=0, rlog_path=route_dir / "rlog.zst"))
  return sorted(out, key=lambda s: s.segment_index)


def discover_routes(root: Path) -> list[RouteRef]:
  routes: list[RouteRef] = []
  for route_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("route_")):
    route_id = _route_id_from_dir(route_dir)
    modern = _modern_segments(route_dir, route_id)
    flat = _flat_segments(route_dir, route_id)
    if modern:
      routes.append(RouteRef(route_id=route_id, route_dir=route_dir, layout="modern", segments=tuple(modern)))
    elif flat:
      routes.append(RouteRef(route_id=route_id, route_dir=route_dir, layout="flat", segments=tuple(flat)))
  return routes
