from __future__ import annotations
import sys, bisect, functools
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # openpilot import robust to CWD
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.framereader import FrameReader
from model_replay_sim import config as C

@dataclass(frozen=True)
class FrameRow:
    segment_num: int
    segment_id: int            # == HEVC frame index in that segment's fcamera.hevc
    frame_id: int              # global, monotonic
    timestamp_eof_s: float
    hevc_frame_count: int
    ecamera_segment_id: int | None

@dataclass(frozen=True)
class FrameAlignment:
    segment_num: int
    segment_id: int
    ecamera_index: int | None

def _seg_dirs(route_id: str):
    return sorted((C.LOG_ROOT / route_id).glob("000000*--*--*"))

def _seg_num(seg: Path) -> int:
    return int(seg.name.rsplit("--", 1)[-1])

@functools.lru_cache(maxsize=8)
def build_frame_timeline(route_id: str) -> tuple:
    rows = []
    for seg in _seg_dirs(route_id):
        fcam = seg / "fcamera.hevc"
        if not fcam.exists():
            continue
        try:
            fcount = int(FrameReader(str(fcam), pix_fmt="nv12").frame_count)
        except Exception:
            continue
        ecam = seg / "ecamera.hevc"                    # wide cam may be absent/short
        try:
            wcount = int(FrameReader(str(ecam), pix_fmt="nv12").frame_count) if ecam.exists() else 0
        except Exception:
            wcount = 0
        wide_by_eof = {}
        road = []
        try:                                          # corrupt/truncated rlog → skip segment
            for m in LogReader(str(seg / "rlog.zst")):
                w = m.which()
                if w == "roadEncodeIdx":
                    e = m.roadEncodeIdx
                    road.append((int(e.segmentNum), int(e.segmentId), int(e.frameId), e.timestampEof * 1e-9))
                elif w == "wideRoadEncodeIdx":
                    e = m.wideRoadEncodeIdx
                    wide_by_eof[round(e.timestampEof * 1e-9, 3)] = int(e.segmentId)
        except Exception:
            continue
        for snum, sid, fid, eof in road:
            if sid >= fcount:                         # truncation guard
                continue
            wsid = wide_by_eof.get(round(eof, 3))     # mirror road guard for wide cam:
            esid = wsid if (wsid is not None and wsid < wcount) else None  # missing/short → None
            rows.append(FrameRow(snum, sid, fid, eof, fcount, esid))
    rows.sort(key=lambda r: r.timestamp_eof_s)
    return tuple(rows)

def map_window_to_frames(route_id: str, mono_times, max_delta_s: float = C.MAX_FRAME_DELTA_S):
    tl = build_frame_timeline(route_id)
    if not tl:
        raise ValueError(f"{route_id}: empty frame timeline")
    eofs = [r.timestamp_eof_s for r in tl]
    out = []
    for t in mono_times:
        i = bisect.bisect_left(eofs, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(tl)]
        best = min(cands, key=lambda j: abs(eofs[j] - t)) if cands else None
        if best is None or abs(eofs[best] - t) > max_delta_s:
            raise ValueError(f"{route_id}: mono {t:.3f}s has no frame within {max_delta_s}s")
        r = tl[best]
        out.append(FrameAlignment(r.segment_num, r.segment_id, r.ecamera_segment_id))
    return out

def read_frame(route_id: str, segment_num: int, segment_id: int, camera: str = "fcamera"):
    """Read one nv12 frame from ``camera`` (default ``fcamera`` = road). (Callers that need
    many frames should open the FrameReader once per segment; this is the simple per-frame
    accessor.) ``camera`` may be ``"fcamera"`` (road) or ``"ecamera"`` (wide)."""
    seg = next(s for s in _seg_dirs(route_id) if _seg_num(s) == segment_num)
    return FrameReader(str(seg / f"{camera}.hevc"), pix_fmt="nv12").get(segment_id)


def read_wide_frame(route_id: str, segment_num: int, ecamera_segment_id: int):
    """Read one wide-camera (``ecamera.hevc``) nv12 frame. The wide frame index
    (``ecamera_segment_id``) is the EOF-aligned wide segmentId from
    :func:`build_frame_timeline` (``FrameRow.ecamera_segment_id`` /
    ``FrameAlignment.ecamera_index``), NOT the road ``segment_id``."""
    return read_frame(route_id, segment_num, ecamera_segment_id, camera="ecamera")
