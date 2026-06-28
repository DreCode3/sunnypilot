import glob
import pytest
from model_replay_sim.alignment import build_frame_timeline, map_window_to_frames

def _route():
    return "route_8d" if glob.glob("explorer_st_logs/route_8d/0000008d--*--1/fcamera.hevc") else None

@pytest.mark.skipif(_route() is None, reason="no route_8d frames")
def test_timeline_uses_segmentId_not_roadcamerastate_count():
    tl = build_frame_timeline("route_8d")
    seg1 = sorted([r for r in tl if r.segment_num == 1], key=lambda r: r.segment_id)
    assert seg1[0].segment_id == 0 and seg1[0].frame_id == 1202     # off-by-one regression
    assert seg1[0].frame_id != 1203                                  # the old (wrong) value
    assert all(r.segment_id < r.hevc_frame_count for r in seg1)      # truncation guard

@pytest.mark.skipif(_route() is None, reason="no route_8d frames")
def test_map_window_resolves_and_rejects_unmappable():
    tl = build_frame_timeline("route_8d")
    seg1 = sorted([r for r in tl if r.segment_num == 1], key=lambda r: r.segment_id)
    monos = [seg1[10].timestamp_eof_s, seg1[20].timestamp_eof_s, seg1[30].timestamp_eof_s]
    aligned = map_window_to_frames("route_8d", monos, max_delta_s=0.03)
    assert [a.segment_id for a in aligned] == [10, 20, 30]
    assert all(a.segment_num == 1 for a in aligned)
    # route_8d spans 43 present segments (1-4, 10-48; seg0 corrupt, 5-9 + 49-59
    # absent), ~2960s, so anchor the unmappable probe to the true end of the
    # timeline (seg1's end is still mid-route).
    end_eof = max(r.timestamp_eof_s for r in tl)
    with pytest.raises(ValueError):
        map_window_to_frames("route_8d", [end_eof + 100.0], max_delta_s=0.03)

@pytest.mark.skipif(_route() is None, reason="no route_8d frames")
def test_boundary_crossing_window_is_accepted():
    tl = build_frame_timeline("route_8d")
    by = {}
    for r in tl: by.setdefault(r.segment_num, []).append(r)
    s1 = sorted(by[1], key=lambda r: r.segment_id); s2 = sorted(by[2], key=lambda r: r.segment_id)
    monos = [s1[-2].timestamp_eof_s, s1[-1].timestamp_eof_s, s2[0].timestamp_eof_s, s2[1].timestamp_eof_s]
    aligned = map_window_to_frames("route_8d", monos, max_delta_s=0.06)
    assert [a.segment_num for a in aligned] == [1, 1, 2, 2]          # boundary crossing OK
    assert [a.segment_id for a in aligned] == [s1[-2].segment_id, s1[-1].segment_id, 0, 1]

@pytest.mark.skipif(_route() is None, reason="no route_8d frames")
def test_missing_ecamera_yields_none_index():
    import glob
    # seg48 has fcamera.hevc but no ecamera.hevc on disk
    assert not glob.glob("explorer_st_logs/route_8d/0000008d--*--48/ecamera.hevc")
    tl = build_frame_timeline("route_8d")
    s48 = [r for r in tl if r.segment_num == 48]
    assert s48 and all(r.ecamera_segment_id is None for r in s48)   # no dangling index into a missing file
