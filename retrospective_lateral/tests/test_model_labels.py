import json

import pandas as pd

from retrospective_lateral.code import model_labels as L


def test_parse_active_bundle_from_json_bytes():
    blob = json.dumps({"index": 54, "internalName": "C210M",
                       "displayName": "CD210 Model (January 31, 2026)", "extra": 1}).encode()
    out = L.parse_active_bundle(blob)
    assert out == {"index": 54, "internal_name": "C210M",
                   "display_name": "CD210 Model (January 31, 2026)"}


def test_parse_active_bundle_accepts_dict_and_str():
    d = {"index": 33, "internalName": "NM", "displayName": "Nevada Model"}
    assert L.parse_active_bundle(d)["internal_name"] == "NM"
    assert L.parse_active_bundle(json.dumps(d))["internal_name"] == "NM"


def test_parse_active_bundle_rejects_garbage():
    assert L.parse_active_bundle(b"not json") is None
    assert L.parse_active_bundle(b"{}") is None          # no internalName
    assert L.parse_active_bundle(json.dumps([1, 2]).encode()) is None


def test_load_labels_roundtrip(tmp_path):
    df = pd.DataFrame({"route_id": ["route_b8", "route_47"], "model": ["C210M", "NM"],
                       "display_name": ["", ""], "index": [54, 33], "source": ["active_bundle"] * 2})
    df.to_csv(tmp_path / "route_model_labels.csv", index=False)
    labels = L.load_labels(tmp_path)
    assert labels == {"route_b8": "C210M", "route_47": "NM"}


def test_hand_labels_cover_opm7_era():
    # the OPM7 reference routes (no ActiveBundle in their era) have a fallback label
    assert L.HAND_LABELS["route_7f"] == "OPM7"
    assert L.HAND_LABELS["route_b8"] == "CD210"
