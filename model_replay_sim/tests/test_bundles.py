import pytest
from model_replay_sim.bundles import bundle_onnx_paths, materialize_bundle
from model_replay_sim import config as C

def test_cd210_is_two_model():
    paths = bundle_onnx_paths("CD210")
    assert any(p.endswith("driving_vision.onnx") for p in paths)
    assert any(p.endswith("driving_policy.onnx") for p in paths)
    assert not any("on_policy" in p for p in paths)   # CD210 is not split

@pytest.mark.skipif(__import__("os").environ.get("MODEL_REPLAY_SKIP_NETWORK") == "1", reason="network materialize")
def test_materialize_cd210_real_onnx_and_provenance():
    prov = materialize_bundle("CD210")
    assert prov["full_sha"] == C.BUNDLES["CD210"]["full_sha"]
    assert prov["tinygrad_sha"]
    assert len(prov["models"]) == 2
    for m in prov["models"]:
        out = C.RESULTS_ROOT/"onnx"/"CD210"/m["name"]
        assert out.exists() and out.stat().st_size > 1_000_000          # real onnx, not a 133B pointer
        assert m["onnx_sha256"] == m["lfs_oid"]                          # integrity (smudge produced the right bytes)
        assert (C.RESULTS_ROOT/"onnx"/"CD210"/m["metadata_path"]).exists() or __import__("pathlib").Path(m["metadata_path"]).exists()
