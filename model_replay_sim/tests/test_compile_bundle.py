from pathlib import Path
from model_replay_sim.compile_bundle import compile_onnx_to_pkl
from model_replay_sim import config as C
def test_compile_stock_policy(tmp_path):
    onnx = C.REPO_ROOT/"selfdrive/modeld/models/driving_policy.onnx"
    rec = compile_onnx_to_pkl(onnx, tmp_path/"policy.pkl", compare_onnxruntime=True)
    assert (tmp_path/"policy.pkl").exists()
    assert rec["compile_ok"] is True                      # compile + pkl written succeeded
    assert "onnxruntime_strict_1e4_passed" in rec          # captured as a FACT (False is acceptable, NOT a gate)
    assert isinstance(rec["onnxruntime_strict_1e4_passed"], bool)
    assert rec["tinygrad_sha"] and rec["onnx_sha256"]
    assert rec["onnx_sha256"] == __import__("hashlib").sha256(onnx.read_bytes()).hexdigest()
