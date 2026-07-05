from __future__ import annotations
import hashlib, subprocess, sys
from pathlib import Path
from model_replay_sim import config as C
from model_replay_sim.env import replay_env

def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _tinygrad_sha() -> str:
    out = subprocess.run(["git", "-C", str(C.TINYGRAD_PATH), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()

def compile_onnx_to_pkl(onnx_path, out_pkl, compare_onnxruntime: bool = True) -> dict:
    """Compile an ONNX to a Mac (DEV=CPU) tinygrad pkl in ONE run. A failing 1e-4
    SELFTEST is recorded as a fact (onnxruntime_strict_1e4_passed=False), NOT a failure."""
    onnx_path = Path(onnx_path).resolve()           # abs path (fetch rejects bare relative)
    out_pkl = Path(out_pkl)
    env = replay_env({"SELFTEST": "1"} if compare_onnxruntime else {})
    cmd = [sys.executable, str(C.TINYGRAD_PATH / "examples" / "openpilot" / "compile3.py"),
           str(onnx_path), str(out_pkl)]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    compile_ok = out_pkl.exists() and "pkl size is" in proc.stdout
    return {
        "onnx": str(onnx_path),
        "out_pkl": str(out_pkl),
        "compile_ok": bool(compile_ok),
        "onnxruntime_strict_1e4_passed": (compare_onnxruntime and proc.returncode == 0),
        "tinygrad_sha": _tinygrad_sha(),
        "onnx_sha256": _sha256(onnx_path),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
    }
