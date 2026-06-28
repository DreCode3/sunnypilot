import subprocess
from model_replay_sim.env import replay_env, replay_python
def test_replay_env_sets_flags_and_pythonpath():
    env = replay_env()
    assert env["DEBUG"] == "0" and env["DEV"] == "CPU" and env["IMAGE"] == "0" and env["THREADS"] == "0"
    assert "tinygrad_repo" in env["PYTHONPATH"]
def test_replay_env_forces_debug_0_even_if_set():
    import os
    env = replay_env()  # replay_env must hard-set DEBUG=0 regardless of inherited DEBUG
    assert env["DEBUG"] == "0"
def test_replay_python_imports_framereader_and_tinygrad():
    proc = subprocess.run(
        replay_python(["-c", "import openpilot.tools.lib.framereader; import tinygrad; print('ok')"]),
        env=replay_env(), text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"
