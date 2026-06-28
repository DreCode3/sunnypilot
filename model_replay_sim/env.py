from __future__ import annotations
import os, sys
from model_replay_sim import config as C
def replay_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(C.TINYGRAD_ENV)                       # hard-set DEBUG=0/DEV=CPU/IMAGE=0/THREADS=0
    existing = env.get("PYTHONPATH", "")
    parts = [str(C.TINYGRAD_PATH), str(C.REPO_ROOT)]
    if existing: parts.append(existing)
    env["PYTHONPATH"] = ":".join(parts)
    if extra: env.update({k: str(v) for k, v in extra.items()})
    return env
def replay_python(args: list[str]) -> list[str]:
    return [sys.executable, *args]
