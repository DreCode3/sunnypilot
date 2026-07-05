from __future__ import annotations
import os, pytest
def require_real_asset(present: bool, what: str):
    if present:
        return
    if os.environ.get("MODEL_REPLAY_ALLOW_MISSING_ASSETS") == "1":
        pytest.skip(f"unit mode: missing {what}")    # FIX 5: skip, never run the body
    raise AssertionError(f"missing required real asset: {what}")
