"""Materialize a model bundle's REAL ONNX files from git-LFS pointers at the bundle's
pinned commit, generate per-model metadata, and record provenance.

The bundle commits live in the local object store but their ONNX are git-LFS *pointers*
(133-byte text). To get the real bytes we (1) fetch the LFS object into the local cache
from the bundle's source remote, then (2) materialize via ``git lfs smudge``, then
(3) verify the materialized sha256 equals the LFS oid. Metadata is NOT shipped at these
commits, so we GENERATE it with ``selfdrive/modeld/get_model_metadata.py`` (standalone,
tinygrad-only) run under ``replay_env()``.

All artifacts are written UNDER ``C.RESULTS_ROOT/onnx/<bundle>/`` (gitignored) — never a
source dir.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from model_replay_sim import config as C
from model_replay_sim.env import replay_env

# Map a bundle's source repo (C.BUNDLES[...]["repo"]) -> the configured git remote that
# hosts its LFS objects. commaai/openpilot -> "openpilot"; sunnypilot/sunnypilot ->
# "upstream". (Remotes are pre-configured; see ``git remote -v``.)
_REPO_TO_REMOTE = {
    "commaai/openpilot": "openpilot",
    "sunnypilot/sunnypilot": "upstream",
}

_MODELS_DIR = "selfdrive/modeld/models"
# The .onnx files each bundle layout ships. A 2-model (non-split) bundle ships
# vision+policy; the split bundle ships vision + on/off policy.
_TWO_MODEL = ["driving_vision.onnx", "driving_policy.onnx"]
_SPLIT_MODEL = ["driving_vision.onnx", "driving_on_policy.onnx", "driving_off_policy.onnx"]

_GET_METADATA = C.REPO_ROOT / "selfdrive" / "modeld" / "get_model_metadata.py"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tinygrad_sha() -> str:
    out = subprocess.run(
        ["git", "-C", str(C.TINYGRAD_PATH), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _bundle_cfg(bundle: str) -> dict:
    if bundle not in C.BUNDLES:
        raise KeyError(f"unknown bundle: {bundle!r} (known: {sorted(C.BUNDLES)})")
    return C.BUNDLES[bundle]


def _remote_for(bundle: str) -> str:
    repo = _bundle_cfg(bundle)["repo"]
    try:
        return _REPO_TO_REMOTE[repo]
    except KeyError:
        raise KeyError(f"no LFS remote mapped for repo {repo!r} (bundle {bundle!r})")


def bundle_onnx_paths(bundle: str) -> list[str]:
    """The ``selfdrive/modeld/models/<name>.onnx`` paths this bundle ships, driven by
    its ``split`` flag in ``C.BUNDLES``."""
    names = _SPLIT_MODEL if _bundle_cfg(bundle)["split"] else _TWO_MODEL
    return [f"{_MODELS_DIR}/{n}" for n in names]


def lfs_oid(bundle: str, onnx_path: str) -> str:
    """The git-LFS oid (sha256 of the real object) for ``onnx_path`` at the bundle's
    pinned commit. Parsed from ``git lfs ls-files --long --include=<path> <sha>`` whose
    first whitespace token is the oid.

    FIX 6: use ``--include=<path>`` — the ``-- <path>`` pathspec form fails (exit 128).
    """
    sha = _bundle_cfg(bundle)["full_sha"]
    out = subprocess.run(
        ["git", "lfs", "ls-files", "--long", f"--include={onnx_path}", sha],
        cwd=str(C.REPO_ROOT), capture_output=True, text=True, check=True,
    )
    line = out.stdout.strip()
    if not line:
        raise RuntimeError(f"no LFS entry for {onnx_path} at {bundle} ({sha})")
    return line.split()[0]


def _fetch_lfs_object(bundle: str, onnx_path: str) -> None:
    """Fetch the LFS object for ``onnx_path`` at the bundle's commit into the local
    cache, from the bundle's source remote."""
    remote = _remote_for(bundle)
    sha = _bundle_cfg(bundle)["full_sha"]
    subprocess.run(
        ["git", "lfs", "fetch", remote, sha, "-I", onnx_path],
        cwd=str(C.REPO_ROOT), capture_output=True, text=True, check=True,
    )


def _smudge_to(onnx_path: str, sha: str, out: Path) -> None:
    """Materialize the real ONNX bytes: pipe the (cached) LFS pointer through
    ``git lfs smudge`` into ``out``."""
    show = subprocess.Popen(
        ["git", "show", f"{sha}:{onnx_path}"],
        cwd=str(C.REPO_ROOT), stdout=subprocess.PIPE,
    )
    with open(out, "wb") as fout:
        smudge = subprocess.run(
            ["git", "lfs", "smudge"],
            cwd=str(C.REPO_ROOT), stdin=show.stdout, stdout=fout,
            stderr=subprocess.PIPE,
        )
    show.stdout.close()
    show.wait()
    if show.returncode != 0:
        raise RuntimeError(f"git show failed for {onnx_path} at {sha}")
    if smudge.returncode != 0:
        raise RuntimeError(f"git lfs smudge failed for {onnx_path}: {smudge.stderr.decode()[-500:]}")


def _generate_metadata(onnx_out: Path) -> Path:
    """Run the standalone ``get_model_metadata.py`` (under replay_env, since it imports
    tinygrad) on the materialized ONNX. It writes ``<stem>_metadata.pkl`` next to the
    ONNX. Returns that path."""
    env = replay_env()
    proc = subprocess.run(
        [sys.executable, str(_GET_METADATA), str(onnx_out)],
        env=env, capture_output=True, text=True,
    )
    meta = onnx_out.parent / (onnx_out.stem + "_metadata.pkl")
    if proc.returncode != 0 or not meta.exists():
        raise RuntimeError(
            f"get_model_metadata.py failed for {onnx_out.name} "
            f"(rc={proc.returncode}): {proc.stderr[-1000:]}"
        )
    return meta


def materialize_bundle(bundle: str, force: bool = False) -> dict:
    """Materialize every ONNX of ``bundle`` (LFS fetch + smudge + sha256==oid integrity
    check), generate each model's metadata pkl, and write a ``provenance.json``. All
    under ``C.RESULTS_ROOT/onnx/<bundle>/`` (gitignored). Returns the provenance dict.

    Re-materialization is skipped per-file when the ONNX + its metadata already exist and
    ``force`` is False (the integrity check still runs against the existing file).
    """
    cfg = _bundle_cfg(bundle)
    sha = cfg["full_sha"]
    out_dir = C.RESULTS_ROOT / "onnx" / bundle
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for onnx_path in bundle_onnx_paths(bundle):
        name = Path(onnx_path).name
        out = out_dir / name
        meta = out_dir / (out.stem + "_metadata.pkl")
        oid = lfs_oid(bundle, onnx_path)

        if force or not out.exists() or not meta.exists():
            _fetch_lfs_object(bundle, onnx_path)
            _smudge_to(onnx_path, sha, out)

        digest = _sha256(out)
        if digest != oid:
            raise RuntimeError(
                f"integrity check FAILED for {bundle}/{name}: "
                f"sha256={digest} != lfs_oid={oid} (smudge did not produce the real bytes)"
            )

        if force or not meta.exists():
            meta = _generate_metadata(out)

        models.append({
            "name": name,
            "onnx_path": onnx_path,
            "onnx_sha256": digest,
            "lfs_oid": oid,
            "metadata_path": meta.name,
        })

    prov = {
        "bundle": bundle,
        "full_sha": sha,
        "repo": cfg["repo"],
        "tinygrad_sha": _tinygrad_sha(),
        "models": models,
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    return prov
