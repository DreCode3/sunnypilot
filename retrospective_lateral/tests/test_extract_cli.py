import json
import subprocess
from pathlib import Path


def test_extract_cli_list_routes(tmp_path):
  route = tmp_path / "route_x1" / "000000x1--abc--0"
  route.mkdir(parents=True)
  (route / "rlog.zst").write_bytes(b"not-a-real-log")
  cmd = [
    ".venv311/bin/python",
    "-m",
    "retrospective_lateral.code.extract",
    "--log-root",
    str(tmp_path),
    "--list-routes",
  ]
  proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=True)
  rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
  assert rows == [{"route_id": "route_x1", "segments": 1, "layout": "modern"}]


def test_extract_cli_writes_failed_status_for_unreadable_route(tmp_path):
  route = tmp_path / "route_bad" / "000000x1--abc--0"
  route.mkdir(parents=True)
  (route / "rlog.zst").write_bytes(b"not-a-real-log")
  cache_root = tmp_path / "cache"
  cmd = [
    ".venv311/bin/python",
    "-m",
    "retrospective_lateral.code.extract",
    "--log-root",
    str(tmp_path),
    "--cache-root",
    str(cache_root),
    "--route",
    "route_bad",
  ]

  proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True, check=True)

  rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
  meta = json.loads((cache_root / "route_bad.json").read_text())
  manifest = json.loads((cache_root / "manifest.json").read_text())
  assert rows == [meta]
  assert manifest == [meta]
  assert meta["success"] is False
  assert meta["sample_count"] == 0
  assert meta["npz_path"] is None
  assert meta["notes"]
  assert not (cache_root / "route_bad.npz").exists()


def test_extract_cli_missing_route_filter_exits_nonzero(tmp_path):
  route = tmp_path / "route_x1" / "000000x1--abc--0"
  route.mkdir(parents=True)
  (route / "rlog.zst").write_bytes(b"not-a-real-log")
  cmd = [
    ".venv311/bin/python",
    "-m",
    "retrospective_lateral.code.extract",
    "--log-root",
    str(tmp_path),
    "--route",
    "definitely_missing",
    "--list-routes",
  ]

  proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], text=True, capture_output=True)

  assert proc.returncode != 0
  assert "definitely_missing" in proc.stderr
