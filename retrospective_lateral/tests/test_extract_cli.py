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
