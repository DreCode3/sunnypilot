from retrospective_lateral.code.run_all import build_pipeline_steps


def test_build_pipeline_steps_uses_cache_and_report_dirs():
  steps = build_pipeline_steps(log_root="logs", cache_root="cache", report_root="reports", routes=["route_b8"])
  assert steps[0][:4] == [".venv311/bin/python", "-m", "retrospective_lateral.code.extract", "--log-root"]
  assert "--route" in steps[0]
  assert steps[1][:3] == [".venv311/bin/python", "-m", "retrospective_lateral.code.metrics"]
