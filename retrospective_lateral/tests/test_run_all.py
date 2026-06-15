from pathlib import Path

from retrospective_lateral.code import run_all


def test_build_pipeline_steps_uses_cache_and_report_dirs():
  steps = run_all.build_pipeline_steps(log_root="logs", cache_root="cache", report_root="reports", routes=["route_b8"])
  assert steps[0][:4] == [".venv311/bin/python", "-m", "retrospective_lateral.code.extract", "--log-root"]
  assert "--route" in steps[0]
  assert steps[1][:3] == [".venv311/bin/python", "-m", "retrospective_lateral.code.metrics"]


def test_build_pipeline_steps_passes_force_and_preserves_route_order():
  steps = run_all.build_pipeline_steps(
    log_root="logs",
    cache_root="cache",
    report_root="reports",
    routes=["route_a", "route_b"],
    force=True,
  )
  extract = steps[0]
  assert "--force" in extract
  route_args = [extract[index + 1] for index, value in enumerate(extract) if value == "--route"]
  assert route_args == ["route_a", "route_b"]


def test_write_pipeline_report_reads_symptom_catalog_and_writes_markdown(tmp_path, monkeypatch):
  report_root = tmp_path / "reports"
  report_root.mkdir()
  (report_root / "symptom_catalog.csv").write_text("route_id,symptom,status\nroute_b8,weave_10_70,ok\n")
  calls = {}

  def fake_write_markdown_report(out_dir, symptom_catalog, scorecard):
    calls["out_dir"] = Path(out_dir)
    calls["symptom_catalog"] = symptom_catalog
    calls["scorecard"] = scorecard
    report_path = Path(out_dir) / "retrospective_lateral_report.md"
    report_path.write_text("# report\n")
    return report_path

  monkeypatch.setattr(run_all, "write_markdown_report", fake_write_markdown_report)

  path = run_all.write_pipeline_report(report_root)

  assert path == report_root / "retrospective_lateral_report.md"
  assert calls["out_dir"] == report_root
  assert calls["symptom_catalog"]["route_id"].tolist() == ["route_b8"]
  assert calls["scorecard"].empty
