# Retrospective Lateral Weave Analysis

This package analyzes existing Explorer ST openpilot/sunnypilot logs for two symptoms:

- Low-speed steering-wheel swing at 1-10 mph.
- Straight/gentle-section path and wheel weave at 10-70 mph.

Run commands from the repository root with `.venv311/bin/python`.

The package is analysis-only. It must not modify vehicle-control code.

## Current Run

Full-corpus command:

```bash
.venv311/bin/python -m retrospective_lateral.code.run_all
```

Primary outputs:

- `retrospective_lateral/results/cache/`
- `retrospective_lateral/results/reports/symptom_catalog.csv`
- `retrospective_lateral/results/reports/retrospective_lateral_report.md`

Generated result files under `retrospective_lateral/results/` are intentionally gitignored.

Observed on the current local corpus:

- Manifest routes: `147`
- Cached route NPZ files: `145`
- Manifest extraction failures: `2`
- Symptom catalog rows: `1557`
- Low-speed wheel-swing rows: `75`
- 10-70 mph weave rows: `1482`

The current report is descriptive analysis. Documented QA residual risks include missing lead/radar gating, route-scoped report isolation on populated caches, incomplete provenance surfacing, and JSON-only extraction failures not yet promoted into `symptom_catalog.csv`.
