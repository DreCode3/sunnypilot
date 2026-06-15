# Cooperative QA Findings

## Commands Run

- `.venv311/bin/python -m pytest retrospective_lateral/tests -q`
  - Result: exit 0, `68 passed in 2.97s`.
- `.venv311/bin/python -m retrospective_lateral.code.run_all --route route_b8`
  - Result: exit 0. Reused/generated `route_b8` cache, `35971` samples across `31` segments, wrote `13` symptom rows, and generated `retrospective_lateral_report.md`.

Inspected:

- `retrospective_lateral/results/cache/route_b8.json`
- `retrospective_lateral/results/reports/symptom_catalog.csv`
- `retrospective_lateral/results/reports/retrospective_lateral_report.md`

## Findings

1. **blocker**: `metrics.py` emitted `confidence = "supported"` for low-speed episodes based only on local detector threshold checks.
   - Why it matters: the design evidence gate reserves terms like `supported`, `refuted`, and `root-cause likely` until cooperative/adversarial QA has passed or residual risk is explicitly documented.
   - Suggested fix: reserve `supported` for post-QA conclusions. Use detector-local labels such as `command_correlated`, `path_correlated`, or `preliminary`.

2. **important**: stage localization is narrower than the design target.
   - `CP:` and `CX1:` parsers exist, but those channels are not yet resampled into cache timelines.
   - Current stage labels compare model y20, desired curvature, actuator curvature, path curvature, and steering, but omit several design stages such as orientationRate-derived curvature, blend/EMA/smooth stages, PI trim, and Ford post-rate-limit command.

3. **important**: `run_all --route route_b8` is not route-isolated on a populated cache.
   - The route filter is passed to extraction only.
   - Metrics still scan every `route_*.npz` in the cache root.

4. **important**: the design output set is only partially implemented.
   - The pipeline writes `symptom_catalog.csv` and a Markdown report.
   - It does not yet generate separate `stage_localization_report`, `historical_scorecard`, April 6/8 case study, or a fully evidence-tiered next-experiment artifact.

5. **important**: cache schema/provenance are written but not enforced or surfaced in metric outputs.
   - Route sidecars record schema, commit, dirty state, and config confidence.
   - Metrics currently load `.npz` files directly without validating sidecar schema or joining provenance into the catalog/report.

## Reconciliation

1. **Accepted and fixed.**
   - Commit `b4081f3f43` changes low-speed detector confidence labels from `supported` to `command_correlated` or `path_correlated`, while preserving `steering_only`.
   - Added regression coverage that command-corroborated detector output no longer uses `supported`.
   - Verification after fix: `.venv311/bin/python -m pytest retrospective_lateral/tests/test_metrics_low_speed.py -q` -> `7 passed`; `.venv311/bin/python -m pytest retrospective_lateral/tests -q` -> `70 passed`.

2. **Accepted as residual scope risk, not fixed in Task 14.**
   - The current implementation provides a first-pass stage label, not the full stage-localization report described in the design.
   - This remains a documented limitation for adversarial QA and later implementation work.

3. **Accepted as residual reproducibility risk, not fixed in Task 14.**
   - Route-scoped smoke runs should use a clean cache root if strict route isolation is needed.
   - Full-corpus runs are unaffected by route filtering because they intentionally scan the full cache.

4. **Accepted as residual scope risk, not fixed in Task 14.**
   - The plan implementation has built the core pipeline, catalog, comparison helpers, report writer, and QA audit hooks.
   - Separate stage-localization, historical-scorecard, and April 6/8 case-study artifacts remain incomplete and should not be represented as complete analysis outputs.

5. **Accepted as residual provenance risk, not fixed in Task 14.**
   - Cache metadata exists and extraction normalizes cache status fields, but metrics/reporting do not yet enforce schema or include provenance columns.
   - This should be addressed before making high-confidence historical claims from full-corpus outputs.
