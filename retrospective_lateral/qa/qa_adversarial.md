# Adversarial QA Findings

## Commands Run

- `.venv311/bin/python -m pytest retrospective_lateral/tests -q`
  - Initial adversarial QA result: exit 0, `70 passed in 3.07s`.
  - Post-fix verification: exit 0, `83 passed in 3.15s`.
- `.venv311/bin/python -m retrospective_lateral.code.run_all --route route_b8`
  - Initial adversarial QA result: exit 0, reused/generated `route_b8` cache with `35971` samples across `31` segments, wrote `13` symptom rows, and generated `retrospective_lateral_report.md`.
  - Post-fix verification: exit 0, rebuilt `route_b8` cache at schema `retrolat-v2`, `35971` samples across `31` segments, wrote `10` symptom rows, and generated `retrospective_lateral_report.md`.

Additional post-fix checks:

- Focused RED tests before the blocker fix failed for missing CP/CX1 cache channels, cache reuse, CP/CX1 exclusion flags, and internal excluded-span reporting.
- Focused GREEN tests after the blocker fix: `.venv311/bin/python -m pytest retrospective_lateral/tests/test_extract.py retrospective_lateral/tests/test_metrics_weave.py -q` -> `25 passed`.
- Route output audit found `16402` finite `cp_final_command` samples, `16428` finite `cx1_command_curvature` samples, and `0` reported route_b8 spans enclosing raw steering override, blinker, lane-change, CP override, CX1 override, or CX1 lane-change samples.

## Findings

1. **blocker**: Stage localization output was not defensible because route_b8 had CP/CX1 stage telemetry, but the cache/report ignored it.
   - Parsers existed for CP/CX1, but extraction only used LC rows for PI config.
   - Metrics localized using yaw-derived path, steering, `act_curvature`, `desired_curvature`, and `model_y20`, missing final Ford command and intermediate controller stages.

2. **blocker**: Eligibility and filtering could leak excluded behavior into route_b8 windows.
   - Metrics filtered full continuous signals with zero-phase filtering, then masked excluded samples.
   - Some route_b8 reported windows enclosed raw override, blinker, or lane-change samples.

3. **important**: Lead-follow contamination is neither extracted nor gated.

4. **important**: Default `run_all --route route_b8` is not fully route-isolated on a populated cache because metrics still scans every `route_*.npz` in the cache root.

5. **important**: Cache/provenance remain too weak for strong route/date/config claims. Sidecars record some provenance, but metrics/reporting do not surface dirty-state or full source provenance.

6. **important**: Extraction failures can disappear from reports when a route has only a failed JSON sidecar and no NPZ.

7. **nit**: The Markdown report heading `Top Symptom Episodes` overstates fixed-grid weave windows.

## Reconciliation

1. **Accepted and fixed.**
   - Commit `a02dd23ecd` extracts CP/CX1 telemetry from `logMessage`, resamples it into route caches, and uses CP/CX1 curvature stages in weave stage localization.
   - Commit `a8d83bbc1f` bumps the cache schema to `retrolat-v2` and requires stage-telemetry cache channels before reusing an existing cache, so normal `run_all --route route_b8` rebuilds pre-fix caches.
   - Focused spec and code-quality re-reviews found no remaining blockers, important findings, or nits for this accepted scope.

2. **Accepted and fixed.**
   - Commit `a02dd23ecd` masks ineligible samples before filtering metric signals.
   - Commit `a8d83bbc1f` adds CP/CX1 override and lane-change flags to the clean mask, splits weave buckets into contiguous eligible spans, and adds regression tests that reported spans do not cross internal excluded samples.
   - Post-fix route_b8 output had zero reported spans enclosing raw excluded flag samples.

3. **Accepted as residual scope risk, not fixed in Task 15.**
   - Lead/radar extraction and headway gating remain absent. Results should not claim the effect is independent of lead-follow context.

4. **Accepted as residual reproducibility risk, not fixed in Task 15.**
   - Route-scoped smoke runs should use a clean cache root when strict route isolation is required.

5. **Accepted as residual provenance risk, not fixed in Task 15.**
   - Sidecar provenance exists, but report/catalog outputs still do not expose all provenance fields or dirty-state caveats.

6. **Accepted as residual data-quality reporting risk, not fixed in Task 15.**
   - Corrupt NPZ files are reported, but extraction JSON-only failure sidecars are not yet promoted into `symptom_catalog.csv`.

7. **Accepted as wording nit, not fixed in Task 15.**
   - The heading remains acceptable for the current first-pass report, but future reporting should distinguish fixed-grid windows from bounded episodes.
