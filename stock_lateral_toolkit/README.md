# stock_lateral_toolkit

Lateral-analysis toolkit for the 2021 Ford Explorer ST on **stock sunnypilot dev
v2026.002.000+** — the clean-baseline measurement kit for the fresh-start effort
(handoff: `docs/superpowers/reports/2026-07-03-stock-fresh-start-handoff.md`).
Seeded 2026-07-03 from the fork-era analysis library's portable core (inventory:
`docs/superpowers/reports/2026-07-02-fresh-start-tooling-logging-inventory.md`);
validated per `validation/ACCEPTANCE.md`.

## Files
- `extract_drive.py` — rlog.zst → per-modelV2-frame npz cache (~20 Hz). Reads each build's
  own carParams (steerRatio etc.). Route glob may point straight at the NAS mount.
- `analyze_compare.py` — matched A/B: GPS-cell + speed-bin paired, model-independent
  straight/curve classification, band-limited 0.10-0.35 Hz weave, robust stats. Add new
  corridors to `CORRIDORS`.
- `signal_utils.py` — band/low-pass filtfilt with gap guards, masked RMS/p2p, GPS cells,
  spectral peak. `shared.py` — weave band-RMS metric + the calibrated stats battery
  (exact/MC permutation, Freedman-Lane partial & within-drive Spearman, BH-FDR).
- `qa_calibration.py`, `calib_candidates.py` — FPR/power calibration harnesses. **Prove any
  new estimator here before trusting its p-values.**
- `config.py` — shared constants (bands, cells, paths). `cache/` — npz caches (gitignored).
- `validation/` — pre-registered port-acceptance gate + negative-control harness.

## Workflow
```sh
# extract a drive (local dir or NAS path):
.venv311/bin/python stock_lateral_toolkit/extract_drive.py \
  "/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock/00000004--3d3385646d" \
  stock_lateral_toolkit/cache/stock_hiram_04.npz
# compare (corridor defined in CORRIDORS):
.venv311/bin/python stock_lateral_toolkit/analyze_compare.py hiram
```

## Discipline (non-negotiable — comparisons flipped 3× without it)
- Speed-match + GPS-cell-match ALWAYS; report per-group speed median/IQR first.
- Robust stats only (median/percentile/sign tests) — never pooled variance.
- Band-limit 0.10-0.35 Hz; steering-angle / yawRate / lane-offset as primaries — NEVER
  aLat (f²-blind at weave frequencies; keep it only as a road-banking control).
- Engagement-gate with `carControl.latActive`; eligibility masks are NOT portable between
  analyses — re-derive each filter's justification per purpose.
- Verify every agent/analysis output independently. Calibrate estimators (FPR/power)
  before trusting significance.

## Provenance
Fork-era originals: `retrospective_lateral/stock_compare/` (proven on stock 2026-07-02),
`retrospective_lateral/code/signal_utils.py`, `learned_param_studies/code/{shared,
qa_calibration}.py`, `longitudinal_weave/calib_candidates.py`. De-couple edits: fork-only
telemetry columns removed (pi_set, lc_int/int_abs/int_railed_frac), obsolete
longitudinal-analyze dependency removed, paths made repo-relative.
