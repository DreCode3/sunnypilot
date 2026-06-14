# longitudinal_weave/ — does the lateral WEAVE grow over time / since a learned-param reset?

Tests the driver's hypothesis: the slow left-right wheel weave GROWS as openpilot's LEARNED params
(paramsd steerRatio/angleOffset, calibration) drift from a fresh/reset state, and DROPS at resets — a sawtooth in
weave vs time, INDEPENDENT of our PI-config tuning. If true, the lever is learned-param drift, not the PI.

## VERDICT (2026-06-14): TIME-TREND **REFUTED** by device reset ground-truth.
The apparent "weave rises +0.025/day with time-since-reset (calibrated blockperm p=0.004, speed-robust)" was an
**ARTIFACT** of inferred angleOffset-jump "resets" (Apr18/May3/May21/May23) that the **device shows never happened**
(pulled 2026-06-14: `LiveParametersV2`+`CalibrationParams` persist normally; `CarParamsPrevRoute` stable since
InstallDate 2025-11-25 → fingerprint unchanged; only real in-window reset = the ~Mar-29 reflash that reset the route
counter 78→12). With GROUND-TRUTH resets the trend is borderline/null and **flips sign** (Mar29-only +0.005/day
p=0.05 ≈ the uncalibratable calendar case; Mar29+Jun7 −0.005/day p=0.07). A real effect doesn't flip sign on one
ambiguous reset → no reliable weave-grows-with-time effect. Today's golden drive bf (Jun14, 3.12@51mph, 94th pctile
speed-adjusted) is genuinely elevated and the driver felt it, but = speed + normal high-variance, not escalation.
Full detail: memory `finding_longitudinal_weave_time.md`.

## Data
- 147 `explorer_st_logs/route_*` dirs (144 with extractable data; 113GB; Feb27→Jun14 2026), BOTH on-disk formats
  (old flat `rlog_N.zst`, new `000000XX--hash--N/rlog.zst`). The device keeps only ~23 recent routes; the local
  store is the fuller history (pulled over months).
- **70 drives have a usable weave value** (n_elig≥3); 74 excluded (48 with no clear straights at all, 26 with only
  1-2 eligible segments) — correctly filtered: >80mph highway-traffic or city (the metric needs 30-80mph clear
  straights). NOT a bug.
- ⚠️ `route_counter` is NOT chronological (device counter reset at the Mar-29 reflash) → ALWAYS order by `wall_date`
  (max-clocks, post-GPS-sync). `calPerc` is 100 throughout → calibration-reset signal is dead here.

## Metric & method (the only configuration that is CALIBRATED)
- WEAVE = the 3-round-QA'd model-indep band-RMS (0.10-0.35Hz of yawRate/vEgo = `weave_path`), reused from
  `learned_param_studies/code/{shared.py, extract_master.py}`; per-drive median over eligible 45s engaged-straight
  segments. SPEED is the dominant confound (Spearman weave↔speed ≈ −0.83) → always controlled.
- Significance: **`blockperm`** (contiguous-block permutation of the speed+config-residualized weave vs the time
  variable), predictor = **`tau` (days-since-reset)**, `epoch_fe=False` → FPR ~0.04 at AR(0.3), rising to ~0.08 at
  AR(0.6) (Monte-Carlo verified, realized epoch sizes). EVERYTHING ELSE is anti-conservative: the CALENDAR predictor
  has NO calibrated test (all methods FPR ~0.14-0.23 even under `epoch_fe`, worse without — random per-epoch baseline
  levels confound a monotone trend); `epoch_fe` and circular-shift both break.
  MBB/sbb/cluster/ar1 are borderline 0.07-0.08. See `calib_candidates.py` for the 5-method comparison.

## Files
- `build.py` — PARALLEL extractor (ProcessPoolExecutor, all cores). One row/drive → weave + speed + centering +
  learned-param START/END trajectory + calPerc + config + date. Resumable (per-route npz cache + skips CSV rows).
  Run: `.venv311/bin/python longitudinal_weave/build.py --out longitudinal_weave/results [--workers N] [--only route_x]`
- `analyze.py` — reset detection (date-ordered; sr/aoa jumps + counter-reset; calPerc) + the trend test + epoch
  assignment. ⚠️ heuristic reset thresholds over-fragment; prefer GROUND-TRUTH reset dates (see verdict).
- `calibrate.py` / `calib_candidates.py` — Monte-Carlo FPR + power of the trend tests (parallel). `--sizes` overrides
  epoch structure. This is how the blockperm-tau calibration (~0.05) and the calendar-uncalibratable (0.14-0.23) were proven.
- `final_trend.py` — the verified blockperm-tau run on real data + re-confirms calibration at the realized epoch structure.
- `results/` — `drive_longitudinal.csv` (one row/drive), `cache/*.npz` (resampled per-drive signals), `drive_epochs.csv`.

## Decisive remaining test (if drift is still suspected)
Deliberate RESET-and-MONITOR: clear `LiveParametersV2`/`CalibrationParams` on device, then drive ONE fixed corridor
repeatedly for ~a week — a controlled reset with KNOWN timing, replacing the after-the-fact inferred resets that
made the observational analysis fragile.
