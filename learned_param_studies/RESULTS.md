# learned_param_studies/ — does any LEARNED parameter correlate with the lateral weave/centering?

Tests whether the slow weave is driven by openpilot's learned/cumulative params — paramsd liveParameters
(steerRatio, angleOffsetAverage, stiffness), the warm-started PI integrator (LaneBiasIntegral), or learned
camera calibration (rpyCalib) — across 10 historical drives / 73 engaged-straight segments. (This 10-drive set is the
per-segment-resolution subset used here; the separate `../longitudinal_weave/` study uses a larger ~70-drive store —
the differing counts are by design, not a discrepancy.)

## VERDICT (2026-06-14, after 3 rounds of adversarial QA): NO learned parameter is a validated weave lever.
- **steerRatio → weave: NULL** (within-drive r=+0.10 p=0.52; across r=−0.09; partial r=−0.31 p=0.44). The 16.4→17.2
  drift is real but doesn't correlate with weave. Underpowered (narrow observed range) — "no detectable effect," not strong disproof.
- **angleOffsetAverage → weave: ns** (within p=0.86). The earlier n=5 "r=−0.80" was a small-n bootstrap-CI artifact
  + speed confound, driven by one drive.
- **PI integrator (LaneBiasIntegral): survives FDR but mechanically REVERSE-CAUSAL** — it correlates with *centering*
  (winds up in RESPONSE to an offset; raw 1Hz telemetry shows |int| lags offset +1-2s), not a cause of weave. Its
  weave link is negative and does not survive global FDR. Not a tuning lever.
- **calibration cal_pitch → weave: REFUTED / ARTIFACT** — the prior positive within-drive correlation (+0.16 in the
  corrected table; an earlier pre-speed-control pass showed +0.29) was a pure SPEED confound (collapses to −0.14
  speed-controlled; within-drive pitch varies ~0.01° = noise floor). cal_yaw ns;
  cal_yaw≡cal_roll is genuine rank-collinearity (calibrationd hard-codes roll=0 → roll is a function of yaw), deduped.
- **Global BH-FDR (effective m=28): 2 robust, both the reverse-causal integrator→centering rows → 0 weave/hunt levers.**
- ⭐ The ONLY robust within-drive signal is a CONFOUND, not a learned param: **lower speed → more weave (r≈−0.65)**.

So "weak performs better now" is NOT explained by a measurable learned-param→weave effect. (The separate
*longitudinal* time-since-reset question — see `../longitudinal_weave/` — was also refuted by device ground truth.)
Full detail: memory `finding_learned_param_drift.md`.

## Method (calibrated, verified)
- Model-indep path-weave (0.1-0.35Hz yawRate/vEgo) + within-drive **detrend + SPEED-control**, significance by
  **Freedman-Lane permutation** (permute the response residual, RE-residualize each permutation), global **BH-FDR**.
- ⚠️ KEY QA LESSON: naive permutation of OLS residuals is ~2× ANTI-CONSERVATIVE (FPR ~9-13%, caught in QA round 1 as
  a BLOCKING bug) → use Freedman-Lane. ALWAYS Monte-Carlo an estimator's FPR before trusting "significant."

## QA history (3 rounds, all PASS after remediation)
- **Round 1:** found the blocking anti-conservative permutation (fixed → Freedman-Lane, FPR re-verified ~0.05) + a
  `time_valid` flag fooled by an outlier timestamp (fixed → cluster-count rule). The broken `wall_date` time-control
  (12/13 drives in a 17s pre-sync window) and the `fmt()` "SIG"-from-CI bug were also fixed.
- **Round 2:** PASS + 4 nits (scale-relative degenerate guards; FDR de-dup of collinear cal_roll; speed-confound
  docstring; harness null-coverage) — all remediated.
- **Round 3:** PASS. Confirmed the only mildly-anti-conservative params (angleOffsetAvg, steerRatio~weave_steer,
  stiffness~hunt_steer ~0.08-0.13 FPR) are IMMATERIAL by disjointness — none coincides with a near-significant real
  result; a 2× inflation stress-model still yields 0 levers.

## Files
- `code/{extract_master.py, shared.py}` — extractor (both on-disk formats, calPerc) + the metric battery +
  the calibrated stats (spearman_perm_p, partial_spearman & within_drive_spearman with Freedman-Lane, bh_fdr).
- `code/{build_table.py, test1_liveparams.py, test2_integrator.py, test3_calibration.py, run_all.py}` — per-param
  tests + `run_all.py` global BH-FDR over the full primary battery.
- `code/qa_calibration.py` — Monte-Carlo FPR/power harness (incl. the assumption-free cross-drive real-pairing null).
- `results/` — segment_table.csv (73 segs), drive_table.csv (10 drives), test*_results.csv, GLOBAL_fdr_summary.csv.
