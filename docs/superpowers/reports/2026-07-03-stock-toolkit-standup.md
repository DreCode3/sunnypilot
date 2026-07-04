# Stock lateral toolkit stand-up — session report (2026-07-03)

Session goal (per `2026-07-03-stock-fresh-start-handoff.md`): stand up a stock-compatible
lateral-analysis toolkit (Step 1), and continue into the simulator-on-stock work (Step 2)
if Step 1 validated. Both steps landed this session.

## Step 0 decision
Kept the branch (`stock-2026.002.001-fresh-start`) as-is — toolkit built in a new
`stock_lateral_toolkit/` dir (handoff option (a)). Clean-stock reset deferred.

Housekeeping commits: the ~130-file NAS-migration deletion of `explorer_st_logs/`
(`e57aeb2886`) and the F3-session leftovers incl. the load-bearing `alignment.py`
corrupt-event fix (`f972bbe445`).

## Step 1 — `stock_lateral_toolkit/` (DONE, validated, committed `acc59b0454`)

Seeded from the portable core: `extract_drive.py`, `analyze_compare.py` (stock_compare pair),
`signal_utils.py`, `shared.py` (stats battery), `qa_calibration.py`, `calib_candidates.py`,
plus a toolkit-local `config.py`. De-couple edits: fork telemetry columns removed
(`pi_set`, `lc_int`/`int_abs`/`int_railed_frac`), obsolete `longitudinal_weave/analyze.py`
dependency removed, absolute paths made repo-relative.

**Pre-registered acceptance gates (`validation/ACCEPTANCE.md`) — ALL PASS:**
- **A. Extractor regression:** stock Hiram 04/05 re-extracted from the NAS are
  **bit-identical** to the proven caches (max |Δ| = 0.0, identical NaN patterns,
  steerRatio 16.8, FORD_EXPLORER_MK6).
- **B. Positive control:** new analyzer output **byte-identical** to the fork-era location;
  the known highway result reproduces (straights band|yawRate| 22-30 m/s: stock 0.0022 vs
  custom-Nevada 0.0043 rad/s ≈ 2×, stock better in 89% of 36 matched cells).
- **C. Negative control:** stock-04 vs stock-05 (same build/corridor, opposite directions)
  through the full matched pipeline + a visit-permutation null (`validation/self_split.py`,
  NSHUFFLE=1000): observed deltas deep inside the null bands (highway delta +0.00001 rad/s),
  win-rates 43–62%. PASS both gated bins.

**Independent verification (4 parallel agents, 0 blockers):**
- Diff audit: 6/6 file pairs clean, 0 leftover fork-coupling greps, constants 6/6 match.
- Adversarial null-harness review: shuffle proven non-vacuous (exchanges ~50% of matched
  cells/draw; a synthetic 2× effect FAILs the control; sensitivity floor ~11–30% at highway);
  seed-stable across 6 seeds at NSHUFFLE=2000.
- Independent re-implementation (from spec only): headline reproduces at 1.92× / 88.6%.
- Stats battery post-surgery: FPR 0.045 (spearman n=8) / 0.055 (within-drive FL), power 0.883,
  bh_fdr exact; `segment_table` runs clean without any fork keys.

**Caveats of record:**
- The validation gates **15–30 m/s only** — the 8–15 m/s bin has ZERO matched cells between
  04 and 05 (low-speed weave remains fully open, as pre-registered).
- The 15–22 m/s bin has n=8 cells (weak power); the statistical weight is the 22–30 bin (n=35).
- The visit-permutation null is a negative-control device, NOT a significance calibration
  for real A/Bs (it omits drive-level common-mode variance; conservative for this purpose).

## Step 2 — simulator on stock (bundle SP002)

**Model identification (the real work — the handoff's "get the SHA" hid three findings):**
1. The device commit `3f8e959` is a **squashed dev release**; its message pins **master
   commit `31dc4d8e520f...`**, which ships the source ONNX as git-LFS. The release itself
   ships only precompiled device pkl chunks (no ONNX).
2. **Layout changed after OPM7:** v2026.002.000 ships `driving_vision.onnx` +
   `driving_on_policy.onnx` (+ big_ variants) — **no `driving_off_policy`**. `bundles.py`
   got a per-bundle `models` override; `infer.py` needed nothing (its split path already
   runs vision → on_policy and skips off_policy by design).
3. **Which model actually drove:** stock drives 04/05's rlog `initData` has **no
   `ModelManager_ActiveBundle`** → the default bundled model ran. Its ONNX LFS oids differ
   from Nevada AND OPM7 → genuinely new model. Registered as **`SP002`**
   (`sunnypilot/sunnypilot@31dc4d8e`, split, vision+on_policy) — commit `3153ac4927`.

**Inputs staged:**
- `route_stock05` (Hiram return drive, 12 segs): local dir with NAS rlog symlinks +
  fcamera/ecamera pulled from device (792 MB; device still held all cameras).
- Eligibility npz via the rich extractor — **no code edit needed** (`REQUIRED_CACHE_CHANNELS`
  only gates cache freshness; cx1/cp channels write empty on stock). 13,142 samples, all
  channels finite.
- Bundle materialized: both ONNX LFS-fetched, sha256==oid verified, metadata generated.
- One code fix: `context.py:_boot_rlog` assumed the fork-era `route_<2-hex>` dir naming;
  now generic (matches `alignment._seg_dirs`).

**Fidelity anchor (gate: corr ≥ 0.95 AND band_ratio ∈ [0.85, 1.15]):**
- Span: 200 warmup + 2,059 compare frames (~103 s contiguous eligible highway, segs 4–6).
- **RESULT: PASS — corr 0.9986, band_ratio 0.993** (band-RMS replayed 1.358e-4 vs logged
  1.367e-4), 2,059/2,059 frames valid, lag-offset profile peaked exactly at 0 (no residual
  shift). The best anchor in the program (CD210 0.993/1.008; Nevada 0.990/1.060).
- Corollaries: (a) SP002 is empirically confirmed as the exact build that drove the stock
  drives (a near-miss build shows the OPM7 corr≈0.92 signature); (b) `img_buffer_length=5`
  / is_20hz correct for SP002; (c) the vision→on_policy split path is sound for the
  post-OPM7 layout. `anchor_validated: True` set in `config.py`.
- Anchor series saved: `retrospective_lateral/results/model_replay/sp002_anchor_series.npz`.
- Fixes required en route (committed `2bbd773888`): `context.py` fork-era route-dir naming
  assumption; `context.py`/`parse.py` unguarded `.which()` on new-format rlogs — parse.py's
  old whole-loop try would have silently truncated the anchor ground truth at the first
  corrupt event.

## Where this leaves the roadmap
- Step 1 toolkit: ready for daily use (`stock_lateral_toolkit/README.md`).
- Step 2 simulator: SP002 registered; anchor verdict below determines `anchor_validated`.
- Step 3 (full library re-validation on a stock corpus) not started — next session.
- Open threads — UPDATED per user direction (2026-07-03, post-session):
  * **Centering: fresh approach, NOT a port** of the fork package (user: don't assume the
    first solution was optimal). Measure the stock deficit first — the logged-|offset|
    measurement is model-frame-confounded (comma model's +0.15 m-left lane-center
    definition), and stock-native levers like `CameraOffset` / the known −3.05° camera-yaw
    mount offset are candidates before any control-loop work.
  * **F3 AOL safeguard: deprioritized to backlog.** User reports zero AOL-like behavior on
    stock across all drives incl. lane changes; the incident's one confirmed amplifier (the
    fork's golden-PI phase lag) doesn't exist on stock, and the open delivery tail hasn't
    reproduced. Remains a real upstream missing-safety-net (candidate upstream contribution),
    not an implicated defect.
  * Low-speed weave characterization (needs a dedicated low-speed matched drive) — unchanged.
