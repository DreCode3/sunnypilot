# Toolkit port acceptance — PRE-REGISTERED criteria (2026-07-03)

Registered BEFORE any validation run (discipline: this program's comparisons flipped 3×
under poor control; criteria fixed in advance so results can't be rationalized after the
fact). The port is ACCEPTED only if ALL of A-C pass. Any failure ⇒ STOP, diagnose, report.

**Data:** stock Hiram drives `00000004--3d3385646d` (26 seg) + `00000005--...` (12 seg),
read from the NAS (`/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock/`), plus the
proven fork-era caches `custom_c5/c7.npz` (copied, not re-extracted — the custom rlogs are
not what's under test).

## A. Extractor regression (port didn't change the numbers)
Re-extract stock_hiram_04 and stock_hiram_05 from NAS rlogs with
`stock_lateral_toolkit/extract_drive.py`. Compare against the proven caches
(`retrospective_lateral/stock_compare/cache/stock_hiram_0{4,5}.npz`):
- **PASS:** same shape, same columns, and max |Δ| < 1e-9 for every column (the code path
  is identical; only path handling changed). steerRatio must read 16.8 (stock), NOT 17.2.
- Row-count mismatch or any |Δ| ≥ 1e-9 ⇒ FAIL.

## B. Positive control (analyzer reproduces the known result)
Run `analyze_compare.py hiram` from the NEW toolkit (new stock caches + copied custom
caches) and from the OLD location (`retrospective_lateral/stock_compare/`, its own caches).
- **PASS:** numerically identical tables (both pipelines are deterministic; if A passed,
  inputs are identical, so any output difference = a port bug). Specifically the known
  highway (22-30 m/s) result must appear: stock calmer on straight-road band|yawRate|,
  stock-better in a clear majority of matched cells (report of record: ~2× / p=0.0007,
  74%/34-cell steer variant, 2026-07-02 comparison).

## C. Negative control (toolkit doesn't invent differences)
`analyze_compare.py hiram_null_04v05` (stock 04 vs stock 05: same build, same corridor,
opposite directions) + `validation/self_split.py` (visit-permutation null, N=200 shuffles,
seed=11, within (cell, speed-bin) — preserves the matching structure exactly).
- **PASS (per speed bin with ≥4 matched cells, primary = band|yawRate| on straights):**
  observed 04-vs-05 delta AND win-rate sit within the shuffle null's [2.5, 97.5] percentile
  band, and win-rate is in [30%, 70%].
- Caveat registered in advance: 04 vs 05 are opposite directions of travel; weave on
  straights is expected direction-agnostic, |offset|/centering may show real crown/direction
  effects — centering is REPORTED but only the weave primary gates acceptance.

## Non-goals of this gate
- No claim about low-speed weave (still an open thread), no new findings — this gate only
  establishes the toolkit measures on stock what it measured before, and measures nothing
  where nothing is.
