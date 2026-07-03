# HANDOFF — Stock sunnypilot fresh start + standing up the stock lateral toolkit

**Read this first.** This is the authoritative starting point for the next session. It hands off
a clean pivot: the 2021 Ford Explorer ST program is **abandoning the custom fork and rebuilding on
stock sunnypilot dev v2026.002.000**, and the immediate job is to **stand up a stock-compatible
lateral-analysis toolkit** so we can quantify and tune from a clean baseline.

Git branch for this effort: **`stock-2026.002.001-fresh-start`** (you should already be on it).

---

## 1. Why we're here (the decision + evidence)

The long-running lateral **weave** + a 2026-07-01 **AOL silent-lateral safety incident** drove a
deep investigation. Conclusion: the custom fork carries real tech debt, its golden PI is a
weave amplifier, and stock is a calmer base. The user flashed stock, drove it, and we did a
matched objective comparison. **Decision: start fresh on stock.**

**Objective stock-vs-custom result** (report:
`docs/superpowers/reports/2026-07-02-stock-vs-custom-lateral-comparison.md`; 5-agent verified,
Powder Springs↔Hiram matched corridor):
- ✅ **Stock weaves ~2× less at highway speed** (SOLID; user confirmed "felt great").
- ⚠️ Low-speed weave: not established better on stock (confounded).
- ❌ **Stock centers WORSE** (user-confirmed subjectively — this is the one real regression;
  custom's hard-won Ford centering = the PI lane-offset loop + EPAS-bias feedforward + steerRatio
  work is a genuine asset stock lacks).
- 🟡 Stock tracks curves a bit more faithfully; custom understeers ~6-9%.
- ✅ EPS delivery is build-independent (weave lives in the commanded path, not actuation).

**The nuance that shapes the roadmap:** the centering-vs-weave tension is partly the *same* lever
(the offset PI). The **static** centering (steerRatio + EPAS constant-bias feedforward + a gentle
P) centers *without* weave; the **dynamic integral** is the weave-generator. So the theoretical
best-of-both = **stock's calmer base + port ONLY the static centering, leave the laggy integral
behind.** That is the eventual tuning target — but FIRST we need the toolkit to measure it.

---

## 2. Current state (what's true right now)

- **Device:** flashed to **stock sunnypilot dev v2026.002.000** (`github.com/sunnypilot/openpilot@3f8e959`,
  steerRatio 16.8). SSH: after the reflash the host key changed (removed the stale local
  known_hosts entry) and SSH auth is re-enabled via the GitHub username in the device UI
  (`DreCode3`). `ssh comma` works (see `reference_device_config.md` for the alias). Passphrase
  for the id_ed25519 key if prompted: `40@5quss`.
- **Fork code:** still on branches `2021_explorer_st-mici` (pre-migration) and
  `clean-v2026.002.001` (the migration). The F3/F2 AOL fixes are committed on worktree branches
  `aol-fixes-f3` (main repo, worktree `/Users/dregilley/Documents/GitHub/sp-aol-f3`) and
  `pi-delag-f2` (opendbc, worktree `/Users/dregilley/Documents/GitHub/opendbc-f2`) — **on hold**,
  but F3's AOL blindness/departure safeguard design ports to any base (see §5).
- **This branch (`stock-2026.002.001-fresh-start`):** holds the fresh-start docs + the portable
  `stock_compare` tooling. It currently still contains the fork's code tree — the actual
  code-level rebase onto a clean stock checkout is YOUR first decision (see §4, step 0).
- **Logs/videos:** the ~130 GB of local logs+videos were **migrated to the NAS** and deleted
  locally to reclaim space. See memory `reference_nas_storage.md` for the path + credentials.
  The device still holds recent drives; re-pull as needed.
- **venv:** `.venv311` (py3.11). Run analyses: `PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python …`.

---

## 3. The toolkit inventory (what to reuse / port / abandon)

Full detail: `docs/superpowers/reports/2026-07-02-fresh-start-tooling-logging-inventory.md`.
Headline: **~70% of the analysis library is portable; coupling is shallow.** steerRatio is NOT
hardcoded (code reads carParams). Logging has **zero cereal-schema debt** (LC/CP/CX1 are all
`logMessage` strings, gone on stock).

**PORTABLE CROWN JEWELS (seed the toolkit from these):**
- `retrospective_lateral/stock_compare/extract_drive.py` + `analyze_compare.py` — matched
  GPS-cell+speed-bin weave/centering/curve A/B, **already build-agnostic and PROVEN on stock this
  session.** This pair is the working template.
- `retrospective_lateral/code/signal_utils.py` — band-limited weave DSP (0.10-0.35 Hz), zero coupling.
- `learned_param_studies/code/shared.py` — weave band-RMS metric + the calibrated stats battery
  (exact/MC permutation, Freedman-Lane partial/within-drive spearman, BH-FDR). **The discipline
  that reversed 3+ findings — do not lose it.**
- `learned_param_studies/code/qa_calibration.py` + `longitudinal_weave/calib_candidates.py` —
  FPR/power calibration harnesses ("prove the estimator before trusting significance").
- `model_replay_sim/` engine (arch-general, targets the vision→policy split stock uses):
  `alignment.py`, `warp.py`, `env.py`, `bundles.py`, `compile_bundle.py`, `metrics.py`,
  `parse.py`, `infer.py`, `context.py`, and `anchor.py`'s general `run_anchor`.

**COUPLED-BUT-WORTH-PORTING (small edits):**
- `model_replay_sim/config.py` → add ONE `BUNDLES` entry for the stock model (commit SHA + repo +
  split). *Single biggest simulator unlock.*
- `model_replay_sim/anchor.py` → generalize `run_cd210_anchor` → `run_anchor(stock_bundle, route)`.
- `retrospective_lateral/code/extract.py` → drop `cp_*`/`cx1_*` from `REQUIRED_CACHE_CHANNELS`.
- `retrospective_lateral/code/model_labels.py` → drop the Nevada/CD210/OPM7 name map (keep hash recovery).
- `retrospective_lateral/stock_compare/analyze_compare.py` → add fresh corridors to `CORRIDORS`.

**OBSOLETE (leave behind):** `retrospective_lateral/code/telemetry.py` (LC/CP/CX1 parsing),
`model_era_weave.py`, `model_replay_sim/compare.py`, all `incident_2026_07_01/scripts/` (harvest
idioms first), `cc_toggle_patch.py`, `test2_integrator.py`, `longitudinal_weave/{analyze,final_trend}.py`.

---

## 4. Stand-up plan (~1.5-2 days, reuses ~70% of the library)

**Step 0 — decide the code base for this branch.** This branch still has the fork tree. Options:
(a) keep it as-is and build the toolkit in a new `stock_lateral_toolkit/` subdir (lowest risk,
recommended to start); (b) reset the branch to a clean stock checkout and re-add only the toolkit
(cleaner long-term, more work). Recommend (a) now, (b) later once the toolkit is stable.

**Step 1 — immediate (< half a day): a working stock A/B.**
- Create `stock_lateral_toolkit/` and copy in: `stock_compare/{extract_drive.py,analyze_compare.py}`,
  `signal_utils.py`, `shared.py`, `qa_calibration.py`, `calib_candidates.py`.
- De-couple edits: none needed for `stock_compare/*` (already generic — just point `CORRIDORS` at
  new caches). Drop pi_set/lc_int columns from `shared.py:segment_table`.
- Validate: pull a couple of stock drives from the device (rlog.zst only), run `extract_drive.py`
  then `analyze_compare.py` on a self-vs-self split (sanity: metrics stable). You now have
  band-limited weave, centering, curve-tracking, matched A/B on stock logs.

**Step 2 — simulator on stock (~0.5-1 day).**
- Add the stock model as a `BUNDLES` entry in `model_replay_sim/config.py` (get the stock model's
  commit SHA + whether it's a vision/policy split; `anchor_validated: False` initially).
- Generalize `run_anchor` (anchor.py) to take (bundle, route). Run the fidelity anchor on a route
  the stock model actually drove (target corr ~0.99, band-ratio ~1.0 like the old CD210/Nevada
  anchors). This validates the sim reproduces the device's model output on stock.

**Step 3 — full library re-validation (~1 day).**
- Re-point `code/extract.py` (drop cp_/cx1_ required channels), re-run detectors/discrimination on
  a stock corpus, confirm the `modeld_v2` parser path matches the fresh base.

**Gate discipline (do not skip — this is why findings held):** every new metric/estimator goes
through the FPR/power calibration harness (`qa_calibration.py`) before you trust a "significant"
result. Always speed-match + GPS-match + robust stats + band-limit 0.10-0.35 Hz. Never aLat as
primary, never pooled variance. (Memory: `feedback_lateral_ab_metrics`.)

---

## 5. Open threads (after the toolkit exists)

1. **The centering regression** (stock's one real weakness). Path: port ONLY the static centering
   (steerRatio tune + EPAS constant-bias feedforward + gentle P) onto stock, NOT the weave-y
   integral. Validate with the toolkit (does centering improve without re-introducing weave?).
   If re-adding the offset PI, re-emit a single trimmed 1 Hz `carlog.info` internal-state line +
   register the `LaneBiasIntegral` param — no cereal schema surgery (logging inventory §logging).
2. **F3 AOL safeguard** — the blindness/departure safeguard designed during the incident work
   (control-residual departure detector: alert when drifting off-center AND the wheel isn't
   arresting it; blindness alert on laneProb collapse). Stock ALSO lacks this (it's an upstream
   gap). Worth porting to the stock base and potentially upstreaming. Design in
   `docs/superpowers/reports/2026-07-01-lateral-safety-investigation-handoff-v3.md` §5 +
   `docs/superpowers/plans/2026-07-02-aol-incident-fixes-f3-f2-ab.md`.
3. **Low-speed weave** — the historically-worst case, and the one the stock comparison could NOT
   prove better. Needs a dedicated low-speed matched drive on stock to characterize.

---

## 6. Key references (all current)

- **This handoff** — start here.
- `docs/superpowers/reports/2026-07-02-stock-vs-custom-lateral-comparison.md` — the objective A/B.
- `docs/superpowers/reports/2026-07-02-fresh-start-tooling-logging-inventory.md` — full tooling/logging map.
- `docs/superpowers/reports/2026-07-01-lateral-safety-investigation-handoff-v3.md` — the AOL
  incident RCA (2-round QA'd) + F3/F2 designs (historical-but-portable).
- `docs/superpowers/plans/2026-07-02-aol-incident-fixes-f3-f2-ab.md` — F3/F2/A-B implementation plan.
- Memory index: `MEMORY.md` → `project_fresh_start_stock.md` (the pivot),
  `reference_device_config.md` (device/SSH/venv/tools), `reference_nas_storage.md` (log archive),
  `feedback_lateral_ab_metrics.md` (the analysis discipline), `finding_aol_silent_lateral_dropout.md`.

---

## 7. Discipline carried forward
- Independently verify every agent/analysis output — this program's comparisons flipped 3× under
  poor control; the calibrated stats battery is why the final findings held.
- Matched (GPS+speed) + robust + band-limited, always. Prove estimators before trusting p-values.
- Car stays treated as a test article; stock does NOT fix the AOL silent-failure gap or the
  open wheel-delivery tail — drive it hands-hovering until the safeguard exists.
- Local Mac = M4 Max 16 cores → parallelize compute (ProcessPoolExecutor). Device SSH = single
  brittle connection, sequential, read-only pulls.
