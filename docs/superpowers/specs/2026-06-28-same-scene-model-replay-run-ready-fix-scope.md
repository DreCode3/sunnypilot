# Run-Ready Fix Scope — Same-Scene Model-Replay Simulator

Date: 2026-06-28
Purpose: scope the fixes required to take the **hardening plan** (`…-simulator-hardening.md`) + the **original plan** (`…-simulator.md`) to a **run-ready** state, per the independent QA verdict (`docs/superpowers/reports/2026-06-28-same-scene-model-replay-hardening-plan-qa-verdict.md`). This is a *scope* (problem → grounded fix → verification), not the implementation.

Status of inputs: hardening plan is **Sound** in Areas 1/2/5; **Needs fixes** in Areas 3/4/6/7/8. All three former "couldn't-verify" points are now grounded (verdict §4). The architecture is sound; these are correctness/integration fixes, not a redesign.

Definition of **run-ready**: with the in-repo stock model and `route_b5` (CD210, local frames), the pipeline can (a) compile a bundle on Mac CPU, (b) align an eligible disengaged window to the correct HEVC frames, (c) run vision→policy with faithful inputs, (d) produce a weave metric, and (e) pass the strict same-model fidelity anchor for CD210 — with Nevada gated on a frame pull and OPM7 as sanity-only. No vehicle-control/`opendbc_repo`/`panda` changes; outputs under `retrospective_lateral/results/model_replay/`.

---

## FIX 1 — Frame alignment (the linchpin) — **rewrite on `roadEncodeIdx`**

**Problem (verified).** The plan indexes HEVC by the sequential count of `roadCameraState` messages → a constant **+1 off-by-one** (HEVC frame 0 = `frameId 1202`, but `roadCameraState` starts at 1203), it can't handle windows that cross a segment boundary, it over-counts truncated HEVC, and it rebuilds the full catalog on every `map_window_to_frames` call (~20 s).

**Grounded mechanics (from real rlogs).** `EncodeIndex` (`cereal/log.capnp:1086`) per road frame gives: `frameId` (global, monotonic across segments), `segmentNum` (which 60 s segment / which per-segment HEVC file), **`segmentId` = the HEVC frame index within the segment** (0..N-1, presentation order — `route_8d` seg1: `segmentId 0 ↔ frameId 1202 ↔ HEVC frame 0`), and `timestampEof` (frame mono time, same clock as NPZ `mono_time = logMonoTime*1e-9`). `roadEncodeIdx` → `fcamera.hevc`; `wideRoadEncodeIdx` → `ecamera.hevc`.

**Fix.**
1. **Frame timeline from `roadEncodeIdx`/`wideRoadEncodeIdx`**, not `roadCameraState` count. For each segment, emit rows `(segment_num, segment_id, timestamp_eof_s, frame_id)`; filter to the full-HEVC `type`; **guard `segment_id < FrameReader(seg_hevc).frame_count`** (truncation) — drop frames the HEVC doesn't contain.
2. **`map_window_to_frames(route, mono_times, max_delta_s=0.03)`**: for each requested `mono_time`, pick the timeline row whose `timestamp_eof_s` is nearest within `max_delta_s` (else the window is unmappable → raise/skip). Return `(segment_num, segment_id)` per frame; **read via `FrameReader(<…--segment_num>/fcamera.hevc).get(segment_id)`**.
3. **Cross-segment continuity** on `timestamp_eof_s` gaps (≈0.05 s at 20 Hz), **NOT** on a global index or per-segment restriction. At a boundary `segment_id` resets 1199→0 while time stays continuous (164.56→164.61) → a `≤ ~0.06 s` gap check accepts boundary-crossing windows (the very thing that broke the `route_b5` anchor).
4. **Performance:** build + cache the per-route timeline once (memoize, or persist `frame_timeline.parquet`); `map_window_to_frames` reads the cache, never rebuilds.
5. **Off-by-one regression test:** assert `segmentId==0` maps to `frameId==1202` on `route_8d` seg1 (and that the old `roadCameraState`-count would give 1203).

**Files:** `model_replay_sim/alignment.py` (+ its tests). **Verify:** on `route_8d`, `map_window_to_frames` for a known window returns the expected `(segment_num, segment_id)`; a boundary-crossing window (seg17→18 of `route_b5`) maps cleanly; a truncated segment yields ≤ `frame_count` frames; one full catalog build ≪ prior ~20 s/call.

---

## FIX 2 — OPM7 `LAT_SMOOTH_SECONDS` (fidelity) — **source from the pinned commit, never a literal**

**Problem (hard-confirmed).** The plan hardcodes OPM7 `lat_smooth = 0.1`, but `git show 052692b2…:selfdrive/modeld/modeld.py:45` → `LAT_SMOOTH_SECONDS = 0.0`. This inflates OPM7's `lat_delay = liveDelay.lateralDelay + LAT_SMOOTH` by ~0.1 s (~29%).

**Fix.** Read each candidate bundle's `LAT_SMOOTH_SECONDS` (and any other model-side lateral constant) **from that bundle's pinned commit** (`git show <ref>:selfdrive/modeld/modeld.py`), recorded into the bundle context/provenance — not a literal. Set OPM7 → 0.0 (and verify CD210/Nevada values from their own commits rather than assuming). Fix the test that encodes `lagd_value + 0.1`.

**Files:** `model_replay_sim/context.py` / `config.py` (bundle constants) + tests. **Verify:** `candidate_overrides("OPM7")["lat"] == 0.0` sourced from `052692b2…`; CD210/Nevada values match their commits; no lateral constant is a literal in the cross-model path.

---

## FIX 3 — Reconcile the two plans (integration) — **one config shape, one module layout**

**Problem (verified).** Hardening and original plans define **incompatible** `model_replay_sim/config.py` (original `BUNDLES` use short ref `55f66e22` + `WEAVE_BAND_HZ`/`ANCHOR_CORR_MIN`/… ; hardening uses full-SHA bundle dicts + `env`/tolerance consts) and **disjoint module layouts** (original: `params_stub/frames/calib/warp/metrics/parse/anchor/infer/run`; hardening: `env/pytest_helpers/alignment/context/assets`). Task 8 only narratively back-ports snippets.

**Fix.** Adopt **one** authoritative module set = hardening's Phase-0 modules (`env`, `compile_bundle`, `alignment`, `context`, `bundles`, `assets`, `pytest_helpers`) **plus** the original's replay/metric modules (`infer`, `parse`, `metrics`, `anchor`, `run`); delete/merge the original's now-superseded `frames.py`/`calib.py`/`params_stub.py` into `alignment`/`context`/`env`. Adopt **one** `config.py`: full-SHA `BUNDLES` (hardening) + the metric/tolerance constants (`WEAVE_BAND_HZ`, `ANCHOR_CORR_MIN`, `ANCHOR_BAND_RATIO`, `GENTLE_CURV_ABS_MAX_1PM`, …) the original tests assert — and update `test_config.py` to the merged shape. Rewrite the original plan's Task 0/4/6/9a/9b/10/11 to import these modules (replace `frames.py`/`calib.py` usage with `alignment`/`context`; replace the GO/NO-GO compile gate with the custom diagnostic; replace `SEG=None`/`skipif` with the `assets.require_*` gates). This supersedes the original plan's File Structure §.

**Files:** the two plan docs + `config.py`/`test_config.py`. **Verify:** a single `model_replay_sim/` file list with no duplicate-purpose modules; `test_config.py` passes against the merged config; no task references a deleted module.

---

## FIX 4 — Wire the CD210 strict anchor to its real window (depends on FIX 1)

**Problem.** `route_b5`'s frames are in segments 16–18 only; the plan's window selection (pre-FIX-1) found 0 aligned windows. The data exists: **68.6 s + 45.9 s eligible windows in the 180 s frame-present range** (lead-reviewer verified).

**Fix.** After FIX 1, have `assets._eligible_aligned_window_count` select eligible windows (corrected disengaged/straight-gentle/speed gate, no blinker/lane-change — note the anchor uses the *replay-scene* gate, not the override mask) that fall in `route_b5`'s frame-present mono range and align via the fixed `map_window_to_frames` (boundary-crossing allowed). Keep `require_same_model_anchor_asset("Nevada")` failing on `needs_frame_pull` (Nevada has no local frames) and OPM7 as `require_sanity_asset` only (no active-bundle provenance). Add an explicit, optional read-only **Nevada frame-pull step** (tar-over-ssh, one route's `fcamera.hevc`/`ecamera.hevc`) so Nevada's strict anchor can be enabled when desired — not required for CD210 run-ready.

**Files:** `model_replay_sim/assets.py` + tests. **Verify:** `anchor_candidates("CD210")` returns `route_b5` with `eligible_aligned_windows_20s ≥ 1`; `require_same_model_anchor_asset("CD210")` passes; Nevada/OPM7 gates still fail loudly as designed.

---

## FIX 5 — Test idioms & ordering

**Problems (verified):** `require_real_asset` unit-mode lets the gate return but the body still runs → errors (not skip); Task 7's `rg "Expected: no matches"` runs *before* Task 8's rewrite (5 matches incl. `NEVADA_SEG=None`); `_frame_count`'s `from openpilot…` import is CWD-fragile (no `sys.path` insert); Task 7 writes no real anchor tests (core replay tests are still `NotImplementedError`/`SEG=None`).

**Fixes.** (a) Make `require_real_asset` in unit-mode actually `pytest.skip(...)` (or return a sentinel the caller checks) so the body doesn't execute on missing assets. (b) Move the consistency-grep to **after** the FIX-3 integration rewrite (run it in the integration task, not before). (c) `_frame_count` (and every module that imports `openpilot.*`) must `sys.path.insert(0, str(C.REPO_ROOT))` at import, not rely on CWD. (d) The replay anchor tests stop being placeholders once the original plan's Tasks 9a/9b/11 are implemented post-FIX-3; the run-ready gate is the CD210 fidelity anchor (FIX 4), not a green unit suite with placeholders.

**Files:** `pytest_helpers.py`, `alignment.py`, the plan task ordering. **Verify:** unit-mode skips (yellow) rather than errors (red); the consistency grep is clean in its (post-rewrite) task; tests pass from a non-repo-root CWD.

---

## FIX 6 — Minor

- **`lfs_oid_for_path`:** `git lfs ls-files --long --include=<path>` (not `-- <path>`, which exits 128); add a test. Non-blocking (provenance enrichment only).
- **Compile-once:** run `compile3.py` a single time with `SELFTEST=1` to get both the pkl and the parity facts (don't compile twice).
- **`DEBUG=release` wording:** reword the premise as environment-defensive (this repo's shell has `DEBUG` unset); keep forcing `DEBUG=0`.

---

## Sequencing for run-ready

1. **FIX 1 (frame alignment)** — unblocks everything; do first.
2. **FIX 2 (OPM7 const)** + **FIX 6 (minor)** — small, independent.
3. **FIX 3 (reconcile plans)** — establishes one module/config set the rest build on.
4. **FIX 4 (CD210 anchor)** — depends on FIX 1 + FIX 3.
5. **FIX 5 (test idioms)** — folds into the FIX-3 integration task.
6. Then implement the original plan's replay core (Tasks 9a/9b/11) on the reconciled modules and run the **CD210 same-model fidelity anchor** = the run-ready gate.

**Out of scope (not required for CD210 run-ready):** the Nevada frame pull (optional, enables Nevada's strict anchor); the full vision-model SELFTEST timing; OPM7 strict fidelity (sanity-only by design). **What stays from the hardening plan unchanged:** Areas 1/2/5 (env, compile diagnostic, provenance) — sound as written.
