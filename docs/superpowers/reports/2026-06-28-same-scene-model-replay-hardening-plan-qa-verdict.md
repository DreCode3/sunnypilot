# Independent QA Verdict — Same-Scene Model-Replay *Hardening* Plan (Round 2)

Date: 2026-06-28
Reviewer role: independent third-party QA (the hardening plan was written by Codex to patch the original simulator plan; this is a fresh review of Codex's work).
Artifacts reviewed:
- `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator-hardening.md` (under review)
- `docs/superpowers/plans/2026-06-28-same-scene-model-replay-simulator.md` (the plan it patches)
- `docs/superpowers/specs/2026-06-28-same-scene-model-replay-simulator-scope.md` (the spec)

Method: 8 parallel area-agents each independently re-derived every load-bearing claim against the real repo/code/data; the lead reviewer re-verified the decisive ones by hand and **ran all three open "couldn't-verify" points fully to ground** (now closed — §4). Read-only QA: no vehicle-control code, `opendbc_repo/`, `panda/`, or `results/` modified; no device run; the only compute performed was one local CPU compile to `/tmp`.

---

## 1. Bottom line

**Do NOT execute as written — but it is a strong, mostly-correct hardening pass that needs ~5 targeted fixes, none fatal.** Codex's environment, compile-diagnostic, and bundle-provenance work is solid and its diagnoses are largely right. The blockers are concentrated in **frame alignment**, one **OPM7 fidelity constant**, and the **integration with the original plan**. After those, a run-ready simulation is achievable (the anchor *data* exists; see §3).

| Area | Verdict |
| --- | --- |
| 1 — Environment & dependencies | **Sound** |
| 2 — Compile path (SELFTEST replacement) | **Sound** |
| 3 — Frame/log alignment | **Needs fixes (the biggest flaw)** |
| 4 — Route context & replay inputs | **Needs fixes (OPM7 constant — now hard-confirmed)** |
| 5 — Bundle provenance & ONNX | **Sound** |
| 6 — Anchor asset selection | **Needs fixes (agent overstated; corrected below)** |
| 7 — Tests & TDD quality | **Needs fixes** |
| 8 — Integration into original plan | **Needs fixes** |

---

## 2. Findings by area (CONFIRMED / BROKEN / OVERSTATED)

**Area 1 — Environment & deps — SOUND.** All three pins exist with macOS-arm64/py3.11 wheels and install (`onnxruntime==1.27.0` [latest, installed], `onnx==1.22.0`, `pyarrow==24.0.0`); `git lfs` present (3.7.1); tinygrad submodule initialized; `replay_env`/`require_real_asset` helpers and test snippets run. *OVERSTATED:* the "this shell has `DEBUG=release`" premise is Codex-environment-specific (this repo's shell has `DEBUG` unset) — but the `DEBUG=0` forcing is a correct, harmless mitigation (a literal `DEBUG=release` does break `vidindex.py:9` and tinygrad `helpers.py:142` with a `ValueError`). Keep the action; reword the rationale.

**Area 2 — Compile path — SOUND.** `compile3.py:140-141` runs `test_vs_onnx(..., 1e-4)` → a single `np.testing.assert_allclose(atol=1e-4, rtol=1e-4)` over the whole output tensor (line 123): an all-or-nothing gate. Replacing it with separate compile-success + ONNXRuntime-parity facts is correct (empirically confirmed in §4 #3). *Minor:* the wrapper compiles twice (once to produce the pkl, once with `SELFTEST=1`) — a single `SELFTEST=1` run yields both; and a latent `>= 0.0` assertion on a parsed max-abs is fragile.

**Area 3 — Frame/log alignment — NEEDS FIXES (the weakest area).**
- **BROKEN — off-by-one frame mapping (hand-confirmed).** The plan indexes HEVC by the sequential count of `roadCameraState` messages, but the authoritative `roadEncodeIdx` says HEVC frame 0 ↔ `frameId 1202`, while the `roadCameraState` stream starts at `frameId 1203` (1202 has an encode index but no `roadCameraState`). Offset is a consistent **+1** → each model is fed the *next* frame. Common-mode, so it partly cancels in a relative model-vs-model comparison, but it is unfaithful and ignores the authoritative index.
- **BROKEN — the NPZ cache cannot supply alignment.** `extract.py` resamples to a synthetic uniform 20 Hz grid and stores only `t` + `mono_time` (no `frameId`/`encodeId`); alignment must come from re-reading the rlogs' `roadEncodeIdx`/`roadCameraState`.
- **OVERSTATED — `alignable_frame_count`** counts by frameId membership and ignores truncated HEVC (e.g. a segment with 1200 road messages but 720 decoded frames reports ~1199 alignable, of which only 720 are mappable).
- **OVERSTATED — performance.** `map_window_to_frames` rebuilds the full segment catalog + frame timeline on *every* call (~0.16 s/segment → ~20 s per call), which is what timed out the Area-6/7 anchor probes.

**Area 4 — Route context — NEEDS FIXES.** All cereal messages and param keys are real, not invented (`liveCalibration.rpyCalib/height`, `liveDelay.lateralDelay` @ `log.capnp:2171`, `CameraOffset`, `PlanplusControl`, `LagdValueCache`, `LagdToggle`, `ModelManager_ActiveBundle`); the `LagdToggle→LagdValueCache else liveDelay` delay-base logic mirrors production. **BROKEN — OPM7 `lat_smooth` hardcoded to 0.1, but the pinned OPM7 commit `052692b2…` has `LAT_SMOOTH_SECONDS = 0.0`** (hand-confirmed, §4 #2). This inflates OPM7's policy delay input by ~0.1 s (~29% over its ~0.349 s base) — a model-specific fidelity error in the model the comparison centers on, and it contradicts both the pinned commit and the spec's "never hardcoded" principle. The test encodes the wrong value.

**Area 5 — Bundle provenance — SOUND.** SHAs are valid and resolve locally; ONNX present at each commit as git-LFS pointers with distinct oids (§4 #1); ONNX written under `results/` (constraint honored); required provenance fields written. *Minor BROKEN:* `lfs_oid_for_path` uses `git lfs ls-files --long -- <path>` which fails (git-lfs treats the post-`--` token as a revision, exit 128) → the `lfs_oid` field is always `None`; non-blocking (graceful) and untested; correct form is `--include=<path>`.

**Area 6 — Anchor assets — NEEDS FIXES (agent overstated; corrected).** The Nevada-needs-frame-pull and OPM7-no-provenance gates are correctly modeled and fail loudly (credit). The Area-6 agent claimed *no model has a usable strict anchor*; **the lead reviewer refuted this**: `route_b5` (CD210) has **68.6 s + 45.9 s contiguous eligible windows (142 s total) within its 180 s frame-present range** (segments 16–18). So the strict CD210 anchor *data exists* — what fails is the plan's window-selection/alignment logic (Area 3), which rejects boundary-crossing windows and mis-maps frames. **Fixable, not fatal.**

**Area 7 — Tests & TDD — NEEDS FIXES.** *BROKEN:* `require_real_asset` unit-mode (`MODEL_REPLAY_ALLOW_MISSING_ASSETS=1`) only makes the gate *return* — the test body still runs and *errors* on the missing asset (it is **not** a hidden-green loophole, as the agent corrected, but it is broken: red, not skip). *BROKEN:* Task 7's `rg … ; Expected: no matches` consistency check fails *in plan order* (it runs before Task 8 does the rewrite — 5 matches incl. the core `NEVADA_SEG = None`). *BROKEN:* `_frame_count` does `from openpilot…` without a `sys.path` insert (CWD-fragile). *BROKEN:* Task 7 only audits — it writes no replacement test code, and the real cross-model replay tests remain `NotImplementedError`/`SEG=None` placeholders in the original plan (unimplemented by either plan).

**Area 8 — Integration — NEEDS FIXES.** *BROKEN:* the hardening plan and the original plan define **incompatible versions of the same files** — `model_replay_sim/config.py` with a different `BUNDLES` shape (short ref `55f66e22` + `WEAVE_BAND_HZ`/`ANCHOR_CORR_MIN`/etc. in the original vs full-SHA bundle dicts in the hardening one) and **disjoint module layouts** (original: `params_stub/frames/calib/warp/metrics/parse/anchor/infer/run`; hardening: `env/pytest_helpers/alignment/context/assets`). Task 8 narratively back-ports a few snippets and does not reconcile these — an implementer following both hits contradictions immediately. *OVERSTATED:* Task 8 flags the original plan's deliberate `NotImplementedError` TDD seams as if they were stale (they are intentional, filled by the original Tasks 9a/9b/11).

---

## 3. Single biggest flaw

**The frame-alignment logic (Area 3).** It is the foundation every downstream stage depends on, and it is currently both *incorrect* (off-by-one feeds the wrong frame and ignores the authoritative `roadEncodeIdx`) and *unable to extract anchor windows that demonstrably exist* (`route_b5`'s 68.6 s window). Fix this and the CD210 strict anchor becomes runnable.

---

## 4. Open verification points — now fully grounded (no remaining gaps)

All three former "couldn't-verify" caveats were run to ground with local resources only (local git object store + one CPU compile; no network, no device). They **reinforce** the findings above.

**#1 — Upstream SHAs — CONFIRMED.** `git cat-file -t` resolves all three (`commit`): CD210 `55f66e22…`, Nevada `3193eac5…`, OPM7 `052692b2…`. ONNX present at each as git-LFS pointers with **distinct oids** (CD210 vision `ee29ee5b…`, Nevada vision `befac016…` → genuinely different models); OPM7's tree shows `driving_vision` + `driving_on_policy` + `driving_off_policy`.onnx (the 3-model split). Checkout + `lfs pull` materialization is valid. *(`git cat-file`/`git show`; seconds; commits already local.)*

**#2 — OPM7 `LAT_SMOOTH_SECONDS` — CONFIRMED BUG.** `git show 052692b2…:selfdrive/modeld/modeld.py` → `LAT_SMOOTH_SECONDS = 0.0` (line 45; used at line 363: `lat_delay = sm["liveDelay"].lateralDelay + LAT_SMOOTH_SECONDS`). The plan's hardcoded **0.1** for OPM7 is wrong. *(One `git show`; seconds.)* → **stays a required fix.**

**#3 — SELFTEST is a bad GO/NO-GO gate — CONFIRMED EMPIRICALLY.** A local CPU compile of `driving_policy.onnx` via `compile3.py` **succeeded** (14.44 MB pkl, 35 kernels JIT-captured, deterministic) but `SELFTEST=1` **FAILED**: `AssertionError: Not equal to tolerance rtol=0.0001, atol=0.0001` — e.g. tinygrad-CPU `1.3540e+03` vs onnxruntime `1.354298e+03` (the model also expects fp16 inputs, fed fp32). So compile + determinism succeed while the 1e-4 all-or-nothing gate fails on backend numeric drift. *(One `DEV=CPU SELFTEST=1 compile3.py` run, ~1–2 min CPU; vision would add a few min but policy already proves it.)* → **Codex's SELFTEST replacement is justified; keep it.** Incidentally confirmed Codex's "path-safe" wrapper is necessary: `fetch()` only accepts a path starting with `/` or `.` (`helpers.py:391`), so a bare relative path fails.

---

## 5. Required changes before execution (prioritized)

1. **[Area 3 — frame alignment] Rewrite using the authoritative `roadEncodeIdx`** (fixes off-by-one); map by `(segment, segmentId)` not a global sequential count; handle continuity *across* contiguous frame-present segments (don't reject boundary-crossing windows when frames are continuous in adjacent per-segment HEVC files); bound `alignable_frame_count` by the actual `FrameReader.frame_count`; and cache the catalog (don't rebuild per call). *Scoped separately.*
2. **[Area 4] Source OPM7's `LAT_SMOOTH_SECONDS` (and all such constants) from the pinned commit (0.0), never a literal.**
3. **[Area 8] Reconcile the hardening and original plans** into one `config.py` shape (single ref format) and one module layout; make the integration step concrete (or restructure the original Phase 0 to import the hardening modules).
4. **[Area 6] After #1, wire the CD210/`route_b5` strict anchor to its real 68.6 s window;** keep Nevada gated on a frame pull, OPM7 sanity-only.
5. **[Area 7] Fix `require_real_asset` unit-mode, Task 7 ordering (run its check after Task 8's rewrite), `_frame_count`'s import,** and replace placeholder anchor tests with real ones.
6. **[Minor] `lfs_oid` syntax (`--include=`), compile-once, reword the `DEBUG=release` premise.**

The architecture is sound and the data supports it; these are correctness/integration fixes, not a redesign.

---

## 6. Credit (what Codex got right)
Dependency pins (all real + arm64 wheels), the `DEBUG/DEV/IMAGE/THREADS` env sanitization, the path-safe compile wrapper, the SELFTEST→custom-diagnostic diagnosis (empirically vindicated), the bundle SHAs/refs/ONNX-under-`results` provenance, the 3-model OPM7 handling, the non-skipping asset gates that correctly model Nevada-needs-pull and OPM7-no-provenance, and the route-context extraction (every param/message real). This is genuinely strong, deep work; the fixes above are scoped, not foundational.
