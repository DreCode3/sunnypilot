# Scope: Same-Scene Model-Replay Simulator (CD210 / OPM7 / Nevada) — Mac-native

Date: 2026-06-28
Status: **SCOPE — twice-QA'd (collaborative + adversarial), converged.** Ready to convert to a build plan.

> **Revision note (post-ADVERSARIAL-QA, 4 fixes):** (1) **BLOCKING — scene source changed from *engaged* to *disengaged*/human-driven segments.** Replaying off-policy models on *engaged* frames judges them on a scene the driving model steered (scene-generation confound); the project's own open-loop test (`model_vs_loop.py`) used `latActive=0` disengaged frames precisely for neutrality. Affects §1/§7/§9. (2) **Phase −1 down-ranked from existential to a routine fetch** — the target ONNX *is* obtainable via the model-build mechanism (`git checkout <ref> && git lfs pull`): CD210 `55f66e22` + Nevada `3193eac5` in commaai/openpilot, OPM7 `052692b2` in sunnypilot/sunnypilot. The real first risk is **throughput**, not provenance. (3) **"Re-implement (small)" reclassified as a *primary* risk** — the recurrent temporal-state core (`plan+planplus`, `prev_desired_curv` feedback, per-family `temporal_idxs` strides, generation/mlsim gating) is the actual hazard and the one thing that can bias one model's weave numerics. (4) **CPU-throughput honesty** — a powered run is ~300k inferences ⇒ budget *overnight*, mandate the Phase-1 micro-benchmark, don't lean on unvalidated Metal.
>
> **Revision note (post-collaborative-QA):** Five material corrections were folded in: (a) `tinygrad_repo`/`msgq_repo` are **uninitialized submodules** (not vendored-and-ready); (b) **the target bundles ship only QCOM-compiled `_tinygrad.pkl`; no ONNX is published on the device or registry** → added a gating **Phase −1 (ONNX provenance)**, the project's existential risk; (c) the macOS build/runtime backend is **`DEV=CPU`**, not Metal (Metal is an optional, unvalidated speedup); (d) reuse is **partial** — `parse_model_outputs`/transforms/`FrameReader` import msgq-free, but `TinygradRunner`/`ModelState`/`fill_model_msg` pull `params_pyx`/`msgq` and need stubbing/bypass; (e) two-tier faithfulness anchor + drop "bit-stable." Also: **OPM7 is a 3-model split** (vision+onPolicy+offPolicy) needing its own adapter; metric expanded to include `orientation_rate` and `model_minus_lane`; the `camera_offset` shear + warp artifact are explicit inputs.

## 1. Objective

A local, analysis-only simulator that re-runs multiple driving-model bundles (CD210, OPM7, Nevada, …) on the **identical recorded camera frames** from existing Explorer ST routes — sourced from **disengaged / human-driven** segments so no model generated the scene — comparing each model's slow-weave behavior with **every confound removed except the model weights** (open-loop; see §6). It answers, without plausible counter-question: *given the exact same road scene, does model X predict a less-weavy path than model Y?* This is the offline gate named in `2026-06-28-...-fresh-full-analysis-review.md` §9 and `...-nevada-vs-cd210-matched-corridor.md` §7 — the test real-drive A/B cannot deliver because two models never see the same input.

### Success criteria ("no questioning" bar)
1. **Two-tier faithfulness (see §8):** (a) the harness reproduces an upstream reference (tinygrad-vs-onnxruntime on the same ONNX) to tight tolerance — proves the *pipeline* is correct; (b) replaying a bundle on a route it natively drove reproduces the **logged** `modelV2` to a **looser, pre-registered** tolerance — proves it's *realistic* despite device(QCOM/FP16)→Mac(CPU) numeric drift.
2. **Reproducibility:** same bundle + frames → outputs stable within a fixed float tolerance (not bit-equality).
3. **Pre-registered metric & decision rule**, locked before any cross-model number is viewed.
4. **Paired comparison:** identical scene ⇒ model-vs-model is paired per window ⇒ maximal power + clean paired test.
5. **Reproducible, provenance-tagged artifacts** under gitignored `retrospective_lateral/results/`.

## 2. Non-goals / Out of scope
- No vehicle-control changes; `opendbc_repo/`, `panda/`, `selfdrive/` *driving* code untouched (we import preprocessing/parsing read-only).
- **No on-device execution** — runs only on the Mac (M4 Max). The Comma 4 is a read-only source for frames/bundles.
- No closed-loop / plant simulation. **Open-loop** model replay (justified §6).
- Not the on-road interleaved A/B (separate, later felt-weave confirmation).
- No model retraining / weight edits.

## 3. Constraints & verified environment
- **Platform:** macOS **arm64**, Apple **M4 Max**, 40-core Metal GPU, **16 CPU cores (12 perf)**, Python 3.11 (`.venv311`).
- **Submodules are NOT initialized:** `tinygrad_repo/` (0 files, `-3501a714…`) and `msgq_repo/` (`-b7688b9…`) are empty. **Day-1: `git submodule update --init tinygrad_repo`** (the sunnypilot tinygrad fork @ `3501a714` has the Metal/CPU backends + `examples/openpilot/compile3.py`). Do **not** build `msgq` (impractical on macOS).
- **`process_replay`/full-modeld-process path is BLOCKED** (`cereal.messaging`→`msgq`, absent). We use direct inference instead.
- **Backend = `DEV=CPU`** on Darwin (verified: `selfdrive/modeld/SConscript:20`, `sunnypilot/modeld_v2/SConscript:26` → `DEV=CPU THREADS=0`). CPU is the **blessed baseline**; `DEV=METAL` is **untested in-repo** and is at most an optional Phase-1 speedup to benchmark, not assumed.
- **Importable msgq-free (verified):** `selfdrive/modeld/parse_model_outputs.Parser`, `common.transformations.model.get_warp_matrix`, `tools.lib.framereader.FrameReader`. **NOT importable as-is (verified):** `TinygradRunner` & `fill_model_msg` (need `common.params_pyx`, an uncompiled Cython ext), `sunnypilot/modeld_v2/modeld.ModelState` (needs `msgq`). Reuse therefore requires a **`Params` stub** and lifting the runner's inference path out of the msgq I/O shell.
- Generated files stay under `retrospective_lateral/results/` (gitignored); analysis is a new package separate from driving code; `onnxruntime`/`onnx` are not installed (needed only for the §8 self-test).

## 4. Architectural decision

**Direct, file-driven tinygrad inference on CPU (Metal optional), maximally reusing existing sunnypilot/openpilot inference pieces, with a `Params` stub replacing on-device config and an ONNX→Mac-pkl compile step per bundle.**

Concretely:
- **Compile:** wrap `tinygrad_repo/examples/openpilot/compile3.py` to compile each bundle's **ONNX** → a Mac (`DEV=CPU`) tinygrad pkl, and use its built-in `SELFTEST` (tinygrad vs onnxruntime) as tier-(a) correctness.
- **Reuse:** `FrameReader` (decode), `get_warp_matrix` + the sunnypilot `camera_offset` shear (warp), `parse_model_outputs`/`parse_model_outputs_split` (tensor→`position`/`desiredCurvature`/`laneLines`/`orientationRate`), and the **inference logic** of `TinygradRunner`/`TinygradSplitRunner` (lifted, with a `Params` stub).
- **Re-implement (the real hazard — see risk #5, *not* small):** the msgq-free driving loop that ModelState normally provides — feed warped YUV + desire/traffic + recurrent `features_buffer` frame-by-frame, manage the temporal-state core (`prev_desired_curv` feedback, per-family `temporal_idxs` strides, generation/`mlsim` gating), and for OPM7 the split merge (`plan = plan + planplus`), then the `LAT_SMOOTH`/`get_curvature_from_plan` post-step. A stride/feedback error here silently corrupts one model's weave numerics — this is the primary correctness risk, guarded by the per-bundle anchor.

Rationale: the full `ModelState`/`process_replay` needs `msgq` (unbuildable here); but ~all the math (decode, warp, runner inference, output parse) exists and is reachable once `params_pyx`/`msgq` are stubbed/avoided. This is the only path that runs on this Mac and is fully under our control — which the §8 anchor then validates.

## 5. Components (proposed package `model_replay_sim/`)
- `config.py` — paths, bands, tolerances, schema, bundle registry.
- `params_stub.py` — minimal in-memory `Params` replacement so the lifted runner imports/runs off-device.
- `compile_bundle.py` — ONNX → Mac `DEV=CPU` pkl via `compile3.py`; record provenance incl. the **tinygrad submodule SHA as a hard field** (the Mac pkl is bound to a tinygrad commit; on-device `test_tinygrad_ref.py` enforces this) so a re-run is reproducible; run onnxruntime self-test.
- `bundles.py` — locate/verify each bundle's vision+policy(+offPolicy for OPM7) artifacts + `*_metadata.pkl`; per-bundle input/output spec from metadata (never hardcoded).
- `frames.py` — `FrameReader` wrapper: road (+wide `ecamera`/`big_img` if the bundle's metadata requires it) hevc → model-input YUV; warmup handling.
- `calib.py` — reconstruct the per-frame warp: `get_warp_matrix(liveCalibration)` **plus** the sunnypilot `camera_offset` shear with its `0.9/0.1` EMA and the device `camera_offset` param (ties to the −3.05° mount). Replicate or rebuild the `warp_*_tinygrad.pkl` (or do the warp in numpy/cv2).
- `infer.py` — lifted `TinygradRunner`/`TinygradSplitRunner` inference (CPU; Metal optional) + recurrent loop; OPM7 split path (`plan = plan + planplus`).
- `parse.py` — thin wrapper over `parse_model_outputs` (skip `fill_model_msg`/capnp).
- `metrics.py` — weave-band RMS + eligibility gate (shared with `retrospective_lateral`).
- `anchor.py` — two-tier fidelity (onnxruntime self-test + replay-vs-logged).
- `run.py` — orchestrator: parallel decode/preprocess/metric pool; per-segment sequential inference; per-(bundle,segment) outputs; pre-registered comparison.
- `tests/` — synthetic + small-segment regression.

## 6. Why open-loop is sufficient (one caveat)
Replay is open-loop (a model's output doesn't steer the recorded frames). Correct **relative** test because (a) the controller faithfully follows `desiredCurvature` (established), so less predicted-path weave ⇒ less commanded ⇒ less felt weave; (b) the disengaged test showed feedback *suppresses* weave (engaged<disengaged), so open-loop over-estimates absolute weave but fairly **ranks** models. Caveat in results: open-loop can't capture closed-loop feedback compounding; the on-road interleaved A/B remains the felt-weave confirmation.

## 7. Inputs & data acquisition
- **ONNX (routine fetch, confirm in Phase −1):** source each bundle's **ONNX** from its build `ref` via the model-build mechanism — `git checkout <ref> && git lfs pull` → `selfdrive/modeld/models/driving_*.onnx`. Refs (adversarial-QA-verified live): **CD210 `55f66e22`** + **Nevada `3193eac5`** in commaai/openpilot; **OPM7 `052692b2`** in sunnypilot/sunnypilot (3-model: `driving_vision`+`driving_on_policy`+`driving_off_policy`.onnx). The device + public registry hold only QCOM `_tinygrad.pkl` (unusable on Mac), so the ONNX comes from the ref commits, not the device. (The device pkl path is a dead end; don't pull bundle pkls.)
- **Scenes:** a pre-registered set of **DISENGAGED / human-driven** straight/gentle segments (`latActive=0`), spanning both eras' roads (and matched corridors); sized for paired power, bounded for compute. **Disengaged is required for neutrality:** on engaged frames the recorded scene was steered by whichever model drove it, so replaying an off-policy model there judges it on a scene it never produced — the same confound `model_vs_loop.py` avoided by using `latActive=0`. (A secondary engaged-scene arm may be registered for contrast, but disengaged is primary.)
- **Frames:** pull `fcamera.hevc` (road) and `ecamera.hevc` (wide). **Read each bundle's `*_metadata.pkl` `input_shapes` in Phase 0 to decide whether any target bundle consumes `big_img` *before* pulling `ecamera` GBs** — if none need wide, skip it and halve the pull. Read-only tar-over-ssh; ~GB/route → scope tightly.
- **Calibration & camera_offset:** route-logged `liveCalibration` + the device `camera_offset` param drive the warp.

## 8. Two-tier faithfulness anchor (linchpin — gates Phase 0/1)

> **Design position — what actually makes the comparison valid:** the cross-model verdict rests on **identical-Mac-inference pairing** — every bundle runs through the *same* Mac pipeline (same compile path, same warp, same frames, same parse), so any constant device→Mac numeric offset cancels in the model-vs-model difference. Tier (b) below (replay vs the QCOM device log) is therefore **reassurance/sanity, not load-bearing**: it guards against a gross pipeline bug, but the ranking does **not** require reproducing the device. The one thing that *would* invalidate the ranking is a Mac-pipeline step that treats one bundle differently from another (e.g. a warp/input-spec/adapter error specific to OPM7's 3-model split) — so per-bundle anchor + metadata-driven (never hardcoded) handling is the real safeguard, not device match.

- **Tier (a) correctness — tight:** for each compiled bundle, `compile3.py SELFTEST` compares tinygrad(CPU) vs onnxruntime on the same ONNX → require near-exact (small pre-set atol/rtol). Proves the compile/runtime is right independent of the device.
- **Tier (b) realism — looser, pre-registered:** replay the **native** bundle on a segment it drove; compare replayed vs **logged** `modelV2` (`desiredCurvature`, `position.y`, `laneLines`) on eligible windows. Pre-register a **realistic** bound (e.g. Pearson ≥ 0.95, weave-band RMS ratio in [0.85, 1.15]) — *not* 0.99 — because device QCOM `FLOAT16=1 IMAGE=2` ≠ Mac `DEV=CPU FLOAT16-off IMAGE=0`, plus warp/`camera_offset` EMA warmup. A controlled systematic offset does **not** invalidate a *relative* model ranking; a failed *shape/phase* match does. Run tier (b) for ≥1 CD210 and ≥1 OPM7/Nevada segment before scaling. **If tier (a) passes but tier (b) can't be brought into bound, investigate warp/camera_offset/warmup; if still failing, stop honestly.**

## 9. Metric, pre-registration, stats
- **Primary:** weave-band (0.10–0.35 Hz) duration-weighted RMS of `desiredCurvature` and `model_y20`; **plus `orientation_rate_curvature` and `model_minus_lane`** (the most discriminating weave signals per the fresh-review), all on identical (disengaged) frames, identical eligibility gate (`latActive=0`, straight/gentle, no blinker/lane-change/near-lead), speed-binned.
- **Pairing:** same windows across models → paired Wilcoxon on per-window model-vs-model differences; report ratio + sign + effect size. Include CD210, OPM7, **and Nevada** (its status is contested — the sim adjudicates whether it returns to the lever list).
- **Pre-register** band, eligibility, window length, warmup, decision rule, and bundle/segment set before viewing any cross-model number.
- **Self-checks:** rerun stability (float tol); null (same bundle twice → ~0 diff); sensitivity (band/window/warmup); independent re-derivation per project QA discipline.

## 10. Parallelization plan (use the M4 Max honestly)
- **CPU is the baseline runtime** (`DEV=CPU`). Inference per segment is **inherently sequential** (recurrent `features_buffer` feeds frame N+1) — cannot parallelize within a segment.
- **Parallelize across segments and across the non-inference stages** (`ProcessPoolExecutor`, spawn-safe top-level workers, ≤12 perf cores): hevc decode, warp, output parse, and all metric computation. With `DEV=CPU`, **running several segments' inference concurrently across cores is the primary parallelism** (each worker its own tinygrad CPU context) — benchmark worker count vs per-frame slowdown in Phase 1.
- **Decode-once / infer-N×:** decode+warp each frame once (bundle-independent), cache, run all bundles on the cached input.
- **Metal (optional):** if `DEV=METAL` validates in Phase 0, it's a single-GPU serial speedup for one segment at a time; pick CPU-parallel-across-segments vs Metal-serial by the Phase-1 micro-benchmark — don't assume.
- **Throughput is the real first risk — budget overnight, not coffee-break.** A powered paired set is ~300k CPU inferences (≈100k eligible frames/bundle × 3); tinygrad-CPU supercombo runs ~0.3–2 s/frame, and 12 concurrent ~250 MB workers contend on the M4 Max's unified-memory bandwidth (won't scale linearly). The **Phase-1 micro-benchmark (worker count × per-frame cost, CPU vs Metal) is mandatory before sizing the set** — do not let the plan implicitly depend on unvalidated Metal. Same-scene pairing keeps power high on a bounded subset, which is the lever if CPU is slow.

## 11. Outputs (gitignored under `retrospective_lateral/results/`)
- `model_replay/compiled/<bundle>/` — Mac pkls + self-test report + provenance.
- `model_replay/<bundle>/<route>_modelv2.parquet` — replayed per-frame outputs.
- `model_replay/anchor_report.csv` — tier-(a) + tier-(b) fidelity.
- `model_replay/weave_by_window.csv`, `cross_model_summary.csv` — per-window metrics + paired decision.
- `model_replay/PREREGISTRATION.md` — locked metric/decision/bundle-segment set.
- A dated findings report under `docs/superpowers/reports/`.

## 12. Risks (ranked) & mitigations
1. **CPU THROUGHPUT (the real first risk).** A powered paired run is ~300k tinygrad-CPU inferences; per-frame cost + unified-memory contention across workers may push it to overnight and the optimistic case leans on unvalidated Metal. *Mitigation:* mandatory Phase-1 micro-benchmark; decode-once/infer-N×; bounded same-scene subset (pairing keeps power); accept overnight; treat Metal as bonus only if Phase 0 validates it.
2. **OPM7 split + recurrent temporal-state mis-replication (primary *correctness* risk).** The split merge (`plan+planplus`), `prev_desired_curv` feedback, per-family `temporal_idxs` strides, and generation/`mlsim` gating are the one place a harness bug biases *one* model's weave numerics. *Mitigation:* metadata-driven (never hardcoded) per-bundle adapter; `TinygradSplitRunner` path; per-bundle-family anchor is the safeguard (per §8 design position).
3. **Faithfulness tier (b) fails** (warp/camera_offset/warmup/precision mismatch). *Mitigation:* tier (a) isolates pipeline correctness from device-match; debug warp+camera_offset+warmup against the in-tree stock model (has ONNX + logs) in Phase 0; pre-registered realistic tolerance; ranking validity rests on identical-Mac pairing, not device match (§8).
4. **Reuse needs `params_pyx`/`msgq` stubbing** (TinygradRunner/ModelState don't import clean). *Mitigation:* `Params` stub; lift runner inference out of the msgq shell; reuse only the clean pieces (`parse_model_outputs`, transforms, FrameReader) directly.
5. **`camera_offset` shear + warp pkl** are extra artifacts/inputs (calib.py bigger than "from liveCalibration"). *Mitigation:* replicate the `0.9/0.1` EMA + param; rebuild or numpy-reimplement the warp; anchor catches errors.
6. **ONNX provenance (DE-RISKED — routine fetch, was feared existential).** ONNX is obtainable via `git checkout <ref> && git lfs pull` (CD210/Nevada in commaai/openpilot, OPM7 in sunnypilot/sunnypilot); the build-workflow mechanism + an in-tree LFS ONNX corroborate it. *Mitigation:* confirm the 3 refs LFS-pull early (Phase −1, now routine); the "fall back to on-road A/B" branch is unlikely to trigger.
7. **Wide camera / warmup / desire inputs** under-specified. *Mitigation:* read `big_img` need + buffer length from metadata; feed desire/traffic pulses correctly (≈0 on straight eligible windows but wired right).
8. **Metal unvalidated.** *Mitigation:* CPU baseline; Metal only if Phase 0 proves it.

## 13. Phasing (each gated)
- **Phase −1 — ONNX fetch (routine, ~1 hr):** `git checkout <ref> && git lfs pull` the three ONNX (CD210 `55f66e22`/Nevada `3193eac5` in commaai/openpilot; OPM7 `052692b2` in sunnypilot/sunnypilot); read each `*_metadata.pkl` `input_shapes` to settle 2-vs-3-model + `big_img` need. (No longer a GO/NO-GO gate — confirmation only; if a ref unexpectedly fails to pull, *then* escalate.)
- **Phase 0 — Pipeline spike on the STOCK model (the real first gate; start now):** `git submodule update --init tinygrad_repo`; compile in-tree stock ONNX → Mac CPU pkl; pass tier-(a) self-test; replay one **disengaged** stock-era segment and pass tier-(b) anchor (build the `Params` stub, warp+camera_offset, frame feed, parse, metric end-to-end). Proves the harness before any bundle pull. **GO/NO-GO is here, not at −1.**
- **Phase 1 — Multi-bundle small + throughput benchmark:** compile CD210/OPM7/Nevada from ONNX (OPM7 via split path); pull a few disengaged segments' hevc; pass both anchor tiers per bundle family; **micro-benchmark CPU worker-count × per-frame cost (vs Metal) and size the registered set from it**; lock pre-registration.
- **Phase 2 — Full registered set:** parallelized run over registered segments × 3 bundles; paired weave comparison + decision.
- **Phase 3 — QA + report:** collaborative + adversarial QA on results; dated findings report; recommendation (incl. whether Nevada returns to the list).

## 14. Open questions for adversarial QA
- Is the ONNX truly unobtainable, or is there a sanctioned route (upstream build repo, a CPU-recompilable artifact, asking the model authors) that de-risks Phase −1?
- Is tier-(b)'s realistic tolerance (corr ≥ 0.95, band ±15%) defensible, or could a systematic device→Mac bias still flip a model ranking? Should the ranking be done *only* relative within the Mac pipeline (model-vs-model on identical Mac inference), making tier-(b) a sanity check rather than a gate?
- Does open-loop predicted-path weave actually track the felt closed-loop weave well enough to be decisive, or is a minimal feedback approximation needed?
- Is CPU throughput on the M4 Max realistically enough for a powered segment set, or does this hinge on Metal working?
- Could differing input/output specs or warp/camera_offset handling across CD210/OPM7/Nevada bias the comparison even on identical frames?
