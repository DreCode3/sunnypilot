# Handoff Prompt — Lateral "Weave" Root-Cause Analysis + Mitigation Plan

> Paste everything below the line into a fresh context session. It is self-contained.

---

You are a senior controls/ML engineer picking up a long-running investigation on a personal **sunnypilot/openpilot fork** running on a **2021 Ford Explorer ST** (Comma 4 device, branch `2021_explorer_st-mici`). A large amount of high-quality analytical work has already been done. **Your job is NOT to start over** — it is to (1) independently verify and synthesize a definitive **root-cause analysis** of the lateral "weave," and (2) produce a ranked, evidence-backed plan of the **best code-change options to mitigate it**. This is a **planning/analysis task: propose, do not implement** vehicle-control changes.

## The problem (symptom, driver-reported)

A small, slow **left-right lateral "weave"** of the car on **straights and gentle curves**: multi-second period (dominant energy ~**0.1–0.35 Hz**, i.e. a ~3–10 s oscillation), small amplitude, but **uncomfortable on long drives and visible to other vehicles**. It is **NOT** high-frequency "hunting" (>0.5 Hz) and **NOT** a static centering offset. It is **worse at low speed** (severe ~30–50 mph, mild ≥60 mph). The driver felt earlier model/config eras "wander" more than others.

## What you must do

1. **Verify & synthesize the root cause** (use the `superpowers:systematic-debugging` skill). Weigh ALL the prior data below, but **independently re-verify every load-bearing claim** — in this project, comparisons have **flipped 3+ times under proper control**, so trust nothing on faith (see `feedback_verify_agent_outputs` and `feedback_lateral_ab_metrics` in memory). Produce a precise, evidence-backed root-cause statement: where the weave originates in the signal chain, the exact mechanism, and the conditions that modulate it.
2. **Enumerate every plausible code-change mitigation** (downstream filtering, speed-gating, controller/PI changes, anticipatory/predicted-curvature blend, lateral planner/MPC, model-level changes, etc.). For each: the mechanism, the exact file/function it touches, expected benefit, and the risk/cost (added lag, curve-response degradation, safety, updater-revert).
3. **Prototype & quantify the promising downstream options OFFLINE using the existing model-replay simulator** (see "Your most important tool" below) **before** recommending anything. Measure both the weave-band-RMS reduction AND the cost (apex lag / curve-response) across multiple validated scenes.
4. **Rank and recommend** the best 1–3 options, each with a concrete validation plan (offline-sim → controlled on-device A/B). **Be brutally honest** about what cannot be cheaply fixed.

**Deliverable:** a root-cause-analysis + mitigation-plan document under `docs/superpowers/reports/` (or `plans/` if it becomes an implementation plan). No code changes to vehicle-control paths.

## The data corpus to weigh (read these; verify the load-bearing ones)

All paths are relative to the repo root `/Users/dregilley/Documents/GitHub/sunnypilot`. Auto-memory lives in `/Users/dregilley/.claude/projects/-Users-dregilley-Documents-GitHub-sunnypilot/memory/` (its `MEMORY.md` index is auto-loaded; read the linked topic files).

**Memory findings (the accumulated conclusions — each is a `.md` topic file):**
- `model_replay_anchor_status.md` — the model-replay simulator + the CD210-vs-Nevada result (most recent, most rigorous).
- `finding_slow_weave_cd210.md` — the original attribution of the slow weave to the **model**, not the PI controller.
- `finding_learned_param_drift.md` — adversarially-QA'd study: **no learned parameter** (steerRatio, angleOffset, PI integrator, calibration) is a validated weave lever. The only robust within-drive signal is a confound: **lower speed → more weave (r≈−0.65)**.
- `finding_b8_centering_not_oscillation.md` — the "golden PI" centering benefit is unproven; oscillation not improved.
- `finding_directional_apex_cutting.md` — right-curve over-yaw = model apex-cutting + a constant ~+0.00035 1/m rightward straight-bias too weak for the PI to cancel.
- `finding_longitudinal_weave_time.md` — the "weave grows with time-since-reset" trend was REFUTED (device-reset ground truth).

**Reports (`docs/superpowers/reports/`):**
- `2026-06-29-cd210-vs-nevada-same-scene-weave.md` — the **balanced 8-scene CD210-vs-Nevada** same-scene weave comparison (the firmed-up result; read the "UPDATED" bottom-line).
- `2026-06-28-retrospective-lateral-fresh-full-analysis-review.md` and the other `2026-06-2x-retrospective-lateral-*` reports — the drill-down analyses (lane geometry, horizon, low-speed source decomposition, model-replay localization).

**Code & tools:**
- `model_replay_sim/` — the **same-scene model-replay simulator** (your test bed; see below). Analysis-only.
- `retrospective_lateral/` — the offline analysis suite (extract → cache NPZ → metrics). `code/signal_utils.py` has `filter_continuous` (the band-limited filter); `code/config.py` has the band/eligibility constants.
- `opendbc_repo/opendbc/car/ford/carcontroller.py` — the **primary lateral control**: the PI lane-centering controller, the `smooth_tau` command EMA, the curvature pipeline, rate limits. **This is where most downstream mitigations would live.**
- `opendbc_repo/opendbc/car/ford/values.py` — `CarControllerParams`, rate limits, steerRatio, smooth_tau.
- `opendbc_repo/opendbc/car/lateral.py` — rate limiting / steer-angle limits.
- `selfdrive/modeld/` and `sunnypilot/modeld_v2/` — the driving-model inference (where `desiredCurvature` is produced from the model plan; `get_curvature_from_output`, `get_action_from_model`).
- `explorer_st_logs/customizations.md` — the full tuning-change history (smooth_tau iterations, steerRatio reverts, PI history, rate limits, safety layer).

## Established conclusions you should START from — then VERIFY, not assume

These are the current best understanding from the prior work. Re-derive/confirm each before building on it:

1. **The weave originates in the driving MODEL's path prediction (`desiredCurvature`), upstream of the controller.** The model-replay simulator replays a model bundle on the device's exact recorded camera frames and reproduces the device's logged `modelV2.action.desiredCurvature` to **corr 0.99+** (CD210 anchor 0.993, Nevada 0.990). The replayed *plan* itself weaves — so the oscillation is in the model output, not the PI/controller/plant. A diagnostic workflow localized the deficit/character to the **near-horizon orientation-rate** (the t≈0 quantity the vision model derives from frame-to-frame motion); it is **broadband + horizon-graded**, NOT a narrow controller resonance.
2. **It is NOT the PI controller, smooth_tau, learned params, or calibration.** Multiple adversarially-QA'd studies found no controller-gain or learned-param lever. The PI integrator is **reverse-causal** (responds to offset, lags it). A simple command low-pass (`smooth_tau`) **cannot** attenuate a ~0.2 Hz weave without unacceptable lag — verify this yourself: an EMA with τ=0.1 s has ~1.6 Hz cutoff, so at 0.2 Hz its attenuation is <2% (the weave band is far below cutoff). Killing a 5 s-period oscillation by low-pass would need τ≈1–2 s = catastrophic apex lag.
3. **Swapping the model does NOT robustly help.** A balanced 8-scene same-scene comparison (4 CD210-native + 4 Nevada-native, all native-leg-validated) found CD210-vs-Nevada weave is **effectively a wash**: median Nevada/CD210 = 0.931 (~7% less) but mean 1.006, 6/8 scenes lean Nevada-less, **sign test p=0.14, bootstrap CI [0.920,1.018]** — not significant, with real road-to-road variation (one road has Nevada weaving +57%). OPM7 could not be cleanly validated (its routes rotated off the device; would need a fresh OPM7 drive). So "just use a different model" is not a proven fix from available bundles.
4. **Speed is the dominant modulator** (lower speed → more weave). Any mitigation should consider speed-gating.

If your independent verification **contradicts** any of these, say so loudly and follow the evidence — that is the most valuable thing you can find.

## Your most important tool: the simulator as a mitigation test bed

`model_replay_sim/` can replay any model bundle on recorded frames and return the per-frame `desiredCurvature` series — i.e. **it reproduces exactly the signal a downstream mitigation would filter.** Use it to **quantify a candidate mitigation offline before any on-device test**:

- `from model_replay_sim.infer import replay_window` → `replay_window(bundle, route_id, mono_times)` returns `{"desired_curvature": (N,), "v_ego": (N,), "lat_action_t", ...}`.
- `from model_replay_sim.anchor import select_anchor_span, run_anchor` → pick a straight/gentle eligible window per route; `run_anchor(bundle, route)` validates fidelity (replay reproduces logged).
- `from model_replay_sim.metrics import weave_band_rms` → the 0.10–0.35 Hz weave-band RMS metric.
- **Validated local scenes with video** (in `explorer_st_logs/`): CD210-native `route_b5, route_bb, route_c0, route_c3`; Nevada-native `route_c4, route_c5, route_c6, route_c7`. Each has a passing same-model anchor (corr 0.978–0.999). The current device model is **Nevada** (NM, "September 07"); CD210 was the prior default (device routes b7–c3 = CD210, c4+ = Nevada).
- Run analysis-only Python via `.venv311/bin/python` from the repo root (Python 3.11).

**Mitigation-prototyping recipe:** for each candidate downstream transform `f` (a filter, blend, speed-gated smoother, notch, etc.), apply `f` to the replayed `desired_curvature` on each validated scene and report **(a)** the weave-band-RMS reduction and **(b)** the cost — e.g. the group delay / apex lag (cross-correlation lag vs the unfiltered signal, and the transient response on a step). A good mitigation reduces the 0.10–0.35 Hz band substantially while adding minimal lag in the >0.5 Hz curve-response band. Test across all 8 scenes (they span speeds and roads, incl. the route_c0 high-weave outlier) and report the distribution, not one scene.

## Candidate mitigation directions to evaluate (non-exhaustive — add your own)

Weigh at least these, and reason about why each would or wouldn't work given "the weave is in the model output, ~0.2 Hz, speed-dependent":
- **Downstream `desiredCurvature` conditioning** in `carcontroller.py`: speed-gated low-pass (accepting lag only at low speed where curve-response demand is low and weave is worst); a phase-compensated / zero-lag filter; an adaptive smoother; a band-limited notch (note the weave frequency may drift). Quantify the lag/benefit trade with the simulator.
- **Anticipatory / predicted-curvature blend** (`pc_blend_ratio`): was retracted before (HF jitter risk) — re-examine with the simulator whether a different blend helps the slow band.
- **PI / lane-centering re-architecture**: the current PI is reverse-causal; consider whether a predictive/feedforward variant could actively *counter* the model's slow oscillation rather than chase it. Be skeptical — prior PI tuning didn't move the slow band.
- **Model-level**: Nevada (wash), a fresh/validated OPM7, a newer comma model bundle, or model-input changes. Note model bundles are **external params** (not a code edit in this repo) and "the model" is hard to change; the simulator showed model inference is faithful, so the weave is the model's *trained behavior*, not a wiring bug.
- **Lateral planner / MPC layer** (if present in this fork): whether a path-smoothing/cost change upstream of `desiredCurvature` would help.
- **Accept-and-bound**: if the weave is genuinely model-bound and uncorrectable downstream without unacceptable lag, say so, and propose the least-bad bounded mitigation (e.g. speed-gated mild smoothing) with its honest cost.

## Standards & constraints (the bar this project holds)

- **Independently verify every claim** — your own and the prior findings'. Re-run numbers, check code/logic yourself. Cooperative AND adversarial agents have both produced confident errors here.
- **Robust lateral-A/B discipline**: same-scene (identical frames) > observational; band-limited 0.10–0.35 Hz RMS (never pooled variance, which is outlier-dominated); speed-match; robust/bootstrap stats; report per-scene distributions. See `feedback_lateral_ab_metrics`.
- **This is PLANNING.** Do **not** modify vehicle-control code, `opendbc_repo/`, `panda/`, or `selfdrive/` driving code. Analysis-only Python in `.venv311`. The working tree has **unrelated pre-existing uncommitted changes** — if you commit anything (docs only), use targeted `git add <specific files>`, never `-A`/`.`.
- **On-device caveat (for the validation plan you propose, not to execute):** the device updater **reverts local code edits** unless `DisableUpdates=1` is set first; model bundles + params survive (they live outside the git repo). Any on-device A/B must reset the integrator each pass, hold speed, interleave A/B/A/B, same corridor, ≥5 passes/config. A controlled on-road A/B is the gold-standard confirmer for any ~5–10% effect (the archival same-scene replay has been pushed to its limit).
- Don't run remote comma update/install/reboot actions. Pulling logs (read-only tar-over-ssh) is fine if needed.

## Suggested output structure for your deliverable

1. **Root cause** — one precise paragraph + the evidence chain (with the numbers you re-verified).
2. **What's been ruled out** — controller/PI, learned params, calibration, model-swap — each with the disconfirming evidence.
3. **Mitigation options table** — option | mechanism | file/function | offline-sim benefit (weave-band-RMS Δ) | cost (lag/curve-response) | risk | verdict.
4. **Ranked recommendation** — the best 1–3, with a concrete validation plan (offline-sim thresholds → on-device A/B protocol).
5. **Honest limits** — what cannot be cheaply fixed, and why.

Begin by reading the corpus and reproducing the core claim (the weave is in the replayed model `desiredCurvature`) on 2–3 validated scenes with the simulator. Then proceed through the four tasks.
