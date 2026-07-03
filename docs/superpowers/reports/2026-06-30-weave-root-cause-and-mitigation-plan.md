# Lateral Weave — Root-Cause Analysis & Mitigation Plan

Date: 2026-06-30
Vehicle: 2021 Ford Explorer ST, sunnypilot fork, branch `2021_explorer_st-mici`, Comma 4.
Scope: **planning/analysis only** — no vehicle-control code was modified. All Python ran in `.venv311`.
Method: `superpowers:systematic-debugging` (root cause before fixes) + the project's lateral-A/B
discipline (band-limited 0.10–0.35 Hz RMS, robust stats, per-scene distributions, verify-everything).

This document picks up the long-running weave investigation, **independently re-verifies every
load-bearing claim from cached/ground-truth data**, and produces a ranked, evidence-backed plan of
code-change mitigations. Where I could re-derive a number I did; where I could not (camera frames
have rotated off the device, so the heavy model replay can't be re-run from scratch) I say so and
fall back to the recorded, provenance-pinned artifacts plus logged device ground truth.

---

## 0. What I re-verified this session (so you can trust the rest)

Frames for the 8 validated scenes are gone from disk, but the prior session cached **(a)** the
replayed `action.desiredCurvature` for all 16 scene×model combinations and **(b)** the full logged
device pipeline (`desired_curvature` → `cp_ema_curvature` → `cp_final_command`, plus `v_ego`,
`cp_predicted_curvature`, `cx1_integral`, `cx1_lane_offset_m`, …) at 20 Hz for each scene. That is
enough to independently reproduce the entire evidence chain without re-running the model.

| Claim | My independent re-derivation | Verdict |
|---|---|---|
| Replay reproduces the **logged** model output | Slide-matched cached replayed series vs logged `desired_curvature`: **route_b5 corr 0.998** (band_ratio 1.008), **route_c5 corr 0.991** (band_ratio 1.064) | ✅ reproduced |
| Weave is **upstream of the controller** | Band-RMS ladder on engaged+gentle ≥30 s windows: median **final/desired = 0.847** (<1 in **7/7** measurable scenes) → controller *attenuates*, doesn't create | ✅ reproduced |
| `smooth_tau` EMA can't touch 0.2 Hz | Analytic 1-pole at dt=0.05 s: τ=0.04 s → **0.3%** atten @0.2 Hz; τ=0.12 s → **1.6%**; 50% atten needs **τ=1.35 s (group delay 1.35 s**, and curve response @0.5 Hz collapses to 0.23) | ✅ reproduced |
| PI integrator is **reverse-causal** | Cross-corr `cx1_integral` vs `cx1_lane_offset_m` (engaged): integral **lags** offset by **+1.0 to +1.75 s** (r 0.6–0.83) on 3 scenes | ✅ reproduced |
| Speed is the dominant modulator | Weave-band RMS of model `desired_curvature` by speed: 9–30 mph ≈ **0.00053–0.00058**, 40–49 mph ≈ 0.00030, 49+ mph ≈ **0.00020** (~2.7× more at low speed) | ✅ reproduced |
| Model swap CD210↔Nevada is a wash | Recomputed 8-scene ratios from cached replays: **median Nev/CD210 = 0.931, mean 1.006, Nevada<CD210 in 6/8, sign-test p=0.145**; route_c0 Nevada **+57%** | ✅ reproduced exactly |
| (new) `pc_blend` predicted-curvature lever | Predicted vs desired weave-band RMS: **median predicted/desired = 0.922** — predicted weaves *slightly less*, so the blend marginally helps; no material lever | ✅ new, rules it out |

Scripts: `scratchpad/mitigation_harness.py` and the inline verifiers in this session's transcript.

---

## 1. Root cause (precise statement)

**The weave is an intrinsic ~0.10–0.35 Hz (≈3–10 s period) oscillation in the driving model's own
predicted path — it is present in `modelV2.action.desiredCurvature` *before* the controller touches
it, and it is the model's trained behavior, not a wiring, controller, calibration, or learned-param
bug.** The model-replay simulator reproduces the device's logged `desiredCurvature` to corr 0.99+ on
its own recorded frames (route_b5 0.998, route_c5 0.991), so the weaving series *is* the model's
output, faithfully reproduced. Downstream, the full controller chain (predicted-curvature blend →
`smooth_tau` EMA → PI lane-centering → rate limits → final Ford command) **attenuates** that
oscillation (median final/desired = 0.847, <1 in every measurable scene) — it passes a wobble it did
not create. The oscillation **does not reproduce by road location** (prior `corridor_repro`: weave
per-cell reproducibility 0.095 vs road-shape 0.76) and **does not collapse when the loop is opened**
(prior open-loop test: disengaged weave ≥ engaged, dis/eng ≈1.5–1.7), which rules out both a fixed
roadway feature and a closed-loop limit-cycle. Prior diagnostic work localized the character to the
**near-horizon orientation-rate** the vision model derives from frame-to-frame motion (it is
broadband and horizon-graded, not a narrow resonance). It is **strongly speed-modulated**: ~2.7× more
weave (in the curvature domain) at 30–50 mph than ≥60 mph.

**Signal-chain location:** originates in `modeld` / the model bundle's path head; surfaces as
`actuators.curvature` entering `carcontroller.py` at
[carcontroller.py:252](../../../opendbc_repo/opendbc/car/ford/carcontroller.py#L252); is then
attenuated (not amplified) by the blend/EMA/PI/rate-limit stages below it.

> ⚠️ One honest nuance on the speed modulation. The metric is in the **curvature** domain (1/m), and
> for a fixed lateral-position wobble of a given period, curvature amplitude scales ~1/v². So part of
> "more weave at low speed" is kinematic inflation of the curvature number, not necessarily a larger
> felt lateral motion. But the driver independently reports it worse at low speed, and the prior
> model-independent path-weave (yawRate/vEgo) study found the same sign (r≈−0.65), so the low-speed
> dominance is real, not purely an artifact. It does mean any speed-gated mitigation is attacking the
> regime where the *curvature command* (what we filter) carries the most weave energy.

---

## 2. What's been ruled out (each with the disconfirming evidence)

- **The PI lane-centering controller / `smooth_tau` / rate limits — NOT the source.**
  - The chain *attenuates* the weave (final/desired 0.847, <1 in 7/7).
  - The PI integrator is **reverse-causal**: it lags lane offset by +1.0–1.75 s, so it *chases*
    offset — it cannot anticipate or actively cancel a 0.2 Hz oscillation (at 0.2 Hz a 1 s lag is
    ~72° of phase; trying to make it "fight" the weave would be destabilizing, not corrective).
  - `smooth_tau` at its real schedule (engage τ = interp [0.12, 0.12, 0.04] over [4, 7, 25] m/s,
    [carcontroller.py:293](../../../opendbc_repo/opendbc/car/ford/carcontroller.py#L293)) attenuates
    **<2%** at 0.2 Hz. The logged `cp_ema/desired ≈ 0.88` is mostly the predicted-blend term
    (predicted weaves 0.92×), not the EMA. Heavier `smooth_tau` was already tried on-device and
    reverted (no benefit). Confirmed analytically and from logs.
- **Learned parameters / calibration — NOT a validated lever.** The prior 3-round adversarially-QA'd
  `learned_param_studies` suite found no learned-param weave lever (steerRatio null, angleOffset ns,
  calibration cal_pitch refuted as a speed confound, integrator reverse-causal and reverse-causal
  again here). The only robust within-drive signal was the **speed confound**, not a parameter.
- **Swapping the model (CD210 → Nevada) — does NOT robustly help.** Reproduced exactly: median
  Nev/CD210 = 0.931, **mean 1.006**, 6/8 lean Nevada-less, **sign-test p=0.145**, bootstrap CI
  includes 1.0, with real road-to-road variance (route_c0: Nevada **+57% worse**). A non-significant
  ~7% lean is not a fix. OPM7 cannot be cleanly validated (its routes rotated off the device); a
  preliminary run suggested OPM7 weaves *more*, but on an unconfirmed build (corr 0.925) — untrusted.
- **The predicted-curvature blend (`pc_blend`) — NOT a lever (new this session).** The predicted
  curvature (`orientationRate.z / vEgo`, blended 10–30%) weaves **0.92×** as much as desired, i.e.
  marginally *less*. Lowering the blend would slightly *increase* weave; raising it buys at most ~8%
  (within noise) and re-introduces the documented HF-jitter / right-curve-overshoot risk. Confirms
  the earlier retraction of `pc_blend` as a weave knob.
- **A fixed roadway feature / closed-loop limit-cycle — refuted** (prior `corridor_repro` &
  open-loop tests; summarized above).

---

## 3. Mitigation options (offline-quantified)

Test bed: the 16 cached replayed `desiredCurvature` series (8 scenes × {CD210, Nevada}), filtered
with each candidate `f`. Benefit = median weave-band (0.10–0.35 Hz) RMS reduction. Costs:
**sweeper-atten%** = attenuation of a *legitimate* 0.15 Hz gentle-sweeper curvature sinusoid
(period 6.7 s, amp 0.004 1/m) — the real-curve energy that lives in the same band; **sharp-keep** =
fraction of 0.5–2 Hz sharp-curve content retained; **step-lag90** = time for `f` to reach 90% of a
curvature step (apex-lag proxy).

| # | Option | Mechanism / file | Weave ↓ | Sweeper atten (cost) | Apex lag | Verdict |
|--:|---|---|--:|--:|--:|---|
| 1 | **EMA τ=0.5 s** (raise `smooth_tau`) | EMA at [carcontroller.py:300](../../../opendbc_repo/opendbc/car/ford/carcontroller.py#L300) | 16% | 11% | 1.15 s | ❌ poor trade; already tried/reverted |
| 2 | **EMA τ=1.0 s** | same | 37% | 28% | **2.3 s** | ❌ catastrophic lag |
| 3 | **EMA τ=2.0 s** | same | 61% | 54% | **4.6 s** | ❌ unusable |
| 4 | **Butterworth LP 0.2 Hz (o2, causal)** | new filter after L252 | 30% | 14% | **2.1 s** | ❌ lag |
| 5 | **Causal bandstop 0.10–0.35 Hz** *(ungated)* | new filter after L252 | **61%** | **89%** | ~0 s | ❌ guts real sweepers/winding roads |
| 6 | **Zero-phase LP/bandstop** | acausal (`filtfilt`) | 17–84% | 6–100% | 0 s | ❌ **cannot run online** (needs future samples); bounds the ceiling only |
| 7 | **Reduce `pc_blend`** | [carcontroller.py:266](../../../opendbc_repo/opendbc/car/ford/carcontroller.py#L266) | −8%→+8% | n/a | n/a | ❌ predicted ≈ desired in band |
| 8 | **PI feedforward to counter weave** | PI block L307–378 | — | — | — | ❌ reverse-causal; can't anticipate model noise |
| 9 | **Swap model (Nevada / OPM7 / newer)** | param (ModelManager), not code | wash / unknown | — | — | ⚠️ Nevada wash; needs a *new* validated bundle |
| 10 | **★ Straight-and-speed-GATED smoother** | gated filter after L252 (see §4) | ~30–60% **on straights** | **~0% on real curves** (gated out) | lag is *free* on straights | ✅ **least-bad, recommend** |
| 11 | **Accept-and-bound** (do nothing in code) | — | 0% | 0% | 0 s | ✅ legitimate fallback |

**The load-bearing finding from the harness:** for every *ungated* filter, **weave-reduction% ≈
sweeper-attenuation%** — because the weave (0.10–0.35 Hz) and *legitimate gentle-curve / winding-road
command* occupy the **same band**, no static downstream filter can remove one without equally
removing the other. The attractive-looking bandstop (61% weave, ~0 lag) attenuates a real 0.15 Hz
sweeper by **89%** — it would make the car badly under-corner on winding roads and long sweepers.
The only escape is **gating**: apply the filter *only where we have proven the in-band energy is
weave, not road* — i.e. on straights (prior `corridor_repro` showed the weave is **not** road-locked,
so on a genuinely straight section the 0.10–0.35 Hz energy is weave by construction).

---

## 4. Ranked recommendation

### #1 (recommended) — Straight-and-speed-gated curvature smoother (downstream, gated)

**Idea.** Insert a smoother on the model curvature right after
[carcontroller.py:252](../../../opendbc_repo/opendbc/car/ford/carcontroller.py#L252), but **crossfade
its output in only when the low-frequency road curvature is near zero (straight) AND speed is low**,
where (a) the in-band energy is provably weave (not a real curve), (b) smoothing lag is *free*
(no maneuver is happening on a straight), and (c) the weave is worst. Hand full authority back to the
raw command before any curve. This is the same logic the existing integral gate already uses
(`abs(apply_curvature) < 0.005`, [carcontroller.py:343](../../../opendbc_repo/opendbc/car/ford/carcontroller.py#L343)).

**Concrete shape (to prototype, not yet implement):**
- Maintain a continuously-running smoother (heavy EMA τ≈0.7–1.0 s, *or* the 0.10–0.35 Hz bandstop) on
  `desired_curvature` so it never cold-starts.
- `gate = straight_weight(road_curv) * speed_weight(v_ego)`, with
  `straight_weight = interp(|lowpass(desired,0.035Hz)|, [0.002, 0.005], [1, 0])` and
  `speed_weight = interp(v_ego, [22, 27], [1, 0])` (full effect ≤50 mph, off ≥60 mph).
- `out = gate * smoothed + (1-gate) * raw`. Crossfade (not a hard switch) so straight↔curve
  transitions don't snap.
- This sits *before* the existing EMA/PI/rate-limit, so all current safety behavior is preserved.

**Offline-quantified benefit/cost (prototyped this session on full logged routes).** I implemented
the gate+crossfade (τ=1 s EMA, `straight_weight` on |lowpass(curv,0.035 Hz)| over [0.002, 0.005],
`speed_weight` over [22, 27] m/s) and ran it on the full logged `desired_curvature` of routes
b5/c3/c5/c7 (mixed straights + curves):

| scene | straight-section weave-RMS raw→gated | curve preservation |
|---|---|---|
| route_b5 | 0.000368 → 0.000246 (**−33%**) | corr 1.0000, peak-keep 1.000 |
| route_c3 | 0.000452 → 0.000303 (**−33%**) | corr 1.0000, peak-keep 1.000 |
| route_c5 | 0.000429 → 0.000283 (**−34%**) | corr 1.0000, peak-keep 1.000 |
| route_c7 | 0.000585 → 0.000422 (**−28%**) | corr 1.0000, peak-keep 1.000 |

So: **~28–34% weave reduction on straight/low-speed sections** (where the driver most notices it and
where smoothing lag is free), while **real curves are left completely untouched** (corr 1.0000,
peak-keep 1.000 — the gate closes on them). A gated bandstop variant would push the straight-section
reduction toward ~60% (per §3) at the cost of more transition ringing. Residual costs (honest):
(a) crossfade transients at straight↔curve boundaries (small; not yet stressed); (b) it does
**nothing** for weave *during* gentle curves (inseparable there); (c) it masks the symptom on
straights, it does not cure the model. Script: `scratchpad/` gated-smoother prototype.

**Validation plan.**
1. *Offline gate (build next):* implement the gate+crossfade in the harness; replay across all 8
   scenes; require **≥30% straight-section weave reduction** AND **<5% attenuation of a synthetic
   gentle sweeper** AND **<0.2 s added lag at the straight→curve transition** before any device test.
2. *On-device A/B (gold standard, only after offline passes):* `DisableUpdates=1` first (the updater
   reverts code edits; models/params survive). One fixed corridor with both straights and gentle
   curves, **interleaved A/B/A/B**, held target speed (~45 mph, the weave-worst regime), integrator
   reset each pass, no lead car, same model bundle, **≥5 passes/config (target 8–10)**. Primary
   metric = duration-weighted 0.10–0.35 Hz RMS of **steering-angle** + model **lane-position**,
   speed-matched, GPS-cell paired, robust stats. Confirm curve response is *not* degraded (apex lag,
   override rate). A ~30–60% straight-only effect should clear road noise with 8–10 matched passes.

### #2 — Pursue a genuinely quieter model (the only real *cure*), validated by the simulator first

The weave is the model's trained behavior, so the only thing that *removes* it (rather than masking
it) is a model whose path head weaves less. Nevada is a wash; OPM7 is unvalidated. **Action:** when a
fresh/newer comma bundle (or a provenance-confirmed OPM7 drive) is available, run it through the
existing `model_replay_sim` anchor + `compare_on_scene` on the 8 validated scenes; adopt only if it
shows a **significant** same-scene weave reduction (sign-test p<0.05 across ≥8 scenes, not the n=3
ceiling that produced the false ~10% earlier). Model swap is a **param** (ModelManager), not a code
edit — it survives the updater and is the lowest-risk *mechanism* if a quieter bundle is found.

### #3 — Accept-and-bound

If #1's offline gate doesn't clear its thresholds, the honest answer is to **accept the weave** (it
is small-amplitude and model-bound) or ship only the mildest gated low-speed smoothing as comfort
trim, logged explicitly as a bounded symptom-mask, not a fix.

---

## 5. Honest limits — what cannot be cheaply fixed

- **There is no cheap downstream filter that removes the weave without cost.** The weave and
  legitimate slow-curve command share the 0.10–0.35 Hz band; ungated filtering removes both equally
  (weave-↓% ≈ sweeper-atten%). A causal low-pass strong enough to halve the weave costs ~1.35 s group
  delay; the only ~0-lag option (bandstop) destroys real sweepers unless gated to straights.
- **Gating limits, but does not eliminate, the problem.** A straight/speed-gated smoother can quiet
  the weave *on straights* (where the driver most notices it and where lag is free), but the weave
  *during gentle curves* is inseparable from the legitimate command and stays.
- **The controller cannot actively cancel it.** The PI is reverse-causal (lags offset ~1 s); a
  feedforward canceller would have to *predict* the model's stochastic weave, which is not possible.
- **Model swap from available bundles is not a fix.** CD210↔Nevada is a statistical wash; OPM7 is
  unvalidated. A real cure needs a *new* bundle that the simulator shows is significantly quieter.
- **The archival same-scene replay is at its limit.** n=8 scenes, no live frames; a ~5–10% model
  effect can't be firmed up further offline. The remaining decisive evidence for *any* option is a
  controlled, interleaved, speed-matched on-device A/B.

---

## Appendix — provenance & reproducibility

- Verifications: `scratchpad/mitigation_harness.py` (filter benefit/cost sweep) + inline verifiers
  (controller ladder, speed bins, model-swap reproduction, EMA math, PI lag, replay-fidelity
  slide-match). Run with `.venv311/bin/python` from repo root.
- Cached inputs: replayed series `exp_<scene>_<bundle>.npz` (16, prior session scratchpad); logged
  device pipeline `retrospective_lateral/results/cache/route_*.npz` (20 Hz). Anchor result
  `retrospective_lateral/results/model_replay/anchor_cd210_route_b5.json` (corr 0.993, tinygrad sha
  pinned & matching).
- Simulator: `model_replay_sim/{config,anchor,infer,metrics,compare}.py`; weave band (0.10, 0.35) Hz,
  fs 20 Hz, anchor gate corr≥0.95 & band_ratio∈[0.85,1.15].
- Prior chain superseded/extended: `finding_slow_weave_cd210`, `finding_learned_param_drift`,
  `model_replay_anchor_status`, reports `2026-06-25-...-weave-root-cause-synthesis.md` and
  `2026-06-29-cd210-vs-nevada-same-scene-weave.md`.
- No vehicle-control code, `opendbc_repo/`, `panda/`, or `selfdrive/` driving code was modified.
