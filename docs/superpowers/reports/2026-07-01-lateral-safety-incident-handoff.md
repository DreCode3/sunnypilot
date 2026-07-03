# Handoff — 2026-07-01 Lateral Safety Incident (silent curve under-steer / lane departure)

**Purpose:** hand off a root-cause investigation of a safety incident to the correct context session. This session was told there is **project progress it was not aware of** — so treat everything below as **evidence + my inferences**, not settled conclusions. Where I mark **[FACT]** it's directly measured from logs/device; **[INFERENCE]** is my reasoning that your fuller context may confirm, refine, or overturn.

**Vehicle/setup:** 2021 Ford Explorer ST, Comma 4, sunnypilot fork. Repo root `/Users/dregilley/Documents/GitHub/sunnypilot` (my local repo branch `2021_explorer_st-mici`). Analysis in `.venv311` (py3.11). Device reachable on home wifi (single SSH ControlMaster, `comma@192.168.98.237`, brittle daemon — one connection only; key needs `ssh-add --apple-load-keychain`).

---

## 1. The incident (as reported)
During a test drive, under **Always-On-Lateral** (MADS; steering active without OP "engaged"), the car **repeatedly failed to steer curves** — under-steered / went straight / jerked toward the other lane — with **NO alarm, warning, or alert**, forcing manual takeover. User statements: **"occurred many times"**, **"worst around Windy Hill Rd traveling Westbound"** (approx GPS `33.89613447555749, -84.58840248370267`, not exact), and **"I've never experienced this before."**

## 2. Data I collected
- **[FACT]** Pulled rlogs (read-only tar-over-ssh) for the two test-drive routes into `explorer_st_logs/`: **`route_ce`** (18 segs, 2026-07-01 14:10) and **`route_cf`** (18 segs, 15:07). Device model = **Nevada (NM, "September 07")**.
- **[FACT]** Pulled video for **`route_cf` seg 9,10,11** (fcamera+ecamera) — the Windy Hill Rd incident window.
- Already-local from prior project work: `route_7f` (OPM7, has video, drove Windy Hill Rd), `route_b5/bb/c0/c3` (CD210), `route_c4/c5/c6/c7` (Nevada) — rlogs; some with video.

## 3. ROOT CAUSE (my conclusion — reconcile with your context)
**[INFERENCE, high-confidence] The device is running the UNVALIDATED `v2026.002.001` migration, and its migrated Ford lateral control clips/rate-limits the curvature command, causing silent under-steer/jerk on curves.** The driving MODEL output is fine; this is a control/safety regression, not a model or tuning-of-the-old-stack issue.

Per memory `project_migration_v2026_002_001`, this migration was supposed to be **deferred to a dev env** ("REAL branches/device untouched", gated behind a build/pytest/recompile/sim-re-anchor gauntlet). **It is deployed on the car.** → **If that deploy was NOT intentional/validated, roll back before driving.** (Your unknown-to-me progress may mean this deploy *is* intentional — in which case the finding becomes "the migration's Ford control regressed; isolate + fix it.")

## 4. Device state I verified [ALL FACT]
```
/data/openpilot: branch=clean-v2026.002.001  commit=acd8518e99 (2026-06-30 12:40)  COMMA_VERSION 0.11.2
opendbc_repo:    commit=849b72a1 "ford: Path B - align to upstream curvature-only safety (v2026...)"  STATUS: DIRTY
                 untracked: opendbc/car/ford/carcontroller.py.bak_pre_goldenPI, .bak_pre_toggle
active model:    ModelManager_ActiveBundle = NM / Nevada / "September 07"
FordCurveMode = 0
```
Ford `values.py` `CURVE_MODE_PARAMS[0]` (active, mode 0): `curvature_error=0.004`, `curvature_rate_gain=1.0` (comment: "was 1.15 — rate lag at apex causes overshoot"), `curvature_lookup_time`=0.35 s @ highway, `smooth_tau=(0.12,0.04)`, `smooth_tau_release=(0.18,0.15)`, **angle rate limits at 25 m/s ≈ 0.00018 (up) / 0.00028 (down) per step, `STEER_STEP=5`**. `carcontroller.apply_ford_curvature_limits` = curvature_error clip (vs current curvature) + `apply_std_steer_angle_limits` (rate) + `MAX_LATERAL_ACCEL` clip; plus an `anti_overshoot` EMA (tau 5 s, blended 5–10 m/s).

## 5. Evidence chain (numbered, with the measured numbers)
1. **[FACT] Not a deactivation or EPS fault.** Across both drives `latActive` stayed True while active; **zero** `steerFaultTemporary/Permanent/steeringDisengage`. Rules out silent lat-dropout and EPS fault.
2. **[FACT] It's Always-On-Lateral.** latActive True with `enabled` False: 6262 AOL frames in ce, 3143 in cf. AOL is **not new** — see #8.
3. **[FACT] "Many times" confirmed.** Forced overrides (steerPressed & |steeringTorque|>2 Nm & vEgo>15): **6 in ~90 s in route_ce** (spread ~3–5 km from Windy Hill), **1 at Windy Hill in cf** (seg10, t=676, torque −2.3/−3.5).
4. **[FACT] A "clean" confidence-blackout case (ce t≈257, 62 mph):** lane-line prob crashed 0.9→**0.01** for ~2.3 s; logged model `desiredCurvature`≈0 (straight) with an erratic far-path (position.y swung +2.7→−14 m); latActive stayed True, no alert; driver override at t≈274. This is the "model went straight on a curve, silently" failure mode.
5. **[FACT, but NOT road-controlled] Confidence-crash rate ~0.4/min on BOTH models** (CD210 0.47/min, Nevada 0.39/min; low-conf frame frac 0.025 vs 0.026). ⚠️ These were each model's OWN drives (different roads), so this does NOT prove model equivalence — see #6 for the controlled test.
6. **[FACT] Controlled same-scene replay → model output is fine and model-agnostic.** Using the model-replay simulator, I replayed **CD210, Nevada, OPM7 on the IDENTICAL Windy Hill Rd curve frames** (route_7f) → all three produce nearly identical, correct curvature through the curve (stds 0.00158/0.00150/0.00156; native OPM7 replay-vs-logged corr 0.957; all track the logged output to ≤0.0002). **⇒ The model is NOT commanding a bad path on curves; switching models is not the fix.**
7. **[FACT] The migrated CONTROL clips the curvature command.** Model→output chain on the cf incident bend (t≈676, 56 mph): **model `desiredCurvature`=+0.00627, but final commanded OUTPUT=+0.00299 (<half)**, then snapped to +0.00671 → lateral accel jumped **1.67→3.71 m/s²** (a jerk). i.e. the car first under-steers the bend, then lurches — matches the felt failure. This is the rate-limit / curvature-only-safety behavior in the migrated Ford control.
8. **[FACT] "Never before" reconciled.** You drove **Windy Hill Rd Westbound under AOL cleanly on OPM7** (route_7f seg9, laneProb 0.85–0.99, steered the same curve) and on older models. AOL and this road are not new; **the migration is the only new thing.**
9. **[FACT] The migration changed logging** — route_cf rlogs have **no `roadEncodeIdx`** (only `driverEncodeIdx`), so the simulator can't frame-align the migration drives. Another symptom of a large, unvalidated change (and a blocker for #11 below).

## 6. What I ruled out [FACT unless noted]
- Silent latActive dropout; EPS/steer fault (none fired).
- Model choice (Nevada vs CD210 vs OPM7) — all identical on the controlled curve replay.
- Always-On-Lateral being new — it isn't.
- The slow "weave" tuning work (separate, unrelated project thread).

## 7. Open questions / what I could NOT isolate (for you to continue)
- **Which migration component regressed?** Candidates: (a) the new Ford "Path B — curvature-only safety" limits, (b) the rate/`curvature_error` tuning in `CURVE_MODE_PARAMS`, (c) the new modeld inference engine + recompiled bundle, (d) the dirty `opendbc` edits. I did not isolate these.
- **Is `CURVE_MODE_PARAMS` (the tight rate limits) the migration's or the user's long-standing tuning?** Unknown to me — your context likely knows. If it's pre-existing and worked before, the regression is elsewhere (engine/safety).
- **Did the new engine degrade the model output on-device?** I could NOT test this: the migration drives lack `roadEncodeIdx`, so I couldn't replay on the actual incident frames. I inferred "model fine" from the OLD-engine replay (#6) + the cf logged model output looking normal in character — but a direct new-vs-old-engine comparison on the same frames was not possible with the current simulator.
- **The ce t≈257 confidence-blackout (model output straight):** genuine model behavior on a hard scene, or new-engine artifact? I didn't pull ce video to replay it.

## 8. Tools, data, and scripts available
- Pulled logs: `explorer_st_logs/route_ce/` `route_cf/` (rlogs), `route_cf/…--{9,10,11}/` (video). Reusable: `route_7f` (OPM7+video), CD210/Nevada routes.
- Model-replay simulator: `model_replay_sim/` — replays any bundle on recorded frames (needs OLD-format `roadEncodeIdx`; works on pre-migration drives, NOT migration drives). Key: `infer.replay_window(bundle, route, mono_times)`, `alignment.build_frame_timeline(route)`, `metrics.weave_band_rms`.
- Scratchpad (session-temp, may be gone): `windyhill_3model.py` (the 3-model same-scene replay). Most other analyses were inline python over the rlogs (re-runnable; the LogReader `.which()` throws on some msgs — wrap in try/except; import from `openpilot.tools.lib.logreader`).
- Relevant cereal fields used: `carControl.{latActive,enabled,actuators.curvature}`, `carOutput.actuatorsOutput.curvature`, `controlsState.{desiredCurvature,curvature}`, `carState.{steerFaultTemporary,steerFaultPermanent,steeringDisengage,yawRate,vEgo,steeringPressed,steeringTorque,steeringAngleDeg}`, `modelV2.{action.desiredCurvature,laneLineProbs,position.y}`, `liveLocationKalman.positionGeodetic` (GPS) + `.calibratedOrientationNED` (heading; 270°=West).

## 9. Cautions
- Device updater **reverts code edits** → set `DisableUpdates=1` before any on-device change; verify the branch/commit actually changed after a rollback.
- Single SSH ControlMaster only (brittle daemon). Pulling logs = read-only, fine. Do NOT run remote update/install/reboot.
- This is safety code — bench/validate before driving.

## 10. IMMEDIATE SAFETY POSITION
Regardless of the above nuances: **the currently-deployed build silently under-steers curves and forced repeated manual takeovers. Do not drive it until the regression is fixed or rolled back.** My recommendation was to roll back to the pre-migration last-known-good software and complete the deferred validation gauntlet before re-deploying the migration — but defer to your fuller project context on whether the migration deploy was intentional and how far its validation has actually progressed.
