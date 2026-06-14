# Controlled-Test Analysis Plan (PRE-REGISTERED)

**Purpose.** Define — *before the drive* — exactly how the controlled-test data will be analyzed, so the
hard-won data gets a rigorous, bias-resistant treatment and the drive never has to be repeated. The
decision criteria in §11 are locked in advance: we commit to the metric, the comparison, the test, and the
thresholds now, so we cannot fish for a result afterward. This is the single most important safeguard —
this investigation produced ≥3 confident findings that reversed under proper control, every time because
the analysis choice was made *after* seeing the data.

This plan combines the clean-room third-party model's methodological strengths (stratified
location+speed matching; drive/stratum-level statistics; model-independent path-curvature metric;
slowly-varying-curvature road gate; capnp-direct extraction) with the lessons from our own investigation
(match the band to the symptom; never per-sample stats; never pooled mean-of-variance; speed is
confounded *and* possibly a mediator; centering ≠ oscillation ≠ wheel-weave; verify every output; the
persisted integrator carries state across drives).

---

## ★★ POST-REVIEW REVISIONS (v2 — an independent 4-lens adversarial review of v1 found blockers; these OVERRIDE the body where they conflict)
1. **DRIVE ≥5 passes/config, target 8–10 (this is the #1 change — affects the drive).** The exact label-permutation has a *floor* min two-sided p = 2/C(2n,n): n=3→0.10 (can never clear p<0.05 → a true win is unprovable), n=4→0.029, n=5→0.008. So **≥4 is the hard floor, ≥5 the working minimum.** Power adds to this: the shakedown's between-pass CV (≈0.15, inflated by route/day heterogeneity) implies ~14–20; the controlled *same-corridor same-day* test will have far lower variance, so **target 8–10** and read the analyzer's printed realized-power to extend in-test if needed. (config.py `MIN_PASSES_FOR_CLAIM=4`, `TARGET_PASSES_FOR_POWER=6`.)
2. **SPACE-ANCHORED matching (not pass-time windows).** v1 tiled 30 s windows (~600 m) into 80 m cells → the *same road* landed in *different* cells across passes → **0 shared strata even weak-vs-weak**. v2 snaps every eligible 50 Hz sample to a fixed global GPS grid (`GPS_CELL_M=500`, ≥ the per-cell seconds the band needs) and computes one band-RMS per (pass × cell). Battery sweeps {300,500,800} m.
3. **PRIMARY metric = P2 (path curvature `yawRate/vEgo`)**, not steering angle. The symptom is *visible path* motion, and steering angle is the one channel the 1/v² speed confound attacks. **P1 (steering) is a corroborator, speed-residualized before it can gate.** A win needs the PRIMARY *and* a significant corroborator.
4. **Mask-AFTER-filter band-RMS.** Filter the continuous pass signal once (tiny-gap-guarded, ≤0.5 s), then RMS over eligibility-masked samples only — never interpolate across gated-out override/curve segments (v1 leaked ~20 %).
5. **Permutation is the BINDING significance gate; CI is descriptive.** The percentile cluster-bootstrap CI over-rejects at n≈4–6; the within-stratum exchangeable permutation (with the (1+c)/(1+n) estimator) is well-calibrated. Decision requires: PRIMARY effect ≤ −15 % **and** direction-explicit CI **and** perm p<0.05 **and** sign-consistency ≥70 % **and** a significant corroborator **and** survival of the §9 battery (enforced on P1 **and** P2). Win and worsen use symmetric bars.
6. **The audit HALTS.** If config-recovered≠declared, multiple build commits, integrator not reset, or camera-calibration (`rpyCalib` yaw) unstable across configs, the verdict is stamped `AUDIT_FAILED` and no causal claim is made. Interleaving and lead-follow are checked/active.
7. **Lead-follow gate, GPS max-gap guard, circular heading, % matched-denominator** all implemented (a lead car weaving in front sits squarely in the weave band; GPS dropouts no longer fabricate cells; N/S corridors no longer mis-stratify).
The code (`code/extract.py`, `code/analyze.py`, `code/config.py`) implements v2 and is validated end-to-end on the shakedown cache (runs, audits, space-anchored matching now yields shared strata, power/decision guards fire correctly).

---

## 0. The test this analyzes — required data structure
The analysis assumes a **counterbalanced factorial** controlled test. The analysis is only as good as the
test; these are hard requirements (the analyzer verifies each in §1 and refuses to proceed if unmet):

- **Same corridor, same session, same day** (one tire/load/weather state).
- **Configurations (factor cells):** at minimum the PI contrast on one model — `CD210/weak` vs
  `CD210/golden`. Strongly preferred: the full 2×2 — `{CD210, OPM7} × {weak, golden}` — so the **model**
  and **controller** main effects and their interaction are separable (the clean room found the *model*
  may matter more than the PI; do not pre-judge).
- **Interleaved / counterbalanced order** (e.g. A B B A …) so time/traffic/warm-up drift cancels and does
  not align with config.
- **≥3 passes per config over the SAME physical segment(s)**, both directions where possible (cancels
  road crown/camber). More passes = more power; see §12.
- **Held speed** in a fixed band (e.g. ACC set to one speed) — removes the speed confound *by design*.
- **Persisted PI integrator reset (or recorded) at the start of each pass** (the warm-start carries state
  across drives — see §6). Record the reset.
- **No lane changes; minimal traffic;** note any lead-follow events.
- **Same software build** across the PI configs (only the PI params differ); the model configs differ only
  in the model param. The analyzer extracts and checks the build commit per pass.

> If the test cannot deliver 4 cells, run `CD210/{weak,golden}` (answers the PI question, which is the
> primary one) plus at least the `OPM7/weak` cell from existing or new data for the model contrast.

---

## 1. Execution verification (run FIRST — gate on it)
Before any weave analysis, confirm the test was executed as designed. The analyzer emits an
`execution_audit` and **halts with a warning if any check fails** (so we don't analyze a broken test):
1. **Config identity per pass, from telemetry** — recover `lc_kp`, `lc_ki`, `int_cap` behavior from the
   controller's `LC:` debug telemetry (P/off, I/int) and the active driving model; confirm each pass ran
   the labeled config. *Never trust the label; prove it from the logs* (this caught a mislabeled config
   before).
3. **Speed actually held** — per-pass speed median/IQR within the intended band; flag passes that drifted.
4. **Interleaving present** — config is not confounded with pass time (check config-vs-time ordering).
5. **Integrator reset** — `int` starts near 0 each pass (from `LC:` telemetry); flag warm-start carryover.
6. **Same build** across PI configs (commit string from `logMessage` ctx).
7. **Corridor overlap** — passes cover the same GPS cells; report the shared-cell census per config pair.
8. **Engagement & override health** — engaged fraction; override events per pass.

---

## 2. Signals (extract a broad superset; choose per metric below)
- **Felt-at-the-wheel:** `carState.steeringAngleDeg` (EPS-measured wheel angle), `steeringRateDeg`,
  `steeringTorque`, `steeringPressed`.
- **Model-INDEPENDENT path motion:** actual path curvature `κ = yawRate / vEgo`. Use **two** yaw sources
  and cross-check: `liveLocationKalman.angularVelocityCalibrated.z` (preferred, calibrated) and
  `carState.yawRate` (CAN). Lateral accel `a_lat = vEgo² · κ`.
- **The control signal chain (for attribution, §8):**
  `modelV2.position`/`laneLines` (the model's desired path & perceived lane) →
  `controlsState.desiredCurvature` (planner target) →
  `carControl.actuators.curvature` (commanded, includes the PI trim) →
  `κ = yawRate/vEgo` (what the car actually did) → `steeringAngleDeg` (the wheel).
- **Controller internals:** the `LC:` telemetry (`off, int, P, I, curv`) — integrator state, PI authority,
  saturation (the proven mechanism difference between configs).
- **Localization & context:** GPS lat/lon, heading (from GPS gradient), `liveCalibration.rpyCalib`
  (camera extrinsics — must be stable across configs or it's a measurement confound), `modelV2.meta`
  (laneChangeState), blinkers, `canValid`, `calStatus`, `carParams`.

---

## 3. Frequency bands
- **PRIMARY weave band: 0.10–0.35 Hz** (period 3–10 s) — the symptom band, independently chosen by the
  clean room and matching our ~0.16–0.20 Hz finding.
- **Sub-bands** (reported, not decisive): 0.05–0.10 (very-slow drift), 0.10–0.20, 0.20–0.35.
- **Control band: 0.5–1.5 Hz** (fast hunt) — reported to confirm the effect is NOT here.
- **Window length ≥ 30 s** (resolves to ≤0.033 Hz); a 45 s variant in the robustness battery.
- **Spectral:** Welch PSD per pass for peak frequency and peakedness (limit-cycle signature).
- Band-pass = zero-phase Butterworth (filtfilt); RMS over the band-passed, eligible samples.

---

## 4. Eligibility gates (a sample/window counts only if ALL hold)
- **Engaged:** `latActive == True`, eroded ±2 s (drop engagement transitions).
- **No override:** `steeringPressed == False`, ±1 s buffer.
- **No lane change / no blinker:** `modelV2.meta.laneChangeState ≤ 0`, blinkers off ±1 s.
- **Speed in the test band** (configurable; default the held setpoint ± a tolerance).
- **Straight / very gentle road:** `|lowpass(κ, 0.035 Hz)| < 0.0015 1/m` (separates *road* curvature from
  *weave*; clean-room idea). A stricter/looser variant in the battery.
- **Lane quality:** `laneLineProbs > 0.5` both sides, lane width ∈ [2.4, 4.6] m (sanity).
- **Data valid:** `canValid`, calibration `calibrated`, finite signals.
- Windows: non-overlapping 30 s; retained only if ≥ 20 s eligible within the window.

---

## 5. Units of analysis & statistics (autocorrelation-honest)
**The cardinal rule: never treat 20–100 Hz samples as independent.** Naive per-sample statistics overstate
significance by ~10–100× (this produced a false "significant" result we had to retract). Hierarchy:
`sample → 30 s window → pass → config`.
- **Primary unit = the PASS** (a single traversal of the corridor under one config). Passes are
  time/space-separated → quasi-independent. With ≥3 passes/config we have a real between-pass variance.
- **Window- and stratum-level values are aggregated to the pass** (median) before any config comparison.
- **Uncertainty:** cluster bootstrap resampling **whole passes** (and, where matched, whole strata);
  report 95% CIs. **Exact label-permutation test** over passes for the config contrast (valid at small n).
- **Robust location:** median / trimmed mean, never mean-of-variance.
- Report effect size (absolute + %), CI, permutation p, **and** per-stratum sign consistency.

---

## 6. Confound controls
- **Speed:** fixed by design — but *verify it held* (§1) and stratify by 2.5 mph speed bin anyway; if a
  residual per-pass speed difference remains, **residualize** (regress metric on speed within config and
  compare adjusted values; ANCOVA). Note: steering *angle* weave scales ≈ 1/v², so any residual speed gap
  biases steering metrics — the path-curvature and displacement metrics are far less speed-sensitive and
  are the tie-breakers.
- **Location / road geometry:** **GPS-cell + heading paired** (default ~80 m cells; battery: 40/80/160 m).
  Compare configs **within the same cell+heading+speed-bin stratum**; aggregate the per-stratum
  differences. Both directions analyzed (crown cancels in the both-direction average; report per-direction).
- **Time / traffic / warm-up drift:** interleaving cancels it; *verify* config ⟂ time; include pass index
  as a covariate; drop passes with lead-follow weave contamination.
- **Build version & camera calibration:** require same commit across PI configs; require `rpyCalib`
  (esp. yaw) stable across configs (a calibration shift would move the model's perceived lane = a
  measurement confound, not a real effect).
- **Persisted integrator warm-start:** verify reset (§1); the integrator state is itself reported as a
  mechanism variable, not a nuisance.
- **Single-pass degeneracy:** any config with <2 passes gets descriptive reporting only — **no causal
  claim** (this is exactly why the b8-single-drive analysis failed).

---

## 7. Weave characterization battery (the symptom from every angle)
Computed per eligible window, aggregated to pass, compared per §5/§11. All are reported; the **primary**
is marked.

| # | metric | channel | what it captures |
|---|---|---|---|
| **P1** ★PRIMARY | **0.10–0.35 Hz band-RMS of `steeringAngleDeg`** | wheel | the felt wheel weave |
| **P2** ★PRIMARY | **0.10–0.35 Hz band-RMS of path curvature `yawRate/vEgo`** | path (model-indep) | the *visible* path weave |
| S1 | estimated peak-to-peak lateral **displacement (cm)** | path | the "visible to traffic" magnitude (from band-passed `a_lat`, or double-integration / model path-y; reported with method) |
| S2 | band-RMS of `a_lat = v²·κ` | body | felt lateral motion |
| S3 | **episode rate / worst-case:** fraction of eligible time with rolling-window band-RMS above a threshold; episodes/min; p90/p95 tails | wheel & path | the bad *episodes* a driver notices (median can hide them) |
| S4 | **steer-per-path gain:** steering-weave ÷ path-weave (deg per 1e-4 1/m, or deg/cm) | ratio | "busy wheel for the same car motion" — a calm vs nervous wheel at equal path |
| S5 | **spectral peak frequency + peakedness** (peak/floor) | wheel & path | limit-cycle signature & whether a config shifts/sharpens it |
| S6 | steering **rate** & **jerk** RMS; reversal rate | wheel | sharpness/character of corrections |
| S7 (secondary axis) | **centering:** signed mean lane offset; sub-0.1 Hz drift; time beyond ±0.3 m | path/model | steady-state position (orthogonal to weave; the proven integrator-cap mechanism acts here) |
| S8 (secondary axis) | **integrator state:** fraction railed at cap, |off| while railed | controller | the proven authority-difference mechanism |

---

## 8. Attribution — model vs PI (the new capability)
Two independent routes; require agreement.
1. **Factorial contrasts** (if 4 cells): PI main effect = ½[(CD210·g − CD210·w) + (OPM7·g − OPM7·w)];
   Model main effect = ½[(OPM7·w − CD210·w) + (OPM7·g − CD210·g)]; plus the interaction. Each on the
   matched, pass-level primary metrics.
2. **Signal-chain localization** — compute the weave-band RMS at each stage of the chain (§2) per config:
   `model desired-path → desiredCurvature → commanded curvature → actual κ → steering`.
   - If the weave is **already present in `modelV2`/`desiredCurvature`** and roughly equal across PI configs
     → it originates in the **MODEL** (PI can't be the lever; matches the clean-room's "OPM7 lowest" hint).
   - If the weave **grows from `desiredCurvature` to `commanded curvature`** → the **PI/controller** is
     injecting it.
   - If it's only in **actual κ / steering** but not in `commanded` → **plant/road**, not software.
   Report the stage-by-stage RMS table per config; this localizes the cause regardless of the factorial.

---

## 9. Robustness battery (the primary conclusion must survive ALL)
Re-run the primary comparison varying one choice at a time:
- weave band: {0.08–0.50, 0.10–0.35, 0.12–0.30, 0.10–0.20, 0.20–0.35}
- window length: {30, 45} s
- road-curvature gate: {0.0010, 0.0015, 0.0020} 1/m
- GPS cell size: {40, 80, 160} m; speed bin: {2.0, 2.5, 5.0} mph
- yaw source: {calibrated, carState}
- statistic: {median, 20% trimmed mean}
- both directions pooled vs per-direction
Report a stability table. A conclusion that flips across these is **not** reported as a finding.

---

## 10. Adversarial self-refutation (mandatory before any "win")
For any claimed difference, the analyzer/automated checks must:
- re-derive it a second, independent way (e.g. P1 and P2 must agree in direction);
- confirm it survives the **strictest** matching (GPS+heading+speed), not just pooled;
- confirm **sign consistency** across ≥70% of matched strata (not driven by a few strata);
- confirm it survives the §9 battery;
- attempt the null: permutation p and CI must clear the §11 thresholds.
A difference that depends on unmatched road/speed, or that is sign-unstable across strata, is reported as
**confounded/insufficient**, not as a finding (this is the exact failure mode that fooled us repeatedly).

---

## 11. ★ PRE-REGISTERED primary analysis & decision criteria (LOCKED)
**Primary question:** Does the **golden** PI reduce the felt/visible weave vs **weak**, on the **same
model (CD210)**, at matched location+speed?

**Primary metric:** the pass-level median of GPS-cell+speed-matched **0.10–0.35 Hz steering-angle band-RMS
(P1)**, corroborated by **path-curvature band-RMS (P2)**.

**Primary statistic:** difference in pass-level medians (golden − weak) within matched strata, with a
**cluster bootstrap over passes** 95% CI and an **exact pass-label permutation** p-value.

**Decision rule (committed now):**
- **"Golden reduces the weave" (a win)** iff ALL hold: (a) matched effect ≤ **−15%** on **P1 AND P2**;
  (b) bootstrap 95% CI excludes 0 on the primary metric; (c) permutation p < 0.05; (d) sign-consistent in
  ≥70% of matched strata; (e) survives the §9 battery; (f) corroborated by ≥1 of {displacement-cm S1,
  episode-rate S3, steer-per-path gain S4} in the same direction.
- **"Golden worsens the weave"** iff the same with sign reversed (≥ +15%, CI excludes 0, etc.).
- **"No difference / inconclusive"** otherwise — explicitly an acceptable, reportable outcome. (We will
  *not* round a confounded or sign-unstable result up to a win.)
- The **−15%** threshold is the provisional perceptibility anchor; it will be finalized from the §12
  power/variance shakedown on existing data **before** the test drive, and frozen thereafter.

**Secondary (same rule template):** model effect (CD210 vs OPM7, matched, weak PI); the factorial PI×model
interaction; centering (S7); the signal-chain attribution (§8).

---

## 12. Power & data sufficiency (set thresholds before driving)
- The shakedown (§13) estimates the **between-pass variance** of P1/P2 on existing data, giving the
  **minimum passes/config and minimum matched cells** needed to detect the −15% effect at ~80% power.
  The analyzer prints this; if the test under-collects, we know *before* concluding "no effect" whether
  it was a true null or just underpowered.
- Pre-committed minimum to attempt a causal claim: **≥3 passes/config** and **≥8 matched GPS+speed
  strata** per contrast (else: descriptive only).

---

## 13. Reproducibility & outputs
- `code/extract.py` — capnp-direct rlog → per-pass signal cache (broad channel set + signal chain + LC
  telemetry + heading + build commit). Verified to run on existing data.
- `code/analyze.py` — implements §1–§11: gating, metrics, matching, pass-level stats, attribution,
  robustness, decision. Emits `results/*.csv`, an `execution_audit.json`, a `decision.json`, and figures.
- `code/config.py` — all parameters (bands, gates, speed band, thresholds, cell sizes) in one place,
  pre-registered values frozen.
- A **shakedown run on the existing b-series + OPM7 logs** validates the pipeline end-to-end and produces
  the §12 power numbers **before** the controlled drive — so the code is proven and the thresholds are set
  in advance. (Existing data can't give the causal answer — that's the whole point of the new test — but
  it fully exercises the code and the variance estimates.)

---

### Provenance of method choices
Carried from the clean-room model: stratified speed×curvature×GPS matching; pass/stratum-level stats with
drive bootstrap + permutation; model-independent `yawRate/vEgo` path metric; 0.035 Hz road-curvature gate;
30 s windows; displacement-cm estimate; build-commit confound check; capnp-direct extraction.
Carried from our investigation: pre-registration & locked thresholds; the signal-chain attribution; the
episode-rate and steer-per-path metrics; spectral peak/peakedness; centering & integrator-saturation as
secondary axes; integrator-reset verification; both-directions crown handling; the "verify config from
telemetry" and "verify every output" discipline; the explicit speed mediator-vs-confounder note (resolved
here by fixing speed in the test design).
