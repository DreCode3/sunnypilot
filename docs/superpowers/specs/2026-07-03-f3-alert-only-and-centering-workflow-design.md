# Design: F3 alert-only safety net + centering RCA/offline-validation workflow (2026-07-03)

User-approved design (this session) for the two workflows brought into scope on the stock
fresh-start base (`stock-2026.002.001-fresh-start`, stock sunnypilot dev v2026.002.000).
Decisions recorded here are the user's; do not relitigate without them.

## Shared context
- Base: stock v2026.002.000 (device commit `3f8e959`, master `31dc4d8e`). No fork ports —
  every optimization is approached fresh (`feedback_fresh_optimization_approach` memory).
- All analysis/testing happens on the Mac first; test drives are the LAST gate of each
  workflow. F3 ships before centering test drives (it is the net under them).
- Discipline: pre-registered acceptance criteria; FPR-calibrated stats
  (`stock_lateral_toolkit/`); GPS+speed-matched A/Bs; independent verification of agent
  outputs.

---

## Workflow 1 — F3 AOL safety net, alert-only port

**Goal:** the incident-designed blindness + departure detectors as a pure safety net on
stock: visual alerts only, zero control-path/panda changes, zero perceptible resource cost.

**User decisions:** visual-only alerts initially (chime is a later one-line decision after
observed fire rates); logging integrated into the existing log stream; no added
computational load (device CPU is limited).

**What ships (adapted from the fork-era F3 design, degrade branch DROPPED):**
1. `sunnypilot/selfdrive/selfdrived/aol_monitor.py` — the pure `AolSafeguardMonitor`
   class, reference implementation and constants exactly as in
   `docs/superpowers/plans/2026-07-02-aol-incident-fixes-f3-f2-ab.md:224-326`
   (low-confidence monitor: conf EMA τ=0.5 s, threshold 0.30, qualifier 1.0 s,
   2.0 s maneuver-blind grace; departure monitor: sighted-only ≥0.6, |offset|>0.40 m AND
   away-rate>0.15 m/s over 0.5 s window, 0.35 m nearest-line backstop, 0.5 s sustain,
   3.0 s post-maneuver grace; active only when lat_active AND v≥10 m/s).
2. Two new events (`cereal/custom.capnp` append after `e2eChime @23`:
   `aolLowLaneConfidence @24`, `aolLaneDeparture @25`) + `EVENTS_SP` entries:
   ET.WARNING, `AudibleAlert.none`, VisualAlert.steerRequired, mid size/priority.
   MADS surfaces WARNING-only events under pure AOL (verified: `sunnypilot/mads/state.py`).
3. Hook in `selfdrived.update_events`, gated on `sm.updated['modelV2']` (selfdrived
   already subscribes to modelV2 — no new subscription). Inputs: laneLineProbs[1,2],
   laneLines[1,2].y[0], meta.laneChangeState, CS.vEgo, blinkers, mads-active.
4. Kill-switch param `AolSafeguardDisabled` (default off ⇒ safeguard on), REGISTERED in
   `common/params_keys.h` before first read (`feedback_param_key_registration`).
5. Telemetry: `cloudlog` line on fire/clear + 1 Hz while an alert is active (monitor
   internals: conf_ema, offset, rate). Rides the existing logMessage path; ~zero bytes on
   clean drives. No new schema, params, files, or processes.

**Resource budget (MEASURED 2026-07-03, benchmark of the reference implementation +
real capnp reads from a stock rlog):** monitor math 0.35 µs/frame worst-case + 6-field
modelV2 read 2.53 µs/frame on M4 ⇒ ~0.006% of an M4 core at 20 Hz; ≈0.06% of one device
core at a conservative 10× derate (~1-2% relative addition to selfdrived's own budget).
Memory ~1.2 KB (one instance, bounded 11-deque). Verification: before/after `selfdrived`
CPU% check on device against recorded perf baselines.

**Offline acceptance gates (all must pass before deploy):**
- G1-G3: committed replay gate (`retrospective_lateral/incident_2026_07_01/scripts/
  f3_replay_check.py`) re-pointed at the NAS corpus — detects the three named incident
  events with designed lead times (G1: departure alert ≥1.0 s before the ce override;
  G2: low-conf ≤2.5 s after cf blind onset; G3: low-conf before the ce lane-change
  override).
- G4: zero departure false-alerts on old clean routes (c5/c7/b5/7f).
- G5 (new): zero departure false-alerts + low-confidence episode census across ALL stock
  drives (00000000-05 + any newer) — false-alert clearance on the build/model that will
  actually run it, and the expected visual-alert frequency, before the user sees one.
- Harness note: the committed gate has hardcoded fork-era paths + worktree import
  (`f3_replay_check.py:21-46`) — re-point to the stock monitor location + NAS corpus.

**Deploy:** device workflow per `reference_device_config` (DisableUpdates=1, double-reboot
gotcha), visual-only. Post-deploy: ~a week of normal drives → review fire log → chime
decision. Device-side pytest/params runtime checks deferred to the deploy step (analysis
venv cannot run them).

---

## Workflow 2 — Centering: RCA → offline-validated solution → road A/B last

**Goal:** definitive root cause of "stock centers worse" and a solution validated as
thoroughly as possible on the Mac before any test drive. NOT a port of the fork centering.

**User decision (ground truth):** BOTH video annotation (primary anchor) and cross-model
consensus (corroboration) — max rigor.

**Key facts constraining the design (verified this session at the stock commit):**
- `CameraOffset` is the ONLY stock lateral-position lever (params_keys.h:232, meters,
  ±0.35, default 0.0 — and it was 0.0 on all stock drives). It acts as a shear on the
  model-input warp (`sunnypilot/modeld_v2/camera_offset_helper.py`), applied to both cams.
- There is NO downstream centering correction (controlsd follows
  `modelV2.action.desiredCurvature` directly; no lateral planner offset).
- Calibration absorbs mount ROTATION only (converged: cal_yaw −2.85° stable on stock
  drives); lateral TRANSLATION is not representable — that residual is CameraOffset's
  domain. `cal_roll` is hard-zero on stock (crown passes through uncorrected).
- Logged offsets are model-frame: physical centering is unidentifiable from logs alone
  (2026-07-02 comparison; stock model "sits +0.15 m left" of its own definition).
- The replay sim reads CameraOffset from route context; SP002/Nevada/CD210 all
  anchor-validated ⇒ same-frame cross-model replay and CameraOffset counterfactuals are
  available offline.

**Phase M — measurement foundation (offline):**
- M1 video ground truth (PRIMARY): sample ~50-100 frames from pulled stock camera files,
  stratified by road/direction/speed/curvature; semi-automated lane-marking annotation
  (detection proposals + human spot-review of a subset by the user); output = physical
  lateral offset (m) with per-frame uncertainty.
- M2 cross-model consensus (CORROBORATION): replay SP002 + Nevada + CD210 on identical
  stock frames; extract each model's laneline geometry; consensus lane-center vs each
  model's own; disagreement map. (Requires extending sim output parsing from
  plan/curvature to lanelines via existing output_slices.)
- M3 reconciliation: decompose stock's logged "+0.15 m left" into physical-position vs
  model-frame-definition components. Method-agreement criteria pre-registered BEFORE
  interpreting (M1 vs M2 vs logs must agree within stated tolerance on overlap samples).

**Phase R — root cause (offline, pre-registered decision tree):**
- R1 translation-vs-preference discriminator: does SP002 misplace the LANE (vs M1 ground
  truth) or correctly place the lane and target off its center?
- R2 crown/roll: direction-paired same-road analysis (cal_roll≡0 ⇒ crown uncompensated);
  offset vs crown-proxy correlation.
- R3 dependence structure: per-corridor/direction/speed breakdown via the GPS-matched
  toolkit; the deficit's shape constrains the mechanism.
- Output: root-cause class (physical translation | model trained bias | crown response |
  mixture w/ fractions) with the evidence pattern that selected it.

**Phase S — solution candidates, offline validation:**
- S0: verify the sim applies the CameraOffset shear identically to production
  (`camera_offset_helper` math is deterministic ⇒ counterfactuals faithful by
  construction once wired; add to sim warp path if absent).
- S1: CameraOffset counterfactual sweep (−0.10…+0.10 m, 0.02 steps) on ≥2 routes:
  predicted centering shift (model lane-center + path), weave-band delta (gate: ≈0
  within calibrated noise), curve-tracking delta (gate: no regression). Runs in PARALLEL
  with Phase M (pure compute, commits nothing).
- S2: non-param candidates only if RCA indicates (each gets fresh design + its own
  offline counterfactual harness + pre-registered acceptance before device deploy).
- Desk-exit criteria (pre-registered): predicted centering improvement sized against the
  M-measured deficit; weave within noise; no curve regression; stats via the calibrated
  battery.

**Phase D — road validation (LAST):** interleaved same-corridor A/B per
`feedback_lateral_ab_metrics` (param toggled between passes, speed-matched, robust
band-limited metrics), F3 already live underneath. Subjective + objective both recorded
against pre-registered success criteria.

**Ordering:** F3 workflow first; centering M and S1 may run concurrently on the Mac.

## Out of scope
- F3 degrade/disengage branch (dropped, not deferred).
- Audible chime (later one-line decision from observed fire rates).
- Any fork centering code (reference literature only).
- Low-speed weave characterization (separate thread; needs a dedicated drive).
