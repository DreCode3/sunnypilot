# Centering road A/B protocol (D-phase) — USER EXECUTES

**HARD DEPENDENCY: F3 AOL safety net (Workflow 1) must be DEPLOYED AND OBSERVED
for at least one normal drive before these runs** (it is the net under them).
Device prep per reference_device_config: `DisableUpdates=1`, double-reboot gotcha.

## Arms
- A (control): CameraOffset = 0.0
- B (treatment): CameraOffset = -0.12
Toggle ONLY while PARKED between passes, via SSH:
  python3 -c "from openpilot.common.params import Params; Params().put('CameraOffset', '<value>')"
  python3 -c "from openpilot.common.params import Params; print(Params().get('CameraOffset'))"
(read-back REQUIRED; modeld EMAs the param in ~0.5 s, but toggling parked removes
all doubt; per feedback_pi_param_drift, verify the param state before each pass).

## Corridor & schedule (feedback_lateral_ab_metrics: same location + speed match)
- Corridor: the hiram corridor (the stock 04/05 route), BOTH directions.
- ≥3 passes per arm per direction, interleaved A,B,A,B,A,B (never blocked AABB).
- Fixed cruise set-speed per corridor leg (pick the leg's normal speed; same both arms).
- Engaged, hands-off-but-ready, no manual corrections except safety; abort a pass on
  rain/heavy traffic/lead-follow (redo it).
- Log sheet per pass: wall time, arm, direction, set speed, weather, any overrides,
  any F3 alerts, subjective centering note (1-5).

## What gets logged (nothing extra to set up)
Normal rlogs+camera on device. After the session: pull rlogs AND fcamera.hevc for all
passes to explorer_st_logs/route_ab_<date>/ (video is REQUIRED — see below).

## Analysis (pre-registered, M0 §6)
1. Extract every pass: stock_lateral_toolkit/extract_drive.py -> per-pass npz.
2. PRIMARY — video ground truth: the running model CANNOT log its own improvement
   (at settle it reads its preferred offset by construction; S1's settling model),
   so re-run the M1 pipeline (frame_sampler with ROUTE pointed at the A/B route,
   annotate, m1_offsets) per arm. Success: treatment median physical |offset| ≤ 0.05 m
   OR reduced by ≥70% of P, sign as predicted by the S1 slope.
3. Weave: matched GPS-cell + speed-bin band|yawRate| (0.10–0.35 Hz) A vs B
   (analyze_compare-style cells). Success: delta not significant (calibrated test)
   AND point estimate ≤ +15% vs control.
4. Safety: zero F3 departure alerts attributable to the offset; logged min distance
   to the nearer lane line not reduced by > 0.05 m (median over matched cells).
5. Subjective: not worse (goal: better), recorded per pass before seeing numbers.
ALL of 2-5 must hold to adopt -0.12 as the standing device value; any
failure -> revert CameraOffset to 0.0 and return to the R-phase evidence.

## Sanity notes
- The A-arm doubles as a fresh baseline P measurement (compare to the M-phase P
  = +0.095 m vehicle-frame, n=34).
- Do not mix in other param changes; this A/B tests exactly one lever.
- Per feedback_lateral_ab_metrics: comparisons have flipped 3x under poor control —
  if pass counts end up unbalanced or speeds unmatched (>2 m/s cell mismatch),
  collect more passes rather than relaxing the matching.
- F3 note: departure detection is SHADOW-only on the deployed build; criterion 4's
  "F3 departure alerts" are read from the cloudlog shadow telemetry, not the UI.
