# DM false-positive evaluation — 2026-07-16 drives

**Verdict: the "distracted" takeovers are PHONE-detector triggers (99.4% of events), not
pose/eye. The model is reading a real visual signature — the arm-sling posture holding the
left hand at chest center plus the lit phone in its mount low-right of the wheel — while the
driver's gaze is verifiably on the road. The policy's phone branch never consults gaze.**

## Data
6 routes 2026-07-16 (139 segments, ~2.3 h; rlogs + full dcamera pulled to
`explorer_st_logs/dm_2026-07-16/`). Build: `960d5de6f` (v2026.002.001 port, deployed 07/05).

## Findings
1. **464 distraction alert events** (`driverDistracted1`/`driverDistracted2` — the .001
   two-stage escalation) across routes 19/1a/1c/1d (52/194/106/112); routes 18/1b zero.
   ~5.4 min of "Pay Attention" displayed over ~63 min of lateral-active driving.
2. **Attribution: 461/464 phone-dominant, 0 pose-dominant, 3 mixed, eye never fired.**
   In alert windows: faceProb 0.91–0.95 (face solidly tracked), face_det 100%,
   phoneProb peaks 0.72–0.91, awareness draining to 14–26% (forced-takeover territory).
3. **Global phoneProb distribution: P50 0.052 / P90 0.574 / P95 0.723 / P99 0.863;
   12.9% of ALL frames exceed the stock threshold 0.5** (fork threshold 0.8 → 2.8%).
   This reproduces and worsens the fork-era measurement (P90 0.53 / P95 0.70) that
   motivated the fork's `_PHONE_THRESH = 0.8` customization ("false positives from
   hand/mount position").
4. **Visual ground truth** (dcamera frames at 4 episodes, phoneProb 0.87–0.91, saved in
   scratchpad dm_frames/): driver's eyes forward on the road in every frame; right arm in
   a sling/brace with the left hand held at chest center; phone mounted with lit screen
   low-right of the wheel, inside the DM crop. No phone in hand in any sampled frame.
5. **Old fork vs current code** (`helpers.py` @ fork tip vs `policy.py` @ .001): pose
   thresholds identical; the operative differences are `_PHONE_THRESH` 0.8→0.5 (fork's
   documented FP fix for this exact cab, not carried per the fresh-start no-port rule) and
   the fork's `_YAW_MIN_OFFSET −0.7` widening (not implicated — pose fired ~0 of events;
   yawCalib offset sits at ~0.156 with the stock clamp… note stock clamp is −0.0246 MIN /
   the learned offset here is +0.15, inside range — consistent with pose being quiet).
6. **The commaai PRs the user found (#37986 const border, #38091 YUV pad=16) are ALREADY
   in the deployed tree** (merge commits present; `border_fill_val=16` in
   compile_dm_warp.py; device dm_warp compiled from this tree on 07/05). They fix DM input
   padding artifacts — orthogonal to this failure mode, and demonstrably not sufficient.

## Mechanism
`policy.py:230`: `distracted_types['phone'] = phone_prob > 0.5`, gated only by
faceProb > 0.7 and low pose-std — **gaze direction is not consulted**. A chest-center hand
(sling posture) + lit mounted screen reads as sustained phone use → two-stage escalation →
takeover, regardless of eyes-on-road.

## Options (user decision — DM is safety layer)
A. **Zero-code (recommended first):** move the phone mount out of the DM camera's
   hand region / screen off while driving. If the sling is temporary, the dominant posture
   signal goes away with it — the fork-era baseline (mount alone) produced far fewer events.
B. **Threshold recalibration** `_PHONE_THRESH 0.5 → 0.8` (fork-validated on this cab;
   trigger-frames 12.9% → 2.8%, episode count drops superlinearly since escalation needs
   sustained accumulation). Cost: genuinely reduced phone-use detection — an explicit
   monitoring-safety tradeoff.
C. Redesign (gaze-gated phone logic / sustained-dwell filter): more principled, more work,
   diverges from stock. Not recommended while A is untried.

Census tooling: scratchpad `dm_census.py` (parallel rlog scan: events, distractedTypes,
phoneProb/pose/calib, alert seconds); frames via ffmpeg select on dcamera.hevc
(NOTE: per-segment rlogs start with an initData stamped with ROUTE-start mono time — use
the first non-initData message for segment t0 when mapping mono→video offsets).

## Post-deploy verification (same day, threshold 0.8 live — routes 1f/20/21/22)

Objective improvement, normalized per lateral-active minute (20.6 min today vs 62.6 on 07/16):
- **Alert episodes: 0.99/min → 0.15/min (−85%; prediction was −87%)** — 3 residual episodes.
- Raw alert events: 7.4/min → 0.73/min (−90%). "Pay Attention" display: 8.6% → 0.8% of engaged time.
- Posture signature UNCHANGED (phoneProb P90 0.574→0.599) → gain is the threshold, as designed.

Residual mode (dcamera frames at all 3 episodes, phoneProb peaks 0.89–0.92): **left hand up
adjusting the sling — two hands + brace bulk at chest center**, reading as a held object. In one
frame the gaze is genuinely down at the hands; these residuals are borderline-legitimate nudges,
not pure FPs. Threshold sensitivity (dwells ≥3.5 s, both days): 0.8 → 8+5; 0.85 → 2+1; 0.9 → 0+0.
0.9 would functionally disable the phone detector for this cab. Standing value: **0.8** (user may
opt to 0.85 if residual nags still intrude; 0.9 not recommended).
