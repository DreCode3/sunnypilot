export const meta = {
  name: 'qa-b8-disparity',
  description: 'Find why b8 golden-PI felt dramatically better (less straight ping-pong) when the aLat analysis showed no change; reconcile subjective vs objective',
  phases: [
    { title: 'Investigate' },
    { title: 'Synthesis' },
  ],
}

const CONTEXT = `
SETUP: 2021 Ford Explorer ST, sunnypilot, CD210 driving model. We restored "golden" lane-centering PI
authority on top of CD210 (single variable vs the prior config) and the user drove route b8.
  WEAK PI (routes b1,b2):  lc_kp 0.0001, int_cap fixed 0.3,            off-gate integral decay 0.98
  GOLD PI (route b8):      lc_kp 0.0005, int_cap np.interp(v,[20,30],[0.3,1.0]), off-gate decay 0.995
  (warm-start persistent integral LaneBiasIntegral present in BOTH). Same CD210 model in b1,b2,b8.
The controller: apply_curvature += lc_kp*lane_offset + lc_ki*integral, where lane_offset is the MODEL's
perceived lane center (EMA-smoothed), integral accumulates ONLY on straights (|apply_curvature|<0.005) when
not overriding, capped at int_cap, decays off-gate. Code: opendbc_repo/opendbc/car/ford/carcontroller.py
(lines ~125 init gains, ~335-360 PI block). PI gates OFF in curves (|apply_curvature|>=0.005).

SUBJECTIVE (driver, trustworthy ground truth):
  - "This drive was impressive, felt very confident the entire time."
  - "The ping-ponging felt DRAMATICALLY REDUCED in straights."
  - "In SHARPER curves it felt like it had LESS AUTHORITY to command sharper turns. I had to take over because
     it wasn't turning sharp enough -- but even then it WASN'T ping-ponging, it just wouldn't turn the wheel
     sharply enough."

OBJECTIVE so far -- TWO analyses, both on engaged (carControl.latActive>=0.9) straight (|aLat|<0.6) 16s windows,
40-80mph, on the shared corridor:
  analyze_b8.py (metric = aLat = liveLocationKalman.angularVelocityCalibrated.z * vEgo; band 0.1-0.5Hz "slow"):
     CD210_WEAK slow 0.121 vs CD210_GOLD slow 0.123 (+2%, FLAT). hunt 0.5-1.5Hz also flat. -> concluded "no change".
     Direct b1/b2-vs-b8 paired test was UNDERPOWERED (only 3-6 shared blocks).
  analyze_b8_v2.py (driver-felt channels, same windows):
     aLat slow:        0.121 -> 0.123  (+2%)   [n 95/56]
     steer slow (steeringAngleDeg 0.1-0.5Hz): 0.993 -> 0.793 (-20%)
     steer STD (deg):  1.197 -> 0.999  (-17%)
     model POS slow (-(laneLines[1].y0+laneLines[2].y0)/2): 0.0787 -> 0.0859 (+9%)
     model POS STD (m): 0.114 -> 0.110 (-3%)
     cmd slow (carControl.actuators.curvature 0.1-0.5Hz): -22%
     cmd STD: -12%
     location-matched same-dir straight blocks: only 2 (underpowered).

LEADING HYPOTHESIS (to validate OR refute): aLat = yawRate*vEgo = lateral ACCELERATION = 2nd derivative of
position, so its band-power weights an oscillation of position-amplitude A at frequency f by ~ (2*pi*f)^2.
Within 0.1-0.5Hz that's a ~25x weighting toward the 0.5Hz end, so a SLOW (~0.12Hz, 8s-period) position/steering
weave -- exactly "ping-pong on a long straight" -- is heavily attenuated in aLat. The driver feels POSITION and
WHEEL motion (steering angle), not acceleration. steeringAngleDeg is model-INDEPENDENT and dropped 17-20%.

DATA: explorer_st_logs/route_b1, route_b2, route_b8 are pulled (rlog.zst, full rate). route_b3..b7 are on the
device but NOT pulled (they ALSO ran weak PI -> additional weak baseline if needed). venv: ./.venv311/bin/python.
`

phase('Investigate')
const FIND_SCHEMA = {
  type: 'object',
  required: ['findings', 'hypothesis_verdict', 'is_pingpong_really_reduced', 'curve_undershoot_explanation', 'confound_risk'],
  properties: {
    findings: { type: 'array', items: { type: 'object', required: ['claim', 'evidence', 'confidence'],
      properties: { claim: { type: 'string' }, evidence: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] } } } },
    hypothesis_verdict: { type: 'string', enum: ['confirmed', 'partly', 'refuted'], description: 'Is the f^2-attenuation explanation for the disparity correct?' },
    is_pingpong_really_reduced: { type: 'string', description: 'Is the ping-pong objectively reduced, or does the wheel just feel calmer while net position wander is unchanged? Which channel is the true measure?' },
    curve_undershoot_explanation: { type: 'string', description: 'Why did golden PI feel like LESS sharp-curve authority (driver had to take over) without ping-pong?' },
    confound_risk: { type: 'string', description: 'Could the steer/cmd -17..22% drop be a route/road confound (b8 vs b1/b2 different roads) rather than the PI? How much do you trust it given only 2 location-matched blocks?' },
  },
}
const [logic, indep] = await parallel([
  () => agent(`${CONTEXT}

YOU ARE REVIEWER A (metric logic & human perception). Do NOT just reload data; REASON about the metrics and
validate the math, with light spot-checks only.
1. Validate the f^2 claim concretely: for y(t)=A*sin(2*pi*f*t), show aLat=y'' has amplitude A*(2*pi*f)^2; compute
   the in-band weighting ratio across 0.1-0.5Hz; estimate how much a 0.12Hz position weave is attenuated in the
   aLat 0.1-0.5Hz band-RMS vs a steering/position metric. Is f^2-attenuation sufficient to explain aLat-flat (+2%)
   while steering dropped -20%?
2. Reconcile the channel disagreement: steering & commanded-curvature oscillation DOWN 12-22%, but model
   lane-POSITION ~flat (+9% slow / -3% std) and aLat flat. What does that combination physically mean? Is the
   ping-pong truly reduced or is the wheel calmer while position wander is unchanged? Which channel best matches
   what a human calls "ping-ponging"? (Consider: steeringAngleDeg is model-independent & is literally the wheel
   the driver watches; model POS is CD210's own possibly-noisy perception; aLat is real accel but f^2-weighted.)
3. Read carcontroller.py PI block. Mechanistically, WHY would STRONGER PI (gold) produce SMOOTHER steering
   (less sawing) rather than more aggressive sawing? (Hint: persistent integral with high cap + slow decay holds
   a steady bias so P isn't re-reacting to every lane-line wiggle; weak PI's low cap + fast decay forces P to
   chase noise.) Is this consistent?
4. Curve undershoot: confirm from code whether the PI is GATED OFF in sharp curves (|apply_curvature|>=0.005),
   i.e. gold==weak in sharp curves, so the under-authority is NOT the PI but CD210's model command + rate limits
   + steerActuatorDelay -- just more salient against calmer straights. OR is there a mechanism by which the
   persistent integral reduces curve authority?
Return structured output.`,
    { label: 'A:logic+perception', phase: 'Investigate', schema: FIND_SCHEMA }),
  () => agent(`${CONTEXT}

YOU ARE REVIEWER B (independent re-analysis & confound). Independently VERIFY the v2 numbers and attack the
route-confound risk. Use ./.venv311/bin/python. Be efficient (target specific segments, don't reload everything
repeatedly).
1. Independently recompute the steering-angle oscillation (steeringAngleDeg std AND a band-limited 0.1-0.5Hz RMS)
   on engaged straights for b1 (or b1+b2) vs b8. Confirm or refute the ~-17..-20% drop. (You can read
   explorer_st_logs/analyze_b8_v2.py for the loader pattern; carState.steeringAngleDeg, carControl.latActive,
   yawRate*vEgo for the straight gate.)
2. CONFOUND CHECK -- the big risk: is the steer drop real PI effect or just b8 driving different/straighter roads
   than b1/b2? Compute GPS overlap between b8 and b1/b2 (how many shared ~275m blocks, same direction). Check
   whether b8's straight windows are at similar speeds and road types. If overlap is poor, quantify how much that
   undermines the absolute-median comparison. Note: b3-b7 (more WEAK-PI drives) are on the device unpulled; would
   pulling them strengthen the baseline? State whether the steer reduction would survive a proper location match.
3. Sanity-check the model POS metric: is -(laneLines[1].y0+laneLines[2].y0)/2 a reliable position measure, or is
   it dominated by CD210 perception noise (compare its noise floor to the steering signal)? This decides whether
   "POS flat" means "position truly unchanged" or "perception too noisy to tell".
4. If cheap, look at one sharp-curve section in b8 where the driver likely took over (high steeringAngleDeg +
   steeringPressed) and characterize: was apply_curvature/commanded curvature saturating below the model's
   desiredCurvature (rate-limit/authority undershoot) vs the model not asking for enough?
Return structured output (same schema; fill curve_undershoot_explanation from what you find or "deferred to A").`,
    { label: 'B:verify+confound', phase: 'Investigate', schema: FIND_SCHEMA }),
])

phase('Synthesis')
const SYN_SCHEMA = {
  type: 'object',
  required: ['disparity_resolved', 'root_cause', 'pingpong_truly_reduced', 'curve_undershoot', 'metric_fix', 'confidence_caveats', 'recommendation'],
  properties: {
    disparity_resolved: { type: 'string', enum: ['yes', 'partly', 'no'] },
    root_cause: { type: 'string', description: 'Why subjective (big improvement) and objective-aLat (flat) disagreed.' },
    pingpong_truly_reduced: { type: 'string', description: 'Best-evidence answer: is straight ping-pong genuinely reduced by golden PI? By how much / in which channel?' },
    curve_undershoot: { type: 'string', description: 'Explanation for the sharp-curve under-authority.' },
    metric_fix: { type: 'string', description: 'What metric SHOULD be the primary lateral-comfort measure going forward (so we never miss this again)?' },
    confidence_caveats: { type: 'string', description: 'Remaining confounds (route overlap, perception-noise) and what data would settle them.' },
    recommendation: { type: 'string', description: 'Keep golden PI on CD210? Tune further? Address curve undershoot how? Pull b3-b7?' },
  },
}
const synth = await agent(`${CONTEXT}

YOU ARE THE SYNTHESIZER. Reconcile the two investigations below into a clear verdict on WHY the subjective
experience (dramatic straight improvement + sharp-curve under-authority) disagreed with the original aLat
analysis (flat), whether the ping-pong is genuinely reduced, the curve-undershoot cause, and what to do next.
Weigh evidence; don't average. Flag every remaining confound honestly (the location-matched N was tiny).

REVIEWER A (logic+perception):
${JSON.stringify(logic, null, 2)}

REVIEWER B (verify+confound):
${JSON.stringify(indep, null, 2)}

Return structured output.`,
  { label: 'synthesis', phase: 'Synthesis', schema: SYN_SCHEMA })

return { logic, indep, synth }
