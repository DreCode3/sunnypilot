export const meta = {
  name: 'qa-model-indep-lateral',
  description: 'QA the model-independent same-road lateral comparison (golden vs weak vs today): cooperative validate, adversarial refute, synthesize',
  phases: [
    { title: 'Cooperative' },
    { title: 'Adversarial' },
    { title: 'Synthesis' },
  ],
}

const CONTEXT = `
PROJECT: 2021 Ford Explorer ST lateral tuning on sunnypilot. We are testing the user's hypothesis that
"lateral performance was better in March/April than now." We built a MODEL-INDEPENDENT same-road comparison
to avoid the trap that lane-offset |A| comes from the driving model's own perception (OPM7 vs CD210 perceive
the lane differently, so |A| is confounded across model changes).

THREE ERAS (all driven on overlapping physical roads, Powder Springs GA <-> Madison AL corridor):
  GOLDEN = strong-PI config (lc_kp 0.0005, int_cap interp 0.3->1.0, curvature_rate_gain 1.15), OPM7 model.
           Routes 7f/95/99/98. THIS WAS THE ACTIVE-TUNING ROAD-TRIP PERIOD (Apr 6-8) -- user was hand-tuning,
           overriding constantly to test/feel.
  WEAK   = weak-PI config (lc_kp 0.0001, fixed int_cap 0.3, rate_gain 1.0), OPM7 model. Routes a0/9b/9d.
           This is the current PI config but with the OLD (OPM7) model.
  TODAY  = weak-PI config + CD210 off-policy model. Routes b1/b2 (driven today, June).

THE SCRIPT UNDER REVIEW: explorer_st_logs/model_indep_ab.py
  - Signals: aLat = liveLocationKalman.angularVelocityCalibrated.value[2] (yawRate) * carState.vEgo.
             override = carState.steeringPressed.
  - Resamples to 20Hz, 4s windows, assigns each window to a ~55m GPS cell (round(lat/0.0005), round(lon/0.0005)).
  - Per window: aLat hunt-band 0.5-1.5Hz RMS (band_rms via rfft), override fraction, speed, median|aLat|.
  - Groups windows by cell, averages within cell, then compares groups on SHARED cells (same physical road)
    with robust median + per-cell sign count + median ratio.
  - Curve cells = max(|aLat|) > 1.0 m/s2; straight cells = both < 0.6.

PRINTED RESULTS (the trailing JSON traceback is post-analysis, numbers are valid):
  cells: GOLDEN 340  WEAK 992  TODAY 201
  ALL cells:    GOLDEN->WEAK(227c) hunt 0.038->0.040(+5%) ovr 13.2%->10.7% | GOLDEN->TODAY(165c) hunt 0.040->0.039(-3%) ovr 15.8%->9.8% | WEAK->TODAY(160c) hunt 0.044->0.039(-12%) ovr 15.5%->11.1%
  CURVE cells:  GOLDEN->WEAK(19c) hunt 0.191->0.171(-10%) ovr 50.0%->25.1% | GOLDEN->TODAY(21c) hunt 0.152->0.169(+11%) ovr 45.8%->24.2% | WEAK->TODAY(21c) hunt 0.171->0.169(-1%) ovr 22.9%->19.8%
  STRAIGHT:     GOLDEN->WEAK(195c) hunt 0.033->0.037(+10%) ovr 7.9%->8.5% | GOLDEN->TODAY(138c) hunt 0.035->0.036(+4%) ovr 10.3%->6.1% | WEAK->TODAY(129c) hunt 0.040->0.036(-9%) ovr 12.5%->8.3%

RELATED PRIOR OUTPUTS (for cross-checking):
  - explorer_st_logs/golden_vs_today.py used MODEL-DEPENDENT |A| lane offset and found TODAY +17% worse centering
    (median cell). That metric is model-confounded (OPM7 vs CD210) -> NOT trusted as real-world centering.
  - gps_matched_ab.py found on smooth_tau b1->b2: hunt 0.5-1.5Hz +8% but SLOW band 0.1-0.5Hz +26%.

HARD-WON METRIC LESSONS (must apply):
  - A_cps / crossings-per-sec metrics are SATURATED (HF noise), DO NOT USE.
  - POOLED mean-of-variance band-RMS is OUTLIER-DOMINATED: a few rough cells flipped a smooth_tau result from
    -55% to +33% when trimmed. ONLY trust SAME-PHYSICAL-LOCATION paired cells + ROBUST stats (median / trimmed /
    sign test). The script uses median + sign count -- check it actually does, and whether N is adequate.
  - The felt complaint is "hunting/wandering, uncomfortable on a long drive." That could be SLOW weave
    (0.1-0.5Hz, period 2-10s) rather than the 0.5-1.5Hz hunt band the script measures.

TENTATIVE CONCLUSIONS TO STRESS-TEST:
  C1: Model-independent physical lateral oscillation (aLat hunt) is statistically FLAT across golden/weak/today
      -> no measurable lateral regression; the "April was better" feeling is not in the motion data.
  C2: Override rate is HIGHEST in golden (~2x today in curves) -> if anything golden was worse, BUT this is
      likely contaminated because golden = active hand-tuning period.
  C3: The script NEVER gates on openpilot engagement -- steeringPressed and aLat are counted whether or not OP
      was engaged. During golden tuning the user drove/intervened manually a lot -> override AND hunt may
      reflect the human, not the system.
`

phase('Cooperative')
const COOP_SCHEMA = {
  type: 'object',
  required: ['checks', 'numbers_reproduce', 'bugs_found', 'overall'],
  properties: {
    checks: { type: 'array', items: { type: 'object', required: ['item', 'status', 'detail'],
      properties: { item: { type: 'string' }, status: { type: 'string', enum: ['ok', 'concern', 'broken'] }, detail: { type: 'string' } } } },
    numbers_reproduce: { type: 'string', description: 'Did an independent targeted recompute land in the same ballpark? Give the numbers you got.' },
    bugs_found: { type: 'array', items: { type: 'string' } },
    overall: { type: 'string', description: 'Is the methodology sound enough to base conclusions on?' },
  },
}
const coop = await agent(`${CONTEXT}

YOU ARE THE COOPERATIVE REVIEWER. Goal: validate that the script computes what it claims and the numbers are real.
Working dir is the repo root. The venv with capnp/openpilot libs: use \`./.venv311/bin/python\` or whatever the
other explorer_st_logs scripts use (check how they're invoked).

DO:
1. Read explorer_st_logs/model_indep_ab.py carefully. Verify: band_rms (rfft Parseval normalization), the 20Hz
   resample, 4s windowing, GPS cell assignment, curve/straight thresholds, the median + sign-count + ratio stats,
   and the override interpolation (np.interp on a 0/1 signal then >0.5 -- is that sound?).
2. Do ONE small INDEPENDENT recompute to confirm a load-bearing number (do NOT re-run the whole script -- too slow).
   Pick ONE golden route (e.g. route_7f) and ONE today route (route_b1). Load liveLocationKalman yawRate*vEgo for a
   handful of overlapping GPS cells and confirm the aLat hunt-band magnitudes are in the ~0.03-0.04 (straight) /
   ~0.15-0.19 (curve) ballpark the script reports. Confirm cell-matching actually finds shared physical cells.
   Keep it cheap: one or two segments is enough to sanity the magnitudes.
3. Report any real bugs (not nitpicks) that would change the conclusions.

Return structured output. Be honest: if it's sound, say so; if a bug invalidates a conclusion, say which.`,
  { label: 'coop:validate', phase: 'Cooperative', schema: COOP_SCHEMA })

phase('Adversarial')
const ADV_SCHEMA = {
  type: 'object',
  required: ['target_conclusion', 'refutations', 'strongest_objection', 'verdict'],
  properties: {
    target_conclusion: { type: 'string' },
    refutations: { type: 'array', items: { type: 'object', required: ['claim', 'attack', 'severity', 'evidence'],
      properties: { claim: { type: 'string' }, attack: { type: 'string' },
        severity: { type: 'string', enum: ['fatal', 'major', 'minor'] }, evidence: { type: 'string' } } } },
    strongest_objection: { type: 'string' },
    verdict: { type: 'string', enum: ['conclusions_hold', 'conclusions_need_revision', 'conclusions_refuted'] },
  },
}
const [advStats, advValidity] = await parallel([
  () => agent(`${CONTEXT}

YOU ARE ADVERSARIAL REVIEWER A (statistics & confounds). Your job is to REFUTE conclusions C1, C2, C3.
Attack hard, but only with attacks you can substantiate. Focus areas:
  - ENGAGEMENT GATING: the script counts steeringPressed and aLat regardless of whether OP was engaged. Quantify
    how badly this distorts override (esp. golden, the tuning period). If you can, load carControl.latActive (or
    controlsState.enabled / selfdriveState.active -- find the right field for these logs) for ONE golden route and
    estimate what fraction of "override" windows were actually disengaged/manual. Cheap spot-check only.
  - SMALL N: curve comparisons rest on 19-21 cells. Is that enough for the +11%/-10%/-1% hunt deltas to mean
    anything? Are the deltas within noise? Is the sign-inconsistency across pairs (GOLDEN->TODAY +11% but
    WEAK->TODAY -1%) evidence the hunt metric is just noise?
  - WRONG BAND: is the felt "wandering, uncomfortable on long drive" actually in the SLOW band 0.1-0.5Hz (smooth_tau
    showed +26% there) rather than 0.5-1.5Hz? If the script measures the wrong band, C1 ("flat") could be hiding a
    real slow-weave regression. Argue this.
  - OUTLIER/ROBUSTNESS: even median can mislead with N=20. Would trimmed-mean or sign-test p-values change the read?
  - Anything else that breaks C1/C2/C3.

Use ./.venv311/bin/python for any spot-check. Return structured output.`,
    { label: 'adv:stats', phase: 'Adversarial', schema: ADV_SCHEMA }),
  () => agent(`${CONTEXT}

YOU ARE ADVERSARIAL REVIEWER B (cross-era validity & same-road integrity). REFUTE the premise that this is a
fair apples-to-apples comparison. Focus areas:
  - CROSS-ERA SENSOR COMPARABILITY: golden is April, today is June. Is liveLocationKalman.angularVelocityCalibrated
    (the yawRate feeding aLat) computed the same way across that span? There was an Apr-25-era locationd/livePose
    backend merge and cherry-picked commits (cam-odo delay #37543, livePose timestamp #37704, locationd filter
    time #37697). Could these shift yawRate magnitude/phase between eras, contaminating a "model-independent"
    metric? Check git log of the locationd/locationd-related files between April and June if you can.
  - SAME-ROAD / SAME-DIRECTION: cells are round(lat,lon) to ~55m but ignore HEADING. On a divided highway the two
    carriageways can fall in the same cell while being different physical lanes/directions; curves differ by
    direction. Does this confound the comparison? How much?
  - SPEED MATCHING: cells are matched by location but is speed matched within cell? A speed mismatch changes aLat
    for the same curvature. Check whether matched cells have similar speeds.
  - ENGAGED + CONFIG PROVENANCE: are golden routes 7f/95/99/98 truly strong-PI OPM7 and actually ENGAGED while
    driving these cells? Are b1/b2 truly CD210? If you can cheaply verify model/config from the logs, do.
  - GROUP IMBALANCE: WEAK has 992 cells, TODAY 201, GOLDEN 340 -- does the uneven coverage bias the shared-cell
    medians?

Use ./.venv311/bin/python for any spot-check. Be concrete. Return structured output.`,
    { label: 'adv:validity', phase: 'Adversarial', schema: ADV_SCHEMA }),
])

phase('Synthesis')
const SYNTH_SCHEMA = {
  type: 'object',
  required: ['conclusions', 'reanalysis_needed', 'bottom_line'],
  properties: {
    conclusions: { type: 'array', items: { type: 'object', required: ['id', 'status', 'reason'],
      properties: { id: { type: 'string' }, status: { type: 'string', enum: ['survives', 'weakened', 'refuted', 'unprovable_as_is'] }, reason: { type: 'string' } } } },
    reanalysis_needed: { type: 'array', items: { type: 'object', required: ['action', 'why'],
      properties: { action: { type: 'string' }, why: { type: 'string' } } } },
    bottom_line: { type: 'string', description: 'In 2-4 sentences: did April lateral actually regress, and is the lever the model or the PI?' },
  },
}
const synth = await agent(`${CONTEXT}

YOU ARE THE SYNTHESIZER. You have the cooperative review and TWO adversarial reviews below. Reconcile them into a
final verdict. Do NOT just average -- weigh evidence. If an adversarial attack is substantiated, let it stand;
if it's speculation the cooperative recompute already addressed, say so. Decide for each conclusion C1/C2/C3
whether it survives, is weakened, refuted, or is unprovable with the current script. Then list the concrete
re-analyses that would actually settle it (e.g., engaged-gated re-run, slow-band 0.1-0.5Hz, heading-split cells,
speed-matched cells). Finish with a 2-4 sentence bottom line on whether April was genuinely better and whether
the real lever is the MODEL (CD210/OPM7) or the PI config.

COOPERATIVE REVIEW:
${JSON.stringify(coop, null, 2)}

ADVERSARIAL A (stats/confounds):
${JSON.stringify(advStats, null, 2)}

ADVERSARIAL B (validity):
${JSON.stringify(advValidity, null, 2)}

Return structured output.`,
  { label: 'synth', phase: 'Synthesis', schema: SYNTH_SCHEMA })

return { coop, advStats, advValidity, synth }
