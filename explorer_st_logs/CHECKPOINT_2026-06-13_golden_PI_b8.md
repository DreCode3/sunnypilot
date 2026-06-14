# CHECKPOINT 2026-06-13 — Golden PI on CD210, b8 tested (HANDOFF / start-here for new context)

Read this first, then `memory/finding_slow_weave_cd210.md` for the full reasoning chain.

---

## ⭐ UPDATE 2026-06-13 PM — THE b8 DISCONNECT IS RESOLVED (this supersedes the "NO VALIDATED WIN / speed-confound only" TL;DR below)
Full writeup: `memory/finding_b8_centering_not_oscillation.md`.
- **The prior analysis measured the WRONG AXIS.** `analyze_b8_v2.py` / `verify_speed_confound.py` measured oscillation AMPLITUDE only — `np.std` removes the mean and detrended `slow_rms` removes the trend — so they were mathematically BLIND to CENTERING (mean offset + slow drift). That is exactly where golden PI acts and exactly what the driver felt. Both the analysis ("no oscillation win") and the driver ("hugely better") were right about DIFFERENT axes.
- **Verified (3 scripts + 9-agent adversarial workflow):** felt win = CENTERING not oscillation; WEAK sits persistently LEFT ~−0.08..−0.13 m, GOLD centered ~−0.02 m (both weak routes worse than b8); **controller-attributable** by difference-in-differences vs each drive's own un-engaged manual baseline at 35-50mph = **+0.06 m toward center, SIGNIFICANT** (human drove centered on both drives; the WEAK controller adds the left bias, golden doesn't); oscillation (hunt/rate/reversals) FLAT-to-WORSE at matched speed.
- **Walked back:** magnitude (−71/−76% → ~−67% speed-controlled, ~−22% n.s. strict-paired = power problem); ">0.3 m excursions −77%" REFUTED (speed-tail artifact; mid >0.15-0.20 m −25-32% survives); overrides slightly WORSE (CD210 curve-undershoot, not the PI). The workflow's own "75% present PI-off → road confound" was ALSO overstated (raw cross-drive un-engaged compare, not baseline-differenced) — the DiD refutes it.
- **Decision UNCHANGED: keep golden PI ON** — now a likely-real centering improvement (not just "no harm"); only the HIGHWAY magnitude/attribution stays unproven (no un-engaged highway data to difference). Decisive test below (Option 2) now also wants a **PI-OFF baseline pass in each config** to difference out road/baseline at highway speed.
- New tools: `reassess_extract.py` (50Hz cache), `reassess_analyze.py`, `reassess_centering_verify.py`, `reassess_overrides.py`, `reassess_pi_off_verify.py` (DiD).

---

## TL;DR (where we are)  ⚠️ partially SUPERSEDED — see the ⭐ UPDATE block above; the "felt-better is just a speed confound" reading was incomplete (it missed the centering axis)
- **Root cause of the lateral regression is SOLVED:** the felt "wandering / uncomfortable on a long drive" is a **0.1–0.5 Hz SLOW WEAVE introduced by the CD210 model** — NOT the PI, and NOT the 0.5–1.5 Hz hunt band we tuned for weeks (that band was flat the whole time). Verified, model-independent, backend-clean.
- **We tried to fix it by restoring "golden" PI authority on CD210** (the strong-centering config from the Apr 6–8 OPM7 "awesome" era). It is **DEPLOYED + LIVE on the device now.**
- **The b8 test drive came back INCONCLUSIVE / NO VALIDATED WIN.** Driver felt it was "dramatically better, very confident," but that turned out to be a **speed confound** (b8 driven ~7 mph faster; the apparent −20% steering win flips to +24% *worse* at matched speed — Simpson's paradox). At matched speed golden PI is flat-to-slightly-worse. Not shown to help, not shown to harm.
- **Open decision (user was choosing when we paused):** how to get a *speed-controlled* verdict. Three options below.

---

## DEVICE STATE (192.168.98.237, home wifi)
- **Config LIVE: CD210 model + GOLDEN PI** in `/data/openpilot/opendbc_repo/opendbc/car/ford/carcontroller.py`:
  - line ~125 `self.lc_kp = 0.0005`
  - line ~345 `int_cap = float(np.interp(CS.out.vEgoRaw, [20., 30.], [0.3, 1.0]))`
  - line ~354 off-gate decay `*= 0.995`
  - (warm-start `LaneBiasIntegral` present; was always there)
- **Backup:** `carcontroller.py.bak_pre_goldenPI` (rollback = `cp` it back + DoReboot).
- **`DisableUpdates=1`** (updater off → on-device edits persist across a SINGLE graceful reboot).
- **Model:** CD210 / `C210M` index 54 (ModelManager_ActiveBundle).
- **Reboot:** `echo -n "1" > /data/params/d/DoReboot` (NEVER `sudo reboot` — EPAS alert).
- **SSH gotcha:** key `~/.ssh/id_ed25519` has a passphrase. Run **`ssh-add --apple-load-keychain`** first, then
  `ssh -o ControlMaster=auto -o ControlPath=~/.ssh/sockets/comma-home -o ControlPersist=600 comma@192.168.98.237`.
  Use ONE ControlMaster socket (parallel SSH crashes the device daemon).
- **ALWAYS grep the device config before trusting any test** (we lost drives b3–b7 to an un-applied edit — see below).

## ROUTES (corridor = Powder Springs GA ↔ Madison AL; all on device `/data/media/0/realdata/`)
| Route | Config | Pulled locally? |
|---|---|---|
| b1, b2 | CD210 + **weak** PI | yes (`explorer_st_logs/route_b1`, `route_b2`) — the weak baseline |
| b3–b7 | CD210 + **weak** PI (ran the OLD config — golden edit wasn't applied yet) | **NO** (on device only) |
| b8 | CD210 + **GOLDEN** PI | yes (`route_b8`, 31 seg, 374 MB) — the test drive |
| 7f, 95, 99, 98 | OPM7 + golden PI (May 2–3) | yes — OPM7 "no-weave" reference (slow ≈0.079) |
| a0, 9b, 9d | OPM7 + weak PI | yes — used in the decomposition |

⚠️ **b3–b7 are 5 weak-PI CD210 drives, NOT a golden test.** On 2026-06-13 we found the golden edit had never been
applied (file mtime Jun 7, no backup) so those drives ran the old config. They are extra weak baseline only.

---

## THE OPEN DECISION (pick up here)
How to get a **speed-controlled** verdict on golden PI (the b8 drive couldn't, due to the 7 mph confound):

1. **Pull b3–b7 first (no new driving)** — RECOMMENDED cheapest. Check if any have **57–62 mph, same-direction**
   segments on b8's roads. If yes → run a speed-matched + location-paired comparison from data we ALREADY have,
   possibly settling it with zero driving. (Only the 57–65 mph same-dir windows add info; more 40–57 mph weak data
   just re-confirms the confounded comparison.)
2. **Controlled A/B drive** — gold standard. Same corridor, same direction, **held at a matched cruise speed**
   (e.g. 60 mph) on golden; then revert to weak (`cp` backup + reboot) and drive it again identically. One reboot
   between. Clean speed+location match → definitive.
3. **Keep golden, accumulate** — leave it on (driver likes it, no harm shown), analyze speed-matched as drives pile
   up. Slowest; risk = natural drives stay speed-confounded.

**Provisional decision already made:** KEEP golden PI ON but logged **UNVALIDATED** (not a confirmed win).

## DECISIVE TEST SPEC (whichever path)
- Same corridor, **speed-matched** (hold same target speed both configs), GPS-binned ~55 m cells, same heading,
  multi-pass / interleaved if possible.
- **Primary metric:** duration-weighted **0.1–0.3 Hz steering-angle RMS** + **model lane-position RMS**, speed-
  controlled, robust stats (median / trimmed / sign-test on per-cell ratios).
- **aLat (yawRate×vEgo) = control channel only** (its slow-hi/banking content tracks road, not controller).

## SEPARATE OPEN THREAD — curve undershoot (do NOT bundle with PI)
Driver: "sharp curves wouldn't turn sharp enough, had to take over, but NOT ping-ponging." This is a **CD210
`desiredCurvature` ceiling** (peak ~0.0062 1/m across b1/b2/b8; controller delivers 99.6% of what the model asks),
**not** the PI (PI gates off in curves, |apply_curvature|≥0.005, gold==weak there). If we want sharper curve
authority: model predicted/desired **blend**, **rate limits** on curve entry, or **steerActuatorDelay** (0.25 s) —
each a SEPARATE single-variable experiment.

---

## METRIC DISCIPLINE (institutionalized — non-negotiable from now on)
- **NEVER** use `aLat = yawRate×vEgo` as the primary ping-pong/comfort metric. It is `position''`, so band power
  weights amplitude by `(2πf)²` → ~17× blind to a 0.12 Hz weave. It made the original analysis read "flat" and
  miss the question entirely. Control channel only.
- **ALWAYS speed-control** (speed-matched resample / per-bin / ANCOVA). A 7 mph offset masqueraded as a −20% win.
- **ALWAYS same-physical-location + heading paired** (GPS ~55 m cells), engagement-gated (`carControl.latActive`
  == `selfdriveStateSP.mads.active`; plain `selfdriveState.active` reads 0 under MADS).
- **NEVER pooled mean-of-variance / pooled band-RMS** — outlier+confound dominated (it has now produced THREE
  false signals that flipped under proper control: smooth_tau −55%→+33% under trimming; b8 steer −20%→+24% under
  speed-matching). Use robust stats.
- Use **≥16 s windows** to resolve 0.1–0.5 Hz (a 4 s window's lowest non-DC bin is 0.25 Hz).

## TOOLS BUILT (explorer_st_logs/)
- `slow_weave_model.py` — the slow-weave methodology (16 s windows, engaged, coarse blocks, slow+hunt bands, same-dir, MW + sign test). The template.
- `analyze_b8.py` — b8 vs baseline using aLat (showed "flat" — aLat is the wrong primary metric).
- `analyze_b8_v2.py` — b8 vs baseline with **driver-felt channels** (steeringAngleDeg, model position, commanded curvature). Showed the (confounded) steering drop.
- `verify_speed_confound.py` — proves the steering drop is a speed artifact (speed bins + speed-matched flip).
- `model_indep_ab.py`, `model_indep_gated.py` — the original model-independent same-road comparison (engaged-gated).
- `qa_model_indep.workflow.js`, `qa_b8_disparity.workflow.js` — the cooperative/adversarial/synthesis QA workflows.
- `DEPLOY_golden_PI_on_CD210.md` — the on-device deploy playbook (edits + verify + reboot + rollback).

## METHOD META-LESSON
Three times now a pooled/uncontrolled comparison produced a confident result that REVERSED under proper control
(trimming, speed-matching). The driver's subjective is trustworthy but can't isolate a variable in an
uncontrolled single drive (speed/road/expectation ride along). Verify every agent output independently — Reviewer
A said "confirmed win," Reviewer B caught the speed confound, and I re-verified it myself before believing it.
