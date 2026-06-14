# Session Checkpoint — Iter 3 Deployed, Awaiting Validation Drive

> ⚠️ **CORRECTION (2026-06-07): Iter 3 was NEVER actually deployed.** Verified on device: git HEAD = `5e0785b987` (2026-06-01, the commit *before* iter3 `fc581df928`), and `values.py` Mode 0 smooth_tau = `(0.12, 0.04)` (old). Drives ab–b0 (Jun 6–7) all ran the OLD config (highway tau telemetry ≈0.04, not 0.12). The Step-2 verification below correctly caught this. The status line just below is SUPERSEDED. To genuinely test iter3, push `fc581df928` to the device, reboot, and re-verify highway tau≈0.12 before driving.

**Status as of 2026-06-03 (SUPERSEDED — see correction above)**: Iter 3 (smooth_tau bump (0.12, 0.04) → (0.25, 0.12)) was *believed* deployed and active on the device following graceful reboot. User is about to do validation drive(s). When user returns, pull logs and run the post-deployment analysis (see "When user returns" below).

---

## 1. TL;DR / Where we are

We've been investigating a subjective complaint of "hunting and jerkiness in moderate-to-sharp curves" on a 2021 Ford Explorer ST (CAN/Q3 harness). After two failed prior iterations (Ki bump, steerRatio bump — both reverted), we built a production-faithful pipeline simulator and ran a speed-controlled analysis that:

1. Identified `cmd_band 0.5-3 Hz` as the validated metric (R²_partial = 0.652 with aLat, after speed-control). cmd_cps was rejected (R²_partial = 0.012).
2. Revealed iter2's "jerky" subjective regression was speed-confounded, not controller-confounded — at matched speed bins, iter2 had LOWER aLat oscillation than baseline.
3. Predicted that a stronger EMA smoothing (`smooth_tau (0.25, 0.12)`) would reduce moderate-curve aLat oscillation 15-20% and sharp-curve 25-30% at a +24-49 ms apex lag cost.
4. After 11 QA rounds catching 30+ bugs in earlier methodology, the simulator was approved and the change deployed.

User reported a specific subjectively-bad curve at GPS **33.92615, -84.63050** during route_aa (pre-deployment). That event maps to route_aa event #24: 55 mph left curve, peak |cmd|=0.00291, with elevated cmd_cps (2.20 vs route median 1.55) and elevated aLat_RMS (1.355 vs median 0.60), smooth wheel motion (dAng_std below median). This event class is exactly what iter3 should help — re-drive same curve post-deployment to compare.

---

## 2. When user returns from validation drive — DO THIS

### Step 1: Pull new logs
```bash
# Device should be on home wifi 192.168.98.237; if not, see SSH section in MEMORY.md
ssh -o ControlPath=~/.ssh/sockets/comma-home comma@192.168.98.237 \
  "ls -1 /data/media/0/realdata/ | grep '^000000' | awk -F'--' '{print \$1\"--\"\$2}' | sort -u | tail -10"
# Identify routes after route_aa (probably ab, ac, ...)

# Pull each new route (preserving segment structure):
for r in ab ac ad; do
  mkdir -p explorer_st_logs/route_$r
  rsync -avh --prune-empty-dirs \
    --include="000000${r}--*/" --include="rlog.zst" --exclude='*' \
    -e "ssh -o ControlPath=~/.ssh/sockets/comma-home" \
    "comma@192.168.98.237:/data/media/0/realdata/" \
    "explorer_st_logs/route_$r/"
done
```

### Step 2: VERIFY ITER3 WAS ACTUALLY ACTIVE
**Critical first check** — if highway p4Tau is still ~0.04, iter3 wasn't loaded and the drive is invalid for validation. Should be ~0.12 at highway.

```bash
.venv311/bin/python -c "
import sys, glob, numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'opendbc_repo')
from explorer_st_logs.analyze_curves_v2 import load_cx1, IDX
for r in ['ab', 'ac', 'ad']:  # whatever the new route IDs are
    segs = sorted(glob.glob(f'explorer_st_logs/route_{r}/000000{r}--*/'))
    if not segs: continue
    prefix = segs[0].rstrip('/').rsplit('--', 1)[0]
    arr, t = load_cx1(prefix)
    if arr is None or len(arr) < 100: continue
    v = arr[:, IDX['v']]
    p4tau = arr[:, IDX['p4Tau']]
    hwy = v > 22
    if hwy.sum() > 30:
        print(f'route_{r}: highway tau median = {np.median(p4tau[hwy]):.3f} (expected 0.12 for iter3 active, 0.04 for old)')
"
```

**If p4Tau shows 0.12 → iter3 was active, proceed. If 0.04 → something went wrong; check git state on device.**

### Step 3: Re-run analysis pipeline
```bash
# 3a. Analyze the new routes — detect events, compute metrics
for r in ab ac ad; do
  segdir=$(ls explorer_st_logs/route_$r/ | head -1)
  routeid=$(echo $segdir | awk -F'--' '{print $1}')
  hashpart=$(echo $segdir | awk -F'--' '{print $2}')
  .venv311/bin/python explorer_st_logs/analyze_curves_v2.py \
    "explorer_st_logs/route_$r/${routeid}--${hashpart}" --no-waveforms \
    --output "explorer_st_logs/route_${r}_curves_v2.json"
done

# 3b. Speed-controlled phase1 validation including new routes
.venv311/bin/python explorer_st_logs/phase1_validation.py \
  --routes route_91,route_9b,route_9c,route_9d,route_9e,route_9f,route_a0,route_a1,route_aa,route_ab,route_ac
# Key output: speed-binned aLat_band comparison; partial R² with new data

# 3c. Drill into specific labeled event re-drive
# If user re-drove 33.92615, -84.63050:
d=route_<new>
prefix=$(ls -d explorer_st_logs/$d/000000*--*/ | head -1 | sed 's|--[0-9]*/||')
.venv311/bin/python explorer_st_logs/find_labeled_event.py "$prefix" \
  --lat 33.92615 --lon -84.63050 --window 4
# Compare to route_aa event #24 metrics:
#   - cmd_cps: was 2.20 (should be lower with iter3 — predicted ~1.5)
#   - aLat_RMS: was 1.355 m/s² (target: 15-25% lower = 1.0-1.15)
#   - peak |aLat|: was 1.73 m/s²
#   - dAng_std: was 20.4 deg/s (probably similar or slightly higher with more lag)
```

### Step 4: Check revert criteria
**Revert iter3 if ANY:**
- Override rate per engaged minute > 0.2/min (memory baseline ~0.08/min). Quick check:
  ```python
  # In phase1 analysis: count ovr 0→1 transitions, divide by t_span_sec / 60
  ```
- User subjectively reports "delayed", "lazy", or "rubber-band feel"
- aLat_band 0.5-3 Hz at matched speed bin INCREASED vs route_aa baseline

### Step 5: Compare aLat_band 0.5-3 Hz at MATCHED speed bins
This is THE validation. Per phase1_validation.py output, look at the speed-binned table comparing new (iter3) routes to baseline (route_aa). Predicted reductions:
- Moderate curves (peak |cmd| 0.002-0.004): -15-20% aLat_band
- Sharp curves (peak |cmd| > 0.004): -25-30% aLat_band

If the data shows reductions in this range AT MATCHED SPEED, iter3 is validated. If reductions are near zero or negative, the prediction failed (cmd_band → aLat regression didn't hold in production, or simulator extrapolation was wrong).

### Step 6: After the iter3 decision — REMIND the user about the CD210 model experiment
The user explicitly asked (2026-06-03) to be reminded, once iter3 validation results are in, to run the **CD210 driving-model experiment**. Do iter3 validation FIRST (single variable); THEN surface this.
- **Why it matters**: the driving model's `desiredCurvature` is the documented ROOT cause of curve hunting (4-5 zero-crossings/sec on moderate curves). We currently run **"OP Model 7" (April 03, 2026, on-policy)**; the stable Comma-4 release (release-mici) instead defaults to **"CD210" (January 31, 2026, off-policy)** — community-noted for **"softer steering"**. ⚠️ CD210 is OLDER than our OP Model 7, NOT an upgrade: this is a softer-model A/B for hunting, not a version bump. Selectable as a **RUNTIME swap in the on-device model selector** — no code change (CD210 confirmed present in the live `driving_models_v16.json`).
- ⚠️ Swapping the model **invalidates the current tuning baseline** (smooth_tau/PI/steerRatio/pc_blend were all tuned vs OP Model 7's output). Treat as a SEPARATE experiment, never stacked on iter3.
- **Protocol**: switch to CD210 in the model UI, keep code untouched (esp. `LAT_SMOOTH_SECONDS=0.1`), re-drive labeled curve 33.92615/-84.63050, compare `desiredCurvature` zero-crossings + `cmd_band`→aLat at matched speed bins vs the OP-Model-7 baseline. Adopt only if measurably smoother; re-tune from there if adopted. Do NOT change `common/model.h` baked default until validated.
- Full detail: MEMORY.md "Backend/Curve Update Survey — results (2026-06-03)".

---

## 3. What was deployed (Iter 3 details)

**Change**: Mode 0 `smooth_tau` in `opendbc_repo/opendbc/car/ford/values.py` line 40:
- **WAS**: `'smooth_tau': (0.12, 0.04),  # (low_speed, high_speed) EMA time constant in seconds`
- **IS**: `'smooth_tau': (0.25, 0.12),  # Iter 3 (2026-06-02) — bumped from (0.12, 0.04). Simulator predicts -15-30% aLat oscillation at +24-49ms apex lag. See customizations.md §16.`

**Commits**:
- Mac opendbc: `cd4f8d8602` (`ford Mode 0: smooth_tau (0.12, 0.04) → (0.25, 0.12)`)
- Mac sunnypilot: `fc581df928` (`bump opendbc — Iteration 3`)
- Device opendbc: `4ab2a885` (in-place commit, detached HEAD on iter2-revert lineage)
- Device sunnypilot: `40ffe89a22` (`bump opendbc — Iter 3`)

**Predicted effect** (per simulator + speed-controlled cmd_band → aLat regression):
- Moderate curves: -15-20% aLat oscillation
- Sharp curves: -25-30% aLat oscillation
- Cost: +24-49 ms median apex lag, 1.5-2.5% peak amplitude loss

**Panda safety**: No `ford.h` change → no panda reflash needed. Normal reboot was sufficient.

---

## 4. Pre-deployment baseline data (already analyzed)

8 routes analyzed (90s, a0-aa). Most-recent baseline = route_aa (43 segs, ~13 km, June 3 2026 AM drive). All pre-iter3.

**Per-route metrics summary** (moderate+sharp curves only):
| Route | Label | N events | cmd_cps med | aLat_band med | jerk_RMS med |
|---|---|---|---|---|---|
| 9b | BASELINE_OK | 15 | 1.49 | 12.84 | 0.61 |
| 9c-f | ITER1_Ki=3e-4 | 29 | 1.44 | 7.87 | 0.70 |
| a0-a1 | ITER2_JERKY | 13 | 1.88 | 28.17 | 0.77 |
| 91 | PI-OFF era | 13 | 1.18 | 24.28 | 0.66 |

**Critical context for new-drive comparison**: aLat_band's strong speed dependence (R²=0.521 with speed) means raw cross-route medians are misleading. **Always use speed-binned comparison** (the speed-binned table in phase1_validation.py).

Existing curves_v2 JSON files: `route_{91,9b,9c,9d,a0,a1}_curves_v2.json` (regenerate for ab/ac/ad/etc. as needed).

---

## 5. Analysis tools — what they do, when to use

| Tool | What it does | When to use |
|---|---|---|
| `analyze_curves_v2.py` | Per-event detection + phase metrics, FFT, cross-route compare | First step for any new route |
| `phase1_validation.py` | Speed-controlled correlations, speed-binned aLat compare, partial R² | Validating cmd_band → aLat after a tuning change |
| `pipeline_simulator.py` | Production-faithful 20Hz replay of carcontroller. `--sweep-tau` for what-if | Predicting effects of tau/PI/blend changes before deploy |
| `find_labeled_event.py` | GPS → curve event → full waveform dump | Re-analyzing a specific user-labeled event |
| `check_lane_offset_convention.py` | Dumps raw modelV2 laneLines for sign-convention investigation | Debugging PI/lane-centering behavior |
| `analyze_drive_v6.py` | Drive-level micro-oscillation + smoothness ratio + PI diagnostics | High-level summary of a single drive |
| `planner_trace.py`, `des_source_trace.py` | Earlier-iteration tools; less useful now | Historical reference |

All tools are in `explorer_st_logs/`. Use `.venv311/bin/python` (project venv).

---

## 6. Known open items (do NOT block this validation)

### A. Mixed sign convention in `lane_offset_raw` blending (QA round 12 finding)
- **Location**: `opendbc_repo/opendbc/car/ford/carcontroller.py:329`
- **Issue**: `lane_offset_raw = pos_y_02 * (1 - laneline_scale) + midpoint * laneline_scale`
  - `pos_y_02` from `model.position.y` uses standard OP positive-LEFT convention
  - `midpoint` from `(laneLines[1].y + laneLines[2].y) / 2` uses positive-RIGHT convention (this codebase's unusual choice; verified empirically: left line at -1.6m, right line at +1.5m)
- **Effect**: Verified `corr(midpoint, pos_y@0.2s) = -0.594` over 8471 samples; 87.5% sign disagreement in labeled event. At conf 0.6 they roughly cancel; at high conf midpoint dominates.
- **Impact on iter3 test**: Minimal (labeled event ran at conf ≥0.8, 86% midpoint-dominated). Safe for now.
- **Follow-up**: Investigate after iter3 validation. Fix is straightforward but needs careful sign verification.

### B. Integral cap saturation 30-52% of route time
- **Status**: NOT a bug per QA. Symmetric +/- distribution (verified 0 cap-to-cap transitions <5s across route_aa). It's tracking real slow drifts (lOff P90 ~0.37m on straights, exceeding Ki×cap=6e-5 correction capacity).
- **Impact**: Constant DC-like contribution during curves (~1.3e-4 in labeled event, 5% of curve magnitude). Doesn't explain "coarse corrections" subjective complaint.
- **Worth investigating later**: Ki cap could be raised, but iter1 (Ki=0.0003) didn't subjectively help. Possibly related to (A) above.

### C. Empty CX1 in some recent routes
- route_a6 and route_a7 had no CX1 telemetry data (load_cx1 returned None). Cause unclear. Doesn't affect iter3 validation since post-deployment routes are separate.

---

## 7. Quick reference

### Verify iter3 still active on device
```bash
ssh -o ControlPath=~/.ssh/sockets/comma-home comma@192.168.98.237 \
  "grep -n smooth_tau /data/openpilot/opendbc_repo/opendbc/car/ford/values.py | head -2"
# Should show: 'smooth_tau': (0.25, 0.12),
```

### Revert iter3 (if criteria triggered)
```bash
ssh -o ControlPath=~/.ssh/sockets/comma-home comma@192.168.98.237 "set -e
sed -i \"s|'smooth_tau': (0.25, 0.12),.*|'smooth_tau': (0.12, 0.04),  # reverted Iter 3|\" \
  /data/openpilot/opendbc_repo/opendbc/car/ford/values.py
cd /data/openpilot/opendbc_repo && git add opendbc/car/ford/values.py && \
  git -c user.email=device@comma -c user.name=device commit -m 'revert Iter 3: smooth_tau back to (0.12, 0.04)'
cd /data/openpilot && git add opendbc_repo && \
  git -c user.email=device@comma -c user.name=device commit -m 'bump opendbc — revert Iter 3'
echo -n '1' > /data/params/d/DoReboot"
```

Then sync the revert to Mac:
```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot/opendbc_repo
# edit values.py line 40 back to (0.12, 0.04), commit
# then cd .. && git add opendbc_repo && commit submodule bump
```

### SSH connectivity reminders (from MEMORY.md)
- Home WiFi: `192.168.98.237` (`~/.ssh/sockets/comma-home`)
- Work WiFi: `10.20.10.66` (`~/.ssh/sockets/comma-work`)
- VPN: `10.10.7.236` (`~/.ssh/sockets/comma-vpn`) — needs `-o IPQoS=none`
- **CRITICAL**: Single ControlMaster only; multiple parallel SSH connections crash the device's sshd
- **Reboot**: Always `echo -n '1' > /data/params/d/DoReboot`. Never `sudo reboot` (causes EPAS alerts)

---

## 8. Deeper context (where to look for full history)

- `/Users/dregilley/.claude/projects/-Users-dregilley-Documents-GitHub-sunnypilot/memory/MEMORY.md` — loaded automatically; comprehensive vehicle/tuning state. Also includes a top-level "Upstream Repos & Update Workflow" section (tracked branches: `upstream/dev` for sunnypilot, `bluepilot/bp-6.0` for Comma 4 Ford features)
- `explorer_st_logs/customizations.md`:
  - **§14**: Iteration 1 (Ki bump) and Iteration 2 (steerRatio bump) — both reverted with full QA history and lessons learned
  - **§15**: Pipeline-origin investigation, methodology evolution, simulator development across 8 QA rounds
  - **§16**: Iter 3 deployment — speed-controlled validation, predicted effects, revert criteria
  - **§17**: Upstream repos + surgical-cherry-pick workflow (remote table, fetch commands, what to pull vs leave alone, sanity-check procedure)
- `opendbc_repo/opendbc/car/ford/carcontroller.py:227-470` — the full lateral control pipeline (blend, deadband, EMA, PI, rate limit, safety)
- `opendbc_repo/opendbc/car/ford/values.py:30-65` — Mode 0/1/2 parameter presets

---

## 9. User preferences and working style (from MEMORY.md)

- Prefers quantitative analysis with metrics before/after changes
- Wants to understand safety implications before modifying safety layer
- **Sequential SSH operations only** (no parallel connections — crashes device sshd)
- Uses graceful reboot (DoReboot param), never `sudo reboot`

---

## 10. Verification this checkpoint is complete

A new Claude session should be able to:
- ☑ Know what's currently deployed (iter3, smooth_tau (0.25, 0.12))
- ☑ Know what to do when user returns from validation drive (pull, verify, analyze, decide)
- ☑ Know how to verify iter3 was actually active during the drive (check p4Tau in CX1)
- ☑ Know the revert procedure and criteria
- ☑ Know the labeled bad event GPS coordinates and pre-deployment metrics for comparison
- ☑ Know the open follow-up items (mixed-convention bug, integral saturation) without confusing them with this iter3 test
- ☑ Have pointers to all relevant tools and docs

If anything in this list is unclear or missing, the new session should ask the user before acting.
