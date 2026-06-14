# Controlled A/B Drive Protocol — CD210, weak-PI vs golden-PI

Goal: collect data that the analysis suite (`ANALYSIS_PLAN.md`) can turn into a **definitive** verdict on whether the
golden PI reduces the slow steering/path weave — so you never have to re-drive. Single factor: the lateral PI controller.

| Config | lc_kp | int_cap | off-gate decay | file |
|---|---|---|---|---|
| **A = WEAK** | 0.0001 | fixed 0.30 | 0.98 | `carcontroller.py.weak` |
| **B = GOLDEN** | 0.0005 | interp 0.30→1.00 | 0.995 | `carcontroller.py.golden` |

---

## 1. Corridor & conditions (pick once, keep identical all session)
- **One stretch, ~5–10 miles**, mostly **straights + gentle curves** (the symptom's habitat), with a **safe U-turn/loop at each end** so you can run it **both directions**.
- **Divided highway preferred** (no oncoming-traffic reactions); **light traffic**.
- **Hold ONE speed** with ACC the entire session — set it to **where the weave bites worst for you (~50–55 mph per our data)**. Do not vary it.
- **One session, one day**, ideally a **low-wind** window with stable lighting (no driving into a setting sun half the passes). Note weather.
- **No lane changes** during a pass; **don't tuck behind a lead car** (a lead weaving ahead corrupts the band — those bits get auto-gated, but avoid it).

## 2. One-time device setup (SSH, before driving)
```bash
ssh-add --apple-load-keychain                       # load key once
SSH="ssh -o ControlMaster=auto -o ControlPath=~/.ssh/sockets/comma-home -o ControlPersist=600 comma@192.168.98.237"
$SSH '
  cd /data/openpilot/opendbc_repo/opendbc/car/ford
  cp carcontroller.py carcontroller.py.golden          # current live file IS golden — save it
  cp carcontroller.py.bak_pre_goldenPI carcontroller.py.weak   # the pre-golden backup IS weak
  grep -n "lc_kp = " carcontroller.py.golden carcontroller.py.weak   # sanity: 0.0005 vs 0.0001
  cat /data/params/d/DisableUpdates                    # must be 1 (so edits persist across reboot)
'
```

## 3. Switching config (between blocks only — this is the only reboot)
```bash
# to WEAK:   CFG=weak ;  to GOLDEN: CFG=golden
$SSH "cd /data/openpilot/opendbc_repo/opendbc/car/ford && cp carcontroller.py.$CFG carcontroller.py \
      && echo -n '0.0' > /data/params/d/LaneBiasIntegral \
      && echo -n '1' > /data/params/d/DoReboot"        # zero the persisted integrator, then EPAS-safe reboot
# wait ~60-90 s for boot, then VERIFY the right config is live before driving:
$SSH 'grep -n "lc_kp = " /data/openpilot/opendbc_repo/opendbc/car/ford/carcontroller.py'   # 0.0001=weak / 0.0005=golden
```
> ⚠️ Always `grep`-verify after the reboot — a mis-applied switch wastes a whole block. (The analyzer also proves
> each pass's config from its own telemetry, so a mislabel is caught, but verifying live saves the drive.)

## 4. Integrator reset (every pass)
The PI integrator persists across drives (warm-start). Reset it **before every pass** so each pass starts clean:
```bash
$SSH "echo -n '0.0' > /data/params/d/LaneBiasIntegral"     # do this while parked, just before you pull out
```
(At a config switch this is already done in step 3. For passes within the same block, run this one line.)

## 5. The pass sequence (counterbalanced & interleaved)
A **pass** = one one-way traversal of the corridor. A **loop** = out-and-back = 2 passes (both directions → cancels road crown).
Run **3–4 loops per config (6–8 passes/config)**; **hard minimum 3 loops/config (6 passes)** — fewer and the statistics
*cannot* return a verdict. **Alternate config every loop, and flip the starting config at the halfway point** (counterbalances
time/tire/traffic drift):

```
WARMUP loop (any config) — DISCARD, just to get tires/system to steady state.
Loop 1: A (out, back)      Loop 5: B (out, back)
Loop 2: B (out, back)      Loop 6: A (out, back)
Loop 3: A (out, back)      Loop 7: B (out, back)
Loop 4: B (out, back)      Loop 8: A (out, back)
```
= 8 loops, 16 passes, 8/config, fully interleaved + order-counterbalanced. (Stop after Loop 6 for the 6/config target.)
A config change happens between loops → **one reboot per config change** (~5 reboots for the 8-loop plan).

## 6. Per-pass checklist (repeat each one-way pass)
1. Parked at the corridor end, correct config verified live (§3).
2. **Zero integrator** (§4).
3. Pull out, get to the **set speed**, **engage** (MADS lateral on), confirm engaged.
4. Drive the stretch **hands-light, no override, no lane change, no tailgating**, holding speed.
5. **Log the pass** (see §7) the moment you finish: pass #, config, direction, the segment/route id, clock time.
6. Turn around; repeat for the return pass.

## 7. Logging (you'll build the analyzer's manifest from this)
For each pass record: `config (A/B)`, `direction (out/back)`, **the on-device route id** of that segment, and start time.
After the session, the route ids become the `folder` column of the manifest. Minimal manifest (`passes.csv`):
```
pass_id,folder,driving_model,pi_set_declared,direction,intended_speed_mph
L1_out,explorer_st_logs/route_<id>,CD210,weak,out,55
L1_back,explorer_st_logs/route_<id>,CD210,weak,back,55
L2_out,explorer_st_logs/route_<id>,CD210,golden,out,55
...
```
(`pi_set_declared` is just a label — the analyzer re-derives it from telemetry and the audit fails loudly if they disagree.)

## 8. After the drive — pull & analyze
```bash
# pull the new passes' rlogs (tar-over-ssh; rlog only, not video):
$SSH "cd /data/media/0/realdata && tar cf - <ids>--*/rlog.zst" | tar xpf - -C explorer_st_logs/_newpasses/
# (reorganize into one folder per route id, fill passes.csv, then:)
.venv311/bin/python controlled_test_analysis/code/extract.py --manifest passes.csv --out controlled_test_analysis/cache_test
.venv311/bin/python controlled_test_analysis/code/analyze.py --cache controlled_test_analysis/cache_test --out controlled_test_analysis/results_test
```
The analyzer prints the **execution audit** (must say `AUDIT_OK: true` — config matches, single build, integrator reset,
calibration stable, interleaved), the **realized power** (if it says you're underpowered, you can add a few more loops on a
later day in the SAME corridor — they append cleanly), and the **decision** (`WIN_REDUCES_WEAVE` / `WORSENS_WEAVE` /
`NO_DIFFERENCE_OR_INCONCLUSIVE`).

## 9. Restore after the test
Set the live config back to whatever you want to keep (golden is currently your daily):
```bash
$SSH "cd /data/openpilot/opendbc_repo/opendbc/car/ford && cp carcontroller.py.golden carcontroller.py \
      && echo -n '1' > /data/params/d/DoReboot"
```
(Leave `DisableUpdates=1` until you're done tuning, then re-enable updates to avoid drift.)

---

### Why these specifics (each maps to an audit gate)
- **Same corridor + held speed** → removes the speed/road confounds that defeated every prior observational analysis (the speed-matched stratum only exists if speed is held).
- **Both directions** → cancels road crown/camber (analyzer reports per-direction + both-direction mean).
- **Interleave + counterbalance** → `interleaved` audit check; decouples config from time-of-day/tire/traffic drift.
- **Integrator reset each pass** → `integrator_reset_ok` check; no warm-start carryover from the prior config.
- **Same build, DisableUpdates=1** → `single_build` check; the only thing differing between configs is the PI.
- **No lead car** → the lead-follow gate; a lead's lateral motion lives in the weave band.
- **≥6 (target 8) passes/config** → the permutation floor (≥4) + power; the analyzer’s realized-power readout tells you if you need more.
