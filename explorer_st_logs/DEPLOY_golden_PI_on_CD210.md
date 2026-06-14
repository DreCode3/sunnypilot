# DEPLOY: Restore Apr-8 golden centering authority on CD210 (slow-weave suppression test)

> **⚠️ HISTORICAL (Jun-7 deploy recipe).** The *rationale* here (CD210 causing a 0.1–0.5 Hz slow weave; the
> OPM7-revert fallback) is **SUPERSEDED** — later analysis found the weave is speed-dominated and **no learned-param
> or PI knob was validated as a lever**, and the golden-vs-weak centering benefit is undecidable from observational
> data (see `EXPLORER_ST.md` §1 + `learned_param_studies/RESULTS.md` + `longitudinal_weave/README.md`). The golden-PI
> deploy STEPS below are still accurate as a how-to; the rationale is outdated.

**Goal:** Test whether restoring the strong lane-centering controller (the config active when OPM7 lateral was "awesome", Apr 6-8) suppresses CD210's 0.1-0.5 Hz slow weave. Single variable: same CD210 model, only the 3 PI authority knobs change.

**Why these 3:** golden→today diff (committed May 23 `6ca2e16e8` "revert PI to memory baseline") gutted exactly these. The LaneBiasIntegral warm-start code is still present today (only the gains were reverted), so no other change needed.

**Why it's clean on highway:** the slow weave is on highway straights (40-80 mph) where laneline confidence >0.8 → laneline_scale=1.0 → the buggy `path_offset_position` term has zero weight. So the known sign-convention bug does NOT affect this test (it only bites low-speed surface streets). No sign fix needed first.

---

## The 3 edits (device file: `opendbc_repo/opendbc/car/ford/carcontroller.py`)

| Line | FROM | TO |
|---|---|---|
| 125 | `self.lc_kp = 0.0001` | `self.lc_kp = 0.0005` |
| 345 | `int_cap = 0.3` | `int_cap = float(np.interp(CS.out.vEgoRaw, [20., 30.], [0.3, 1.0]))` |
| 354 | `self.lane_centering_integral *= 0.98  # reverted from 0.995 …` | `self.lane_centering_integral *= 0.995  # GOLDEN restore` |

---

## On-device deploy (DisableUpdates recipe — updater already disabled, single reboot persists)

```sh
# 1. SSH in (single ControlMaster; home wifi shown — swap host/socket per network)
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/sockets/comma-home -o ControlPersist=600 comma@192.168.98.237

# 2. CONFIRM updater is disabled (must print 1; if not, see GOTCHA below)
cat /data/params/d/DisableUpdates; echo

# 3. Locate the file + back it up
F=$(find /data/openpilot -path '*ford/carcontroller.py' | head -1); echo "$F"
cp "$F" "$F.bak_pre_goldenPI"

# 4. VERIFY the 3 target lines are present BEFORE editing
grep -n 'self.lc_kp = 0.0001' "$F"
grep -n 'int_cap = 0.3' "$F"
grep -n 'reverted from 0.995' "$F"

# 5. Apply the 3 edits (unique anchors; .* absorbs the em-dash comments)
sed -i 's|self.lc_kp = 0.0001.*|self.lc_kp = 0.0005  # GOLDEN restore (Apr-8) - CD210 weave test|' "$F"
sed -i 's|int_cap = 0.3|int_cap = float(np.interp(CS.out.vEgoRaw, [20., 30.], [0.3, 1.0]))|' "$F"
sed -i 's|self.lane_centering_integral \*= 0.98  # reverted from 0.995.*|self.lane_centering_integral *= 0.995  # GOLDEN restore|' "$F"

# 6. VERIFY edits took (expect 0.0005, the np.interp cap, and *= 0.995)
grep -n 'self.lc_kp = 0.0005' "$F"
grep -n 'int_cap = float(np.interp' "$F"
grep -n '\*= 0.995  # GOLDEN restore' "$F"
# sanity: the OTHER decays must be UNCHANGED (still 0.97 @352, 0.98 @375/377)
grep -n 'lane_centering_integral \*= 0.9' "$F"

# 7. python-syntax check before reboot (catch a bad edit)
cd /data/openpilot && python -c "import ast,sys; ast.parse(open('$F').read()); print('syntax OK')"

# 8. Graceful reboot (NEVER sudo reboot — causes EPAS alert)
echo -n "1" > /data/params/d/DoReboot
```

**After reboot — verify it persisted + CD210 still active:**
```sh
ssh ... comma@192.168.98.237
F=$(find /data/openpilot -path '*ford/carcontroller.py' | head -1)
grep -n 'self.lc_kp = 0.0005\|int_cap = float(np.interp\|0.995  # GOLDEN' "$F"   # all 3 present = persisted
cat /data/params/d/ModelManager_ActiveBundle 2>/dev/null | head -c 200; echo    # confirm CD210 still loaded
```

**GOTCHA (only if step 2 printed not-1):** the first reboot after enabling DisableUpdates can still revert (a finalized update was staged) — re-apply steps 5-8 and reboot a 2nd time; after that single reboots stick. Per current memory, updater is already disabled, so one reboot should persist.

**ROLLBACK:** `cp "$F.bak_pre_goldenPI" "$F"` then DoReboot.

---

## Drive plan
- Same corridor as b1/b2 (the Madison/Powder-Springs roads). Highway straights are the measurement zone; include both directions if practical.
- Engaged (MADS active) the whole highway stretch — the weave only shows engaged.
- 10-20 min of highway is plenty for the slow-band stats.
- Note subjective: is the center-wander reduced? any new comfort issue (the May-23 revert was triggered by a comfort regression on strong Kp — watch for it)?

## Analysis when logs are back (call it route_b3+)
- Compare route_b3 (CD210 + golden PI) vs b1/b2 (CD210 + weak PI) on shared corridor, engaged straights:
  - **PASS = slow band 0.1-0.5 Hz drops** (toward golden's ~0.06-0.08) while **hunt band 0.5-1.5 Hz stays flat** (control).
  - Tools: `explorer_st_logs/slow_weave_model.py` (add b3 to GROUPS), `model_indep_gated.py`.
- If slow weave does NOT drop (or rises = perception-chasing) → recipe fails on CD210 → fall back to reverting CD210→OPM7 (known no-weave).
