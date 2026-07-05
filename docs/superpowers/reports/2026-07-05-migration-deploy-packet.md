# Deploy Packet — stock-2026.002.001-agnos18 (sunnypilot v2026.002.001 + AGNOS 18.4)

**Date:** 2026-07-05 · **Status:** DESK PHASES COMPLETE — awaiting user go for device deploy
**Branch:** `stock-2026.002.001-agnos18` (43 commits over pure stock `31dc4d8e52`)
**Supersedes:** `stock-2026.002.001-fresh-start` @ `bbf6bd7f38` (currently on device @ `f3fcd668a`)

---

## 0. ⚠️ Finding you must know before driving the CURRENT build

The deployed fresh-start branch is **fork-lineage, not stock-based**: it pins the **fork
opendbc** (`8fd74f47`, the 885-line carcontroller **with the golden PI / LaneBiasIntegral**),
and its main-repo base is April-2026-era master (hence AGNOS 17.2). Verified live on the
device 2026-07-05: `/data/openpilot/opendbc_repo` is at `8fd74f47`; carcontroller contains
the PI. **The pending F3 shakedown drive would NOT have been on stock lateral.**

- No drives have occurred on this build (F3 shakedown was still pending) → no contaminated data.
- All desk validation (toolkit gates, SP002 anchor, centering RCA, S1) used NAS rlogs from the
  true stock flash (v2026.002.000 release) → unaffected.
- This packet's branch **fixes it permanently**: pure stock base, stock opendbc pin `b9712d20e`.
- **Recommendation: do not drive engaged on the current build; deploy this packet's branch first.**

## 1. What the new branch is

- Base: `31dc4d8e52` = sunnypilot master v2026.002.001 (pins AGNOS 18.4, opendbc `b9712d20e`,
  tinygrad `ac1632ab9`, panda `d994e8e80`).
- Construction: 1 import commit (analysis trees from the fork boundary: `model_replay_sim/`,
  `retrospective_lateral/`, `docs/superpowers/`, merged `.gitignore`) + 42 cherry-picks of the
  fresh-start program work. Dropped as obsolete: explorer_st_logs mirror removal; fonts raylib-6
  + loggerd libva fixes (both verified already upstreamed at .001).
- **Code delta vs pure stock = F3 only, +325/−0 lines across 7 files**: `cereal/custom.capnp`
  (+2 enum: aolLowLaneConfidence @24, aolLaneDeparture @25), `common/params_keys.h`
  (+1: AolSafeguardDisabled — fork keys NOT carried), `selfdrive/selfdrived/selfdrived.py` (+60
  wiring), `sunnypilot/selfdrive/selfdrived/aol_monitor.py` (new), `.../events.py` (+16),
  tests (+145). Blindness-only ship state intact (departure = shadow/cloudlog).
- Program tree (toolkit, sim, centering, docs) **bit-identical** to the fresh-start branch.
- Key identity proof: the v2026.002.000 release that produced all stock drive logs is
  **Python/schema-identical** to this base (model engine, selfdrived, capnp, services all empty
  diffs; release differs only in packaging). Model weights identical too (in-tree ONNX LFS oids
  `ee29ee5b…`/`78477124…` = SP002 = CD210). ⇒ desk results transfer by construction; gates below
  re-verify empirically.

## 2. Verification & gate evidence (all re-run on this branch, 2026-07-05)

| Check | Result | Reference |
|---|---|---|
| Port mechanical integrity | ✅ range-diff: all picks content-equal; program tree bit-identical; code delta = F3 only | this session |
| Semantic verify (4 independent lenses + adjudication) | ✅ 0 blockers; 2 concerns dispositioned (historical scripts point at migrated data; adversarial model-mismatch premise refuted by weight-oid identity) | workflow wf_7783449d |
| VM build gauntlet (Ubuntu, uv 3.12) | ✅ core build clean (font-gen fails headless only, expected) | /tmp/port_gauntlet_*.log on devvm |
| Test gauntlet | ✅ **1,426 passed + 10,004 safety subtests, 0 failed** — Ford safety+MADS 174p/9120s, Honda control 760p, modeld_v2 91p, monitoring 18p, cereal 339p + upstream wire-compat OK, params 13p | VM run 2026-07-05 |
| F3 unit tests | ✅ 11/11 (VM and local) | |
| F3 G1–G4 replay gate | ✅ regression EXACT match to accepted 2026-07-04 packet: G2 PASS, G3 PASS; G1 first-alert 212.09 (0.45 s pre-override) and G4 count 17 — both departure-side, shipped as SHADOW per user decision | f3_replay_check.py output |
| F3 G5 census | ✅ per-route EXACT match (02: 6 dep eps, 20 lowconf/57.4 s; 04: 0, 7/4.2 s; 05: 2, 10/9.3 s). New NAS rlog coverage (routes 00/01/03, were qlog-only) adds 2 dep eps on route 01 + census 00: 2/0.5 s, 01: 9/12.9 s — new data, not behavior change. Departure = shadow. | f3_stock_census.py output |
| R2 calibration (pinned 3-drive R2_DRIVES) | ✅ EXACT match: FPR 0.0 (max 0.07), power 0.95@0.03 m / 1.0@0.06 m, 54 windows | r2_calibration.py output |
| R2 crown | ✅ EXACT match: p 0.4046, not significant; slope −1.288, crown component −0.024 m | r2_crown.py output |
| SP002 sim anchor on route_stock05 (pkls recompiled under .001 tinygrad `ac1632ab9`) | ✅ **PASS: corr 0.9986090, band_ratio 0.99292, n 2059/2059** — identical to the 2026-07-03 reference (0.9986 / 0.993); the tinygrad version change is numerically transparent. (Anchor JSON's `tinygrad_sha` field reads the 2026-07-03 materialization provenance.json — stale label, not the runtime compiler.) | scratchpad/anchor_sp002_agnos18.json |
| S1 CameraOffset sweep → δ* re-derivation (fresh full re-run of all 33 points, new tinygrad; old results preserved as `s1_ref_2026-07-04`) | ✅ **ALL GATES PASS: δ* = −0.12 m** (raw −0.11538; slope −0.825 vs ref −0.824; ρ −1.000; determinism repeat delta 0.00e+00; d_center@δ* +0.102 m ≈ vehicle-frame P). Road A/B arms 0 vs −0.12 stand. | results/s1/s1_report.{json,md} |

Sim-layer note: the compiled sim pkls (tinygrad JIT artifacts) from 2026-07-03 do NOT load under
.001's tinygrad (`Ops.UNIQUE` unpickle assertion) — exactly the anticipated "old-format bundles
won't load," at the artifact layer. SP002 pkls recompiled; CD210/Nevada/OPM7 artifacts remain
old-format (`.tinygrad3501.bak`) — recompile before any future cross-model run.

## 3. Deploy runbook (on explicit user go — no step before it)

Pre-push:
1. Local: confirm branch head + clean tree; tag `pre-migration-2026-07-05` on the fresh-start
   branch for reference.
2. Push `stock-2026.002.001-agnos18` to `origin` (DreCode3/sunnypilot). LFS objects resolve from
   the absolute gitlab endpoint in `.lfsconfig` — fork remote is fine.

Device (single ControlMaster SSH; poll ≤60 s; never leave a step unmonitored):
3. Preconditions: `DisableUpdates=1` confirmed (currently set ✓); note current state for rollback
   (branch `stock-2026.002.001-fresh-start` @ `f3fcd668a`, opendbc `8fd74f47`, AGNOS 17.2).
4. `cd /data/openpilot && git fetch origin stock-2026.002.001-agnos18 && git checkout` it;
   `git submodule sync`.
5. **MANDATORY submodule sweep** (release→dev transition gotchas):
   a. Remove stale submodule dirs before init (`opendbc_repo`, `msgq_repo`, `teleoprtc_repo`,
      `panda`, `rednose_repo`, `tinygrad_repo`, `sunnypilot/neural_network_data` as needed).
   b. `git submodule update --init --recursive --force`.
   c. tinygrad pin `ac1632ab9` lives on `refs/pull/2/head` of sunnypilot/tinygrad — if the SHA
      fetch fails: `git fetch origin refs/pull/2/head` inside the submodule first.
   d. **Sweep EVERY submodule worktree for non-empty content** — especially
      `sunnypilot/neural_network_data` (empty ⇒ card.py FileNotFoundError at car start,
      "sunnypilot unavailable"). Verify `sunnypilot/neural_network_data/neural_network_lateral_control`
      exists and opendbc carcontroller is the 202-line stock file (no LaneBiasIntegral).
6. Reboot via `echo -n "1" > /data/params/d/DoReboot` (never `sudo reboot`).
7. **First boot flashes AGNOS 18.4** (~1 GB download + reboot cycle). Expect extended downtime;
   poll at ≤60 s. Then scons rebuild runs, including **on-device driving-model compile**
   (driving_tinygrad.pkl from in-tree ONNX, QCOM; plus dm_warp) — the slowest step, expect
   15–30+ min on first build. Do not interrupt.
8. Param note: only param-level reboot needed beyond the above (no /data/params wipe — learned
   state preserved).

Post-boot verification (all before calling it done):
- `/VERSION` = 18.4; `git -C /data/openpilot log -1` = pushed head; `DisableUpdates` = 1.
- `git -C /data/openpilot/opendbc_repo rev-parse HEAD` = `b9712d20e…`; carcontroller stock.
- All submodule worktrees non-empty; NNLC dir present.
- `selfdrive/modeld/models/driving_tinygrad.pkl.chunk*` exist; modeld + dmonitoringmodeld start;
  no crash loops in `/data/community/crashes` (or equivalent); onroad UI comes up with F3 alerts
  registered (AolSafeguardDisabled param readable, default 0).

Rollback: `git checkout stock-2026.002.001-fresh-start` + submodule update + DoReboot-reboot
(AGNOS re-downgrade will trigger — accept), or re-flash stock release. Device params untouched
either way. Note rollback returns to the fork-opendbc build (§0) — engaged driving not recommended there.

## 4. Road phases after deploy (user)

1. **F3 shakedown drive** on the new build; then `f3_cpu_check.py` on a post-drive rlog
   (selfdrived CPU budget vs the 18.5% baseline).
2. **Centering road A/B** per `docs/superpowers/plans/2026-07-03-centering-road-ab-protocol.md`,
   arms 0 vs **δ* = −0.12 m** (re-derived on this branch 2026-07-05; identical to the protocol's value).
