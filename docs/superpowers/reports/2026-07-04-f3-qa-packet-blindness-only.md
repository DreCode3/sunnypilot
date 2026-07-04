# F3 QA packet — blindness-only ship (A11 checkpoint, 2026-07-04)

**Ship decision (user, 2026-07-04): Option A — blindness-only.** The low-confidence
("Lane detection lost / Take control") visual alert ships; the departure detector runs in
**shadow mode** (cloudlog telemetry only, no driver-facing event) to collect redesign data.
Deployment (plan Task 12) proceeds ONLY on explicit user approval of this packet.

## 1. What ships (9 commits, branch `stock-2026.002.001-fresh-start`, base `c4d1e66b74`)

| commit | content |
|---|---|
| 461ddd3045 | cereal: `aolLowLaneConfidence @24`, `aolLaneDeparture @25` (append-only) |
| b8df3e1af3 | 11 pure-stdlib monitor unit tests (TDD, committed failing) |
| 5ace596c52 | `AolSafeguardMonitor` — byte-identical reference implementation |
| d13a62fbbd | EVENTS_SP entries, both `AudibleAlert.none` (visual-only; departure entry now dormant) |
| f90b1932e7 | `AolSafeguardDisabled` kill switch registered (default 0 ⇒ safeguard ON; read once at startup) |
| 3cb24acbf6 | selfdrived wiring: 20 Hz monitor step gated on `sm.updated['modelV2']`, cloudlog telemetry, latched 100 Hz event re-add (review fixes vs plan: ISC002 single-line f-strings; alive-guarded latch to stop onroadEventsSP 20/100 Hz flapping) |
| 63e6941c28 / 93647b646d | G1–G4 gate re-pointed at this checkout + NAS; G5 stock census gate (new) |
| defee42a50 | **blindness-only**: only `aolLowLaneConfidence` is ever raised; departure detections stay in cloudlog. `low_conf` (ema<0.30) and `departure` (ema≥0.6) are structurally mutually exclusive, so the shadow branch can never mask a blindness alert |

Diffstat (F3 files): 9 files, +479/−22. No control-path, panda, or longitudinal changes;
no new subscriptions, processes, params reads in the loop, or schema messages.

## 2. Verification evidence

- **Unit tests: 11/11 pass** (`pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py --noconftest -o addopts=""`), re-run independently by the controller.
- **Ship gates (blindness detector) — PASS as pre-registered:**
  - **G2 PASS** — cf blowout (deep-blind 682.21): 22 alerts within [682.21, 684.71].
  - **G3 PASS** — ce blind-in-lane-change: first alert 258.75, before the 260.18 override (2 s maneuver suppression lifted correctly).
- **Shadow-mode characterization (departure detector) — reported, not ship gates:**
  - G1: decisive ce event detected at 212.09, **0.45 s before the override** (pre-registered 1.0 s lead is unachievable: |offset| crossed 0.40 m only ~1 s pre-override and the 0.5 s sustain consumes half; the sanctioned 0.35 m tune reaches only ~0.6 s).
  - G4: 17 episodes on fork-era routes — adjudicated frame-by-frame: 2 driver-caused (steering pressed), 12 genuine blinkerless sweep-and-return excursions (>0.5 m at 0.19–0.62 m/s), 1 sustained drift. Zero data artifacts (modelV2 median dt exactly 0.050 s, zero gaps; no line-proximity backstop misfires).
  - G5: 10 episodes across ALL stock drives (full corpus incl. device-pulled 00/01/03 rlogs).
  - Conclusion: the "expect 0" criteria were never reality-tested (fork plan never executed); brief genuine excursions occur on both builds. Redesign deferred; detector logs shadow data for it.
- **Full-corpus low-confidence census (the expected visual-alert frequency):**

| route | drive min | active min | low-conf eps | alert s | shadow-dep eps |
|---|---|---|---|---|---|
| 00000000 | 38.8 | 0.7 | 2 | 0.5 | 0 |
| 00000001 | 17.9 | 7.6 | 9 | 12.9 | 2 |
| 00000002 | 63.3 | 16.5 | 20 | 57.4 | 6 |
| 00000003 | 7.1 | 0.0 | 0 | 0.0 | 0 |
| 00000004 | 24.9 | 5.1 | 7 | 4.2 | 0 |
| 00000005 | 10.9 | 5.0 | 10 | 9.3 | 2 |
| **total** | **162.9** | **34.9** | **48** | **84.3** | **10** |

  ≈1 brief (~1.8 s avg) visual alert per 45 engaged seconds, concentrated on suburban
  route 02. These are TRUE low-confidence moments by construction. If judged fatiguing
  after the first drives, the sanctioned knob is CONF_QUALIFIER 1.0→1.5 s (Task 10 Rule 3;
  G2/G3 re-run required — both have ≥1.5 s margin).

## 3. Resource budget (measured)

Monitor math 0.35 µs/frame + 6-field modelV2 capnp read 2.53 µs/frame ≈ **2.9 µs/frame on
M4** at 20 Hz; memory ~1.2 KB (one instance, bounded 11-deque); ≈**0.06% of one device core**
at a conservative 10× derate. Pre-F3 device `selfdrived` baseline: **18.5% CPU median**
(procLog, `stock/00000005--ef46fdca62--3`). Post-deploy check: `f3_cpu_check.py`
(acceptance: within +1.0 pp of baseline).

## 4. Deploy plan (Task 12 — runs only on explicit approval)

`DisableUpdates=1` → push branch → device fetch/checkout → `scons` rebuild (capnp enum needs
C++ recompile) → param-reboot → on-device checks (params default, 11 unit tests, alert
construction incl. enum ints 24/25) → first supervised drive → telemetry grep
(`aol_safeguard` in swaglog) → CPU check → ~1 week observation → chime decision (one-line
`AudibleAlert.none` change, later). Rollback: `AolSafeguardDisabled=1` + ignition cycle
(no rebuild), or full `git reset --hard c4d1e66b74` + rebuild.

## 5. Known deviations from the plan text (all reviewed + committed with rationale)

1. Telemetry f-strings single-line (repo ruff bans ISC002 multiline implicit concat).
2. Latched 100 Hz event re-add with `sm.alive['modelV2']` guard (fixes onroadEventsSP duty-cycle flapping; visual behavior identical).
3. Blindness-only ship (user decision after gate adjudication); departure = shadow.
