# F3 Alert-Only AOL Safety Net — Stock Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 2026-07-01-incident-designed AOL blindness + lane-departure detectors to stock sunnypilot as a pure safety net: **visual-only alerts** (`AudibleAlert.none`), **no degrade/disengage branch** (dropped, not deferred), a registered kill-switch param, `cloudlog` telemetry on the existing log stream, and **five offline replay gates (G1-G5) that must all pass before any device deployment**. Authoritative spec: `docs/superpowers/specs/2026-07-03-f3-alert-only-and-centering-workflow-design.md` (Workflow 1). Reference implementation source: `docs/superpowers/plans/2026-07-02-aol-incident-fixes-f3-f2-ab.md` lines 224-326 (copied verbatim into Task 4).

**Architecture:** `AolSafeguardMonitor` is a pure Python class (stdlib only — one `collections.deque`) living at `sunnypilot/selfdrive/selfdrived/aol_monitor.py`. It is stepped inside `SelfdriveD.update_events` **only when `sm.updated['modelV2']`** (selfdrived's loop is 100 Hz, modelV2 is 20 Hz — the monitor's constants assume `dt=0.05`). It returns `(low_confidence_alert, departure_alert)` booleans; selfdrived raises the corresponding `EventsSP` events (`aolLowLaneConfidence` / `aolLaneDeparture`, new `custom.capnp` enum values @24/@25). Both events are `ET.WARNING`-only, which MADS surfaces under pure AOL (verified: `sunnypilot/mads/state.py:132-133` appends `ET.WARNING` to the selfdrive state machine's `current_alert_types` whenever MADS is active; `selfdrive/selfdrived/selfdrived.py:522` then builds SP alerts from that list). Alerts are 0.2 s duration re-raised every monitor frame ⇒ continuous while the condition holds. Telemetry is a compact `cloudlog.info` string on fire/clear edges + 1 Hz while active — rides the existing `logMessage` path, ~zero bytes on clean drives. Offline gates replay NAS rlogs through the exact committed monitor class.

**Tech Stack:** Python 3.11 (`/Users/dregilley/Documents/GitHub/sunnypilot/.venv311`), pytest (isolated: `--noconftest -o addopts=""` — this venv lacks `pytest-xdist` which the repo `addopts` demands, and `msgq`/`params_pyx` are not compiled so `events.py`/`selfdrived.py` cannot be *imported* here, only `py_compile`d; the functional gates are the replay harnesses which need only `LogReader` + the pure monitor — both verified importable in this venv this session). Repo: `/Users/dregilley/Documents/GitHub/sunnypilot`, branch `stock-2026.002.001-fresh-start` (all commits land here; nothing is pushed until the user-gated deploy task). Log corpus: NAS SMB mount `/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/` (all six fork-era route dirs verified present this session: `route_ce` 18 rlogs, `route_cf` 18, `route_c5` 13, `route_c7` 36, `route_b5` 43, `route_7f` 18 — each laid out `<route>/<segdir>/rlog.zst`; stock drives under `stock/` — routes `00000002` 64 rlogs, `00000004` 26, `00000005` 12; routes `00000000`, `00000001`, `00000003` are **qlog-only on the NAS** and are skipped-with-report by G5). Resource budget (MEASURED 2026-07-03, cited from the spec): monitor math 0.35 µs/frame + 6-field modelV2 capnp read 2.53 µs/frame ≈ **2.9 µs/frame on M4**, memory ~**1.2 KB** (one instance, bounded 11-deque); ≈0.06% of one device core at a conservative 10× derate. Deploy checklist includes a before/after `selfdrived` CPU% check (pre-F3 stock baseline measured this session from procLog: **18.5% median** on `stock/00000005--ef46fdca62--3`).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `cereal/custom.capnp` | Modify | Append `aolLowLaneConfidence @24;` and `aolLaneDeparture @25;` to `OnroadEventSP.EventName` (currently ends `e2eChime @23;` at line 344) |
| `sunnypilot/selfdrive/selfdrived/aol_monitor.py` | Create | The pure `AolSafeguardMonitor` class — verbatim reference implementation (constants: CONF_EMA_TAU=0.5 s, CONF_THRESHOLD=0.30, CONF_QUALIFIER=1.0 s, MANEUVER_BLIND_GRACE=2.0 s, DEPART_CONF_MIN=0.6, DEPART_OFFSET=0.40 m, DEPART_RATE=0.15 m/s, DEPART_SUSTAIN=0.5 s, LINE_DIST_MIN=0.35 m, MANEUVER_DEPART_GRACE=3.0 s, MIN_SPEED=10 m/s, RATE_WINDOW=11) |
| `sunnypilot/selfdrive/selfdrived/tests/__init__.py` | Create | Empty package marker so pytest collection is unambiguous |
| `sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py` | Create (Test) | 11 pure-stdlib unit tests: synthetic 20 Hz sequences covering blindness qualifier timing, maneuver blind-grace, departure rate+sustain, approach-to-center non-fire, line-dist backstop, reset semantics, post-maneuver departure grace |
| `sunnypilot/selfdrive/selfdrived/events.py` | Modify | Two `EVENTS_SP` entries: ET.WARNING only, `AudibleAlert.none`, `VisualAlert.steerRequired`, `AlertStatus.userPrompt`, `AlertSize.mid`, `Priority.MID`, duration 0.2 s (inserted after the `laneTurnRight` entry at lines ~203-209) |
| `common/params_keys.h` | Modify | Register `{"AolSafeguardDisabled", {PERSISTENT, BOOL, "0"}}` (typed-struct format verified; MUST be registered before any read — unregistered keys crash writes / silently return None) |
| `selfdrive/selfdrived/selfdrived.py` | Modify | Import + instantiate the monitor, read the kill switch once at startup, step the monitor in `update_events` gated on `sm.updated['modelV2']`, raise the SP events, emit cloudlog telemetry |
| `retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py` | Modify | G1-G4 acceptance gate: re-point the committed fork-era harness — monitor now imports from THIS checkout (no worktree, no importlib hack), corpus now on the NAS |
| `retrospective_lateral/incident_2026_07_01/scripts/f3_stock_census.py` | Create | G5 acceptance gate (new): zero departure alerts across ALL stock drives with rlog coverage + low-confidence episode census (expected visual-alert frequency) |
| `retrospective_lateral/incident_2026_07_01/scripts/f3_cpu_check.py` | Create | Deploy verification: `selfdrived` CPU% from procLog, pre-F3 stock baseline vs first post-deploy drive |

---

## Task 1: Preflight — branch, NAS mount, venv sanity

**Files:** none (environment checks only)

- [ ] **Step 1: Confirm the working branch and record the starting commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git branch --show-current && git log --oneline -1
```

Expected: `stock-2026.002.001-fresh-start` and the current tip (was `c370480384 spec: F3 alert-only safety net + centering ...` when this plan was written). Record the tip hash — it is the device rollback point in Task 12.

- [ ] **Step 2: Confirm the NAS corpus is mounted with the expected routes**

```bash
for r in route_ce route_cf route_c5 route_c7 route_b5 route_7f; do
  echo -n "$r: "; ls "/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/$r"/*/rlog.zst 2>/dev/null | wc -l
done
ls "/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock/" | sed 's/--[0-9]*$//' | sort -u
```

Expected rlog counts: `route_ce: 18`, `route_cf: 18`, `route_c5: 13`, `route_c7: 36`, `route_b5: 43`, `route_7f: 18`. Expected stock route IDs: `00000000--c181384b0c`, `00000001--e80958d9ca`, `00000002--5da5840d8d`, `00000003--b37c613b41`, `00000004--3d3385646d`, `00000005--ef46fdca62` (plus any drives pulled since 2026-07-03). If the mount is absent, mount SMB share `RAID_6_HDD` on `datacore.local` first (creds in memory `reference_nas_storage.md`).

- [ ] **Step 3: Confirm the analysis venv can drive the gates**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && \
PYTHONPATH=$PWD .venv311/bin/python -c "from cereal import custom; print('capnp OK:', int(custom.OnroadEventSP.EventName.e2eChime))" && \
PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python -c "from openpilot.tools.lib.logreader import LogReader; print('LogReader OK')"
```

Expected: `capnp OK: 23` and `LogReader OK` (both verified working this session). If capnp fails, run `git submodule update --init --recursive` and retry.

---

## Task 2: New event enum values in cereal

**Files:**
- Modify: `cereal/custom.capnp` (enum `OnroadEventSP.EventName` ends `e2eChime @23;` — verified line 344 this session; re-confirm with the grep in Step 1)

- [ ] **Step 1: Locate the enum tail**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && grep -n "e2eChime @23;" cereal/custom.capnp
```

Expected: exactly one hit at line ~344.

- [ ] **Step 2: Append the two entries (append-only; never renumber existing values)**

In `cereal/custom.capnp`, change:

```capnp
    speedLimitPending @22;
    e2eChime @23;
  }
```

to:

```capnp
    speedLimitPending @22;
    e2eChime @23;
    aolLowLaneConfidence @24;
    aolLaneDeparture @25;
  }
```

- [ ] **Step 3: Verify the enum loads (pycapnp parses the .capnp at import — no scons needed on the Mac)**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD .venv311/bin/python -c "from cereal import custom; print(int(custom.OnroadEventSP.EventName.aolLowLaneConfidence), int(custom.OnroadEventSP.EventName.aolLaneDeparture))"
```

Expected: `24 25`. (On the DEVICE this enum change requires a `scons` rebuild — that is Task 12 Step 5; C++ consumers compile the schema.)

- [ ] **Step 4: Commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add cereal/custom.capnp && git commit -m "cereal: add aolLowLaneConfidence/aolLaneDeparture OnroadEventSP events" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Monitor unit tests — failing first (TDD)

**Files:**
- Create: `sunnypilot/selfdrive/selfdrived/tests/__init__.py` (empty)
- Test: `sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py`

The test file is pure stdlib (`math` only — no numpy, no capnp, no params) so it runs in the analysis venv AND on the device unchanged. `sunnypilot/selfdrive/selfdrived/` already has an `__init__.py`; the `openpilot/` symlink shim at the repo root makes `openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor` resolve to this checkout (import chain verified stdlib-clean this session).

- [ ] **Step 1: Create the package marker**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && mkdir -p sunnypilot/selfdrive/selfdrived/tests && touch sunnypilot/selfdrive/selfdrived/tests/__init__.py
```

- [ ] **Step 2: Write the failing tests (complete file)**

Write `sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py`:

```python
"""
Unit spec for AolSafeguardMonitor (F3 alert-only stock port, 2026-07-03).

Pure-stdlib tests (no numpy/capnp/params) so they run in the analysis venv and on-device.

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD \
      .venv311/bin/python -m pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py \
      -v --noconftest -o addopts=""

Timing arithmetic behind the assertion bounds (constants: CONF_EMA_TAU=0.5, CONF_THRESHOLD=0.30,
CONF_QUALIFIER=1.0, MANEUVER_BLIND_GRACE=2.0, DEPART_CONF_MIN=0.6, DEPART_OFFSET=0.40,
DEPART_RATE=0.15, DEPART_SUSTAIN=0.5, LINE_DIST_MIN=0.35, MANEUVER_DEPART_GRACE=3.0,
MIN_SPEED=10.0, RATE_WINDOW=11, dt=0.05):
  - EMA alpha = 0.05/(0.5+0.05) = 0.0909. From ~0.95, with inner_prob=0.02 the EMA crosses
    0.30 after ~13 frames ((0.93)*(0.909)^k < 0.28 => k >= 13), then the 1.0 s qualifier
    (20 frames) => first possible low-conf fire ~33 frames after collapse onset.
  - Departure needs the 11-sample offset window full (0.5 s), |offset| > 0.40 m, moving
    away from center at |rate| > 0.15 m/s, sustained 0.5 s (10 more frames).
  - Post-maneuver departure grace 3.0 s + 0.5 s deque refill + 0.5 s sustain => a drift
    already qualifying at maneuver end first fires ~4.0 s after the maneuver ends.
If a constant changes (Task 10 tuning), recompute these bounds BY HAND and update them.
"""
import math

from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor

DT = 0.05  # 20 Hz modelV2 cadence


def run(mon, seq):
  """seq: list of dicts of step kwargs; returns list of (low_conf, departure)."""
  out = []
  t = 0.0
  for s in seq:
    kw = dict(t=t, lat_active=True, v_ego=25.0, inner_prob=0.95, lane_offset=0.0,
              left_line_dist=1.7, right_line_dist=1.7, maneuver=False)
    kw.update(s)
    out.append(mon.update(**kw))
    t += DT
  return out


def steps(n, **kw):
  return [dict(kw) for _ in range(n)]


class TestLowConfidence:
  def test_sustained_collapse_alerts_after_qualifier(self):
    mon = AolSafeguardMonitor()
    seq = steps(40) + steps(60, inner_prob=0.02)  # 2s good, 3s blind
    res = run(mon, seq)
    fired = [i for i, r in enumerate(res) if r[0]]
    assert fired, "low-conf alert never fired on a 3 s collapse"
    # EMA (tau .5) needs ~0.6 s to fall below 0.30 from 0.95, then 1.0 s qualifier:
    # expect first fire between 1.2 s and 2.2 s after collapse onset (frames 64..84)
    assert 64 <= fired[0] <= 84, f"first fire at frame {fired[0]}, expected 64..84"

  def test_short_flicker_no_alert(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(8, inner_prob=0.02) + steps(40))  # 0.4 s dip
    assert not any(r[0] for r in res)

  def test_suppressed_first_2s_of_maneuver_then_fires(self):
    mon = AolSafeguardMonitor()
    # maneuver starts, goes blind immediately, blindness persists 4 s into maneuver
    seq = steps(40) + steps(80, inner_prob=0.02, maneuver=True)
    res = run(mon, seq)
    fired = [i for i, r in enumerate(res) if r[0]]
    assert fired, "persistent blindness in a maneuver must still alert"
    # suppressed for 2.0 s of maneuver (frames 40..79); EMA+qualifier already
    # elapsed by then, so first fire right after suppression lifts (frames 80..86)
    assert fired[0] >= 80
    assert fired[0] <= 86

  def test_inactive_or_slow_never_alerts_and_resets(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(60, inner_prob=0.02, lat_active=False))
    assert not any(r[0] for r in res)
    res = run(mon, steps(60, inner_prob=0.02, v_ego=5.0))
    assert not any(r[0] for r in res)


class TestDeparture:
  def test_drift_with_rate_alerts_before_07m(self):
    mon = AolSafeguardMonitor()
    # replicate the decisive event: offset ramps 0 -> -1.0 m at -0.24 m/s (car
    # drifting RIGHT of center; midpoint convention + = left of center)
    drift = [dict(lane_offset=-0.24 * i * DT) for i in range(84)]
    res = run(mon, steps(40) + drift)
    fired = [i - 40 for i, r in enumerate(res) if r[1]]
    assert fired, "departure alert never fired on a 0.24 m/s drift"
    t_fire = fired[0] * DT
    # |off|>0.40 at ~1.67 s, +0.5 s sustain -> ~2.17 s; offset then is ~0.52 m,
    # well before the 0.70 m the driver reacted to in the real event
    assert 1.9 <= t_fire <= 2.6, f"fired at {t_fire:.2f}s"
    assert abs(-0.24 * (fired[0] * DT)) < 0.70

  def test_static_offset_no_alert(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(100, lane_offset=-0.55))  # parked off-center, no rate
    assert not any(r[1] for r in res)

  def test_approach_to_center_no_alert(self):
    mon = AolSafeguardMonitor()
    # drift OUT to -0.8 m at 0.10 m/s (below DEPART_RATE, never fires), then recover
    # TOWARD center at 0.30 m/s: |offset| > 0.40 with a big rate, but moving toward
    # center (rate * offset < 0) -> must never fire
    out = [dict(lane_offset=-0.10 * i * DT) for i in range(160)]           # 0 -> -0.8 over 8 s
    back = [dict(lane_offset=-0.8 + 0.30 * i * DT) for i in range(40)]     # -0.8 -> -0.2 over 2 s
    res = run(mon, steps(40) + out + back + steps(40, lane_offset=-0.2))
    assert not any(r[1] for r in res)

  def test_inner_line_proximity_backstop(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(30, right_line_dist=0.25, lane_offset=-1.2))
    assert any(r[1] for r in res)

  def test_lane_change_sweep_suppressed(self):
    mon = AolSafeguardMonitor()
    # commanded lane change: offset sweeps to -1.6 m and back over 4 s, maneuver=True,
    # then settles; no departure alert anywhere (3 s post-maneuver grace covers the tail)
    sweep = [dict(lane_offset=-1.6 * math.sin(math.pi * i / 80), maneuver=True) for i in range(80)]
    settle = [dict(lane_offset=-0.3 + 0.3 * min(1.0, i / 40)) for i in range(60)]
    res = run(mon, steps(40) + sweep + settle)
    assert not any(r[1] for r in res)

  def test_post_maneuver_departure_grace_then_fires(self):
    mon = AolSafeguardMonitor()
    # 2 s clean, 1 s maneuver, then a drift that already qualifies the moment the
    # maneuver ends (|offset| > 0.40, moving away at 0.20 m/s). Grace suppresses
    # appends for 3.0 s after maneuver end; deque refill 0.5 s; sustain 0.5 s
    # => first fire ~4.0 s after maneuver end (maneuver ends at frame 60).
    pre = steps(40) + steps(20, maneuver=True)
    drift = [dict(lane_offset=-(0.45 + 0.20 * i * DT)) for i in range(100)]  # 5 s
    res = run(mon, pre + drift)
    fired = [i for i, r in enumerate(res) if r[1]]
    assert fired, "post-grace qualifying drift must fire"
    t_after_end = (fired[0] - 60) * DT
    assert t_after_end >= 3.9, f"fired {t_after_end:.2f}s after maneuver end (inside grace/refill)"
    assert t_after_end <= 4.3, f"fired too late: {t_after_end:.2f}s after maneuver end"

  def test_blind_never_fires_departure(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(60, inner_prob=0.05, lane_offset=-1.0))
    assert not any(r[1] for r in res)  # low-conf monitor owns the blind case
```

- [ ] **Step 3: Run the tests, verify they fail on import**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD .venv311/bin/python -m pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py -x -q --noconftest -o addopts=""
```

Expected: `ModuleNotFoundError: No module named 'openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor'` (collection error — the implementation does not exist yet).

- [ ] **Step 4: Commit the failing tests**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add sunnypilot/selfdrive/selfdrived/tests/ && git commit -m "test: AOL safeguard monitor spec (failing)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Monitor implementation — verbatim reference class

**Files:**
- Create: `sunnypilot/selfdrive/selfdrived/aol_monitor.py`

This is the reference implementation from `docs/superpowers/plans/2026-07-02-aol-incident-fixes-f3-f2-ab.md:224-326`, copied **verbatim** per the spec ("reference implementation and constants exactly as in" that plan). It is import-dependency-free: stdlib `collections.deque` only.

- [ ] **Step 1: Write the implementation (complete file)**

```python
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

AOL/MADS lateral safeguard monitors (2026-07-01 incident, handoff v3 §5).

Two pure monitors stepped at modelV2 rate (20 Hz):
  low-confidence: model blind while steering -> alert instead of silent continuation
  departure:      sighted but drifting off-center with velocity -> alert ~1.5 s
                  before the takeover point observed in the incident

Sign conventions (verified in the investigation, handoff v3 §1.2):
  lane_offset (lane-line midpoint y): + = car LEFT of lane center
  a drift "away from center" means sign(d(offset)/dt) == sign(offset)
"""
from collections import deque


class AolSafeguardMonitor:
  # low-confidence monitor
  CONF_EMA_TAU = 0.5          # s
  CONF_THRESHOLD = 0.30
  CONF_QUALIFIER = 1.0        # s below threshold before alerting
  MANEUVER_BLIND_GRACE = 2.0  # s of maneuver during which blindness is expected

  # departure monitor
  DEPART_CONF_MIN = 0.6
  DEPART_OFFSET = 0.40        # m
  DEPART_RATE = 0.15          # m/s, away from center
  DEPART_SUSTAIN = 0.5        # s
  LINE_DIST_MIN = 0.35        # m, absolute backstop to the nearest inner line
  MANEUVER_DEPART_GRACE = 3.0 # s after a maneuver ends

  MIN_SPEED = 10.0            # m/s
  RATE_WINDOW = 11            # samples (0.5 s at 20 Hz)

  def __init__(self, dt: float = 0.05):
    self.dt = dt
    self._reset()

  def _reset(self):
    self.conf_ema = 1.0
    self.low_conf_since = None
    self.depart_since = None
    self.offsets = deque(maxlen=self.RATE_WINDOW)
    self.maneuver_start_t = None
    self.maneuver_end_t = -1e9

  def update(self, t: float, lat_active: bool, v_ego: float, inner_prob: float,
             lane_offset: float, left_line_dist: float, right_line_dist: float,
             maneuver: bool) -> tuple[bool, bool]:
    """Step once per modelV2 frame. Returns (low_confidence_alert, departure_alert)."""
    if not lat_active or v_ego < self.MIN_SPEED:
      self._reset()
      return False, False

    # maneuver edge tracking
    if maneuver and self.maneuver_start_t is None:
      self.maneuver_start_t = t
    elif not maneuver and self.maneuver_start_t is not None:
      self.maneuver_end_t = t
      self.maneuver_start_t = None

    # ---- low-confidence monitor ----------------------------------------
    alpha = self.dt / (self.CONF_EMA_TAU + self.dt)
    self.conf_ema = alpha * inner_prob + (1.0 - alpha) * self.conf_ema

    blind_suppressed = (self.maneuver_start_t is not None
                        and (t - self.maneuver_start_t) < self.MANEUVER_BLIND_GRACE)
    low_conf = False
    if self.conf_ema < self.CONF_THRESHOLD:
      if self.low_conf_since is None:
        self.low_conf_since = t
      low_conf = (t - self.low_conf_since) >= self.CONF_QUALIFIER and not blind_suppressed
    else:
      self.low_conf_since = None

    # ---- departure monitor ----------------------------------------------
    departure = False
    depart_suppressed = (self.maneuver_start_t is not None
                         or (t - self.maneuver_end_t) < self.MANEUVER_DEPART_GRACE)
    if self.conf_ema >= self.DEPART_CONF_MIN and not depart_suppressed:
      self.offsets.append(lane_offset)
      cond = min(left_line_dist, right_line_dist) < self.LINE_DIST_MIN
      if not cond and len(self.offsets) == self.RATE_WINDOW:
        rate = (self.offsets[-1] - self.offsets[0]) / ((self.RATE_WINDOW - 1) * self.dt)
        moving_away = rate * lane_offset > 0
        cond = abs(lane_offset) > self.DEPART_OFFSET and moving_away and abs(rate) > self.DEPART_RATE
      if cond:
        if self.depart_since is None:
          self.depart_since = t
        departure = (t - self.depart_since) >= self.DEPART_SUSTAIN
      else:
        self.depart_since = None
    else:
      self.offsets.clear()
      self.depart_since = None

    return low_conf, departure
```

- [ ] **Step 2: Run the tests, verify all pass**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD .venv311/bin/python -m pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py -v --noconftest -o addopts=""
```

Expected: **11 passed** (4 in TestLowConfidence, 7 in TestDeparture). If a timing assertion fails by 1-2 frames, fix the TEST bound only if hand-computation of the EMA/qualifier arithmetic (docstring at the top of the test file) supports it; otherwise the implementation deviated from the verbatim reference — re-diff against Task 4 Step 1.

- [ ] **Step 3: Commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add sunnypilot/selfdrive/selfdrived/aol_monitor.py && git commit -m "sunnypilot: AOL safeguard monitor (low-confidence + departure), alert-only stock port" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: Alert definitions in EVENTS_SP — visual-only

**Files:**
- Modify: `sunnypilot/selfdrive/selfdrived/events.py` (insert after the `laneTurnRight` entry, lines ~203-209)

Per the user decision in the spec: **`AudibleAlert.none`** initially — the chime is a later one-line decision after observed fire rates. `Priority`, `ET`, `Alert` are already imported in this file from `events_base` (line 10); `AlertStatus`/`AlertSize`/`VisualAlert`/`AudibleAlert` at lines 15-18.

- [ ] **Step 1: Locate the insertion anchor**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && grep -n "laneTurnRight" sunnypilot/selfdrive/selfdrived/events.py
```

Expected: one `EventNameSP.laneTurnRight:` hit at line ~203.

- [ ] **Step 2: Insert the two entries after the `laneTurnRight` entry's closing `},`**

```python
  EventNameSP.aolLowLaneConfidence: {
    ET.WARNING: Alert(
      "Lane detection lost",
      "Take control",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.none, .2),
  },

  EventNameSP.aolLaneDeparture: {
    ET.WARNING: Alert(
      "Lane departure",
      "Take control",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.none, .2),
  },
```

(Duration 0.2 s + the monitor re-raising every modelV2 frame ⇒ continuous visual alert while the condition holds; it self-clears within 0.2 s of the condition ending. `EventsSP.__init__` builds `event_counters` from `EVENTS_SP.keys()`, so the new entries are picked up automatically.)

- [ ] **Step 3: Verify — syntax + structural checks (this venv CANNOT import events.py: `msgq` is not compiled here, verified this session; the runtime construction check runs on-device in Task 12 Step 7)**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && \
.venv311/bin/python -m py_compile sunnypilot/selfdrive/selfdrived/events.py && echo "py_compile OK" && \
grep -c "aolLowLaneConfidence\|aolLaneDeparture" sunnypilot/selfdrive/selfdrived/events.py && \
grep -A 6 "EventNameSP.aolLowLaneConfidence" sunnypilot/selfdrive/selfdrived/events.py | grep -c "AudibleAlert.none" && \
grep -A 6 "EventNameSP.aolLaneDeparture" sunnypilot/selfdrive/selfdrived/events.py | grep -c "AudibleAlert.none"
```

Expected: `py_compile OK`, then `2`, `1`, `1` (both entries present, both audibly silent).

- [ ] **Step 4: Commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add sunnypilot/selfdrive/selfdrived/events.py && git commit -m "sunnypilot: AOL safeguard alerts (lane detection lost / lane departure), visual-only" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: Kill-switch param registration

**Files:**
- Modify: `common/params_keys.h`

⚠️ Per project discipline (memory `feedback_param_key_registration`): the key MUST be registered before any read/write — an unregistered key raises UnknownKeyName on write and silently returns None on read. Format verified this session: this branch uses the TYPED struct `{"key", {FLAGS, TYPE, "default"}}` (e.g. `{"DisableUpdates", {PERSISTENT | BACKUP, BOOL, "0"}}` at line ~33).

- [ ] **Step 1: Register `AolSafeguardDisabled`**

Insert after the `{"AlphaLongitudinalEnabled", ...}` line (line ~40, just before `{"enable_lane_positioning", ...}` at ~41):

```c
    {"AolSafeguardDisabled", {PERSISTENT, BOOL, "0"}},
```

Default `"0"` = false ⇒ **safeguard ON by default** (Task 7 reads `not get_bool("AolSafeguardDisabled")`).

- [ ] **Step 2: Verify registration line (runtime Params check is a device gate — `params_pyx` is not compiled in this venv)**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && grep -n '"AolSafeguardDisabled", {PERSISTENT, BOOL, "0"}' common/params_keys.h && echo "registered"
```

Expected: prints the line + `registered`.

- [ ] **Step 3: Commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add common/params_keys.h && git commit -m "params: register AolSafeguardDisabled (F3 kill switch, default off = safeguard on)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: Wire the monitor into selfdrived + cloudlog telemetry

**Files:**
- Modify: `selfdrive/selfdrived/selfdrived.py`

Anchors verified this session at the branch tip (`c370480384`) — re-verify each with grep before editing:
- import block: `from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP` at line **32**
- `__init__`: `self.params = Params()` at **55**; `self.events_sp = EventsSP()` at **166**; `self.mads = ModularAssistiveDrivingSystem(self)` at **169**; `self.car_events_sp = CarSpecificEventsSP(self.CP, self.CP_SP)` at **172**
- `update_events(self, CS)` starts at **176**; the laneTurn block ends at **318** (`self.events_sp.add(custom.OnroadEventSP.EventName.laneTurnRight)`); the pandaStates loop starts at **320**
- SubMaster already subscribes `'modelV2'` (line **98**) — **no new subscription**
- `LaneChangeState` already imported (line **43**); `cloudlog` already imported (line **14**); `self.active` initialized line **126**, set line **586**; `self.mads.active` exists (published at line **564**)
- `self.sm.logMonoTime['modelV2']` — SubMaster per-service last-recv mono-time dict, verified present in `cereal/messaging/__init__.py` lines 162/229 on this branch
- **CADENCE (critical):** selfdrived runs at 100 Hz (`Ratekeeper(100)`, line **144**) but modelV2 is 20 Hz. The `self.sm.updated['modelV2']` guard is ESSENTIAL — the monitor's `dt=0.05` (EMA alpha, rate window) assumes it is stepped only on fresh modelV2 frames. Do NOT step it every control frame.

- [ ] **Step 1: Add the import (below line 32)**

```python
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
```

- [ ] **Step 2: Instantiate in `__init__` (immediately after `self.car_events_sp = CarSpecificEventsSP(self.CP, self.CP_SP)` at line ~172)**

```python
    # AOL lateral safeguard (2026-07-01 incident, alert-only stock port).
    # Kill switch is read once at startup: toggling AolSafeguardDisabled requires
    # an openpilot restart (ignition cycle or reboot) to take effect.
    self.aol_monitor = AolSafeguardMonitor()
    self.aol_safeguard_enabled = not self.params.get_bool("AolSafeguardDisabled")
    self.aol_alert_prev = None  # None | "lowConf" | "departure" (telemetry edge tracking)
    self.aol_log_ctr = 0
```

(`self.params` is the existing `Params()` from line 55 — reuse it, do NOT create a second instance.)

- [ ] **Step 3: Step the monitor in `update_events` — insert between the laneTurn block (ends line ~318) and the pandaStates loop (line ~320)**

```python
    # AOL lateral safeguard (2026-07-01 incident): visual alert instead of silently
    # steering through model blindness or a developing lane departure.
    # Stepped ONLY on fresh modelV2 frames (20 Hz) — the monitor's dt assumes it.
    if self.aol_safeguard_enabled and self.sm.updated['modelV2']:
      md = self.sm['modelV2']
      lat_active = self.mads.active or self.active
      inner_prob = min(md.laneLineProbs[1], md.laneLineProbs[2]) if len(md.laneLineProbs) >= 3 else 1.0
      if len(md.laneLines) >= 3 and len(md.laneLines[1].y) and len(md.laneLines[2].y):
        left_y, right_y = md.laneLines[1].y[0], md.laneLines[2].y[0]   # +y = RIGHT
        lane_offset = (left_y + right_y) / 2.0                          # + = car LEFT of center
        left_dist, right_dist = abs(left_y), abs(right_y)
      else:
        lane_offset, left_dist, right_dist = 0.0, 10.0, 10.0
      maneuver = (md.meta.laneChangeState != LaneChangeState.off
                  or CS.leftBlinker or CS.rightBlinker)
      low_conf, departure = self.aol_monitor.update(
        t=self.sm.logMonoTime['modelV2'] * 1e-9, lat_active=lat_active, v_ego=CS.vEgo,
        inner_prob=inner_prob, lane_offset=lane_offset,
        left_line_dist=left_dist, right_line_dist=right_dist, maneuver=maneuver)
      if departure:
        self.events_sp.add(custom.OnroadEventSP.EventName.aolLaneDeparture)
      elif low_conf:
        self.events_sp.add(custom.OnroadEventSP.EventName.aolLowLaneConfidence)

      # telemetry: cloudlog on fire/clear edges + 1 Hz while an alert is active.
      # Rides the existing logMessage path; ~zero bytes on clean drives.
      alert_kind = "departure" if departure else ("lowConf" if low_conf else None)
      if alert_kind is None:
        if self.aol_alert_prev is not None:
          cloudlog.info(f"aol_safeguard clear alert={self.aol_alert_prev} "
                        f"conf_ema={self.aol_monitor.conf_ema:.3f} offset={lane_offset:+.2f} "
                        f"v_ego={CS.vEgo:.1f}")
        self.aol_log_ctr = 0
      else:
        self.aol_log_ctr += 1
        if alert_kind != self.aol_alert_prev or self.aol_log_ctr >= 20:
          offs = self.aol_monitor.offsets
          rate = (offs[-1] - offs[0]) / ((len(offs) - 1) * self.aol_monitor.dt) if len(offs) >= 2 else 0.0
          edge = "fire" if alert_kind != self.aol_alert_prev else "active"
          cloudlog.info(f"aol_safeguard {edge} alert={alert_kind} "
                        f"conf_ema={self.aol_monitor.conf_ema:.3f} offset={lane_offset:+.2f} "
                        f"rate={rate:+.2f} v_ego={CS.vEgo:.1f}")
          self.aol_log_ctr = 0
      self.aol_alert_prev = alert_kind
```

Notes:
- The `elif` (departure wins over low-conf) is deliberate: never stack both alerts. (The monitor makes them near-mutually-exclusive anyway — departure requires conf ≥ 0.6, low-conf requires EMA < 0.30.)
- `CS` is the `carState` argument of `update_events(self, CS)`; `CS.leftBlinker`/`CS.rightBlinker`/`CS.vEgo` are standard carState fields.
- The telemetry `rate` is recomputed from the monitor's exposed `offsets` deque so the monitor class itself stays byte-identical to the verbatim reference.
- `self.aol_log_ctr >= 20` at the 20 Hz monitor cadence = 1 Hz "active" heartbeat while an alert holds.
- Fallback if device runtime disagrees on the `logMonoTime` API (could not be runtime-checked in this venv — `msgq` absent): `self.sm.frame * DT_CTRL` is a valid substitute — the monitor uses `t` ONLY for elapsed-time differences, so any monotonic seconds source works.

- [ ] **Step 4: Syntax + structural verification (full import/pytest of selfdrived is impossible in this venv — `params_pyx`/`msgq` uncompiled; the functional gates for this wiring are Tasks 8-9, which drive the monitor with the exact same field extraction; on-device runtime checks are Task 12)**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && \
.venv311/bin/python -m py_compile selfdrive/selfdrived/selfdrived.py && echo "py_compile OK" && \
grep -n "AolSafeguardMonitor\|aol_safeguard_enabled\|aolLaneDeparture\|aolLowLaneConfidence" selfdrive/selfdrived/selfdrived.py
```

Expected: `py_compile OK` plus hits for: the import (~line 33), the two `__init__` lines (~173-176), the `update_events` block (~320-360).

- [ ] **Step 5: Commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add selfdrive/selfdrived/selfdrived.py && git commit -m "selfdrived: wire AOL safeguard monitor into update_events (visual alerts + cloudlog telemetry)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 8: Gates G1-G4 — re-point the committed replay harness at this checkout + the NAS

**Files:**
- Modify: `retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py`

The committed harness (commit `95ad2c1796`) has two fork-era assumptions to remove (its lines 21-46): (1) it loads the monitor from the `sp-aol-f3` worktree via an `importlib` path hack — the monitor now lives in THIS checkout and the `openpilot/` symlink shim makes a plain import resolve here; (2) it globs rlogs from the deleted local `explorer_st_logs/` — the corpus is now on the NAS. Gates, replay logic, segment-ordering fix, and episode merging stay **identical**.

- [ ] **Step 1: Replace the file contents (complete file)**

```python
#!/usr/bin/env python3
"""F3 acceptance gate: replay recorded rlogs through AolSafeguardMonitor.

STOCK-PORT VERSION (2026-07-03): the monitor imports from THIS checkout
(sunnypilot/selfdrive/selfdrived/aol_monitor.py via the openpilot/ symlink shim —
no worktree), and the rlog corpus reads from the NAS mount.

MUST-PASS gates (from handoff v3 §5/§9b):
  G1 ce decisive event (override at mono 212.54): departure alert >= 1.0 s before override.
  G2 cf blowout (deep-blind at 682.21): low-conf alert within 2.5 s of collapse onset.
  G3 ce blind-in-lane-change (blind 257.31, override 260.18): low-conf alert BEFORE the
     override despite the maneuver (suppression must lift after 2 s).
  G4 departure-alert count on OLD routes (c5/c7/b5/7f) == 0 (no real departures occurred).
REPORT-ONLY (user tunes thresholds if excessive):
  R1 low-conf alert episode count + total alert seconds per route (fatigue check).

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo \
      .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py
"""
import glob, os, re, sys

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
CORPUS = "/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")

from openpilot.tools.lib.logreader import LogReader
import openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor as am_mod
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
print("monitor module:", am_mod.__file__)
assert os.path.realpath(am_mod.__file__).startswith(os.path.realpath(ROOT)), \
    "monitor must come from this checkout"
assert os.path.isdir(CORPUS), \
    f"NAS corpus not mounted: {CORPUS} (mount smb://datacore.local/RAID_6_HDD first)"


def _seg_num(path):
    # Sort segments by numeric index so the monitor sees a continuous chronological
    # time stream, exactly like selfdrived. Lexical sort interleaves --10 before --2,
    # injecting large mono-time discontinuities across segment boundaries.
    m = re.search(r"--(\d+)/rlog\.zst$", path)
    return int(m.group(1)) if m else 0


def replay(route):
    """Feed the monitor exactly what selfdrived would see; return alert timelines."""
    rows = []      # (t, kind) kind in {"lc","dep"}
    cs_state = dict(v=0.0, lb=False, rb=False)
    lat = dict(active=False)
    mon = AolSafeguardMonitor()
    files = sorted(glob.glob(f"{CORPUS}/{route}/*/rlog.zst"), key=_seg_num)
    print(f"  {route}: {len(files)} segments")
    for rl in files:
        try:
            lr = LogReader(rl)
        except Exception:
            continue
        for m in lr:
            try:
                w = m.which()
            except Exception:
                continue
            if w == "carState":
                c = m.carState
                cs_state = dict(v=float(c.vEgo), lb=bool(c.leftBlinker), rb=bool(c.rightBlinker))
            elif w == "carControl":
                lat = dict(active=bool(m.carControl.latActive))
            elif w == "modelV2":
                md = m.modelV2
                t = m.logMonoTime * 1e-9
                p = list(md.laneLineProbs)
                inner = min(p[1], p[2]) if len(p) >= 3 else 1.0
                ll = list(md.laneLines)
                if len(ll) >= 3 and len(ll[1].y) and len(ll[2].y):
                    ly, ry = ll[1].y[0], ll[2].y[0]
                    off, ld, rd = (ly + ry) / 2.0, abs(ly), abs(ry)
                else:
                    off, ld, rd = 0.0, 10.0, 10.0
                maneuver = (str(md.meta.laneChangeState) != "off") or cs_state["lb"] or cs_state["rb"]
                lc, dep = mon.update(t=t, lat_active=lat["active"], v_ego=cs_state["v"],
                                     inner_prob=inner, lane_offset=off,
                                     left_line_dist=ld, right_line_dist=rd, maneuver=maneuver)
                if dep:
                    rows.append((t, "dep"))
                elif lc:
                    rows.append((t, "lc"))
    return rows


def episodes(rows, kind):
    ts = [t for (t, k) in rows if k == kind]
    eps = []
    for t in ts:
        if eps and t - eps[-1][1] < 1.0:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


results = {r: replay(r) for r in ("route_ce", "route_cf", "route_c5", "route_c7", "route_b5", "route_7f")}

print("\n===== REPORT (R1: alert load per route) =====")
for r, rows in results.items():
    lc_eps, dep_eps = episodes(rows, "lc"), episodes(rows, "dep")
    lc_secs = sum(e[1] - e[0] for e in lc_eps)
    print(f"  {r}: low-conf episodes={len(lc_eps)} ({lc_secs:.1f}s total)  departure episodes={len(dep_eps)}")

print("\n===== GATES =====")
ce_dep = [t for (t, k) in results["route_ce"] if k == "dep"]
g1 = [t for t in ce_dep if 205.0 < t < 212.54]
print(f"G1 decisive-event departure alert: {'PASS' if g1 and (212.54 - g1[0]) >= 1.0 else 'FAIL'}"
      f"  (first alert {'%.2f' % g1[0] if g1 else 'none'}; need <= 211.54)")

cf_lc = [t for (t, k) in results["route_cf"] if k == "lc"]
g2 = [t for t in cf_lc if 682.21 <= t <= 684.71]
print(f"G2 blowout low-conf alert: {'PASS' if g2 else 'FAIL'}  (alerts in [682.2, 684.7]: {len(g2)})")

ce_lc = [t for (t, k) in results["route_ce"] if k == "lc"]
g3 = [t for t in ce_lc if 257.31 <= t <= 260.18]
print(f"G3 blind-in-lane-change alert before override: {'PASS' if g3 else 'FAIL'}"
      f"  (first {'%.2f' % g3[0] if g3 else 'none'}; override 260.18)")

old_dep = sum(len(episodes(results[r], "dep")) for r in ("route_c5", "route_c7", "route_b5", "route_7f"))
print(f"G4 departure alerts on OLD routes: {'PASS' if old_dep == 0 else 'FAIL'}  (count {old_dep})")
sys.exit(0 if (g1 and (212.54 - g1[0]) >= 1.0 and g2 and g3 and old_dep == 0) else 1)
```

- [ ] **Step 2: Run the gate**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py
```

Expected output shape (runtime ~15-30 min — six full-route rlog passes over SMB):
- `monitor module: /Users/dregilley/Documents/GitHub/sunnypilot/openpilot/sunnypilot/selfdrive/selfdrived/aol_monitor.py`
- segment counts: `route_ce: 18`, `route_cf: 18`, `route_c5: 13`, `route_c7: 36`, `route_b5: 43`, `route_7f: 18` (verified on the NAS this session — a shortfall means an incomplete NAS copy: STOP and reconcile before trusting any gate)
- **G1 PASS** (first departure alert in (205.0, 212.54), at or before 211.54 — ≥1.0 s before the override)
- **G2 PASS** (≥1 low-conf alert within [682.21, 684.71] on route_cf)
- **G3 PASS** (≥1 low-conf alert in [257.31, 260.18] on route_ce — blind-in-lane-change, suppression lifts after 2 s)
- **G4 PASS** (0 departure episodes across route_c5/c7/b5/7f)
- exit status 0

R1 sanity expectation: low-conf episodes on the old routes ≈ their genuine non-maneuver blindness episodes (order 5-15 per route; these are TRUE positives — the user judges fatigue in Task 11 and may raise CONF_QUALIFIER via Task 10). If any gate FAILS, go to Task 10 before touching anything else.

- [ ] **Step 3: Commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py && git commit -m "analysis: re-point F3 G1-G4 replay gate at stock monitor + NAS corpus" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 9: Gate G5 — stock-drive false-alert clearance + low-confidence census (new)

**Files:**
- Create: `retrospective_lateral/incident_2026_07_01/scripts/f3_stock_census.py`

G1-G4 validate detection + specificity on the OLD build's logs. G5 is the clearance on the build/model that will actually run the safeguard: **zero departure alerts across all stock drives**, plus the low-confidence episode census — the expected visual-alert frequency the user reviews before ever seeing one in the car. Data reality (verified 2026-07-03): stock routes `00000002` (64 segs), `00000004` (26), `00000005` (12) have full `rlog.zst` coverage; `00000000` (5 segs), `00000001` (3), `00000003` (3) are **qlog-only on the NAS** — the script skips-and-reports them because qlog modelV2 is decimated (~4 Hz), which violates the monitor's 20 Hz `dt` assumption and would distort every time constant.

- [ ] **Step 1: Write the census script (complete file)**

```python
#!/usr/bin/env python3
"""G5 acceptance gate: AOL safeguard on ALL stock drives — false-alert clearance + census.

GATE (must pass before deploy):
  ZERO departure-alert episodes across every stock route with rlog coverage.
REPORT (expected on-device visual-alert frequency; user reviews before deploy):
  per-route low-conf episode count, total low-conf alert-seconds, active-driving time.

Corpus: /Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock/
  Segment dirs are RRRRRRRR--HHHHHHHHHH--N; a route = all segments sharing the
  RRRRRRRR--HHHHHHHHHH prefix. Routes are auto-discovered, so drives pulled after
  2026-07-03 are included automatically — re-run this gate after pulling new drives.
  Routes with NO rlog.zst (qlog-only on the NAS: 00000000, 00000001, 00000003 as of
  2026-07-03) are SKIPPED and reported — qlog modelV2 is decimated (~4 Hz), which
  violates the monitor's 20 Hz dt assumption and would distort every time constant.

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo \
      .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_stock_census.py
Exit status: 0 = gate PASS (zero departure episodes), 1 = FAIL.
"""
import os, re, sys
from collections import defaultdict

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
STOCK = "/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")

from openpilot.tools.lib.logreader import LogReader
import openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor as am_mod
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
print("monitor module:", am_mod.__file__)
assert os.path.realpath(am_mod.__file__).startswith(os.path.realpath(ROOT)), \
    "monitor must come from this checkout"
assert os.path.isdir(STOCK), \
    f"NAS corpus not mounted: {STOCK} (mount smb://datacore.local/RAID_6_HDD first)"

SEG_RE = re.compile(r"^([0-9a-fA-F]{8}--[0-9a-fA-F]{10})--(\d+)$")


def discover_routes():
    routes = defaultdict(list)   # route_id -> [(seg_num, seg_dir), ...] sorted by seg_num
    for d in sorted(os.listdir(STOCK)):
        m = SEG_RE.match(d)
        if m and os.path.isdir(os.path.join(STOCK, d)):
            routes[m.group(1)].append((int(m.group(2)), os.path.join(STOCK, d)))
    return {rid: sorted(segs) for rid, segs in routes.items()}


def replay(rlogs):
    """Feed the monitor exactly what selfdrived would see; return alert rows + context."""
    rows = []           # (t, kind) kind in {"lc", "dep"}
    active_frames = 0   # modelV2 frames with lat_active and v >= MIN_SPEED (alerts possible)
    total_frames = 0
    cs_state = dict(v=0.0, lb=False, rb=False)
    lat = dict(active=False)
    mon = AolSafeguardMonitor()
    for rl in rlogs:
        try:
            lr = LogReader(rl)
        except Exception:
            continue
        for m in lr:
            try:
                w = m.which()
            except Exception:
                continue
            if w == "carState":
                c = m.carState
                cs_state = dict(v=float(c.vEgo), lb=bool(c.leftBlinker), rb=bool(c.rightBlinker))
            elif w == "carControl":
                lat = dict(active=bool(m.carControl.latActive))
            elif w == "modelV2":
                md = m.modelV2
                t = m.logMonoTime * 1e-9
                p = list(md.laneLineProbs)
                inner = min(p[1], p[2]) if len(p) >= 3 else 1.0
                ll = list(md.laneLines)
                if len(ll) >= 3 and len(ll[1].y) and len(ll[2].y):
                    ly, ry = ll[1].y[0], ll[2].y[0]
                    off, ld, rd = (ly + ry) / 2.0, abs(ly), abs(ry)
                else:
                    off, ld, rd = 0.0, 10.0, 10.0
                maneuver = (str(md.meta.laneChangeState) != "off") or cs_state["lb"] or cs_state["rb"]
                lc, dep = mon.update(t=t, lat_active=lat["active"], v_ego=cs_state["v"],
                                     inner_prob=inner, lane_offset=off,
                                     left_line_dist=ld, right_line_dist=rd, maneuver=maneuver)
                total_frames += 1
                if lat["active"] and cs_state["v"] >= AolSafeguardMonitor.MIN_SPEED:
                    active_frames += 1
                if dep:
                    rows.append((t, "dep"))
                elif lc:
                    rows.append((t, "lc"))
    return rows, active_frames * 0.05, total_frames * 0.05


def episodes(rows, kind):
    ts = [t for (t, k) in rows if k == kind]
    eps = []
    for t in ts:
        if eps and t - eps[-1][1] < 1.0:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


routes = discover_routes()
total_dep = 0
scanned = 0
print(f"\n{'route':24s} {'segs':>5s} {'rlogs':>5s} {'drive_min':>9s} {'active_min':>10s} "
      f"{'lowconf_eps':>11s} {'lowconf_s':>9s} {'depart_eps':>10s}")
for rid, segs in sorted(routes.items()):
    rlogs = [os.path.join(d, "rlog.zst") for (_, d) in segs
             if os.path.exists(os.path.join(d, "rlog.zst"))]
    if not rlogs:
        print(f"{rid:24s} {len(segs):5d} {0:5d}  SKIPPED (qlog-only on NAS — pull rlogs from device for coverage)")
        continue
    scanned += 1
    rows, active_s, total_s = replay(rlogs)
    lc_eps, dep_eps = episodes(rows, "lc"), episodes(rows, "dep")
    lc_secs = sum(e[1] - e[0] for e in lc_eps)
    total_dep += len(dep_eps)
    print(f"{rid:24s} {len(segs):5d} {len(rlogs):5d} {total_s/60:9.1f} {active_s/60:10.1f} "
          f"{len(lc_eps):11d} {lc_secs:9.1f} {len(dep_eps):10d}")
    for e in dep_eps:
        print(f"    DEPARTURE EPISODE mono t=[{e[0]:.2f}, {e[1]:.2f}] — pull this window's "
              f"speed/blinker/laneChangeState context before any threshold change")

print(f"\nscanned {scanned} routes with rlog coverage")
print(f"G5 departure alerts on stock drives: {'PASS' if total_dep == 0 else 'FAIL'}  (count {total_dep})")
sys.exit(0 if total_dep == 0 else 1)
```

- [ ] **Step 2: Run the gate**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_stock_census.py
```

Expected (runtime ~10-20 min — 102 rlog segments over SMB):
- table rows for `00000002--5da5840d8d` (64/64 rlogs), `00000004--3d3385646d` (26/26), `00000005--ef46fdca62` (12/12); SKIPPED lines for `00000000--c181384b0c`, `00000001--e80958d9ca`, `00000003--b37c613b41`
- **`G5 departure alerts on stock drives: PASS (count 0)`**, exit status 0
- low-conf census: no hard bound — this is the REPORT the user reviews in Task 11. Plausible order: 0-15 episodes per route (true blindness moments). If it is dramatically noisier than the old routes' R1 numbers, that is a stock-model behavior difference worth flagging to the user, and CONF_QUALIFIER tuning (Task 10) is the lever.

If departure count > 0: FAIL — go to Task 10. Record every printed DEPARTURE EPISODE window.

- [ ] **Step 3: Commit**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add retrospective_lateral/incident_2026_07_01/scripts/f3_stock_census.py && git commit -m "analysis: F3 G5 stock-drive false-alert clearance + low-confidence census gate" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 10: Threshold tuning — ONLY if a gate fails (decision rules)

**Files:**
- Modify (only on a documented gate failure): `sunnypilot/selfdrive/selfdrived/aol_monitor.py` (constants), `sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py` (recomputed bounds)

These rules come from the fork plan's validated tuning guidance. **Any constant change invalidates ALL prior gate results: re-run Task 4 Step 2 (unit tests, with bounds recomputed by hand per the test-file docstring), Task 8, AND Task 9 — every gate, every time.**

- [ ] **Rule 1 — G1 marginal** (departure alert exists but < 1.0 s before the 212.54 override): FIRST re-check the offset-rate window arithmetic in the replay (segment ordering, mono-time continuity) — a harness bug is more likely than a detector deficiency. Only if the arithmetic is confirmed clean: lower `DEPART_OFFSET` 0.40 → 0.35, then re-run G4 + G5 to confirm zero false-alert cost (and G2/G3, which are insensitive to it, as the blanket rule requires).
- [ ] **Rule 2 — G4 or G5 departure false alerts**: for each offending timestamp, pull that window's context (speed, blinkers, `laneChangeState`) from the rlog. If the trigger is an unmarked maneuver (blinkerless lane change, merge lane), tighten the maneuver definition at the selfdrived wiring (Task 7 Step 3) rather than the monitor; if it is genuine offset noise, raise `DEPART_RATE` 0.15 → 0.20. NEVER exclude the offending route from the gate.
- [ ] **Rule 3 — low-conf census too noisy** (user judges the Task 9/R1 alert load fatiguing): raise `CONF_QUALIFIER` 1.0 → 1.5 s. Note this delays the G2/G3 alerts by the same 0.5 s — G2's window [682.21, 684.71] must still pass.
- [ ] **After ANY change**: hand-recompute the affected unit-test bounds (the derivations are in the test-file docstring), then re-run in order: unit tests → G1-G4 → G5. All must pass on the SAME constants before Task 11.

---

## Task 11: User QA checkpoint — STOP

**Files:** none

- [ ] **Step 1: Assemble the review packet** — present to the user:
  1. the full diff: `git diff <starting-commit-from-Task-1>..HEAD --stat` plus the monitor + wiring hunks
  2. unit test output (11 passed)
  3. G1-G4 gate output (four PASS lines + the R1 alert-load table)
  4. G5 output (PASS + the per-route low-conf census — this is the expected visual-alert frequency in the car)
  5. the resource-budget statement: ~2.9 µs/frame measured on M4, ~1.2 KB memory, ≈0.06% of one device core at 10× derate; pre-F3 device `selfdrived` baseline 18.5% CPU median (from `stock/00000005--ef46fdca62--3` procLog)
- [ ] **Step 2: STOP.** Device deployment (Task 12) proceeds ONLY after explicit user approval in that session. Record the user's chime decision input if offered (the audible upgrade remains a later one-line change to the two `AudibleAlert.none` fields).

---

## Task 12: Device deploy — USER-GATED

**Files:**
- Create: `retrospective_lateral/incident_2026_07_01/scripts/f3_cpu_check.py`
- Deploy: branch `stock-2026.002.001-fresh-start` to device `/data/openpilot`

⚠️ **ASK THE USER BEFORE TOUCHING THE DEVICE. Do not begin any step of this task without the user's explicit go-ahead given after Task 11.** Device SSH connection details are in memory `reference_device_config.md` (host key changed post-reflash; GitHub-username auth). All device work in a single SSH session; device SSH is single-connection/sequential.

- [ ] **Step 1: Disable the updater FIRST (it reverts code edits)**

On the device:

```bash
echo -n "1" > /data/params/d/DisableUpdates
```

⚠️ Known gotcha (memory `reference_device_config.md`): the FIRST reboot after setting this can still revert the checkout. After the Step 6 reboot, verify (Step 7); if reverted, re-apply this param, re-checkout, and reboot a SECOND time.

- [ ] **Step 2: Push the branch from the Mac**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git push origin stock-2026.002.001-fresh-start
```

Expected: branch updated on the GitHub remote the device fetches from. If the push remote differs from the device's remote, check `git remote -v` on BOTH ends and push to the one the device's `/data/openpilot` origin points at.

- [ ] **Step 3: Fetch + checkout on the device**

```bash
cd /data/openpilot && git remote -v && \
git fetch origin stock-2026.002.001-fresh-start && \
git checkout stock-2026.002.001-fresh-start && \
git reset --hard origin/stock-2026.002.001-fresh-start && \
git submodule update --init --recursive && \
git log --oneline -1
```

Expected: HEAD at the Task 11-approved commit. Record the PREVIOUS device commit first (`git log --oneline -1` before fetching) — it is the rollback target.

- [ ] **Step 4: Rebuild (REQUIRED — the `custom.capnp` enum change must be compiled for C++ consumers)**

```bash
cd /data/openpilot && scons -j$(nproc)
```

Expected: build completes without errors (several minutes). Do not skip: a stale schema on the C++ side is undefined behavior for the new enum values.

- [ ] **Step 5: Reboot via the param (NEVER `sudo reboot`)**

```bash
echo -n "1" > /data/params/d/DoReboot
```

- [ ] **Step 6: Post-reboot verification (reconnect SSH)**

```bash
cd /data/openpilot && git log --oneline -1 && cat /data/params/d/DisableUpdates
```

Expected: still the approved commit, and `1`. If the updater reverted the checkout: re-run Step 1, Step 3, Step 5 (the second reboot sticks — known behavior).

- [ ] **Step 7: On-device runtime checks (the checks the analysis venv could not run)**

```bash
# (a) param registered + kill switch default (safeguard ON):
cd /data/openpilot && python3 -c "from openpilot.common.params import Params; print('AolSafeguardDisabled =', Params().get_bool('AolSafeguardDisabled'))"
# expected: AolSafeguardDisabled = False

# (b) unit tests on the device python:
cd /data/openpilot && python3 -m pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py -q --noconftest -o addopts=""
# expected: 11 passed
# if pytest is unavailable on the device, run this dependency-free sanity instead:
cd /data/openpilot && python3 -c "
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
mon = AolSafeguardMonitor()
fired = []
t = 0.0
for i in range(100):
    lc, dep = mon.update(t=t, lat_active=True, v_ego=25.0,
                         inner_prob=(0.95 if i < 40 else 0.02), lane_offset=0.0,
                         left_line_dist=1.7, right_line_dist=1.7, maneuver=False)
    if lc:
        fired.append(i)
    t += 0.05
assert fired and 64 <= fired[0] <= 84, fired[:1]
print('device monitor sanity OK, first low-conf fire at frame', fired[0])
"

# (c) alert entries construct at runtime (msgq/params exist here):
cd /data/openpilot && python3 -c "
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP, EventNameSP, ET
for ev in (EventNameSP.aolLowLaneConfidence, EventNameSP.aolLaneDeparture):
    a = EVENTS_SP[ev][ET.WARNING]
    print(int(ev), repr(a.alert_text_1), '| audible:', a.audible_alert)
"
# expected: two lines (enum ints 24 and 25), audible: none for both
```

- [ ] **Step 8: Alert-render verification policy — first-drive verification (accepted)**

A parked bench trigger is IMPOSSIBLE without violating the design: the monitor hard-gates on `v_ego >= 10 m/s`, and we do not modify constants for testing (a test-lowered `CONF_THRESHOLD` would not be the shipped detector). Verification is therefore: (i) Step 7's on-device detector + alert-construction checks, plus (ii) the first supervised drive — the G5 census predicts the low-conf visual-alert frequency, so the "Lane detection lost / Take control" mid-size alert should be observed naturally within normal driving; the user notes every alert (expected vs surprising). After the drive, confirm the telemetry landed:

```bash
# on the device — cloudlog lines go to the local swaglog files AND the rlog logMessage stream:
grep -a aol_safeguard /data/log/swaglog* | tail -5
```

Expected: `aol_safeguard fire/active/clear alert=lowConf conf_ema=... offset=... rate=... v_ego=...` lines matching the observed alerts. ZERO lines on a fully clean drive is the designed outcome (~zero bytes on clean drives), not a failure — cross-check against whether any alert was seen on screen.

- [ ] **Step 9: CPU% before/after check — write and run the comparison script**

Write `retrospective_lateral/incident_2026_07_01/scripts/f3_cpu_check.py` on the Mac:

```python
#!/usr/bin/env python3
"""Deploy verification: selfdrived CPU%% from procLog — pre-F3 baseline vs post-deploy.

The selfdrived process appears in procLog with its name truncated to 15 chars:
'selfdrive.selfd' (verified on stock rlogs 2026-07-03). CPU%% = d(cpuUser+cpuSystem)/dt
between consecutive procLog samples; we report the median.

Baseline (measured 2026-07-03 on stock/00000005--ef46fdca62--3): 18.5%% median.
ACCEPTANCE: post-deploy median within +1.0 percentage point of the baseline
(the monitor's measured cost is ~2.9 us/frame at 20 Hz -- far below procLog noise).

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
      retrospective_lateral/incident_2026_07_01/scripts/f3_cpu_check.py <post_deploy_rlog.zst>
"""
import statistics, sys

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader

BASELINE = ("/Volumes/RAID_6_HDD/OpenPilot Data/explorer_st_logs/stock/"
            "00000005--ef46fdca62--3/rlog.zst")


def selfdrived_cpu(rlog):
    samples = []
    for m in LogReader(rlog):
        try:
            w = m.which()
        except Exception:
            continue
        if w == "procLog":
            for pr in m.procLog.procs:
                if pr.name.startswith("selfdrive.selfd"):
                    samples.append((m.logMonoTime * 1e-9, pr.cpuUser + pr.cpuSystem))
    if len(samples) < 2:
        raise SystemExit(f"not enough procLog samples in {rlog} ({len(samples)})")
    rates = [100 * (samples[i + 1][1] - samples[i][1]) / (samples[i + 1][0] - samples[i][0])
             for i in range(len(samples) - 1)]
    return statistics.median(rates), len(samples)


base_med, base_n = selfdrived_cpu(BASELINE)
print(f"baseline (pre-F3):  selfdrived {base_med:.1f}% CPU  ({base_n} procLog samples)")
if len(sys.argv) > 1:
    after_med, after_n = selfdrived_cpu(sys.argv[1])
    print(f"after F3:           selfdrived {after_med:.1f}% CPU  ({after_n} procLog samples)")
    ok = after_med <= base_med + 1.0
    print("CPU check:", "PASS" if ok else "FAIL — investigate before continuing")
    sys.exit(0 if ok else 1)
else:
    print("pass a post-deploy rlog.zst path as argv[1] to compare")
```

Then, after the first post-deploy drive, pull one steady-driving rlog segment from the device (`/data/media/0/realdata/<route>--<seg>/rlog.zst`, e.g. via `scp`) and run:

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_cpu_check.py <local-path-of-the-pulled-post-deploy-rlog.zst>
```

Expected: `baseline (pre-F3): selfdrived 18.5% CPU` (± NAS-segment choice) and post-deploy median within +1.0 percentage point → `CPU check: PASS`. Commit the script:

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add retrospective_lateral/incident_2026_07_01/scripts/f3_cpu_check.py && git commit -m "analysis: F3 deploy CPU% check (selfdrived procLog, pre vs post)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 10: Post-deploy observation window** — per the spec: ~a week of normal drives → review the `aol_safeguard` fire log (Step 8's grep + rlog logMessage extraction) against the G5 census prediction → then the user makes the chime decision (a later one-line change: the two `AudibleAlert.none` fields in `events.py`).

**Rollback (either level, user-initiated):**
- *Kill switch (no rebuild):* on device `cd /data/openpilot && python3 -c "from openpilot.common.params import Params; Params().put_bool('AolSafeguardDisabled', True)"`, then ignition cycle or `echo -n "1" > /data/params/d/DoReboot` (the switch is read once at selfdrived startup).
- *Full rollback:* on device `cd /data/openpilot && git reset --hard <pre-F3 commit recorded in Step 3> && scons -j$(nproc)` then `echo -n "1" > /data/params/d/DoReboot`.

---

## Execution-order summary

1. Task 1 (preflight) → Tasks 2-7 (code, TDD; ~2-2.5 h desk work)
2. Task 8 (G1-G4, ~15-30 min run) → Task 9 (G5, ~10-20 min run) — Task 10 only on a gate failure (every constant change re-runs unit tests + ALL gates)
3. Task 11 — user QA, **STOP for approval**
4. Task 12 — device deploy (**user-gated**), first supervised drive, CPU check, ~1 week observation → chime decision

**Estimated effort:** Tasks 1-7 ≈ 2-2.5 h; Tasks 8-9 ≈ 1-1.5 h (dominated by NAS replay wall-time); Task 11 ≈ 15 min; Task 12 ≈ 1-2 h device session + first drive. Total ≈ one focused day to the QA checkpoint, plus the deploy session.

## Self-review notes (writing-plans checklist)

- Spec coverage vs Workflow 1: monitor verbatim (T4), two capnp events + visual-only EVENTS_SP entries (T2, T5), selfdrived hook gated on `sm.updated['modelV2']` with no new subscription (T7), kill-switch registered before read (T6), cloudlog telemetry on the existing logMessage path (T7), degrade branch ABSENT by construction, gates G1-G5 (T8-T9), resource budget cited + device CPU check (T12 Step 9). Chime, fork centering, and the degrade branch are out of scope per the spec.
- Every repo anchor (line numbers, formats, import chains, NAS layouts, procLog naming, venv limitations) was re-verified against `stock-2026.002.001-fresh-start` @ `c370480384` and the live NAS mount on 2026-07-03; implementers must still eyeball each anchor grep before editing.
- Types/signatures consistent: `AolSafeguardMonitor.update(...)` kwargs identical in tests (T3), implementation (T4), selfdrived wiring (T7), and both replay harnesses (T8, T9).
- Known open risks: (1) stock routes 00000000/01/03 are qlog-only on the NAS — G5 covers 00000002/04/05 (102 segments); if full coverage is wanted, pull those routes' rlogs from the device before Task 11 and re-run G5 (auto-discovery picks them up once placed under `stock/`). (2) G1-G4 pass/fail on the NAS copy has not been re-executed since the corpus moved off local disk — Task 8 Step 2's segment-count check guards against an incomplete copy. (3) `sm.logMonoTime` could not be runtime-exercised in the analysis venv; T7 documents the `self.sm.frame * DT_CTRL` fallback.
