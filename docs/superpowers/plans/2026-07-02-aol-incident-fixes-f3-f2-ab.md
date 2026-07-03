# AOL Incident Fixes (F3 Safeguard + F2 PI De-lag + A/B Protocol) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the three log-validated fixes from the 2026-07-01 AOL incident investigation (handoff v3): an AOL blindness/departure alert safeguard (F3), the PI phase-lag reduction (F2), and the instrumented A/B drive protocol — each offline-validated against the incident logs before any device deployment.

**Architecture:** F3 is a pure-python monitor class raising two new `EventsSP` warning events from `selfdrived` (MADS already appends `ET.WARNING` when active, so alerts fire under pure AOL — verified on `clean-v2026.002.001`). F2 is a 3-line-scale parameter change to the Ford PI block in opendbc, with the two behavior changes live-toggleable via the existing `/data/lc_pi_config` mechanism so drive legs need no rebuild. Everything is validated by replaying the recorded incident rlogs offline before the car moves.

**Tech Stack:** Python 3.11 (`.venv311`), pytest, capnp (cereal), LogReader replay. Branches: main repo `aol-fixes-f3` off `clean-v2026.002.001`; opendbc `pi-delag-f2` off `849b72a1a`. Device deploys only after all gates pass AND user approval.

**Context documents:** `docs/superpowers/reports/2026-07-01-lateral-safety-investigation-handoff-v3.md` (root cause, §4-6 fix rationale, §9b per-event table). Investigation working tree (`2021_explorer_st-mici` + dirty analysis files) must NOT be disturbed — all code work happens in worktrees.

**Verified sign conventions (do not re-derive):** curvature/carOutput **+ = RIGHT**; `laneLines` **+y = RIGHT** (left line y0 ≈ −1.7, right ≈ +1.7); lane-center offset (midpoint) **+ = car LEFT of center**; `steeringAngleDeg` + = LEFT.

**Out of scope (deferred):** F1a/F1b (curvature_rate FF). Go/no-go comes from A/B Leg 1: if the current build fails the nudge-release arrest test the way the decisive event did, F1b (ungated small-command rate FF) gets its own plan + Ford safety suite work; if Leg 1 is clean, F1a remains a curve-entry-feel nicety only.

---

## Part A — F3: AOL blindness/departure safeguard (main repo)

Design (from handoff v3 §5, QA-round-1-corrected):
- **Low-confidence monitor**: EMA(inner laneProb, tau 0.5 s) < 0.30 for ≥ 1.0 s, lateral active, v > 10 m/s → warning alert with chime, re-raised while the condition holds. Suppressed only during the FIRST 2.0 s of a maneuver (lane-change line reassignment is legitimately blind ~1-2 s; blindness persisting past 2 s into a maneuver still alerts — required for the ce 260.2 event).
- **Departure monitor**: confidence ≥ 0.6 (sighted) AND [ |offset| > 0.40 m AND offset moving away from center at > 0.15 m/s, sustained 0.5 s ] OR [ nearest inner line < 0.35 m ] → warning alert with chime. Fully suppressed during a maneuver and for 3.0 s after (offset legitimately sweeps ±1.7 m during lane changes).
- "Maneuver" = `modelV2.meta.laneChangeState != off` OR either blinker.
- Kill-switch param `AolSafeguardDisabled` (unset ⇒ safeguard ON).

### Task A1: Worktree + branch setup

**Files:** none (environment)

- [ ] **Step 1: Create the main-repo worktree on a new branch**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot
git worktree add ../sp-aol-f3 -b aol-fixes-f3 clean-v2026.002.001
```

Expected: `Preparing worktree (new branch 'aol-fixes-f3')`, HEAD at `acd8518e99` (or current clean-v2026.002.001 tip).

- [ ] **Step 2: Verify the worktree python environment works**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot/../sp-aol-f3
PYTHONPATH=$PWD /Users/dregilley/Documents/GitHub/sunnypilot/.venv311/bin/python -c "from cereal import custom; print(custom.OnroadEventSP.EventName.e2eChime)"
```

Expected: prints `e2eChime` (or its enum repr). If capnp import fails, run `git submodule update --init --recursive` in the worktree first and retry.

Note for all Part A tasks: **WORKTREE = `/Users/dregilley/Documents/GitHub/sp-aol-f3`**, **VENV_PY = `/Users/dregilley/Documents/GitHub/sunnypilot/.venv311/bin/python`**, and run python as `cd $WORKTREE && PYTHONPATH=$PWD:$PWD/opendbc_repo $VENV_PY ...` (opendbc_repo on the path — cereal's log.capnp imports car.capnp).

**ENVIRONMENT REALITY (confirmed 2026-07-02):** this is an analysis venv, not a device build. `common/params_pyx` is NOT compiled and `pytest-xdist` is NOT installed (repo-wide, not this branch). Consequences:
- The repo-root `conftest.py` imports `params_pyx` and `pyproject.toml addopts` demand `-n auto --dist=loadgroup` → any bare `pytest` that loads them dies before reaching tests.
- Pure-python unit tests run in isolation with: `... -m pytest <path> --noconftest -o addopts=""` (used for the monitor tests — they need only numpy + `collections.deque`).
- Modules that import `openpilot.common.params` (that's `selfdrive/selfdrived/selfdrived.py`, `sunnypilot/.../events.py` via its import chain) CANNOT be imported or pytest-run in this venv. For A5/A7, verify with `py_compile` (syntax) + AST/structural checks + the **A8 replay harness** (the real functional gate — it needs only LogReader+cereal+the monitor, no params). The full `selfdrive/selfdrived/` pytest suite is a **device/CI gate**, added to Task D1 pre-deploy.

### Task A2: New event enum values in cereal

**Files:**
- Modify: `cereal/custom.capnp` (enum `OnroadEventSP.EventName`, currently ends at `e2eChime @23;` — verified line ~344)

- [ ] **Step 1: Add the two new enum entries (append-only; never renumber)**

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

- [ ] **Step 2: Verify the enum loads**

```bash
cd $WORKTREE && PYTHONPATH=$PWD $VENV_PY -c "from cereal import custom; print(int(custom.OnroadEventSP.EventName.aolLowLaneConfidence), int(custom.OnroadEventSP.EventName.aolLaneDeparture))"
```

Expected: `24 25`

- [ ] **Step 3: Commit**

```bash
cd $WORKTREE && git add cereal/custom.capnp && git commit -m "cereal: add aolLowLaneConfidence/aolLaneDeparture OnroadEventSP events"
```

### Task A3: The monitor class — failing tests first

**Files:**
- Test: `sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py` (create; note `sunnypilot/selfdrive/selfdrived/` has an `__init__.py` already; create `tests/__init__.py` if pytest collection needs it)

- [ ] **Step 1: Write the failing tests (complete file)**

```python
import numpy as np

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

  def test_inner_line_proximity_backstop(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(30, right_line_dist=0.25, lane_offset=-1.2))
    assert any(r[1] for r in res)

  def test_lane_change_sweep_suppressed(self):
    mon = AolSafeguardMonitor()
    # commanded lane change: offset sweeps to -1.6 m and back over 4 s, maneuver=True,
    # then 3 s grace; no departure alert anywhere
    sweep = [dict(lane_offset=-1.6 * np.sin(np.pi * i / 80), maneuver=True) for i in range(80)]
    settle = [dict(lane_offset=-0.3 + 0.3 * min(1.0, i / 40)) for i in range(60)]
    res = run(mon, steps(40) + sweep + settle)
    assert not any(r[1] for r in res)

  def test_blind_never_fires_departure(self):
    mon = AolSafeguardMonitor()
    res = run(mon, steps(40) + steps(60, inner_prob=0.05, lane_offset=-1.0))
    assert not any(r[1] for r in res)  # low-conf monitor owns the blind case
```

- [ ] **Step 2: Run tests, verify they fail on import**

```bash
cd $WORKTREE && PYTHONPATH=$PWD $VENV_PY -m pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py -x -q
```

Expected: `ModuleNotFoundError: No module named 'openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor'`

- [ ] **Step 3: Commit the failing tests**

```bash
cd $WORKTREE && git add sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py && git commit -m "test: AOL safeguard monitor spec (failing)"
```

### Task A4: The monitor class — implementation

**Files:**
- Create: `sunnypilot/selfdrive/selfdrived/aol_monitor.py`

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
cd $WORKTREE && PYTHONPATH=$PWD $VENV_PY -m pytest sunnypilot/selfdrive/selfdrived/tests/test_aol_monitor.py -v
```

Expected: all 9 tests PASS (4 in TestLowConfidence, 5 in TestDeparture). Use the isolation flags per the ENVIRONMENT REALITY note: `... -m pytest <path> -v --noconftest -o addopts=""`. If a timing assertion fails by 1-2 frames, fix the TEST bound only if hand-computation of the EMA/qualifier arithmetic supports it; otherwise fix the code.

- [ ] **Step 3: Commit**

```bash
cd $WORKTREE && git add sunnypilot/selfdrive/selfdrived/aol_monitor.py && git commit -m "sunnypilot: AOL safeguard monitor (low-confidence + departure)"
```

### Task A5: Alert definitions in EVENTS_SP

**Files:**
- Modify: `sunnypilot/selfdrive/selfdrived/events.py` (the `EVENTS_SP` dict; follow the `laneTurnLeft` entry pattern at ~line 195)

- [ ] **Step 1: Add the two event entries**

Insert into the `EVENTS_SP` mapping (after the `laneTurnRight` entry):

```python
  EventNameSP.aolLowLaneConfidence: {
    ET.WARNING: Alert(
      "Lane detection lost",
      "Take control",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.promptRepeat, .2),
  },

  EventNameSP.aolLaneDeparture: {
    ET.WARNING: Alert(
      "Lane departure",
      "Take control",
      AlertStatus.userPrompt, AlertSize.mid,
      Priority.MID, VisualAlert.steerRequired, AudibleAlert.warningSoft, .2),
  },
```

(duration .2 s + the monitor re-raising every frame ⇒ continuous alert while the condition holds; `AudibleAlert.promptRepeat` loops the chime.)

- [ ] **Step 2: Run the existing alert-construction test suites**

```bash
cd $WORKTREE && PYTHONPATH=$PWD $VENV_PY -m pytest selfdrive/selfdrived/tests/test_alerts.py -q
cd $WORKTREE && PYTHONPATH=$PWD $VENV_PY -m pytest sunnypilot/selfdrive/selfdrived/ -q
```

Expected: PASS (test_alerts iterates every event's alerts; a malformed entry fails here).

- [ ] **Step 3: Commit**

```bash
cd $WORKTREE && git add sunnypilot/selfdrive/selfdrived/events.py && git commit -m "sunnypilot: AOL safeguard alerts (lane detection lost / lane departure)"
```

### Task A6: Param kill-switch registration

**Files:**
- Modify: `common/params_keys.h`

⚠️ Per project discipline (memory `feedback_param_key_registration`): the key MUST be registered before any read/write — an unregistered key raises UnknownKeyName on write and silently returns None on read.

- [ ] **Step 1: Register `AolSafeguardDisabled`**

⚠️ VERIFIED FORMAT (controller, 2026-07-02): this branch's `params_keys.h` uses a TYPED struct — `{"key", {FLAGS, TYPE, "default"}}` — NOT the bare `{"Key", PERSISTENT}` older format. The sunnypilot lateral keys live together (see `{"enable_lane_positioning", {PERSISTENT, BOOL, "0"}}` at line ~41, and `FordPath4Enabled`/`LaneBiasIntegral` right below). Add the new key in that cluster:

```c
    {"AolSafeguardDisabled", {PERSISTENT, BOOL, "0"}},
```

Default `"0"` = false ⇒ safeguard ON by default (A7 reads `not get_bool("AolSafeguardDisabled")`).

- [ ] **Step 2: Verify registration**

Note: the Params C++ keys are read from `params_keys.h` at import; the analysis venv can construct a Params instance with a temp path. If `get_bool` isn't importable in this venv (params_pyx uncompiled — see ENVIRONMENT REALITY), fall back to a grep assertion that the line is present and well-formed, and defer the runtime check to CI/device.

```bash
cd $WORKTREE && grep -n '"AolSafeguardDisabled", {PERSISTENT, BOOL, "0"}' common/params_keys.h && echo "registered"
```

Expected: prints the line + `registered`.

- [ ] **Step 3: Commit**

```bash
cd $WORKTREE && git add common/params_keys.h && git commit -m "params: register AolSafeguardDisabled"
```

### Task A7: Wire the monitor into selfdrived

**Files:**
- Modify: `selfdrive/selfdrived/selfdrived.py`
  - imports (top, near `from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP` at ~line 32)
  - `__init__` (near `self.events_sp = EventsSP()` at ~line 170)
  - `update_events` (immediately after the laneTurn block that ends at ~line 341 with `self.events_sp.add(custom.OnroadEventSP.EventName.laneTurnRight)`)

- [ ] **Step 1: Add import**

```python
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
```

- [ ] **Step 2: Instantiate in `__init__` (next to `self.events_sp = EventsSP()`)**

```python
    self.aol_monitor = AolSafeguardMonitor()
    self.aol_safeguard_enabled = not self.params.get_bool("AolSafeguardDisabled")
```

(`self.params` already exists in selfdrived's `__init__`; if the attribute is named differently on this branch, use the existing Params instance — do NOT create a second one.)

- [ ] **Step 3: Step the monitor in `update_events` after the laneTurn block**

```python
    # AOL lateral safeguard (2026-07-01 incident): alert instead of silently
    # steering through model blindness or a developing lane departure.
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
```

Adaptation notes (CONTROLLER-VERIFIED 2026-07-02 against this branch — anchors confirmed):
- `LaneChangeState` is already imported (used at line 322). ✓
- `CS` is the carState arg of `update_events(self, CS)` (line 180). ✓
- `self.active` (line 128, set line 609) and `self.mads.active` (line 587) both exist. ✓ `self.params` is the existing Params() (line 57) — reuse it. ✓ Import anchor line 32, laneTurn block ends line 341. ✓
- **CADENCE (critical):** `selfdrived` runs at 100 Hz (`Ratekeeper(100)`, line 148) but `modelV2` is 20 Hz. The `if ... self.sm.updated['modelV2']` guard is ESSENTIAL — it steps the monitor only on fresh modelV2 frames (≈20 Hz), which is what the monitor's `dt=0.05` (EMA alpha, rate window) assumes. Do NOT step it every control frame.
- **`t` source:** `self.sm.logMonoTime['modelV2'] * 1e-9` is the standard SubMaster API (a per-service last-recv-time dict; long-stable). It could not be runtime-checked here (`msgq` absent in the analysis venv). If access differs on-device, a valid fallback is `self.sm.frame * DT_CTRL` — the monitor uses `t` ONLY for elapsed-time differences, so any monotonic seconds source works.
- The `elif` (departure wins over low-conf) is deliberate: never stack both alerts.

- [ ] **Step 4: Syntax check (analysis venv) — full import/suite is a CI/device gate**

```bash
cd $WORKTREE && PYTHONPATH=$PWD:$PWD/opendbc_repo $VENV_PY -m py_compile selfdrive/selfdrived/selfdrived.py sunnypilot/selfdrive/selfdrived/events.py && echo "py_compile OK"
```

Expected: `py_compile OK`. NOTE (ENVIRONMENT REALITY): `import selfdrive.selfdrived.selfdrived` and `pytest selfdrive/selfdrived/tests/` CANNOT run in this analysis venv (they pull `openpilot.common.params` → uncompiled `params_pyx`; suite also needs the declared-but-absent `hypothesis`/`pytest-xdist`). The functional gate for the wiring is Task A8 (replay harness, which drives the monitor with the exact selfdrived inputs). The full `selfdrive/selfdrived/` pytest suite runs at the **D1 pre-deploy CI/device gate**.

- [ ] **Step 5: Commit**

```bash
cd $WORKTREE && git add selfdrive/selfdrived/selfdrived.py && git commit -m "selfdrived: wire AOL safeguard monitor into update_events"
```

### Task A8: Offline replay validation against the incident logs (the acceptance gate)

**Files:**
- Create: `retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py` (in the MAIN checkout, not the worktree — it's an analysis script; it imports the monitor from the worktree path)

- [ ] **Step 1: Write the replay harness (complete file)**

```python
#!/usr/bin/env python3
"""F3 acceptance gate: replay recorded rlogs through AolSafeguardMonitor.

MUST-PASS gates (from handoff v3 §5/§9b):
  G1 ce decisive event (override at mono 212.54): departure alert >= 1.0 s before override.
  G2 cf blowout (deep-blind at 682.21): low-conf alert within 2.5 s of collapse onset.
  G3 ce blind-in-lane-change (blind 257.31, override 260.18): low-conf alert BEFORE the
     override despite the maneuver (suppression must lift after 2 s).
  G4 departure-alert count on OLD routes (c5/c7/b5/7f) == 0 (no real departures occurred).
REPORT-ONLY (user tunes thresholds if excessive):
  R1 low-conf alert episode count + total alert seconds per route (fatigue check).

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo:/Users/dregilley/Documents/GitHub/sp-aol-f3 \
      .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py
(the sp-aol-f3 worktree path LAST so production code resolves from the main tree and only
 aol_monitor comes from the branch — verify with the printed module path)
"""
import glob, sys
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
WORKTREE = "/Users/dregilley/Documents/GitHub/sp-aol-f3"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo"); sys.path.append(WORKTREE)

from openpilot.tools.lib.logreader import LogReader
sys.path.insert(0, WORKTREE)
from openpilot.sunnypilot.selfdrive.selfdrived import aol_monitor as am_mod
from openpilot.sunnypilot.selfdrive.selfdrived.aol_monitor import AolSafeguardMonitor
print("monitor module:", am_mod.__file__)
assert WORKTREE in am_mod.__file__, "monitor must come from the aol-fixes-f3 worktree"


def replay(route):
    """Feed the monitor exactly what selfdrived would see; return alert timelines."""
    rows = []      # (t, kind) kind in {"lc","dep"}
    cs_state = dict(v=0.0, lb=False, rb=False)
    lat = dict(active=False)
    mon = AolSafeguardMonitor()
    for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/{route}/*/rlog.zst")):
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
```

- [ ] **Step 2: Run it**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py
```

Expected: `G1..G4 PASS`. Runtime ~10-15 min (six full-route rlog passes).
- If G1 fails marginally (alert < 1.0 s early): re-check the offset-rate window arithmetic first; only then consider DEPART_OFFSET 0.40→0.35 — and re-run G4 + R1 to confirm no false-alert cost.
- If G4 fails: print the offending timestamps, pull that window's context (speed/blinker/laneChangeState), and tighten the maneuver detection or DEPART_RATE — never gate G4 out.
- R1 sanity expectation: low-conf episodes on OLD routes ≈ their genuine non-maneuver blindness episodes (order 5-15 per route; these are TRUE positives — the user judges fatigue and may raise CONF_QUALIFIER).

- [ ] **Step 3: Commit the harness (main checkout, investigation branch — it's an analysis artifact)**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && git add retrospective_lateral/incident_2026_07_01/scripts/f3_replay_check.py && git commit -m "analysis: F3 safeguard replay acceptance gate" 2>/dev/null || echo "commit on investigation branch per its usual flow"
```

(If the investigation tree's commit conventions differ, leave the file uncommitted — the user manages that tree.)

### Task A9: F3 review checkpoint (user QA)

- [ ] Present to the user: the monitor diff, the R1 alert-load table, and the G1-G4 gate results. **STOP — device deployment only after user approval** (deployment steps in Part D).

---

## Part B — F2: PI de-lag (opendbc, branch off the device pin)

Design (handoff v3 §4, QA-2-corrected scope — this is the WEAVE fix, not the takeover fix):
- **B-1 (unconditional):** integral cap back to fixed ±0.30 (kills the highway integral windup: the decisive event's I-term at cap was ±0.000163; at 0.30 it is ±0.00006).
- **B-2 (live-toggleable, `golden2` in `/data/lc_pi_config`):** (i) P-term freeze while the EMA'd offset disagrees in sign with the RAW offset (the EMA is stale ~1.5 s through every reversal); (ii) zero-crossing integral decay 0.97 → 0.90 per 20 Hz frame (unwinds a saturated integral in ~1 s instead of ~3.4 s).
- NO hard integral reset (would re-expose the structural +0.22 m left bias the integral cancels).

### Task B1: opendbc worktree + baseline — DONE by controller (2026-07-02)

Worktree created: `/Users/dregilley/Documents/GitHub/opendbc-f2` at `849b72a1a` (branch `pi-delag-f2`, verified HEAD 849b72a1a).

**Scope clarification (important):** F2 modifies `opendbc/car/ford/carcontroller.py` (the CONTROLS layer) — it does NOT touch `opendbc/safety/**` / `ford.h` (the panda SAFETY layer) and changes no safety limits or message packing. So the user's "must pass the upstream Ford **safety** suite" requirement (panda `test_ford.py` safety) binds only the DEFERRED F1b, NOT F2/F3.

**Test-suite reality (confirmed):** the opendbc car-behavior suite `opendbc/car/ford/tests/test_ford.py` requires `hypothesis` (declared in pyproject `hypothesis ==6.47.*`, absent in the analysis venv) → it **cannot run here**; it is a **CI/device gate** (Task D2 pre-deploy). The analysis-venv acceptance gate for F2 is Task B4 (the PI counterfactual replay of the decisive event on the recorded logs — needs only LogReader, runs fully). Do NOT pip-install into the shared venv without user sign-off.

### Task B2: B-1 — integral cap to fixed 0.30

**Files:**
- Modify: `opendbc/car/ford/carcontroller.py:347` (verified content at 849b72a1a)

- [ ] **Step 1: Apply the change**

Line 347, change:

```python
              int_cap = 0.30 if self._pi_cfg == "weak" else float(np.interp(CS.out.vEgoRaw, [20., 30.], [0.3, 1.0]))
```

to:

```python
              # 2026-07-01 incident (handoff v3): the speed-interp cap to 1.0 was never
              # highway-validated and let the integral store ±0.82 of wrong-way authority
              # at 62 mph (I-term ±0.000163 vs model corrections ~0.0003-0.001). No route's
              # telemetry needs >0.3 to hold center (c7 centered median |int| 0.13).
              int_cap = 0.30
```

- [ ] **Step 2: Import check + commit**

```bash
cd /Users/dregilley/Documents/GitHub/opendbc-f2 && PYTHONPATH=$PWD /Users/dregilley/Documents/GitHub/sunnypilot/.venv311/bin/python -c "import opendbc.car.ford.carcontroller"
git -C /Users/dregilley/Documents/GitHub/opendbc-f2 add opendbc/car/ford/carcontroller.py
git -C /Users/dregilley/Documents/GitHub/opendbc-f2 commit -m "ford PI: fix integral cap at 0.30 (2026-07-01 incident, weave amplifier)"
```

### Task B3: B-2 — `golden2` reversal handling (live-toggleable)

**Files:**
- Modify: `opendbc/car/ford/carcontroller.py` lines 353-354 (zero-crossing decay), 368-377 (cfg parser + P-term)

- [ ] **Step 1: Extend the config parser (lines 368-375)**

Change:

```python
            self._pi_cfg_ctr += 1
            if self._pi_cfg_ctr % 50 == 0:
              try:
                _c = open("/data/lc_pi_config").read().strip().lower()
                self._pi_cfg = _c if _c in ("weak", "golden") else "golden"
              except Exception:
                self._pi_cfg = "golden"
            self.lc_kp = 0.0001 if self._pi_cfg == "weak" else 0.0005
```

to:

```python
            self._pi_cfg_ctr += 1
            if self._pi_cfg_ctr % 50 == 0:
              try:
                _c = open("/data/lc_pi_config").read().strip().lower()
                self._pi_cfg = _c if _c in ("weak", "golden", "golden2") else "golden"
              except Exception:
                self._pi_cfg = "golden"
            self.lc_kp = 0.0001 if self._pi_cfg == "weak" else 0.0005
```

- [ ] **Step 2: Faster zero-crossing decay under golden2 (lines 353-354)**

Change:

```python
              if lane_offset * self.lane_centering_integral < 0:  # offset and integral disagree
                self.lane_centering_integral *= 0.97  # gentle decay toward zero
```

to:

```python
              if lane_offset * self.lane_centering_integral < 0:  # offset and integral disagree
                # golden2 (2026-07-01 incident): 0.97/frame leaves a saturated integral
                # anti-correcting for ~2 s after every reversal; 0.90 unwinds it in ~1 s.
                self.lane_centering_integral *= 0.90 if self._pi_cfg == "golden2" else 0.97
```

- [ ] **Step 3: P-term freeze on stale EMA under golden2 (line 376)**

Change:

```python
            pi_p = self.lc_kp * lane_offset
```

to:

```python
            pi_p = self.lc_kp * lane_offset
            # golden2: the 1.5 s EMA is stale through every reversal — while its sign
            # disagrees with the raw offset, its P push is wrong-way; zero it.
            if self._pi_cfg == "golden2" and lane_offset * lane_offset_raw < 0:
              pi_p = 0.0
```

(`lane_offset_raw` is in scope — computed a few lines above the EMA at ~line 331.)

- [ ] **Step 4: Import check + Ford suite vs baseline**

```bash
cd /Users/dregilley/Documents/GitHub/opendbc-f2 && PYTHONPATH=$PWD /Users/dregilley/Documents/GitHub/sunnypilot/.venv311/bin/python -m pytest opendbc/car/ford/ -q 2>&1 | tail -3
diff <(tail -1 /tmp/f2_baseline.txt) <(echo "<paste the new tail line>") || true
```

Expected: identical pass/fail counts to the Task B1 baseline (F2 touches no safety limits, no message packing).

- [ ] **Step 5: Commit**

```bash
git -C /Users/dregilley/Documents/GitHub/opendbc-f2 add opendbc/car/ford/carcontroller.py
git -C /Users/dregilley/Documents/GitHub/opendbc-f2 commit -m "ford PI: golden2 config — P-freeze on stale-EMA reversal + faster integral zero-crossing decay"
```

### Task B4: Pre-drive falsification — replay the decisive event through old vs new PI params

**Files:**
- Create: `retrospective_lateral/incident_2026_07_01/scripts/f2_replay_check.py` (main checkout)

- [ ] **Step 1: Write the check (complete file)**

```python
#!/usr/bin/env python3
"""F2 acceptance gate: re-run the decisive event's PI state (integral + P) under the
old params vs B-1/B-2, using the RECORDED raw/EMA offsets from route_ce as input.
The PI recursion is self-contained (offset -> ema -> integral -> P+I), so this is an
exact counterfactual for the PI contribution — not a vehicle simulation.

GATES (decisive event, override at mono 212.54; onset = 209.2, the center crossing):
  G1 golden+cap0.30 (B-1): |PI wrong-way contribution at onset| <= 0.00016
     (was +0.000248 measured; cap 0.30 bounds I to 0.00006 and P is unchanged)
  G2 golden2 (B-1+B-2): PI wrong-way contribution during 209.2..211.2 mean <= 0.00006
     AND the integral crosses 0 within 1.2 s of the raw-offset sign flip (was ~3.4 s)
  G3 steady-state: over route_ce straights (|desired|<0.0005, conf>0.8), mean |ema|
     under golden2 differs from golden by < 0.03 m  (centering not degraded)

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
      retrospective_lateral/incident_2026_07_01/scripts/f2_replay_check.py
"""
import glob, math, sys
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader

KI = 0.0002
KP = 0.0005
DT = 0.05


def pi_series(raw_offsets, cfg):
    """cfg in {'golden','golden_cap30','golden2'}; replicates carcontroller lines 331-378."""
    ema = 0.0
    integral = 0.0
    out = []
    alpha = 1.0 - math.exp(-DT / 1.5)
    for raw in raw_offsets:
        ema = alpha * raw + (1 - alpha) * ema
        integral += ema * DT                      # accumulate-gate assumed open (straight, unpressed)
        cap = 0.84 if cfg == "golden" else 0.30   # 0.84 = interp cap at the event's 27.7 m/s
        integral = float(np.clip(integral, -cap, cap))
        if ema * integral < 0:
            integral *= 0.90 if cfg == "golden2" else 0.97
        pi_p = KP * ema
        if cfg == "golden2" and ema * raw < 0:
            pi_p = 0.0
        out.append((pi_p, KI * integral, ema, integral))
    return out


# -- load the event window's recorded raw midpoint offsets (202..213 s) ---------------
raws, ts = [], []
for rl in sorted(glob.glob(f"{ROOT}/explorer_st_logs/route_ce/*/rlog.zst")):
    try:
        lr = LogReader(rl)
    except Exception:
        continue
    for m in lr:
        try:
            w = m.which()
        except Exception:
            continue
        if w == "modelV2":
            t = m.logMonoTime * 1e-9
            if 195.0 <= t <= 213.0:
                ll = list(m.modelV2.laneLines)
                if len(ll) >= 3 and len(ll[1].y):
                    ts.append(t); raws.append((ll[1].y[0] + ll[2].y[0]) / 2.0)
ts = np.array(ts); raws = np.array(raws)
print(f"event window frames: {len(ts)}")

res = {c: pi_series(raws, c) for c in ("golden", "golden_cap30", "golden2")}
onset = np.searchsorted(ts, 209.2)
w12 = (ts >= 209.2) & (ts <= 211.2)

for c, series in res.items():
    pi = np.array([p + i for (p, i, _, _) in series])
    # wrong-way = PI positive (rightward) while the correct direction is left (raw < 0)
    ww_onset = pi[onset]
    ww_mean = np.mean(np.clip(pi[w12], 0, None))
    print(f"{c:14s} PI@onset={ww_onset:+.6f}  wrong-way mean(209.2-211.2)={ww_mean:+.6f}")

g1 = abs(res["golden_cap30"][onset][0] + res["golden_cap30"][onset][1]) <= 0.00016
pi2 = np.array([p + i for (p, i, _, _) in res["golden2"]])
g2a = np.mean(np.clip(pi2[w12], 0, None)) <= 0.00006
flip_i = onset  # raw offset flips sign at the center crossing = onset by construction
ints2 = np.array([itg for (_, _, _, itg) in res["golden2"]])
zc = np.where(np.sign(ints2[flip_i:]) != np.sign(ints2[flip_i]))[0]
g2b = len(zc) > 0 and (ts[flip_i + zc[0]] - ts[flip_i]) <= 1.2
print(f"G1 cap-0.30 onset PI bound: {'PASS' if g1 else 'FAIL'}")
print(f"G2 golden2 wrong-way + integral-unwind: {'PASS' if (g2a and g2b) else 'FAIL'} (mean ok={g2a}, unwind ok={g2b})")
print("G3: run the whole-route steady-state comparison per the docstring (add if time permits; "
      "QA2 already bounded the cost analytically at <= 0.05-0.1 m).")
```

- [ ] **Step 2: Run + record numbers**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f2_replay_check.py
```

Expected: G1 PASS, G2 PASS, and the printed golden line should reproduce ≈ +0.00024-0.00026 at onset (cross-check against the measured event — if it does not, the recursion has a bug; fix before trusting G1/G2).

- [ ] **Step 3: Commit (main checkout, same convention as Task A8 step 3)**

### Task B5: F2 review checkpoint (user QA)

- [ ] Present: the 3-hunk diff, baseline-vs-after test counts, f2_replay_check numbers. **STOP — deployment with Part D only after approval.** Drive semantics: `echo golden > /data/lc_pi_config` = B-1 only; `echo golden2 > /data/lc_pi_config` = B-1+B-2 — togglable between A/B legs without rebuild.

---

## Part C — A/B drive protocol + analysis tooling

### Task C1: Arrest-metrics analysis script

**Files:**
- Create: `retrospective_lateral/incident_2026_07_01/scripts/arrest_metrics.py` (main checkout)

- [ ] **Step 1: Write the script (complete file)**

```python
#!/usr/bin/env python3
"""A/B drive analysis: nudge-release arrest metrics + PI/EPS health, per route.

For each nudge-release episode (steeringPressed with |torque|>1 Nm for 0.3-2 s, then
released, latActive throughout, v>18, laneProb>0.6):
  - injected lateral velocity (offset slope over the 0.5 s after release)
  - peak |offset| excursion within 8 s of release
  - time-to-arrest: release -> first time |offset| stops growing AND |offset rate|<0.05 m/s
  - wheel-tracking: corr + best lag between carOutput.curvature and the baseline-relative
    achieved wheel curvature (VehicleModel) over release..release+4 s
  - PI state from LC telemetry (integral trajectory through the episode)
Prints one row per episode + per-route medians. Compare Leg-1 (baseline) vs Leg-2 (F2).

RUN:  PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python \
      retrospective_lateral/incident_2026_07_01/scripts/arrest_metrics.py <route_dir_name>...
"""
import glob, bisect, math, re, sys
import numpy as np

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/opendbc_repo")
from openpilot.tools.lib.logreader import LogReader
from opendbc.car.vehicle_model import VehicleModel

# (loader identical in shape to event_attribution.py — carState/carOutput/carControl/
#  modelV2/LC-telemetry sorted lists; reuse that file's load() by import:)
sys.path.insert(0, ROOT + "/retrospective_lateral/incident_2026_07_01/scripts")
from event_attribution import load, win, ach_curv  # noqa: E402


def episodes(d):
    """nudge-release: pressed>=1Nm for 0.3-2.0s then released with latActive, v>18."""
    cs = d["cs"]
    eps = []
    i = 0
    while i < len(cs):
        t, v, sa, sp, tq, blk = cs[i]
        if sp and abs(tq) > 1.0 and v > 18:
            j = i
            while j < len(cs) and cs[j][3]:
                j += 1
            dur = cs[min(j, len(cs) - 1)][0] - t
            if 0.3 <= dur <= 2.0 and j < len(cs):
                eps.append((cs[j][0], v))     # release time
            i = j + 1
        else:
            i += 1
    return eps


def analyze(route):
    d = load(route)
    mvt = [x[0] for x in d["mv"]]; cst = [x[0] for x in d["cs"]]; cot = [x[0] for x in d["co"]]
    print(f"\n==== {route} ====")
    rows = []
    for (tr, v) in episodes(d):
        mvw = win(d["mv"], mvt, tr, tr + 8.0)
        offs = [(x[0], x[3]) for x in mvw if np.isfinite(x[3]) and x[2] > 0.6]
        if len(offs) < 60:
            continue
        t0s = np.array([o[0] - tr for o in offs]); os_ = np.array([o[1] for o in offs])
        v_lat = np.polyfit(t0s[t0s < 0.5], os_[t0s < 0.5], 1)[0] if (t0s < 0.5).sum() > 4 else np.nan
        peak = float(np.max(np.abs(os_)))
        # time-to-arrest: |offset| stops growing and rate small
        arrest = np.nan
        for k in range(10, len(os_) - 10):
            r = np.polyfit(t0s[k - 5:k + 5], os_[k - 5:k + 5], 1)[0]
            if abs(r) < 0.05 and abs(os_[k]) >= abs(os_[k - 5]) - 0.02:
                arrest = t0s[k]; break
        # wheel tracking over release..+4s
        cw = win(d["co"], cot, tr, tr + 4.0)
        aw = win(d["cs"], cst, tr, tr + 4.0)
        base = np.median([ach_curv(x[2], x[1]) for x in win(d["cs"], cst, tr - 1.5, tr)])
        cmds = np.interp([x[0] for x in aw], [x[0] for x in cw], [x[1] for x in cw])
        achs = np.array([ach_curv(x[2], x[1]) - base for x in aw])
        lags = range(0, 30)
        best = max(((np.corrcoef(cmds[:-L or None], achs[L:])[0, 1] if L else np.corrcoef(cmds, achs)[0, 1], L)
                    for L in lags), key=lambda z: (z[0] if np.isfinite(z[0]) else -9))
        rows.append((tr, v, v_lat, peak, arrest, best[1] * 0.01, best[0]))
        print(f"  release t={tr:8.2f} v={v * 2.237:3.0f}mph  vlat={v_lat:+.2f}m/s  "
              f"peak={peak:.2f}m  arrest={arrest if np.isfinite(arrest) else float('nan'):.2f}s  "
              f"wheel lag={best[1] * 0.01:.2f}s corr={best[0]:.2f}")
    if rows:
        a = np.array([[r[3], r[4], r[6]] for r in rows if np.isfinite(r[4])])
        if len(a):
            print(f"  MEDIANS: peak={np.median(a[:, 0]):.2f}m  arrest={np.median(a[:, 1]):.2f}s  wheelCorr={np.median(a[:, 2]):.2f}")


for r in (sys.argv[1:] or ["route_ce"]):
    analyze(r)
```

- [ ] **Step 2: Smoke-test on the incident route (the decisive event's nudge should appear)**

```bash
cd /Users/dregilley/Documents/GitHub/sunnypilot && PYTHONPATH=$PWD:$PWD/opendbc_repo .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/arrest_metrics.py route_ce
```

Expected: at least the t≈206-207 release row (the decisive event's nudge at 62 mph) with peak ≈ 0.85 m and arrest = nan-or->3 s (it was never arrested — the driver overrode). This validates the extractor against known ground truth. Note: the aw/cmds/achs interpolation assumes carState 100 Hz vs carOutput 100 Hz — if array lengths mismatch, align on the shorter.

- [ ] **Step 3: Commit (main checkout convention)**

### Task C2: The written drive protocol

**Files:**
- Create: `docs/superpowers/plans/2026-07-02-ab-drive-protocol.md`

- [ ] **Step 1: Write the protocol document (complete content)**

```markdown
# A/B Drive Protocol — AOL incident fixes (run only after F3 is deployed + approved)

Safety rules (all legs): empty multi-lane road, dry, daylight, no traffic within 5 s
headway; hands hovering the wheel; abort a maneuver at ANY doubt. DisableUpdates=1.
F3 safeguard ACTIVE on every leg. Car otherwise stays parked.

## Leg 0 — F3 shakedown (30 min, any familiar route incl. one highway stretch)
Purpose: F3 alert behavior in the real car (timing feel, false-alert rate, volume).
Config: current build + F3 branch; /data/lc_pi_config = golden.
Record: normal drive; note every alert (expected vs surprising).
Pass: no alert storms; lane changes suppressed correctly; any blindness alert matches
a real display line-loss.

## Leg 1 — nudge-release arrest baseline (the mechanism-C probe)
Same corridor both directions, 3 speeds (45 / 55 / 62 mph), latActive (AOL), lc_pi_config=golden.
Maneuver x4 per speed per direction: from centered lane-keeping, apply a firm 0.5-1 s
nudge (~like the incident: enough to visibly displace), RELEASE fully, hands hover,
let the system arrest. Take over at 0.9 m displacement or lane edge, whichever first.
Analysis: arrest_metrics.py on the new route. KEY QUESTION: do any episodes reproduce
the decisive event's signature (peak >0.7 m, wheel-tracking corr <0.8 / lag >0.3 s)?
  -> If yes at any speed: mechanism C reproduces on the current build -> F1b planning
     goes live; collect ≥6 such episodes for the F1b baseline.
  -> If no: C was situational (road/EPS-state that day); F2 + F3 close the incident.

## Leg 2 — F2 A/B (same corridor, same day if possible)
2a: echo golden  > /data/lc_pi_config   (B-1 only: cap 0.30) — repeat Leg-1 maneuvers x2/speed
2b: echo golden2 > /data/lc_pi_config   (B-1+B-2) — repeat x2/speed
Plus 10 min of plain lane-keeping per config at highway speed (weave/centering feel).
Analysis: arrest_metrics.py legs 1 vs 2a vs 2b + the standard weave band metric.
Success = peak excursions and arrest times shrink toward the c7 baseline distribution
(0.19-0.70 m); centering (median |offset|) within 0.05 m of Leg 1. NOT expected: takeover
prevention (that is mechanism C, not the PI).

## Data to pull after each leg
rlogs for the whole drive (+ fcamera for any surprising event), then run:
  arrest_metrics.py, f3 alert-load report, LC/CX1 telemetry extraction.
```

- [ ] **Step 2: Commit (main checkout convention)**

---

## Part D — Deployment / rollback (each step user-approved, in order)

### Task D1: Deploy F3 (after Task A9 approval)

- [ ] **Step 1:** Push branches: `git -C $WORKTREE push origin aol-fixes-f3` (main repo only — F3 touches no opendbc code).
- [ ] **Step 2:** On device (single SSH session): `cd /data/openpilot && git fetch origin aol-fixes-f3 && git checkout aol-fixes-f3` — confirm `DisableUpdates=1` still set; reboot deliberately.
- [ ] **Step 3:** Bench check while PARKED: engage AOL in the driveway (wheels straight, engine on), cover the road camera → within ~2 s the "Lane detection lost / Take control" alert must appear + chime. Uncover → alert clears.
- [ ] **Rollback:** `git checkout clean-v2026.002.001` + reboot.

### Task D2: Deploy F2 (after Task B5 approval + Leg 1 complete)

- [ ] **Step 1:** Push: `git -C /Users/dregilley/Documents/GitHub/opendbc-f2 push origin pi-delag-f2`.
- [ ] **Step 2:** On device: update the opendbc checkout to `pi-delag-f2` (same mechanism the migration used for 849b72a1 — fetch + checkout in the opendbc tree), reboot.
- [ ] **Step 3:** Confirm active config: `cat /data/lc_pi_config` (absent/`golden` = B-1 only).
- [ ] **Rollback:** checkout `849b72a1a` in the opendbc tree + reboot. Config-level rollback: `echo golden > /data/lc_pi_config` reverts B-2 instantly without rebuild.

---

## Execution-order summary

1. Part A (F3) → user QA (A9) → D1 deploy → **Leg 0 shakedown**
2. Part C (tools + protocol) — can run in parallel with Part A
3. **Leg 1** (baseline arrest probe, decides F1b's fate)
4. Part B (F2) → user QA (B5) → D2 deploy → **Leg 2** A/B
5. F1a/F1b: separate plan if and only if Leg 1 reproduces mechanism C

## Self-review notes (writing-plans checklist)

- Spec coverage: F3 monitors+alerts+param+gates (A2-A8), F2 three changes+falsification (B2-B4), A/B protocol+tooling (C1-C2), deployment+rollback (D1-D2), F1 explicitly deferred with go/no-go. Matches handoff v3 §4-6.
- Line numbers verified against `849b72a1a` (B tasks) and `clean-v2026.002.001` (A tasks) this session; implementer must still eyeball each anchor before editing (Adaptation notes in A7).
- Types/signatures consistent: `AolSafeguardMonitor.update(...)` kwargs identical in tests (A3), implementation (A4), selfdrived wiring (A7), and replay harness (A8).
- Known softness (deliberate): A7's exact variable names inside `update_events` must be matched to the file at execution time (flagged in-task); f2_replay_check G3 is analytical-only (QA2 already bounded it) — listed as report-not-gate.
```
