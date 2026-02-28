import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_scheduler_module(monkeypatch):
  class FakeCloudlog:
    def __init__(self):
      self.events = []
      self.errors = []
      self.exceptions = []
      self.warnings = []

    def event(self, name, **kwargs):
      self.events.append((name, kwargs))

    def error(self, msg):
      self.errors.append(msg)

    def exception(self, msg):
      self.exceptions.append(msg)

    def warning(self, msg):
      self.warnings.append(msg)

  class FakePubMaster:
    def __init__(self, _services):
      self.sent = []

    def send(self, service, msg):
      self.sent.append((service, msg))

  class FakeSubMaster:
    def __init__(self, _services, poll=None):
      self.poll = poll
      self.updated = {"carState": False}
      self.logMonoTime = {"carState": 0}
      self._data = {
        "carState": SimpleNamespace(
          steeringPressed=False,
          leftBlinker=False,
          rightBlinker=False,
          vEgo=0.0,
        ),
        "selfdriveState": SimpleNamespace(active=False, enabled=False),
        "carControl": SimpleNamespace(latActive=False, currentCurvature=0.0),
      }

    def __getitem__(self, key):
      return self._data[key]

    def update(self, _timeout):
      return None

  class FakeParams:
    def __init__(self):
      self.put_calls = []
      self.put_bool_calls = []

    def put(self, key, value):
      self.put_calls.append((key, value))

    def put_bool(self, key, value):
      self.put_bool_calls.append((key, value))

  cloudlog = FakeCloudlog()

  cereal_mod = ModuleType("cereal")
  messaging_mod = ModuleType("cereal.messaging")
  messaging_mod.PubMaster = FakePubMaster
  messaging_mod.SubMaster = FakeSubMaster
  messaging_mod.new_message = lambda name: {"which": name}
  cereal_mod.messaging = messaging_mod

  openpilot_mod = ModuleType("openpilot")
  openpilot_common_mod = ModuleType("openpilot.common")
  openpilot_params_mod = ModuleType("openpilot.common.params")
  openpilot_params_mod.Params = FakeParams
  openpilot_swaglog_mod = ModuleType("openpilot.common.swaglog")
  openpilot_swaglog_mod.cloudlog = cloudlog
  openpilot_common_mod.params = openpilot_params_mod
  openpilot_common_mod.swaglog = openpilot_swaglog_mod
  openpilot_mod.common = openpilot_common_mod

  monkeypatch.setitem(sys.modules, "cereal", cereal_mod)
  monkeypatch.setitem(sys.modules, "cereal.messaging", messaging_mod)
  monkeypatch.setitem(sys.modules, "openpilot", openpilot_mod)
  monkeypatch.setitem(sys.modules, "openpilot.common", openpilot_common_mod)
  monkeypatch.setitem(sys.modules, "openpilot.common.params", openpilot_params_mod)
  monkeypatch.setitem(sys.modules, "openpilot.common.swaglog", openpilot_swaglog_mod)

  script_path = Path(__file__).resolve().parents[1] / "tune_distance_scheduler.py"
  module_name = "test_tune_distance_scheduler_module"
  if module_name in sys.modules:
    del sys.modules[module_name]

  spec = importlib.util.spec_from_file_location(module_name, script_path)
  module = importlib.util.module_from_spec(spec)
  assert spec is not None and spec.loader is not None
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module, cloudlog, FakeParams, FakeSubMaster


def _write_plan(tmp_path, params=None, distance_miles=1.0):
  if params is None:
    params = {"SomeFloat": 1.25}
  plan = {
    "name": "test_plan",
    "count_mode": "active_lat",
    "steps": [
      {
        "label": "A",
        "distance_miles": distance_miles,
        "params": params,
      }
    ],
  }
  plan_path = tmp_path / "plan.json"
  plan_path.write_text(json.dumps(plan), encoding="utf-8")
  return plan_path


def _make_args(tmp_path, plan_path, **overrides):
  data = {
    "plan": str(plan_path),
    "state_path": str(tmp_path / "state.json"),
    "events_log": str(tmp_path / "events.log"),
    "errors_log": str(tmp_path / "errors.log"),
    "tone_file": "",
    "enable_tone": False,
    "enable_bookmark_toast": False,
    "resume": True,
    "reset_when_done": True,
    "dry_run": False,
    "min_speed_mps": 15.0,
    "safe_curvature": 0.0012,
    "count_mode": "active_lat",
    "loop": False,
  }
  data.update(overrides)
  return argparse.Namespace(**data)


def test_params_put_preserves_python_types(monkeypatch):
  module, _cloudlog, FakeParams, _FakeSubMaster = _load_scheduler_module(monkeypatch)
  params = FakeParams()

  module.Scheduler._params_put(params, "BoolParam", True)
  module.Scheduler._params_put(params, "FloatParam", 1.5)
  module.Scheduler._params_put(params, "IntParam", 3)
  module.Scheduler._params_put(params, "StringParam", "x")

  assert params.put_bool_calls == [("BoolParam", True)]
  assert params.put_calls[0] == ("FloatParam", 1.5)
  assert isinstance(params.put_calls[0][1], float)
  assert params.put_calls[1] == ("IntParam", 3)
  assert isinstance(params.put_calls[1][1], int)
  assert params.put_calls[2] == ("StringParam", "x")


def test_apply_step_handles_param_write_failures(monkeypatch, tmp_path):
  module, _cloudlog, _FakeParams, _FakeSubMaster = _load_scheduler_module(monkeypatch)
  plan_path = _write_plan(tmp_path, params={"okFloat": 1.0, "badFloat": 2.0, "okBool": True})
  args = _make_args(tmp_path, plan_path)
  scheduler = module.Scheduler(args)

  class MixedParams:
    def __init__(self):
      self.put_calls = []
      self.put_bool_calls = []

    def put(self, key, value):
      if key == "badFloat":
        raise TypeError("bad write")
      self.put_calls.append((key, value))

    def put_bool(self, key, value):
      self.put_bool_calls.append((key, value))

  scheduler.params = MixedParams()
  scheduler._apply_step(0)

  assert scheduler.state.current_step_idx == 0
  assert scheduler.state.pending_step_idx is None
  assert scheduler.params.put_calls == [("okFloat", 1.0)]
  assert scheduler.params.put_bool_calls == [("okBool", True)]

  events_text = Path(args.events_log).read_text(encoding="utf-8")
  errors_text = Path(args.errors_log).read_text(encoding="utf-8")
  assert "Applied step 1/1 'A'" in events_text
  assert "skipped 1 params: badFloat" in events_text
  assert "Failed writing param 'badFloat'" in errors_text


def test_done_state_resets_when_requested(monkeypatch, tmp_path):
  module, _cloudlog, _FakeParams, _FakeSubMaster = _load_scheduler_module(monkeypatch)
  plan_path = _write_plan(tmp_path)
  args = _make_args(tmp_path, plan_path, reset_when_done=True, resume=True)

  plan_doc = json.loads(plan_path.read_text(encoding="utf-8"))
  plan_id = module.Scheduler._compute_plan_id(plan_doc)
  state_blob = {
    "plan_id": plan_id,
    "current_step_idx": 0,
    "pending_step_idx": None,
    "distance_on_current_m": 10.0,
    "total_distance_m": 20.0,
    "qualifying_distance_m": 20.0,
    "cycle": 1,
    "done": True,
  }
  Path(args.state_path).write_text(json.dumps(state_blob), encoding="utf-8")

  scheduler = module.Scheduler(args)
  assert scheduler.state.done is False
  assert scheduler.state.current_step_idx is None
  assert scheduler.state.pending_step_idx == 0
  events_text = Path(args.events_log).read_text(encoding="utf-8")
  assert "resetting scheduler to step 1" in events_text


def test_run_recovers_after_update_exception(monkeypatch, tmp_path):
  module, _cloudlog, _FakeParams, _FakeSubMaster = _load_scheduler_module(monkeypatch)
  plan_path = _write_plan(tmp_path, params={"SomeFloat": 1.0}, distance_miles=100.0)
  args = _make_args(tmp_path, plan_path, dry_run=True)
  scheduler = module.Scheduler(args)

  class FlakySubMaster:
    def __init__(self, outer):
      self.outer = outer
      self.calls = 0
      self.updated = {"carState": True}
      self.logMonoTime = {"carState": 0}
      self._data = {
        "carState": SimpleNamespace(
          steeringPressed=False,
          leftBlinker=False,
          rightBlinker=False,
          vEgo=20.0,
        ),
        "selfdriveState": SimpleNamespace(active=True, enabled=True),
        "carControl": SimpleNamespace(latActive=True, currentCurvature=0.0),
      }

    def __getitem__(self, key):
      return self._data[key]

    def update(self, _timeout):
      self.calls += 1
      if self.calls == 1:
        raise RuntimeError("boom")
      self.logMonoTime["carState"] += int(0.1 * 1e9)
      if self.calls >= 3:
        self.outer.running = False

  scheduler.sm = FlakySubMaster(scheduler)
  rc = scheduler.run()
  assert rc == 0
  errors_text = Path(args.errors_log).read_text(encoding="utf-8")
  assert "RuntimeError: boom" in errors_text


def test_qualifies_for_distance_modes(monkeypatch, tmp_path):
  module, _cloudlog, _FakeParams, _FakeSubMaster = _load_scheduler_module(monkeypatch)
  plan_path = _write_plan(tmp_path)
  args = _make_args(tmp_path, plan_path)
  scheduler = module.Scheduler(args)

  scheduler.sm._data["carState"] = SimpleNamespace(
    steeringPressed=False,
    leftBlinker=False,
    rightBlinker=False,
    vEgo=16.0,
  )
  scheduler.sm._data["selfdriveState"] = SimpleNamespace(active=True, enabled=True)
  scheduler.sm._data["carControl"] = SimpleNamespace(latActive=True, currentCurvature=0.0)
  scheduler.count_mode = "active_lat"
  assert scheduler._qualifies_for_distance() is True

  scheduler.sm._data["carControl"] = SimpleNamespace(latActive=False, currentCurvature=0.0)
  assert scheduler._qualifies_for_distance() is False

  scheduler.count_mode = "total"
  assert scheduler._qualifies_for_distance() is True
