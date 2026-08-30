"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Every field of the CarControlSP CAPNP schema must have a twin in the plain-dataclass
mirror in opendbc/car/structs.py.

WHY THIS EXISTS: selfdrive/car/helpers.py::convert_carControlSP splats EVERY capnp field
into structs.CarControlSP(**...), and that dataclass raises on an unknown kwarg. capnp
emits primitives unconditionally, so a new primitive field appears in to_dict() even at its
default -- which means adding one field to the schema and forgetting the mirror crashes
card.py:283 on EVERY FRAME, for EVERY CAR, whether or not the feature using it is enabled.
Adding accelTrim did exactly that, and it was caught by review rather than by a test.

The repo's own selfdrive/car/tests/test_car_interfaces.py exercises convert_carControlSP,
but it needs a built tree. These tests need only pycapnp and pure-Python modules, so they
run in an unbuilt worktree -- which is where a schema edit is actually made.

MAINTAINER WARNING -- DO NOT hand-roll `capnp.load("custom.capnp", imports=[<your own dir>])`
here. openpilot.cereal and opendbc both resolve `/include/c++.capnp` against
<opendbc>/car, and loading it a SECOND time under any other import root aborts the
interpreter from C++ with `Duplicate ID @0xbdf87d7bb8304e81` -- a SIGABRT, not a Python
exception, so pytest reports a crashed worker rather than a failure. Always take the schema
from `openpilot.cereal`, which is the same object controlsd_ext and card use in production.
"""
import dataclasses

from opendbc.car import structs
from openpilot.cereal import custom
from openpilot.selfdrive.car.helpers import convert_carControlSP

SCHEMA = custom.CarControlSP.schema


def _all_fields_materialised():
  """An all-defaults CarControlSP with every pointer field initialised.

  capnp omits null pointer (struct/list) fields from to_dict() but always emits
  primitives, so initialising everything gives the widest dict convert_carControlSP can
  ever be handed -- the worst case for an unknown-kwarg crash.
  """
  msg = custom.CarControlSP.new_message()
  for name in SCHEMA.fieldnames:
    for args in ((name,), (name, 0)):  # struct field, then list field
      try:
        msg.init(*args)
        break
      except Exception:  # primitive field, nothing to initialise
        continue
  return msg


def test_every_capnp_field_has_a_dataclass_twin():
  schema_fields = {f for f in SCHEMA.fieldnames if not f.endswith("DEPRECATED")}
  mirror_fields = {f.name for f in dataclasses.fields(structs.CarControlSP)}
  missing = schema_fields - mirror_fields
  assert not missing, (
      f"CarControlSP capnp fields with no twin in opendbc/car/structs.py: {sorted(missing)}. "
      + "convert_carControlSP would raise TypeError on every frame in card.py:283.")


def test_the_real_converter_accepts_a_fully_populated_message():
  """The actual failure path: exactly what card.py:283 calls, on the widest possible message."""
  converted = convert_carControlSP(_all_fields_materialised().as_reader())
  assert isinstance(converted, structs.CarControlSP)


def test_accel_trim_survives_the_real_converter():
  """Regression pin for the field whose omission from the mirror motivated this file."""
  msg = _all_fields_materialised()
  msg.accelTrim = 0.25
  assert convert_carControlSP(msg.as_reader()).accelTrim == 0.25


def test_the_schema_is_not_read_vacuously():
  """Guards the guard: if SCHEMA.fieldnames ever came back empty the checks above would
  pass for the wrong reason. Also pins the field-numbering subtlety that made accelTrim's
  `@5` look like a collision -- CarControlSP's nested `struct Param` (@0-@3) and
  `enum ParamType` (@0-@6) number INDEPENDENTLY of the outer struct, so their members must
  never appear in the outer struct's fieldnames."""
  fieldnames = set(SCHEMA.fieldnames)
  assert "accelTrim" in fieldnames
  assert {"mads", "params", "leadOne", "leadTwo", "intelligentCruiseButtonManagement"} <= fieldnames
  assert not fieldnames & {"key", "type", "value", "string", "bytes"}, (
      "nested Param/ParamType members leaked into the outer struct's fieldnames")
