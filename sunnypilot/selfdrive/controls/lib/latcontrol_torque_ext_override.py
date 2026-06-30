"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog


class LatControlTorqueExtOverride:
  def __init__(self, CP):
    self.CP = CP
    self.params = Params()
    self.enforce_torque_control_toggle = self.params.get_bool("EnforceTorqueControl")  # only during init
    self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
    self.last_torque_override_enabled = self.torque_override_enabled
    self.last_applied_override_params = None
    self.frame = -1

  def update_override_torque_params(self, torque_params) -> bool:
    if not self.enforce_torque_control_toggle:
      return False

    self.frame += 1
    if self.frame % 300 == 0:
      self.torque_override_enabled = self.params.get_bool("TorqueParamsOverrideEnabled")
      if self.torque_override_enabled != self.last_torque_override_enabled:
        cloudlog.event("torque_override_enabled_changed",
                       enabled=self.torque_override_enabled,
                       frame=self.frame,
                       car_fingerprint=self.CP.carFingerprint)
        self.last_torque_override_enabled = self.torque_override_enabled

      if not self.torque_override_enabled:
        return False

      lat_accel_factor = float(self.params.get("TorqueParamsOverrideLatAccelFactor", return_default=True))
      lat_accel_offset = float(self.params.get("TorqueParamsOverrideLatAccelOffset", return_default=True))
      friction = float(self.params.get("TorqueParamsOverrideFriction", return_default=True))

      torque_params.latAccelFactor = lat_accel_factor
      torque_params.latAccelOffset = lat_accel_offset
      torque_params.friction = friction

      current_params = (lat_accel_factor, lat_accel_offset, friction)
      if current_params != self.last_applied_override_params:
        cloudlog.event("torque_override_params_applied",
                       lat_accel_factor=lat_accel_factor,
                       lat_accel_offset=lat_accel_offset,
                       friction=friction,
                       frame=self.frame,
                       car_fingerprint=self.CP.carFingerprint)
        self.last_applied_override_params = current_params

      return True

    return False
