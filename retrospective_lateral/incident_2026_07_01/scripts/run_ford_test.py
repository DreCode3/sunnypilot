import sys, types
# Stub the heavy carcontroller import chain: test only needs MAX_LATERAL_ACCEL float.
# Recompute it the same way carcontroller.py does.
ISO_LATERAL_ACCEL = 3.0  # from opendbc.car.lateral
ACCELERATION_DUE_TO_GRAVITY = 9.81
AVERAGE_ROAD_ROLL = 0.06
stub = types.ModuleType("opendbc.car.ford.carcontroller")
stub.MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (ACCELERATION_DUE_TO_GRAVITY * AVERAGE_ROAD_ROLL)
sys.modules["opendbc.car.ford.carcontroller"] = stub
import pytest
sys.exit(pytest.main(["opendbc/safety/tests/test_ford.py", "-p","no:cacheprovider","-o","addopts=", "-q"]))
