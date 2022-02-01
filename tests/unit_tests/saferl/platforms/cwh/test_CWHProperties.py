from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from saferl.platforms.cwh.cwh_properties import PositionProp, ThrustProp, VelocityProp

#saferl/platforms/cwh/cwh_properties.py


def test_ThrustProp():
    obj = ThrustProp()
    assert obj.name == 'thrust'
    assert obj.low == [-1.0]
    assert obj.high == [1.0]
    assert obj.unit == ["newtons"]
    assert obj.description == 'Direct Thrust Control'


def test_PositionProp():
    obj = PositionProp()
    assert obj.name == 'position'
    assert obj.low == [-80000.0] * 3
    assert obj.high == [80000.0] * 3
    assert obj.unit == ["m"] * 3
    assert obj.description == "Position Sensor Properties"


def test_VelocityProp():
    obj = VelocityProp()
    assert obj.name == 'velocity'
    assert obj.low == [-10000.0] * 3
    assert obj.high == [10000.0] * 3
    assert obj.unit == ["m/s"] * 3
    assert obj.description == "Velocity Sensor Properties"
