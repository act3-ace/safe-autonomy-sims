"""
Tests for the CWHProperties module
"""
import pytest

from saferl.platforms.cwh.cwh_properties import PositionProp, ThrustProp, VelocityProp


@pytest.mark.unit_test
def test_ThrustProp():
    """
    Assess that ThrustProp entities are what they are expected to be
    """
    obj = ThrustProp()
    assert obj.name == 'thrust'
    assert obj.low == [-1.0]
    assert obj.high == [1.0]
    assert obj.unit == ["newtons"]
    assert obj.description == 'Direct Thrust Control'


@pytest.mark.unit_test
def test_PositionProp():
    """
    Assess that PositionProp entities are what they are expected to be
    """
    obj = PositionProp()
    assert obj.name == 'position'
    assert obj.low == [-80000.0] * 3
    assert obj.high == [80000.0] * 3
    assert obj.unit == ["m"] * 3
    assert obj.description == "Position Sensor Properties"


@pytest.mark.unit_test
def test_VelocityProp():
    """
    Assess that VelocityProp entities are what they are expected to be
    """
    obj = VelocityProp()
    assert obj.name == 'velocity'
    assert obj.low == [-10000.0] * 3
    assert obj.high == [10000.0] * 3
    assert obj.unit == ["m/s"] * 3
    assert obj.description == "Velocity Sensor Properties"
