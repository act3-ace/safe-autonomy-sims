"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------

Tests for the CWHProperties module
"""
import pytest

from safe_autonomy_sims.platforms.cwh.cwh_properties import PositionProp, ThrustProp, VelocityProp


@pytest.mark.unit_test
def test_ThrustProp():
    """
    Assess that ThrustProp entities are what they are expected to be
    """
    obj = ThrustProp()
    assert obj.name == 'thrust'
    assert obj.low == [-1.0]
    assert obj.high == [1.0]
    assert obj.unit == "newtons"
    assert obj.description == 'Direct Thrust Control'


@pytest.mark.unit_test
def test_PositionProp():
    """
    Assess that PositionProp entities are what they are expected to be
    """
    obj = PositionProp()
    assert obj.name == 'position'
    assert obj.low == [-10000.0] * 3
    assert obj.high == [10000.0] * 3
    assert obj.unit == "m"
    assert obj.description == "Position Sensor Properties"


@pytest.mark.unit_test
def test_VelocityProp():
    """
    Assess that VelocityProp entities are what they are expected to be
    """
    obj = VelocityProp()
    assert obj.name == 'velocity'
    assert obj.low == [-1000.0] * 3
    assert obj.high == [1000.0] * 3
    assert obj.unit == "m/s"
    assert obj.description == "Velocity Sensor Properties"
