from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

import saferl.platforms.cwh.cwh_properties as cwh_props
from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.platforms.cwh.cwh_sensors import CWHSensor, PositionSensor, VelocitySensor
from saferl_sim.cwh.cwh import CWHSpacecraft


@pytest.mark.unit_test
def test_CWHSensor_interface():
    """
    Test the CWHSensor interface - _calculate_measurement method
    """
    platform = MagicMock()
    sensor = CWHSensor(platform, {}, cwh_props.ThrustProp)
    dummy_state = np.array([0., 0., 0.])
    with pytest.raises(NotImplementedError) as excinfo:
        sensor._calculate_measurement(dummy_state)


@pytest.fixture(name='pos_sensor')
def setup_pos_sensor(cwh_platform_pos):
    """
    Fixture that creates a PositionSensor
    """

    config = {}
    measurement_props = cwh_props.PositionProp
    pos_sensor = PositionSensor(cwh_platform_pos, config, measurement_props)
    return pos_sensor


pos_sensor_tests = [
    ([0., 0., 0.], np.array([0., 0., 0.])), ([1., 2., 3.], np.array([1., 2., 3.])),
    ([10000., 10000., 10000.], np.array([10000., 10000., 10000.]))
]


@pytest.mark.unit_test
@pytest.mark.parametrize("pos_input,pos_expected", pos_sensor_tests, indirect=True)
def test_PositionSensor_calc_msmt(pos_sensor, pos_input, pos_expected):
    """
    parametrized test for the _calculate_measurement method of the Position Sensor
    """
    state = np.array([0., 0., 0.])
    calced = pos_sensor._calculate_measurement(state)
    assert np.array_equiv(calced, pos_expected)


"""
Tests for velocity sensor
"""


@pytest.fixture(name='vel_sensor')
def setup_vel_sensor(cwh_platform_vel):
    """
    A method that sets up a VelocitySensor
    """
    config = {}
    measurement_props = cwh_props.VelocityProp
    vel_sensor = VelocitySensor(cwh_platform_vel, config, measurement_props)
    return vel_sensor


# pos,
vel_sensor_tests = [
    ([0., 0., 0.], np.array([0., 0., 0.])), ([1., 2., 3.], np.array([1., 2., 3.])), ([10., 10., 10.], np.array([10., 10., 10.])),
    ([10000., 10000., 10000.], np.array([10000., 10000., 10000.]))
]


@pytest.mark.unit_test
@pytest.mark.parametrize("vel_input,vel_expected", vel_sensor_tests, indirect=True)
def test_VelocitySensor_calc_msmt(vel_sensor, vel_input, vel_expected):
    """
    Tests for the _calculate_measurement method of a VelocitySensor
    """

    state = np.array([0., 0., 0.])
    calced = vel_sensor._calculate_measurement(state)
    assert np.array_equiv(calced, vel_expected)
