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

 Tests for the cwh_sensors module
"""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

import safe_autonomy_sims.platforms.cwh.cwh_properties as cwh_props
from safe_autonomy_sims.platforms.cwh.cwh_sensors import CWHSensor, PositionSensor, VelocitySensor
from test.conftest import delimiter, read_test_cases


@pytest.mark.unit_test
def test_CWHSensor_interface():
    """
    Test the CWHSensor interface - _calculate_measurement method
    """
    platform = MagicMock()
    sensor = CWHSensor(platform, {}, cwh_props.ThrustProp)
    dummy_state = np.array([0., 0., 0.])
    with pytest.raises(NotImplementedError):
        sensor._calculate_measurement(dummy_state)  #pylint: disable=W0212


@pytest.fixture(name='pos_sensor')
def setup_pos_sensor(cwh_platform_pos):
    """
    Fixture that creates a PositionSensor
    """

    config = {}
    measurement_props = cwh_props.PositionProp
    pos_sensor = PositionSensor(cwh_platform_pos, config, measurement_props)
    return pos_sensor


class TestPositionSensor:
    """
    The class defines unit tests for the PositionSensor
    """
    test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/platforms/cwh/position_sensor_test_cases.yaml")
    parameterized_fixture_keywords = ["pos_input", "pos_expected"]
    test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

    @pytest.mark.unit_test
    @pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
    def test_PositionSensor_calc_msmt(self, pos_sensor, pos_expected):
        """
        parametrized test for the _calculate_measurement method of the Position Sensor
        """
        state = np.array([0., 0., 0.])
        calced = pos_sensor._calculate_measurement(state).m  #pylint: disable=W0212
        assert np.array_equiv(calced, pos_expected)


#Tests for velocity sensor
@pytest.fixture(name='vel_sensor')
def setup_vel_sensor(cwh_platform_vel):
    """
    A method that sets up a VelocitySensor
    """
    config = {}
    measurement_props = cwh_props.VelocityProp
    vel_sensor = VelocitySensor(cwh_platform_vel, config, measurement_props)
    return vel_sensor


class TestVelocitySensor:
    """
    The class defines unit tests for the VelocitySensor
    """
    test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/platforms/cwh/velocity_sensor_test_cases.yaml")
    parameterized_fixture_keywords = ["vel_input", "vel_expected"]
    test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

    @pytest.mark.unit_test
    @pytest.mark.parametrize("vel_input,vel_expected", test_configs, indirect=True, ids=IDs)
    def test_VelocitySensor_calc_msmt(self, vel_sensor, vel_expected):
        """
        Tests for the _calculate_measurement method of a VelocitySensor
        """

        state = np.array([0., 0., 0.])
        calced = vel_sensor._calculate_measurement(state).m  #pylint: disable=W0212
        assert np.array_equiv(calced, vel_expected)
