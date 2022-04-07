"""
This module defines unit tests for the dubins SpeedSensor.

Author: Justin Merrick
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.core.platforms.dubins.sensors.dubins_sensors import SpeedSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/SpeedSensor_test_cases.yaml")
parameterized_fixture_keywords = ["speed", "expected_speed"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='speed')
def fixture_velocity(request):
    """
    Obtains speed value from parameter list
    """
    return request.param


@pytest.fixture(name='expected_speed')
def fixture_expected_velocity(request):
    """
    Obtains expected speed value from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(speed):
    """
    Returns a mock of a DubinsPlatform with a 'velocity' attribute
    """
    mock_platform = mock.MagicMock()
    mock_platform.v = speed

    return mock_platform


@pytest.fixture(name='speed_sensor')
def setup_velocity_sensor(dubins_platform):
    """
    Fixture that creates a SpeedSensor
    """

    config = {}
    speed_sensor = SpeedSensor(dubins_platform, config)
    return speed_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test__calculate_measurement(speed_sensor, expected_speed):
    """
    parametrized test for the _calculate_measurement method of the SpeedSensor
    """
    state = np.array([0., 0., 0.])
    result = speed_sensor._calculate_measurement(state)  #pylint: disable=W0212
    assert np.array_equiv(result, expected_speed)
