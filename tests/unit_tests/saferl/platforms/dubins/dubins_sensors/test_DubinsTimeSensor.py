"""
This module defines unit tests for the dubins DubinsTimeSensor.

Author: Jamie Cunningham
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.sensors.dubins_sensors import DubinsTimeSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/DubinsTimeSensor_test_cases.yaml"
)
parameterized_fixture_keywords = ["sim_time", "expected_sim_time"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='sim_time')
def fixture_sim_time(request):
    """
    Obtains sim_time angle value in degrees from parameter list
    """
    return request.param


@pytest.fixture(name='expected_sim_time')
def fixture_expected_sim_time(request):
    """
    Obtains expected sim_time output value in radians from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(sim_time):
    """
    Returns a mock of a DubinsPlatform with a 'sim_time' attribute
    """
    mock_platform = mock.MagicMock()
    mock_platform.sim_time = sim_time

    return mock_platform


@pytest.fixture(name='time_sensor')
def setup_time_sensor(dubins_platform):
    """
    Fixture that creates a DubinsTimeSensor
    """

    config = {}
    time_sensor = DubinsTimeSensor(dubins_platform, config)
    return time_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_calculate_and_cache_measurement(time_sensor, expected_sim_time):
    """
    parametrized test for the calculate_and_cache_measurement method of the VelocitySensor
    """
    state = np.array([0., 0., 0.])
    time_sensor.calculate_and_cache_measurement(state)  #pylint: disable=W0212
    result = time_sensor.get_measurement()
    assert np.allclose(result, expected_sim_time)
