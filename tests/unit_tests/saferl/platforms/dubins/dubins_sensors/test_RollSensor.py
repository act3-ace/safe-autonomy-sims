"""
This module defines unit tests for the dubins RollSensor.

Author: Jamie Cunningham
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.dubins_sensors import RollSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/RollSensor_test_cases.yaml")
parameterized_fixture_keywords = ["roll", "expected_roll"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='roll')
def fixture_roll(request):
    """
    Obtains roll angle value in degrees from parameter list
    """
    return request.param


@pytest.fixture(name='expected_roll')
def fixture_expected_roll(request):
    """
    Obtains expected roll output value in radians from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(roll):
    """
    Returns a mock of a DubinsPlatform with a 'roll' attribute
    """
    mock_platform = mock.MagicMock()
    mock_platform.roll = roll

    return mock_platform


@pytest.fixture(name='roll_sensor')
def setup_roll_sensor(dubins_platform):
    """
    Fixture that creates a RollSensor
    """

    config = {}
    roll_sensor = RollSensor(dubins_platform, config)
    return roll_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_calculate_and_cache_measurement(roll_sensor, expected_roll):
    """
    parametrized test for the calculate_and_cache_measurement method of the VelocitySensor
    """
    state = np.array([0., 0., 0.])
    roll_sensor.calculate_and_cache_measurement(state)  #pylint: disable=W0212
    result = roll_sensor.get_measurement()
    assert np.allclose(result, expected_roll)
