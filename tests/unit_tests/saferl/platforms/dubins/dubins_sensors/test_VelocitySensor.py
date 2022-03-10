"""
This module defines unit tests for the dubins VelocitySensor.

Author: John McCarroll
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.dubins_sensors import VelocitySensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/VelocitySensor_test_cases.yaml")
parameterized_fixture_keywords = ["velocity", "expected_velocity"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='velocity')
def fixture_velocity(request):
    """
    Obtains velocity value from parameter list
    """
    return request.param


@pytest.fixture(name='expected_velocity')
def fixture_expected_velocity(request):
    """
    Obtains expected velocity value from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(velocity):
    """
    Returns a mock of a DubinsPlatform with a 'velocity' attribute
    """
    mock_platform = mock.MagicMock()
    mock_platform.velocity = np.array(velocity)

    return mock_platform


@pytest.fixture(name='velocity_sensor')
def setup_velocity_sensor(dubins_platform):
    """
    Fixture that creates a VelocitySensor
    """

    config = {}
    velocity_sensor = VelocitySensor(dubins_platform, config)
    return velocity_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test__calculate_measurement(velocity_sensor, expected_velocity):
    """
    parametrized test for the _calculate_measurement method of the VelocitySensor
    """
    state = np.array([0., 0., 0.])
    result = velocity_sensor._calculate_measurement(state)  #pylint: disable=W0212
    assert np.array_equiv(result, expected_velocity)
