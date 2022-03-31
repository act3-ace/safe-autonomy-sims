"""
This module defines unit tests for the dubins QuaternionSensor.

Author: Jamie Cunningham
"""

import os
from unittest import mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from saferl.platforms.dubins.sensors.dubins_sensors import QuaternionSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/QuaternionSensor_test_cases.yaml"
)
parameterized_fixture_keywords = ["orientation", "expected_quaternion"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='orientation')
def fixture_orientation(request):
    """
    Obtains orientation euler zyx rotation value from parameter list
    """
    return request.param


@pytest.fixture(name='expected_quaternion')
def fixture_expected_quaternion(request):
    """
    Obtains expected quaternion output value from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(orientation):
    """
    Returns a mock of a DubinsPlatform with a 'orientation' attribute
    """
    mock_platform = mock.MagicMock()
    mock_platform.orientation = Rotation.from_euler("zyx", orientation)

    return mock_platform


@pytest.fixture(name='quaternion_sensor')
def setup_quaternion_sensor(dubins_platform):
    """
    Fixture that creates a QuaternionSensor
    """

    config = {}
    quaternion_sensor = QuaternionSensor(dubins_platform, config)
    return quaternion_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_calculate_and_cache_measurement(quaternion_sensor, expected_quaternion):
    """
    parametrized test for the calculate_and_cache_measurement method of the VelocitySensor
    """
    state = np.array([0., 0., 0.])
    quaternion_sensor.calculate_and_cache_measurement(state)  #pylint: disable=W0212
    result = quaternion_sensor.get_measurement()
    assert np.allclose(result, expected_quaternion)
