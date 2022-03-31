"""
This module defines unit tests for the dubins HeadingSensor.

Author: John McCarroll
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.sensors.dubins_sensors import HeadingSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/HeadingSensor_test_cases.yaml")
parameterized_fixture_keywords = ["heading", "expected_heading"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='heading')
def fixture_heading(request):
    """
    Obtains heading value from parameter list
    """
    return request.param


@pytest.fixture(name='expected_heading')
def fixture_expected_heading(request):
    """
    Obtains expected heading value from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(heading):
    """
    Returns a mock of a DubinsPlatform with a 'heading' attribute
    """
    mock_platform = mock.MagicMock()
    mock_platform.heading = np.array(heading)

    return mock_platform


@pytest.fixture(name='heading_sensor')
def setup_heading_sensor(dubins_platform):
    """
    Fixture that creates a HeadingSensor
    """

    config = {}
    heading_sensor = HeadingSensor(dubins_platform, config)
    return heading_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test__calculate_measurement(heading_sensor, expected_heading):
    """
    parametrized test for the _calculate_measurement method of the HeadingSensor
    """
    state = np.array([0., 0., 0.])
    result = heading_sensor._calculate_measurement(state)  #pylint: disable=W0212
    assert np.array_equiv(result, expected_heading)
