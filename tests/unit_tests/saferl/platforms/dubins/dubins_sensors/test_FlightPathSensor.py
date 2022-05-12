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

This module defines unit tests for the dubins FlightPathSensor.

Author: John McCarroll
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.core.platforms.dubins.sensors.dubins_sensors import FlightPathSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/FlightPathSensor_test_cases.yaml"
)
parameterized_fixture_keywords = ["flight_path", "expected_flight_path"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='flight_path')
def fixture_flight_path(request):
    """
    Obtains flight_path value from parameter list
    """
    return request.param


@pytest.fixture(name='expected_flight_path')
def fixture_expected_flight_path(request):
    """
    Obtains expected flight_path value from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(flight_path):
    """
    Returns a mock of a DubinsPlatform with a 'flight_path_angle' attribute
    """
    mock_platform = mock.MagicMock()
    mock_platform.flight_path_angle = np.array(flight_path)

    return mock_platform


@pytest.fixture(name='flight_path_sensor')
def setup_flight_path_sensor(dubins_platform):
    """
    Fixture that creates a FlightPathSensor
    """

    config = {}
    flight_path_sensor = FlightPathSensor(dubins_platform, config)
    return flight_path_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test__calculate_measurement(flight_path_sensor, expected_flight_path):
    """
    parametrized test for the _calculate_measurement method of the FlightPathSensor
    """
    state = np.array([0., 0., 0.])
    result = flight_path_sensor._calculate_measurement(state)  #pylint: disable=W0212
    assert np.array_equiv(result, expected_flight_path)
