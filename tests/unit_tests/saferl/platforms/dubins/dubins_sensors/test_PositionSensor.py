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

This module defines unit tests for the dubins PositionSensor.

Author: John McCarroll
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.core.platforms.dubins.sensors.dubins_sensors import PositionSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/PositionSensor_test_cases.yaml")
parameterized_fixture_keywords = ["position", "expected_position"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='position')
def fixture_position(request):
    """
    Obtains position value from parameter list
    """
    return request.param


@pytest.fixture(name='expected_position')
def fixture_expected_position(request):
    """
    Obtains expected position value from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def mock_dubins_platform(position):
    """
    Returns a mock for a DubinsPlatform with a position attirbute
    """
    mock_platform = mock.MagicMock()
    mock_platform.position = np.array(position)

    return mock_platform


@pytest.fixture(name='position_sensor')
def setup_position_sensor(dubins_platform):
    """
    Fixture that creates a PositionSensor
    """

    config = {}
    position_sensor = PositionSensor(dubins_platform, config)
    return position_sensor


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_calculate_measurement(position_sensor, expected_position):
    """
    parametrized test for the _calculate_measurement method of the PositionSensor
    """
    state = np.array([0., 0., 0.])
    result = position_sensor._calculate_measurement(state)  #pylint: disable=W0212
    assert np.array_equiv(result, expected_position)
