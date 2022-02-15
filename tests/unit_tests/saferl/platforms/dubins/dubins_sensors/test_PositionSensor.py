"""
This module defines unit tests for the dubins PositionSensor.

Author: John McCarroll
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.dubins_sensors import PositionSensor
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../../test_cases/dubins_platform_test_cases/PositionSensor_test_cases.yaml"
)
parameterized_fixture_keywords = ["position", "expected_position"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='position')
def fixture_position(request):
    """
    obtain pos_input, the value to set a CWH platform postion at,
    from parameter list
    """
    return request.param


@pytest.fixture(name='expected_position')
def fixture_expected_position(request):
    """
    obtain pos_input, the value to set a CWH platform postion at,
    from parameter list
    """
    return request.param


@pytest.fixture(name='dubins_platform')
def setup_dubins_platform_pos(position):
    """
    based off a CWHSpacecraft set at a certain position create the appropriate
    CWHPlatform
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
def test_PositionSensor_calc_msmt(position_sensor, expected_position):
    """
    parametrized test for the _calculate_measurement method of the PositionSensor
    """
    state = np.array([0., 0., 0.])
    calced = position_sensor._calculate_measurement(state)  #pylint: disable=W0212
    assert np.array_equiv(calced, expected_position)
