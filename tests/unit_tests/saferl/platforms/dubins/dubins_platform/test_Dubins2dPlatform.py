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

This module defines unit tests for Dubins2dPlatform.

Author: John McCarroll
"""

from unittest import mock

import numpy as np
import pytest
from safe_autonomy_dynamics.dubins import Dubins2dAircraft

from saferl.platforms.dubins.dubins_platform import Dubins2dPlatform


@pytest.fixture(name="platform")
def get_platform(platform_name):
    """
    Returns a mock platform object
    """
    return Dubins2dAircraft(name=platform_name)


@pytest.fixture(name="dubins_2d_platform")
def get_dubins_2d_platform(platform_name, platform):
    """
    Returns an instantiated Dubins2dPlatform.

    Parameters
    ----------
    platform_name : str
        The name of the CUT
    platform : mock.MagicMock
        A mock platform
    """
    config = []
    return Dubins2dPlatform(platform_name=platform_name, platform=platform, parts_list=config)


# Test constructor
@pytest.mark.unit_test
def test_constructor(platform_name, platform):
    """
    This test instantiates a Dubins2dPlatform and asserts its attributes and properties are setup properly.

    Parameters
    ----------
    platform_name : str
        name of the platform
    platform : mock.MagicMock
        the mock platform passed to the Dubins2dPlatform constructor
    """
    with mock.patch("corl.simulators.base_platform.BasePlatform._get_part_dict") as func:
        sensor = mock.MagicMock()
        sensor.name = "test_sensor"
        controller = mock.MagicMock()
        controller.name = "test_controller"
        sensors = {sensor.name: sensor}
        controllers = {controller.name: controller}
        func.side_effect = [sensors, controllers]

        config = []

        cut = Dubins2dPlatform(platform_name=platform_name, platform=platform, parts_list=config)

        assert np.array_equal(cut._last_applied_action, np.array([0, 0], dtype=np.float32))  # pylint: disable=W0212
        assert cut._sim_time == 0.0  # pylint: disable=W0212
        assert cut._platform == platform  # pylint: disable=W0212
        assert cut._name == platform_name  # pylint: disable=W0212
        assert cut._sensors == sensors  # pylint: disable=W0212
        assert cut._controllers == controllers  # pylint: disable=W0212

        assert hasattr(cut, "position")
        assert hasattr(cut, "velocity")
        assert hasattr(cut, "heading")
        assert hasattr(cut, "orientation")
        assert hasattr(cut, "sim_time")


# Tests for save_action_to_platform
@pytest.fixture(name="action")
def get_action():
    """
    Returns an action array (numpy.ndarray)
    """
    return np.array([10000.0, -54])


# axis
@pytest.mark.unit_test
def test_Dubins2dPlatform_saveActionToPlatform(dubins_2d_platform, action):  # pylint: disable=W0212
    """
    Tests for CWHPlatform method - save_action_to_platform(),
    This is a parametrized test, where the tests are in the action_to_platform_tests
    """
    # test no axis path
    dubins_2d_platform.save_action_to_platform(action)
    assert np.array_equal(dubins_2d_platform._last_applied_action, action)  # pylint: disable=W0212

    # test axis path
    new_action = 49.536
    axis = 1
    dubins_2d_platform.save_action_to_platform(new_action, axis=axis)
    action[axis] = new_action

    assert np.array_equal(dubins_2d_platform._last_applied_action, action)  # pylint: disable=W0212
