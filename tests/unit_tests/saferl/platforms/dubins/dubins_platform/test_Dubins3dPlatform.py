"""
This module defines unit tests for Dubins3dPlatform.

Author: John McCarroll
"""

from unittest import mock

import numpy as np
import pytest

from saferl.core.platforms.dubins.dubins_platform import Dubins3dPlatform


@pytest.fixture(name="dubins_3d_platform")
def get_dubins_3d_platform(platform_name, platform):
    """
    Returns an instantiated Dubins3dPlatform.

    Parameters
    ----------
    platform_name : str
        The name of the CUT
    platform : mock.MagicMock
        A mock platform
    """
    config = {}
    return Dubins3dPlatform(platform_name=platform_name, platform=platform, platform_config=config)


# Test constructor
@pytest.mark.unit_test
def test_constructor(platform_name, platform):
    """
    This test instantiates a Dubins3dPlatform and asserts its attributes and properties are setup properly.

    Parameters
    ----------
    platform_name : str
        name of the platform
    platform : mock.MagicMock
        the mock platform passed to the Dubins3dPlatform constructor
    """
    with mock.patch("saferl.core.platforms.dubins.dubins_platform.BasePlatform._get_part_list") as func:
        sensors = mock.MagicMock()
        sensors.name = "sensors"
        controllers = mock.MagicMock()
        controllers.name = "controllers"
        func.side_effect = [sensors, controllers]

        config = {}

        cut = Dubins3dPlatform(platform_name=platform_name, platform=platform, platform_config=config)

        assert np.array_equal(cut._last_applied_action, np.array([0., 0., 0.], dtype=np.float32))  # pylint: disable=W0212
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
        assert hasattr(cut, "flight_path_angle")
        assert hasattr(cut, "roll")


# Tests for save_action_to_platform
@pytest.fixture(name="action")
def get_action():
    """
    Returns an action array (numpy.ndarray)
    """
    return np.array([10000.0, -54, 7777])


# axis
@pytest.mark.unit_test
def test_Dubins3dPlatform_saveActionToPlatform(dubins_3d_platform, action):  # pylint: disable=W0212
    """
    Tests for CWHPlatform method - save_action_to_platform(),
    This is a parametrized test, where the tests are in the action_to_platform_tests
    """
    # test no axis path
    dubins_3d_platform.save_action_to_platform(action)
    assert np.array_equal(dubins_3d_platform._last_applied_action, action)  # pylint: disable=W0212

    # test axis path
    new_action = 49.536
    axis = 1
    dubins_3d_platform.save_action_to_platform(new_action, axis=axis)
    action[axis] = new_action

    assert np.array_equal(dubins_3d_platform._last_applied_action, action)  # pylint: disable=W0212
