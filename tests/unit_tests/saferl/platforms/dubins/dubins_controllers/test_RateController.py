"""
Tests for the RateControllers module of the dubins platform.

Author: John McCarroll
"""

from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.dubins_controllers import RateController
from saferl.platforms.dubins.dubins_platform import Dubins3dPlatform


@pytest.fixture(name="control")
def get_control():
    """
    Return control value
    """

    return 10


@pytest.fixture(name="config")
def get_config():
    """
    Setup config dict
    """

    axis = 2
    return {"axis": axis}


@pytest.fixture(name="control_properties")
def get_control_properties():
    """
    Return a mock of control_properties
    """
    return mock.MagicMock(return_value="props")


@pytest.fixture(name='dubins_platform')
def setup_dubins_platform():
    """
    Set up basic Dubins3dPlatform, with default values
    """

    platform_name = 'friendly_platform'
    platform_config = []
    aircraft = mock.MagicMock()

    platform_obj = Dubins3dPlatform(platform_name, aircraft, platform_config)
    return platform_obj


@pytest.fixture(name="rate_controller")
def setup_rate_controller(dubins_platform, control_properties, config):
    """
    Set up RateController with default values
    """

    return RateController(control_properties=control_properties, parent_platform=dubins_platform, config=config)


# Unit Tests
@pytest.mark.unit_test
def test_constructor(control_properties, config):
    """
    Simple test to ensure construction and attribute assignment functions appropriately
    """
    parent_platform = mock.MagicMock()

    cut = RateController(control_properties=control_properties, parent_platform=parent_platform, config=config)

    assert cut.config.axis == config.get("axis")
    assert cut._properties == control_properties()  # pylint: disable=W0212
    assert cut._parent_platform == parent_platform  # pylint: disable=W0212


@pytest.mark.unit_test
def test_apply_control(rate_controller, config, control):
    """
    Simple test for the apply_control method of RateController
    """
    rate_controller.apply_control(control)
    assert rate_controller.parent_platform._last_applied_action[config['axis']] == control  # pylint: disable=W0212


@pytest.mark.unit_test
def test_get_applied_action(rate_controller):
    """
    Test for the get_applied_control method of RateController
    """

    dummy_arr = np.array([10., 20., 30.])
    rate_controller.parent_platform._last_applied_action = dummy_arr  # pylint: disable=W0212

    assert np.array_equal(rate_controller.get_applied_control(), dummy_arr)  # pylint: disable=W0212
