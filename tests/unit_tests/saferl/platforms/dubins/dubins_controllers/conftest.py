"""
This module defines common fixtures for dubins controller tests

Author: John McCarroll
"""

from unittest import mock

import pytest

from saferl.platforms.dubins.dubins_platform import Dubins3dPlatform


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
