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

This module defines common fixtures for dubins controller tests

Author: John McCarroll
"""

from unittest import mock

import pytest
from safe_autonomy_dynamics.dubins import Dubins3dAircraft

from saferl.platforms.dubins.dubins_platform import Dubins3dPlatform


@pytest.fixture(name='dubins3d_platform')
def setup_dubins_platform():
    """
    Set up basic Dubins3dPlatform, with default values
    """

    platform_name = 'friendly_platform'
    parts_list = []
    aircraft = Dubins3dAircraft(name=platform_name)

    platform_obj = Dubins3dPlatform(platform_name, aircraft, parts_list)
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
