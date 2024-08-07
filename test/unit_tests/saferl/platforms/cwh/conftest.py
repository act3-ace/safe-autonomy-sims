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

Conftest file for the CWH platform tests
"""

import pytest
import numpy as np
from safe_autonomy_simulation.sims.spacecraft import CWHSpacecraft

from safe_autonomy_sims.platforms.cwh.cwh_platform import CWHPlatform


@pytest.fixture(name='cwh_spacecraft')
def setup_CWHSpacecraft():
    """
    Setup a basic cwh_spacecraft
    """
    add_args = {'name': 'CWH'}
    spcft = CWHSpacecraft(**add_args)

    return spcft


@pytest.fixture(name='cwh_platform')
def setup_cwhplatform(cwh_spacecraft):
    """
    set up basic cwhplatform, with default values
    """

    platform_name = 'blue0'
    parts_list = []

    platform_obj = CWHPlatform(platform_name, cwh_spacecraft, parts_list)
    return platform_obj


@pytest.fixture(name='pos_input')
def fixture_pos_input(request):
    """
    obtain pos_input, the value to set a CWH platform postion at,
    from parameter list
    """
    return request.param


@pytest.fixture(name='pos_expected')
def fixture_pos_expected(request):
    """
    obtain pos_expected from a parameter list
    """
    return request.param


@pytest.fixture(name='cwh_spacecraft_pos')
def setup_CWHSpacecraft_pos(pos_input):
    """
    based off a certain position construct a CWHSpacecraft at a certain position
    """
    add_args = {'name': 'CWH', 'position': np.array(pos_input)}
    spcft = CWHSpacecraft(**add_args)

    return spcft


@pytest.fixture(name='cwh_platform_pos')
def setup_cwhplatform_pos(cwh_spacecraft_pos):
    """
    based off a CWHSpacecraft set at a certain position create the appropriate
    CWHPlatform
    """
    platform_name = 'blue0'
    platform_config = []

    platform_obj = CWHPlatform(platform_name, cwh_spacecraft_pos, platform_config)
    return platform_obj


@pytest.fixture(name='vel_input')
def fixture_vel_input(request):
    """
    obtain vel_input, the value to set a CWH platform velocity at,
    from parameter list
    """
    return request.param


@pytest.fixture(name='vel_expected')
def fixture_vel_expected(request):
    """
    obtain vel_expected from parameter list
    """
    return request.param


@pytest.fixture(name='cwh_spacecraft_vel')
def setup_CWHSpacecraft_vel(vel_input):
    """
    setup a CWHSpacecraft at a certain velocity
    """
    add_args = {'name': 'CWH', 'velocity': np.array(vel_input)}
    spcft = CWHSpacecraft(**add_args)

    return spcft


@pytest.fixture(name='cwh_platform_vel')
def setup_cwhplatform_vel(cwh_spacecraft_vel):
    """
    setup a CWHPlatform at a certain velocity
    """
    platform_name = 'blue0'
    platform_config = []

    platform_obj = CWHPlatform(platform_name, cwh_spacecraft_vel, platform_config)
    return platform_obj
