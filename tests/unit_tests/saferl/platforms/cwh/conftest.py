"""
Conftest file for the CWH platform tests
"""

import pytest

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl_sim.cwh.cwh import CWHSpacecraft


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
    platform_config = []

    platform_obj = CWHPlatform(platform_name, cwh_spacecraft, platform_config)
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
    add_args = {'name': 'CWH', 'x': pos_input[0], 'y': pos_input[1], 'z': pos_input[2]}
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
    add_args = {'name': 'CWH', 'xdot': vel_input[0], 'ydot': vel_input[1], 'zdot': vel_input[2]}
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
