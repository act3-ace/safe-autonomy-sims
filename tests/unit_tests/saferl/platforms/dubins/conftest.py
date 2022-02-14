"""
Conftest file for the Dubins platform tests
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


# @pytest.fixture(name='dubins_spacecraft')
# def setup_DubinsSpacecraft():
#     """
#     Setup a basic dubins_spacecraft
#     """
#     add_args = {'name': 'Dubins'}
#     spcft = DubinsSpacecraft(**add_args)
#
#     return spcft
#
#
# @pytest.fixture(name='dubins_platform')
# def setup_dubinsplatform(dubins_spacecraft):
#     """
#     set up basic dubinsplatform, with default values
#     """
#
#     platform_name = 'blue0'
#     platform_config = []
#
#     platform_obj = DubinsPlatform(platform_name, dubins_spacecraft, platform_config)
#     return platform_obj
#
#
# @pytest.fixture(name='pos_input')
# def fixture_pos_input(request):
#     """
#     obtain pos_input, the value to set a Dubins platform postion at,
#     from parameter list
#     """
#     return request.param
#
#
# @pytest.fixture(name='pos_expected')
# def fixture_pos_expected(request):
#     """
#     obtain pos_expected from a parameter list
#     """
#     return request.param
#
#
# @pytest.fixture(name='dubins_spacecraft_pos')
# def setup_DubinsSpacecraft_pos(pos_input):
#     """
#     based off a certain position construct a DubinsSpacecraft at a certain position
#     """
#     add_args = {'name': 'Dubins', 'x': pos_input[0], 'y': pos_input[1], 'z': pos_input[2]}
#     spcft = DubinsSpacecraft(**add_args)
#
#     return spcft
#
#
# @pytest.fixture(name='dubins_platform_pos')
# def setup_dubinsplatform_pos(dubins_spacecraft_pos):
#     """
#     based off a DubinsSpacecraft set at a certain position create the appropriate
#     DubinsPlatform
#     """
#     platform_name = 'blue0'
#     platform_config = []
#
#     platform_obj = DubinsPlatform(platform_name, dubins_spacecraft_pos, platform_config)
#     return platform_obj
#
#
# @pytest.fixture(name='vel_input')
# def fixture_vel_input(request):
#     """
#     obtain vel_input, the value to set a Dubins platform velocity at,
#     from parameter list
#     """
#     return request.param
#
#
# @pytest.fixture(name='vel_expected')
# def fixture_vel_expected(request):
#     """
#     obtain vel_expected from parameter list
#     """
#     return request.param
#
#
# @pytest.fixture(name='dubins_spacecraft_vel')
# def setup_DubinsSpacecraft_vel(vel_input):
#     """
#     setup a DubinsSpacecraft at a certain velocity
#     """
#     add_args = {'name': 'Dubins', 'xdot': vel_input[0], 'ydot': vel_input[1], 'zdot': vel_input[2]}
#     spcft = DubinsSpacecraft(**add_args)
#
#     return spcft
#
#
# @pytest.fixture(name='dubins_platform_vel')
# def setup_dubinsplatform_vel(dubins_spacecraft_vel):
#     """
#     setup a DubinsPlatform at a certain velocity
#     """
#     platform_name = 'blue0'
#     platform_config = []
#
#     platform_obj = DubinsPlatform(platform_name, dubins_spacecraft_vel, platform_config)
#     return platform_obj
