"""
Tests for the CWHPlatform module
"""

from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl_sim.cwh.cwh import CWHSpacecraft
"""
Tests for get_applied_action
"""


@pytest.mark.unit_test
def test_CWHPlatform_getAppliedAction_default(cwh_platform):
    """
    Test for CWHPlatform - get_applied_action method,
    testing to see whether it returns default values
    """

    platform_obj = cwh_platform
    expected = np.array([0., 0., 0.])
    assert np.array_equiv(platform_obj.get_applied_action(), expected)


get_applied_tests = [(np.array([15., 16., 17.])), (np.array([100., 100., 100.])), (np.array([5000., 5000., 5000.]))]


@pytest.fixture(name='applied_action')
def applied_action(request):
    """
    fixture to obtain the parameter 'applied_action' from a list
    """
    return request.param


@pytest.mark.unit_test
@pytest.mark.parametrize('applied_action', get_applied_tests, indirect=True)
def test_CWHPlatform_getAppliedAction(cwh_platform, applied_action):
    """
    Test CWHPlatform - get_applied_action() method
    This is a parametrized test with 'get_applied_tests' being the test cases
    """
    cwh_platform._last_applied_action = applied_action

    result = cwh_platform.get_applied_action()
    assert np.array_equiv(result, applied_action)


"""
Tests for save_action_to_platform
"""

action_to_platform_tests = [(np.array([5., 6., 7.])), (np.array([10., 10., 10.])), (np.array([1000., 1000., 1000.]))]


@pytest.mark.unit_test
@pytest.mark.parametrize('applied_action', action_to_platform_tests, indirect=True)
def test_CWHPlatform_saveActionToPlatform(cwh_platform, applied_action):
    """
    Tests for CWHPlatform method - save_action_to_platform(),
    This is a parametrized test, where the tests are in the action_to_platform_tests
    """
    cwh_platform.save_action_to_platform(applied_action[0], 0)
    cwh_platform.save_action_to_platform(applied_action[1], 1)
    cwh_platform.save_action_to_platform(applied_action[2], 2)

    assert cwh_platform._last_applied_action[0] == applied_action[0]
    assert cwh_platform._last_applied_action[1] == applied_action[1]
    assert cwh_platform._last_applied_action[2] == applied_action[2]


"""
Test position attribute
"""

pos_attr_tests = [
    (np.array([0., 0., 0.]), np.array([0., 0., 0.])), (np.array([50., 100., 150.]), np.array([50., 100., 150.])),
    (np.array([10000., 10000., 10000.]), np.array([10000., 10000., 10000.]))
]


@pytest.mark.unit_test
@pytest.mark.parametrize("pos_input,pos_expected", pos_attr_tests, indirect=True)
def test_position_attribute(cwh_platform_pos, pos_expected):
    """
    Test CWHPlatform - position property
    """
    assert np.array_equiv(cwh_platform_pos.position, pos_expected)


"""
Tests for velocity attribute
"""

# pos,
vel_attr_tests = [
    ([0., 0., 0.], np.array([0., 0., 0.])), ([1., 2., 3.], np.array([1., 2., 3.])), ([10., 10., 10.], np.array([10., 10., 10.])),
    ([1000., 1000., 1000.], np.array([1000., 1000., 1000.]))
]


@pytest.mark.unit_test
@pytest.mark.parametrize("vel_input,vel_expected", vel_attr_tests, indirect=True)
def test_velocity_attrbute(cwh_platform_vel, vel_expected):
    """
    Test for velocity property getter method
    """
    calced = cwh_platform_vel.velocity
    assert np.array_equiv(calced, vel_expected)


"""
Tests for sim_time attribute , accessor and mutator
"""


@pytest.mark.unit_test
def test_sim_time_default(cwh_platform):
    """
    Test for sim_time , default case , must be equal to 0.0
    """
    simulation_time = cwh_platform.sim_time
    assert simulation_time == 0.0


@pytest.fixture(name='set_to_time')
def setup_set_to_time(request):
    """
    fixture to obtain sim_time parameter from a parameter list
    """
    return request.param


@pytest.fixture(name='simtime_platform')
def setup_sim_time_platform(cwh_platform, set_to_time):
    """
    fixture to set sim_time with respect to the cwh_platform
    """
    cwh_platform._sim_time = set_to_time
    return cwh_platform


time_prop_tests = [(10.5), (100.), (200.)]


@pytest.mark.unit_test
@pytest.mark.parametrize("set_to_time", time_prop_tests, indirect=True)
def test_simtime_property(simtime_platform, set_to_time):
    """
    parametrized test for to test if sim_time property getter method
    """
    assert simtime_platform.sim_time == set_to_time


time_accsr_tests = [(10.5), (100.), (200.)]


@pytest.mark.unit_test
@pytest.mark.parametrize("set_to_time", time_accsr_tests, indirect=True)
def test_simtime_setter(cwh_platform, set_to_time):
    """
    parametrized test for to test if sim_time property setter method
    """
    cwh_platform.sim_time = set_to_time
    assert cwh_platform._sim_time == set_to_time


"""
Tests for 'operable' property
"""


@pytest.mark.unit_test
def test_is_operable(cwh_platform):
    """
    validate that cwh_platform is always operable
    """
    assert cwh_platform.operable is True
