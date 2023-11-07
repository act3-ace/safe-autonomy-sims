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

Tests for the CWHPlatform module
"""

import os

import numpy as np
import pytest

from test.conftest import delimiter, read_test_cases


# tests for get_applied_action
@pytest.mark.unit_test
def test_CWHPlatform_getAppliedAction_default(cwh_platform):
    """
    Test for CWHPlatform - get_applied_action method,
    testing to see whether it returns default values
    """

    platform_obj = cwh_platform
    expected = np.array([0., 0., 0.])
    assert np.array_equiv(platform_obj.get_applied_action(), expected)


@pytest.fixture(name='applied_action')
def get_applied_action(request):  # pylint: disable=W0621
    """
    fixture to obtain the parameter 'applied_action' from a list
    """
    return request.param


class Test_get_applied_action:
    """
    Class to define parameterized applied_action tests and fixtures
    """
    # Define test assay
    test_cases_file_path = os.path.join(
        os.path.split(__file__)[0], "../../../../test_cases/platforms/cwh/platform_get_applied_action_test_cases.yaml"
    )
    parameterized_fixture_keywords = ["applied_action"]
    test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

    @pytest.mark.unit_test
    @pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
    def test_CWHPlatform_getAppliedAction(self, cwh_platform, applied_action):  # pylint: disable=W0621
        """
        Test CWHPlatform - get_applied_action() method
        This is a parametrized test with 'get_applied_tests' being the test cases
        """
        cwh_platform._last_applied_action = applied_action  # pylint: disable=W0212

        result = cwh_platform.get_applied_action()
        assert np.array_equiv(result, applied_action)


class Test_save_action_to_platform:
    """
    Tests for save_action_to_platform
    """
    # Define test assay
    test_cases_file_path = os.path.join(
        os.path.split(__file__)[0], "../../../../test_cases/platforms/cwh/platform_save_action_to_platform_test_cases.yaml"
    )
    parameterized_fixture_keywords = ["applied_action"]
    test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

    @pytest.mark.unit_test
    @pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
    def test_CWHPlatform_saveActionToPlatform(self, cwh_platform, applied_action):  # pylint: disable=W0621
        """
        Tests for CWHPlatform method - save_action_to_platform(),
        This is a parametrized test, where the tests are in the action_to_platform_tests
        """

        cwh_platform.save_action_to_platform(applied_action[0][0], 0)  # TODO: fix needless 2d array
        cwh_platform.save_action_to_platform(applied_action[0][1], 1)
        cwh_platform.save_action_to_platform(applied_action[0][2], 2)

        assert cwh_platform._last_applied_action[0] == applied_action[0][0]  #pylint: disable=W0212
        assert cwh_platform._last_applied_action[1] == applied_action[0][1]  #pylint: disable=W0212
        assert cwh_platform._last_applied_action[2] == applied_action[0][2]  #pylint: disable=W0212


class Test_position_attribute:
    """
    Test position attribute
    """
    # Define test assay
    test_cases_file_path = os.path.join(
        os.path.split(__file__)[0], "../../../../test_cases/platforms/cwh/platform_position_attribute_test_cases.yaml"
    )
    parameterized_fixture_keywords = ["pos_input", "pos_expected"]
    test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

    @pytest.mark.unit_test
    @pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
    def test_position_attribute(self, cwh_platform_pos, pos_expected):
        """
        Test CWHPlatform - position property
        """
        assert np.array_equiv(cwh_platform_pos.position, pos_expected)


class Test_velocity_attribute:
    """
    Tests for velocity attribute
    """
    test_cases_file_path = os.path.join(
        os.path.split(__file__)[0], "../../../../test_cases/platforms/cwh/platform_velocity_attribute_test_cases.yaml"
    )
    parameterized_fixture_keywords = ["vel_input", "vel_expected"]
    test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

    @pytest.mark.unit_test
    @pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
    def test_velocity_attrbute(self, cwh_platform_vel, vel_expected):
        """
        Test for velocity property getter method
        """
        calced = cwh_platform_vel.velocity
        assert np.array_equiv(calced, vel_expected)


#Tests for sim_time attribute , accessor and mutator
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
    cwh_platform._sim_time = set_to_time  #pylint: disable=W0212
    return cwh_platform


class Test_simtime_property:
    """
    Test simtime property
    """
    time_prop_tests = [(10.5), (100.), (200.)]

    @pytest.mark.unit_test
    @pytest.mark.parametrize("set_to_time", time_prop_tests, indirect=True)
    def test_simtime_property(self, simtime_platform, set_to_time):
        """
        parametrized test for to test if sim_time property getter method
        """
        assert simtime_platform.sim_time == set_to_time


class Test_simtime_setter_property:
    """
    Test the setter method for simtime property
    """
    time_accsr_tests = [(10.5), (100.), (200.)]

    @pytest.mark.unit_test
    @pytest.mark.parametrize("set_to_time", time_accsr_tests, indirect=True)
    def test_simtime_setter(self, cwh_platform, set_to_time):
        """
        parametrized test for to test if sim_time property setter method
        """
        cwh_platform.sim_time = set_to_time
        assert cwh_platform._sim_time == set_to_time  #pylint: disable=W0212


#Tests for 'operable' property
@pytest.mark.unit_test
def test_is_operable(cwh_platform):
    """
    validate that cwh_platform is always operable
    """
    assert cwh_platform.operable is True
