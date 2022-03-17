"""
This module defines unit tests for DubinsPlatform.

Author: John McCarroll
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.dubins_platform import DubinsPlatform
from tests.conftest import delimiter, read_test_cases


@pytest.fixture(name="dubins_platform")
def get_dubins_platform(platform_name, platform):
    """
    Returns an instantiated DubinsPlatform.

    Parameters
    ----------
    platform_name : str
        The name of the CUT
    platform : mock.MagicMock
        A mock platform
    """
    config = {}
    return DubinsPlatform(platform_name=platform_name, platform=platform, platform_config=config)


# Test constructor
@pytest.mark.unit_test
def test_constructor(platform_name, platform):
    """
    This test instantiates a DubinsPlatform and asserts its attributes and properties are setup properly.

    Parameters
    ----------
    platform_name : str
        name of the platform
    platform : mock.MagicMock
        the mock platform passed to the DubinsPlatform constructor
    """
    with mock.patch("saferl.platforms.dubins.dubins_platform.BasePlatform._get_part_list") as func:
        sensors = mock.MagicMock()
        sensors.name = "sensors"
        controllers = mock.MagicMock()
        controllers.name = "controllers"
        func.side_effect = [sensors, controllers]

        config = {}

        cut = DubinsPlatform(platform_name=platform_name, platform=platform, platform_config=config)

        assert cut._last_applied_action is None  # pylint: disable=W0212
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


# Test __eq__
class Test__eq__:
    """
    This class defines fixtures and values used to evaluate DubinsPlatform's overwritten __eq__ method.
    """

    # Define test assay
    test_cases_file_path = os.path.join(
        os.path.split(__file__)[0], "../../../../../test_cases/platforms/dubins/DubinsPlatform__eq__test_cases.yaml"
    )
    parameterized_fixture_keywords = [
        "velocity1",
        "velocity2",
        "position1",
        "position2",
        "orientation1",
        "orientation2",
        "heading1",
        "heading2",
        "sim_time1",
        "sim_time2",
        "expected"
    ]
    test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

    @pytest.fixture(name="other_dubins_platform")
    def get_other_dubins_platform(self):
        """
        Returns another instantiated DubinsPlatform
        """
        platform_name = "other_cut"
        config = {}
        other_platform = mock.MagicMock(name="other_platform")
        return DubinsPlatform(platform_name=platform_name, platform=other_platform, platform_config=config)

    @pytest.fixture(name="velocity1")
    def get_velocity1(self, request):
        """
        Returns a parameterized velocity array (numpy.ndarray)
        """
        return request.param

    @pytest.fixture(name="velocity2")
    def get_velocity2(self, request):
        """
        Returns a parameterized velocity array (numpy.ndarray)
        """
        return request.param

    @pytest.fixture(name="position1")
    def get_position1(self, request):
        """
        Returns a parameterized position array (numpy.ndarray)
        """
        return request.param

    @pytest.fixture(name="position2")
    def get_position2(self, request):
        """
        Returns a parameterized position array (numpy.ndarray)
        """
        return request.param

    @pytest.fixture(name="orientation1")
    def get_orientation1(self, request):
        """
        Returns a parameterized orientation (scipy.Rotation)
        """
        return request.param

    @pytest.fixture(name="orientation2")
    def get_orientation2(self, request):
        """
        Returns a parameterized orientation (scipy.Rotation)
        """
        return request.param

    @pytest.fixture(name="heading1")
    def get_heading1(self, request):
        """
        Returns a parameterized heading value (float)
        """
        return request.param

    @pytest.fixture(name="heading2")
    def get_heading2(self, request):
        """
        Returns a parameterized heading value (float)
        """
        return request.param

    @pytest.fixture(name="sim_time1")
    def get_sim_time1(self, request):
        """
        Returns a parameterized time value (float)
        """
        return request.param

    @pytest.fixture(name="sim_time2")
    def get_sim_time2(self, request):
        """
        Returns a parameterized  time value (float)
        """
        return request.param

    @pytest.fixture(name="expected")
    def get_expected(self, request):
        """
        Returns a parameterized boolean for the expected result of the comparison between DubinsPlatform objects
        """
        return request.param

    @pytest.mark.unit_test
    @pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
    def test__eq__(
        self,
        dubins_platform,
        other_dubins_platform,
        velocity1,
        velocity2,
        position1,
        position2,
        orientation1,
        orientation2,
        heading1,
        heading2,
        sim_time1,
        sim_time2,
        expected
    ):
        """
        A parameterized test which instantiates two DubinsPlatform objects checks their equality. This tests the logic
        of the overwriten __eq__ method of the DubinsPlatform class.

        Parameters
        ----------
        dubins_platform : DubinsPlatform
            A CUT for comparison
        other_dubins_platform : DubinsPlatform
            A CUT for comparison
        velocity1 : numpy.ndarray
            The velocity array assigned to the first DubinsPlatform object
        velocity2 : numpy.ndarray
            The velocity array assigned to the second DubinsPlatform object
        position1 : numpy.ndarray
            The position array assigned to the first DubinsPlatform object
        position2 : numpy.ndarray
            The position array assigned to the second DubinsPlatform object
        orientation1 : scipy.Rotation
            The orientation object assigned to the first DubinsPlatform object
        orientation2 : scipy.Rotation
            The orientation object assigned to the second DubinsPlatform object
        heading1 : float
            The heading value assigned to the first DubinsPlatform object
        heading2 : float
            The heading value assigned to the second DubinsPlatform object
        sim_time1 : int
            The sim_time value assigned to the first DubinsPlatform object
        sim_time2 : int
            The sim_time value assigned to the second DubinsPlatform object
        expected : bool
            The expected result of checking the equality of the two DubinsPlatform objects

        Returns
        -------

        """

        dubins_platform._platform.velocity = velocity1  # pylint: disable=W0212
        dubins_platform._platform.position = position1  # pylint: disable=W0212
        dubins_platform._platform.orientation = orientation1  # pylint: disable=W0212
        dubins_platform._platform.heading = heading1  # pylint: disable=W0212
        dubins_platform.sim_time = sim_time1

        other_dubins_platform._platform.velocity = velocity2  # pylint: disable=W0212
        other_dubins_platform._platform.position = position2  # pylint: disable=W0212
        other_dubins_platform._platform.orientation = orientation2  # pylint: disable=W0212
        other_dubins_platform._platform.heading = heading2  # pylint: disable=W0212
        other_dubins_platform.sim_time = sim_time2

        assert (dubins_platform == other_dubins_platform) == expected

    @pytest.mark.unit_test
    def test__eq__class_mismatch(self, dubins_platform):
        """
        A simple test to check comparisons with objects that are not DubinsPlatforms
        """

        class NotADubinsPlatform:
            """Dummy class for equality comparison"""

        other = NotADubinsPlatform()
        assert dubins_platform != other


# Tests for save_action_to_platform
@pytest.fixture(name="action")
def get_action():
    """
    Returns an action array (numpy.ndarray)
    """
    return np.array([10000, 3.0, -54])


# axis
@pytest.mark.unit_test
def test_DubinsPlatform_saveActionToPlatform(dubins_platform, action):  # pylint: disable=W0212
    """
    Tests for CWHPlatform method - save_action_to_platform(),
    This is a parametrized test, where the tests are in the action_to_platform_tests
    """
    # test no axis path
    dubins_platform.save_action_to_platform(action)
    assert np.array_equal(dubins_platform._last_applied_action, action)  # pylint: disable=W0212

    # test axis path
    new_action = 49.536
    axis = 1
    dubins_platform.save_action_to_platform(new_action, axis=axis)
    action[axis] = new_action

    assert np.array_equal(dubins_platform._last_applied_action, action)  # pylint: disable=W0212
