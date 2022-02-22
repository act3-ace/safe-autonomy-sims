"""
This module defines fixture common to the docking_dones test modules.
"""

from unittest import mock

import numpy as np
import pytest


@pytest.fixture(name='target_name')
def fixture_target_name():
    """
    Fixture for returning target's name.

    Returns
    -------
    str
        name of the target platform
    """
    return "test_target"


@pytest.fixture(name='target_velocity')
def fixture_target_velocity():
    """
    Default target velocity array.

    Returns
    -------
    numpy.ndarray
        Three element array describing target's 3D velocity vector
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='target_position')
def fixture_target_position():
    """
    Default target position array.

    Returns
    -------
    NoneType
        None
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='target')
def fixture_target(mocker, target_position, target_velocity, target_name):
    """
    A fixture to create a mock platform with a position property

    Parameters
    ----------
    mocker : fixture
        A pytest-mock fixture which exposes unittest.mock functions
    target_position : numpy.ndarray
        The platform's 3D positional vector
    target_velocity : numpy.ndarray
        The platform's 3D velocity vector
    target_name : str
        The name of the agent

    Returns
    -------
    test_target_platform : MagicMock
        A mock of a platform with a position property
    """
    test_target_platform = mocker.MagicMock(name=target_name)
    test_target_platform.position = target_position
    test_target_platform.velocity = target_velocity
    return test_target_platform


@pytest.fixture()
def call_results(cut, observation, action, next_observation, next_state, platform):
    """
    A fixture responsible for calling the appropriate component under test and returns the results.
    This variation of the function is specific to docking_dones

    Parameters
    ----------
    cut : DoneFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    next_state : StateDict
        The StateDict that the DockingVelocityLimitDoneFunction mutates
    platform : MagicMock
        The mock platform to be returned to the DockingVelocityLimitDoneFunction when it uses get_platform_by_name()

    Returns
    -------
    results : DoneDict
        The resulting DoneDict from calling the DockingVelocityLimitDoneFunction
    """
    with mock.patch("saferl.dones.docking_dones.get_platform_by_name") as func:
        with mock.patch("saferl.utils.get_platform_by_name") as func1:
            func.return_value = platform
            func1.return_value = platform
            results = cut(observation, action, next_observation, next_state)
            return results
