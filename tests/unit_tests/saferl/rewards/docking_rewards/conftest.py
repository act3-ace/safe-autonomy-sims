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

This module defines fixtures common to docking rewards testing.
"""

from unittest import mock

import numpy as np
import pytest


@pytest.fixture(name='expected_value')
def fixture_expected_value(request):
    """
    Parameterized fixture for comparison to the expected value of reward to be found corresponding to the agent_name (the key)
    in the RewardDict returned by the DockingSuccessRewardFunction.

    Returns
    -------
    bool
        The expected value of the reward function
    """
    return request.param


@pytest.fixture(name='sim_time')
def fixture_sim_time():
    """
    Default sim_time
    """
    return 0


@pytest.fixture(name='platform_velocity')
def fixture_platform_velocity():
    """
    Default platform velocity

    Returns
    -------
    numpy.ndarray
        placeholder platform 3D velocity vector
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='platform_position')
def fixture_platform_position():
    """
    Default platform position

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D position
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='platform')
def fixture_platform(mocker, platform_position, platform_velocity, sim_time, agent_name):
    """
    A fixture to create a mock platform with a position property

    Parameters
    ----------
    mocker : fixture
        A pytest-mock fixture which exposes unittest.mock functions
    platform_position : numpy.ndarray
        The platform's 3D position
    agent_name : str
        The name of the agent

    Returns
    -------
    test_platform : MagicMock
        A mock of a platform with a position property
    """
    test_platform = mocker.MagicMock(name=agent_name)
    test_platform.position = platform_position
    test_platform.velocity = platform_velocity
    test_platform.sim_time = sim_time
    return test_platform


@pytest.fixture(name='call_results')
def fixture_call_results(
    cut,
    observation,
    action,
    next_observation,
    state,
    next_state,
    observation_space,
    observation_units,
    platform,
):
    """
    A fixture responsible for calling the DockingSuccessRewardFunction and returning the results.

    Parameters
    ----------
    cut : DockingSuccessRewardFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    state : numpy.ndarray
        Placeholder array for state
    next_state : StateDict
        The StateDict that the DockingSuccessRewardFunction mutates
    observation_space : numpy.ndarray
        Placeholder array for observation space
    observation_units : numpy.ndarray
        Placeholder array for observation units
    platform : MagicMock
        The mock platform to be returned to the DockingSuccessRewardFunction when it uses get_platform_by_name()

    Returns
    -------
    results : RewardDict
        The resulting RewardDict from calling the DockingSuccessRewardFunction
    """
    with mock.patch("saferl.core.rewards.docking_rewards.get_platform_by_name") as func:
        with mock.patch("saferl.core.utils.get_platform_by_name") as func1:
            func.return_value = platform
            func1.return_value = platform
            results = cut(observation, action, next_observation, state, next_state, observation_space, observation_units)
            return results
