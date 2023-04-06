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

This module defines fixtures common to unit tests on DoneFunctions.

Author: John McCarroll
"""

from unittest import mock

import gym
import pytest


@pytest.fixture()
def expected_value(request):
    """
    Parameterized fixture for comparison to the expected boolean to be found corresponding to the agent_name (the key)
    in the DoneDict returned

    Returns
    -------
    bool
        The expected value of the boolean assigned to describe if the agent is done or not
    """
    return request.param


@pytest.fixture()
def expected_status(request):
    """
    Parameterized fixture for comparison to the expected DoneStatus to be assigned to the next_state StateDict

    Returns
    -------
    None or DoneStatusCodes
        The expected value assigned to describe the agent's episode status in the next_state StateDict
    """
    return request.param


@pytest.fixture(name='observation_space')
def fixture_observation_space():
    """
    A fixture to return a gym.spaces.dict.Dict.
    """
    return gym.spaces.dict.Dict()


@pytest.fixture(name='observation_untis')
def fixture_observation_units():
    """
    A fixture to return a gym.spaces.dict.Dict.
    """
    return gym.spaces.dict.Dict()


# TODO: generalize patch so that this function need not be duplicated by lower level conftests. same function is being patched, just the
# file it is being imported into changes in each lower dir.
@pytest.fixture()
def call_results(cut, observation, action, next_observation, next_state, observation_space, observation_units, platform):
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
    with mock.patch("safe_autonomy_sims.core.dones.cwh.docking_dones.get_platform_by_name") as func:
        func.return_value = platform
        results = cut(observation, action, next_observation, next_state, observation_space, observation_units)
        return results
