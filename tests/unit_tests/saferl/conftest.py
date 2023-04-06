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

This module holds fixtures common to the saferl package tests.

Author: John McCarroll
"""
from unittest import mock

import numpy as np
import pytest
from corl.libraries.state_dict import StateDict


@pytest.fixture(name='observation')
def fixture_observation():
    """
    Generic fixture for creating a naive observation for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='action')
def fixture_action():
    """
    Generic fixture for creating a naive action for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='next_observation')
def fixture_next_observation():
    """
    Generic fixture for creating a naive observation for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='state')
def fixture_state():
    """
    A fixture for creating a StateDict populated with the structure expected by the DoneFunction.

    Parameters
    ----------
    agent_name : str
        The name of the agent
    cut_name : str
        The name of the component under test

    Returns
    -------
    state : StateDict
        The populated StateDict
    """
    state = StateDict({})
    return state


@pytest.fixture(name='next_state')
def fixture_next_state(agent_name, cut_name):
    """
    A fixture for creating a StateDict populated with the structure expected by the DoneFunction.

    Parameters
    ----------
    agent_name : str
        The name of the agent
    cut_name : str
        The name of the component under test

    Returns
    -------
    state : StateDict
        The populated StateDict
    """
    state = StateDict({"episode_state": {agent_name: {cut_name: None}}})
    return state


@pytest.fixture(name='observation_space')
def fixture_observation_space():
    """
    A fixture for creating a StateDict populated with the structure expected by the RewardFunction.

    Parameters
    ----------
    agent_name : str
        The name of the agent
    cut_name : str
        The name of the component under test

    Returns
    -------
    state : StateDict
        The populated StateDict
    """
    state = StateDict({})
    return state


@pytest.fixture(name='observation_units')
def fixture_observation_units():
    """
    A fixture for creating a StateDict populated with the structure expected by the RewardFunction.

    Parameters
    ----------
    agent_name : str
        The name of the agent
    cut_name : str
        The name of the component under test

    Returns
    -------
    state : StateDict
        The populated StateDict
    """
    state = StateDict({})
    return state


@pytest.fixture(name='agent_name')
def fixture_agent_name():
    """
    Fixture to define a common agent name for tests.

    Returns
    -------
    str
        The common agent name
    """
    return "blue0"


@pytest.fixture(name='platform_name')
def fixture_platform_name():
    """
    Fixture to define a common platform name for tests.

    Returns
    -------
    str
        The common platform name
    """
    return "blue0_ctrl"


@pytest.fixture(name='cut_name')
def fixture_cut_name():
    """
    Fixture to define a common name for the Component Under Test.

    Returns
    -------
    str
        The common cut name
    """
    return "cut"


@pytest.fixture(name='platform_position')
def fixture_platform_position():
    """
    placeholder fixture to be overridden by testing modules for returning platform position.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D position
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='platform_velocity')
def fixture_platform_velocity():
    """
    placeholder fixture to be overridden by testing modules for returning platform velocity.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D velocity
    """
    return np.array([0, 0, 0])


@pytest.fixture(name='sim_time')
def fixture_sim_time():
    """
    placeholder fixture to be overridden by testing modules for returning platform sim time.

    Returns
    -------
    float
        Current simulation time
    """
    return 0.0


@pytest.fixture(name='platform')
def fixture_platform(platform_position, platform_velocity, sim_time, agent_name):
    """
    A fixture to create a mock platform with a position property and velocity property

    Parameters
    ----------
    mock : fixture
        A pytest-mock fixture which exposes unittest.mock functions
    platform_position : numpy.ndarray
        The platform's 3D positional vector
    platform_velocity : numpy.ndarray
        The platform's 3D velocity vector
    sim_time : float
        The platform's simulation time
    agent_name : str
        The name of the agent

    Returns
    -------
    test_platform : MagicMock
        A mock of a platform with a position property
    """
    test_platform = mock.MagicMock(name=agent_name)
    test_platform.position = platform_position
    test_platform.velocity = platform_velocity
    test_platform.sim_time = sim_time
    return test_platform
