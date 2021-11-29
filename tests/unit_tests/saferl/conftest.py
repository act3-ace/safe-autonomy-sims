"""
This module holds fixtures common to the saferl package tests.

Author: John McCarroll
"""

import numpy as np
import pytest
import pytest_mock
from act3_rl_core.libraries.state_dict import StateDict


@pytest.fixture
def observation():
    """
    Generic fixture for creating a naive observation for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture
def action():
    """
    Generic fixture for creating a naive action for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture
def next_observation():
    """
    Generic fixture for creating a naive observation for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture()
def next_state(agent_name, cut_name):
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


@pytest.fixture
def agent_name():
    """
    Fixture to define a common agent name for tests.

    Returns
    -------
    str
        The common agent name
    """
    return "safety_buddy"


@pytest.fixture()
def cut_name():
    """
    Fixture to define a common name for the Component Under Test.

    Returns
    -------
    str
        The common cut name
    """
    return "cut"


@pytest.fixture()
def platform_position():
    """
    placeholder fixture to be overridden by testing modules for returning platform position.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D position
    """
    return np.array([0, 0, 0])


@pytest.fixture()
def platform_velocity():
    """
    placeholder fixture to be overridden by testing modules for returning platform velocity.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D velocity
    """
    return np.array([0, 0, 0])


@pytest.fixture()
def platform(mocker, platform_position, platform_velocity, agent_name):
    """
    A fixture to create a mock platform with a position property

    Parameters
    ----------
    mocker : fixture
        A pytest-mock fixture which exposes unittest.mock functions
    platform_position : numpy.ndarray
        The platform's 3D positional vector
    platform_velocity : numpy.ndarray
        The platform's 3D velocity vector
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
    return test_platform
