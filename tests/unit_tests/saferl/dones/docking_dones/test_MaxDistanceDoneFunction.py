"""
This module holds unit tests and fixtures for the MaxDistanceDoneFunction.

Author: John McCarroll
"""

from unittest import mock

import numpy as np
import pytest
import pytest_mock
from act3_rl_core.dones.done_func_base import DoneStatusCodes
from act3_rl_core.libraries.state_dict import StateDict

from saferl.dones.docking_dones import MaxDistanceDoneFunction

test_configs = [
    # (platform_position, max_distance, expected_value, expected_status),
    (np.array([9, 0, 0]), 10, False, None),
    (np.array([10, 0, 0]), 10, False, None),
    (np.array([11, 0, 0]), 10, True, DoneStatusCodes.LOSE),
]


@pytest.fixture()
def platform_position(request):
    """
    Parameterized fixture for returning platform position defined in test_configs.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D position
    """
    return request.param


@pytest.fixture()
def max_distance(request):
    """
    Parameterized fixture for returning the max_distance passed to the MaxDistanceDoneFunction's constructor, as defined
    in test_configs.

    Returns
    -------
    int
        The max allowed distance in a docking episode
    """
    return request.param


@pytest.fixture()
def expected_value(request):
    """
    Parameterized fixture for comparison to the expected boolean to be found corresponding to the agent_name (the key)
    in the DoneDict returned by the MaxDistanceDoneFunction.

    Returns
    -------
    bool
        The expected value of the boolean assigned to describe if the agent is done or not
    """
    return request.param


@pytest.fixture()
def expected_status(request):
    """
    Parameterized fixture for comparison to the expected DoneStatus to be assigned to the next_state StateDict by the
    MaxDistanceDoneFunction.

    Returns
    -------
    None or DoneStatusCodes
        The expected value assigned to describe the agent's episode status in the next_state StateDict
    """
    return request.param


@pytest.fixture()
def platform(mocker, platform_position, agent_name):
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
    return test_platform


@pytest.fixture()
def cut(cut_name, agent_name, max_distance):
    """
    A fixture that instantiates a MaxDistanceDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    max_distance : int
        The max distance passed to the MaxDistanceDoneFunction constructor

    Returns
    -------
    MaxDistanceDoneFunction
        An instantiated component under test
    """
    return MaxDistanceDoneFunction(name=cut_name, max_distance=max_distance, agent_name=agent_name)


@pytest.fixture()
def next_state(agent_name, cut_name):
    """
    A fixture for creating a StateDict populated with the structure expected by the MaxDistanceDoneFunction.

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


@pytest.fixture()
def call_results(cut, observation, action, next_observation, next_state, platform):
    """
    A fixture responsible for calling the MaxDistanceDoneFunction and returning the results.

    Parameters
    ----------
    cut : MaxDistanceDoneFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    next_state : StateDict
        The StateDict that the MaxDistanceDoneFunction mutates
    platform : MagicMock
        The mock platform to be returned to the MaxDistanceDoneFunction when it uses get_platform_by_name()

    Returns
    -------
    results : DoneDict
        The resulting DoneDict from calling the MaxDistanceDoneFunction
    """
    with mock.patch("saferl.dones.docking_dones.get_platform_by_name") as func:
        func.return_value = platform
        results = cut(observation, action, next_observation, next_state)
        return results


@pytest.mark.unit_test
@pytest.mark.parametrize("platform_position,max_distance,expected_value,expected_status", test_configs, indirect=True)
def test_call(call_results, next_state, agent_name, cut_name, expected_value, expected_status):
    """
    A parameterized test to ensure that the MaxDistanceDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the MaxDistanceDoneFunction
    next_state : StateDict
        The StateDict that may have been mutated by the MaxDistanceDoneFunction
    agent_name : str
        The name of the agent
    cut_name : str
        The name of the component under test
    expected_value : bool
        The expected bool corresponding to whether the agent's episode is done or not
    expected_status : None or DoneStatusCodes
        The expected status corresponding to the status of the agent's episode
    """
    assert call_results[agent_name] == expected_value
    assert next_state.episode_state[agent_name][cut_name] is expected_status
