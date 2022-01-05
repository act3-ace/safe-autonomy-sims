"""
This module holds unit tests and fixtures for the TimeoutDoneFunction.

Author: John McCarroll
"""

import pytest
from act3_rl_core.dones.done_func_base import DoneStatusCodes
from act3_rl_core.libraries.state_dict import StateDict

from saferl.dones.common_dones import TimeoutDoneFunction

test_configs = [
    # (sim_time,max_sim_time,expected_value,expected_status),
    (9, 10, False, None),
    (10, 10, True, DoneStatusCodes.LOSE),
    (11, 10, True, DoneStatusCodes.LOSE),
]


@pytest.fixture(name='max_sim_time')
def fixture_max_sim_time(request):
    """
    Parameterized fixture for returning the, max_sim_time, max allowed time in the simulation,
    which is passed to the TimeoutDoneFunction's constructor, as defined in test_configs.

    Returns
    -------
    int
        The max allowed distance in a docking episode
    """
    return request.param


@pytest.fixture(name='sim_time')
def fixture_sim_time(request):
    """
    Parameterized fixture for returning the 'sim_time', i.e. current time of the simulation,
    passed to the TimeoutDoneFunction's constructor, as  defined in test_configs.

    Returns
    -------
    int
        The max allowed time in simulation in a docking episode
    """
    return request.param


@pytest.fixture(name='next_state')
def fixture_next_state(mocker, agent_name, cut_name, sim_time):
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
    mock_platform = mocker.MagicMock()
    mock_platform.sim_time = sim_time
    state = StateDict({"episode_state": {agent_name: {cut_name: None}}, "sim_platforms": [mock_platform]})
    return state


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, max_sim_time):
    """
    A fixture that instantiates a TimeoutDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    max_sim_time : float

    Returns
    -------
    TimeoutDoneFunction
        An instantiated component under test
    """
    return TimeoutDoneFunction(name=cut_name, agent_name=agent_name, max_sim_time=max_sim_time)


@pytest.mark.unit_test
@pytest.mark.parametrize("sim_time,max_sim_time,expected_value,expected_status", test_configs, indirect=True)
def test_call(call_results, next_state, agent_name, cut_name, expected_value, expected_status):
    """
    A parameterized test to ensure that the TimeoutDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the TimeoutDoneFunction
    next_state : StateDict
        The StateDict that may have been mutated by the TimeoutDoneFunction
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
