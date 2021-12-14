"""
Unit tests for the CWHDistanceChangeReward function from the docking_rewards module
"""

from unittest import mock

import numpy as np
import pytest

# this is very much needed
from act3_rl_core.libraries.state_dict import StateDict

from saferl.rewards.docking_rewards import CWHDistanceChangeReward

test_configs = [
    # case 1
    (np.array([0, 0, 0]), np.array([1, 0, 0]), 2, 2),
    # case 2
    (np.array([1, 1, 1]), np.array([4, 4, 4]), 0.5, 0.5 * np.sqrt(27)),
    # case 3 - large number case
    (np.array([100, 100, 100]), np.array([1000, 1000, 1000]), 0.4, 0.4 * np.sqrt(3 * (900 * 900))),
    # edge case 1
    (np.array([0, 0, 0]), np.array([0, 0, 0]), 2, 0),
    # edge case 2
    (np.array([1, 1, 1]), np.array([1, 1, 1]), 1.5, 0),
]

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


@pytest.fixture(name='platform_position1')
def fixture_platform_position1(request):
    """
    Parameterized fixture for returning platform position defined in test_configs.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D position
    """
    return request.param


@pytest.fixture(name='platform_position2')
def fixture_platform_position2(request):
    """
    Parameterized fixture for returning platform position defined in test_configs.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D position
    """
    return request.param


@pytest.fixture(name='scale')
def fixture_scale(request):
    """
    Get the 'scale' parameter from the test config
    """
    return request.param


@pytest.fixture(name='expected_value')
def fixture_expected_value(request):
    """
    Parameterized fixture for comparison to the expected boolean to be found corresponding to the agent_name (the key)
    in the RewardDict returned by the MaxDistanceDoneFunction.

    Returns
    -------
    float
        The expected value of the reward function
    """
    return request.param


@pytest.fixture(name='platform')
def fixture_platform(mocker, platform_position, agent_name):
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


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, scale):
    """
    A fixture that instantiates a CWHDistanceChangeReward Function and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    scale : float
        value by which rewards can be scaled

    Returns
    -------
    CWHDistanceChangeReward
        An instantiated component under test
    """
    return CWHDistanceChangeReward(name=cut_name, scale=scale, agent_name=agent_name)


@pytest.fixture(name='next_state')
def fixture_next_state(agent_name, cut_name):
    """
    A fixture for creating a StateDict populated with the structure expected by the CWHDistanceChangeReward Function.

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


@pytest.fixture(name='call_results')
def fixture_call_results(
    cut,
    platform_position1,
    platform_position2,
    observation,
    action,
    next_observation,
    state,
    next_state,
    observation_space,
    observation_units,
    platform
):
    """
    A fixture responsible for calling the CWHDistanceChangeReward and returning the results.

    Parameters
    ----------
    cut : CWHDistanceChangeReward
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
    results : RewardDict
        The resulting RewardDict from calling the MaxDistanceDoneFunction
    """
    with mock.patch("saferl.rewards.docking_rewards.get_platform_by_name") as func:
        platform.position = platform_position1
        func.return_value = platform
        _ = cut(observation, action, next_observation, state, next_state, observation_space, observation_units)

        platform.position = platform_position2
        func.return_value = platform
        results2 = cut(observation, action, next_observation, state, next_state, observation_space, observation_units)
        return results2


# expected value -- idk
# platform position,
#far awy from docking region, mid way to docking region , close to docking region, then at docking region


@pytest.mark.unit_test
@pytest.mark.parametrize("platform_position1, platform_position2, scale, expected_value", test_configs, indirect=True)
def test_reward_function(call_results, agent_name, expected_value):
    """
    A parameterized test to ensure that the CWHDistanceChangeReward behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the CWHDistanceChangeReward
    next_state : StateDict
        The StateDict that may have been mutated by the CWHDistanceChangeReward
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
