"""
Unit tests for the DockingSuccessReward function from the docking_rewards module
"""

from unittest import mock

import numpy as np
import pytest
from act3_rl_core.libraries.state_dict import StateDict

from saferl.rewards.docking_rewards import DockingSuccessReward

# modify for the test cases

# test input
# "platform_position, velocity, scale, sim_time, timeout, docking_region_radius, max_vel_constraint, expected_value"

test_configs = [
    # successful docking test 1
    (np.array([0, 0, 0]), 5.0, 1.0, 200, 2000, 0.5, 10, 1.9),
    # successful docking test 2
    (np.array([0.2, 0.2, 0.2]), 7.0, 1.0, 200, 2000, 0.5, 10, 1.9),
    # unsuccessful docking test 1
    (np.array([1, 1, 1]), 5.0, 1.0, 400, 2000, 0.5, 10, 0),
    # unsuccessful docking test 2
    (np.array([0.2, 0.2, 0.2]), 20.0, 1.0, 200, 2000, 0.5, 10, 0),
    # unsuccessful docking test 3
    (np.array([0.5, 0.5, 0.5]), 7.0, 1.0, 200, 2000, 0.5, 10, 0),
]

#
# @pytest.fixture
# def observation():
#     """
#     Generic fixture for creating a naive observation for running Done and Reward function tests.
#
#     Returns
#     -------
#     numpy.ndarray
#         Placeholder array
#     """
#     return np.array([0, 0, 0])


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
def expected_value(request):
    """
    Parameterized fixture for comparison to the expected boolean to be found corresponding to the agent_name (the key)
    in the DoneDict returned by the DockingFailureRewardFunction.

    Returns
    -------
    bool
        The expected value of the boolean assigned to describe if the agent is done or not
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
def scale(request):
    """
    Get 'scale' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def timeout(request):
    """
    Get the 'timeout' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def timeout_reward(request):
    """
    Get the 'timeout_reward' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def distance_reward(request):
    """
    Get the 'distance_reward' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def crash_reward(request):
    """
    Get the 'crash_reward' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def max_goal_distance(request):
    """
    Get the 'max_goal_distance' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def docking_region_radius(request):
    """
    Get the 'docking_region_radius' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def max_vel_constraint(request):
    """
    Get the 'max_vel_constraint' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def sim_time(request):
    """
    Get the 'sim_time' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def velocity(request):
    """
    Get the 'velocity' parameter from the test config input
    """
    return request.param


@pytest.fixture()
def cut(cut_name, scale, agent_name, timeout, docking_region_radius, max_vel_constraint):
    """
    A fixture that instantiates a DockingFailureRewardFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent

    Returns
    -------
    DockingFailureRewardFunction
        An instantiated component under test
    """
    return DockingSuccessReward(
        name=cut_name,
        agent_name=agent_name,
        scale=scale,
        timeout=timeout,
        docking_region_radius=docking_region_radius,
        max_vel_constraint=max_vel_constraint
    )


# @pytest.fixture()
# def next_state(agent_name, cut_name):
#     """
#     A fixture for creating a StateDict populated with the structure expected by the CWHDistanceChangeReward
#
#     Parameters
#     ----------
#     agent_name : str
#         The name of the agent
#     cut_name : str
#         The name of the component under test
#
#     Returns
#     -------
#     state : StateDict
#         The populated StateDict
#     """
#     state = StateDict({"episode_state": {agent_name: {cut_name: None}}})
#     return state


@pytest.fixture()
def call_results(
    cut,
    platform_position,
    observation,
    action,
    next_observation,
    state,
    next_state,
    observation_space,
    observation_units,
    platform,
    sim_time,
    velocity,
):
    """
    A fixture responsible for calling the DockingFailureRewardFunction and returning the results.

    Parameters
    ----------
    cut : DockingFailureRewardFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    next_state : StateDict
        The StateDict that the DockingFailureRewardFunction mutates
    platform : MagicMock
        The mock platform to be returned to the DockingFailureRewardFunction when it uses get_platform_by_name()

    Returns
    -------
    results : DoneDict
        The resulting DoneDict from calling the DockingFailureRewardFunction
    """
    with mock.patch("saferl.rewards.docking_rewards.get_platform_by_name") as func:
        platform.position = platform_position
        platform.velocity = velocity
        platform.sim_time = sim_time
        func.return_value = platform
        results = cut(observation, action, next_observation, state, next_state, observation_space, observation_units)
        return results


#timeout_reward,distance_reward,crash_reward,timeout_reward,max_goal_distance,docking_region_radius,max_vel_constraint
@pytest.mark.unit_test
@pytest.mark.parametrize(
    "platform_position, velocity, scale, sim_time, timeout, docking_region_radius, max_vel_constraint, expected_value",
    test_configs,
    indirect=True
)
def test_reward_function(call_results, agent_name, expected_value):
    """
    A parameterized test to ensure that the DockingFailureRewardFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the DockingFailureRewardFunction
    next_state : StateDict
        The StateDict that may have been mutated by the DockingFailureRewardFunction
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
