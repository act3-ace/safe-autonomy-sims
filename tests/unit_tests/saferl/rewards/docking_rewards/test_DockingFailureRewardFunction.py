from collections import OrderedDict
from unittest import mock

import numpy as np
import pytest
import pytest_mock
from act3_rl_core.libraries.state_dict import StateDict

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.rewards.docking_rewards import DockingFailureReward


# modify for the test cases

# test input
# ['platform_position', 'velocity', 'sim_time', 'timeout', 'timeout_reward', 'distance_reward', 'crash_reward', 'max_goal_distance', 'docking_region_radius', 'max_vel_constraint', 'expected_value']
test_configs = [
                # test for crash
                (np.array([0, 0, 0]),5.0,200,2000,-1,-1,-1, 40000,0.5,0,-1),
                # test for max distance
                (np.array([50000, 50000, 50000]),5.0,200,2000,-1,-1,-1, 40000,0.5,0,-1),
                # test for max velocity exceeded
                (np.array([0.01, 0.01, 0.01]),20.0,200,2000,-1,-1,-1, 40000,0.5,10,-1),
                # test for a successful case, in docking therefore no failure reward,
                (np.array([500, 500, 500]),8.0,200,2000,-1,-1,-1, 40000,0.5,10,0)    
                ]



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
def platform(mocker, platform_position,  agent_name):
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
def timeout(request):
    return request.param

@pytest.fixture()
def timeout_reward(request):
    return request.param


@pytest.fixture()
def distance_reward(request):
    return request.param

@pytest.fixture()
def crash_reward(request):
    return request.param


@pytest.fixture()
def max_goal_distance(request):
    return request.param

@pytest.fixture()
def docking_region_radius(request):
    return request.param

@pytest.fixture()
def max_vel_constraint(request):
    return request.param

@pytest.fixture()
def sim_time(request):
    return request.param

@pytest.fixture()
def velocity(request):
    return request.param

@pytest.fixture()
def cut(cut_name, agent_name, timeout, timeout_reward,distance_reward,crash_reward,max_goal_distance,docking_region_radius,max_vel_constraint):
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
    return DockingFailureReward(name=cut_name,agent_name=agent_name, timeout=timeout, timeout_reward=timeout_reward, distance_reward=distance_reward,crash_reward=crash_reward, max_goal_distance=max_goal_distance,docking_region_radius=docking_region_radius,max_vel_constraint=max_vel_constraint)


@pytest.fixture()
def next_state(agent_name, cut_name):
    """
    A fixture for creating a StateDict populated with the structure expected by the CWHDistanceChangeReward

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
@pytest.mark.parametrize("platform_position, velocity, sim_time, timeout, timeout_reward, distance_reward, crash_reward, max_goal_distance, docking_region_radius, max_vel_constraint, expected_value", test_configs, indirect=True)
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
