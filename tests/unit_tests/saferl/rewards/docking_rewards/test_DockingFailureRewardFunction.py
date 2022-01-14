"""
Unit tests for the DockingFailureReward function from the docking_rewards module
"""
import os

import pytest
from act3_rl_core.libraries.state_dict import StateDict

from saferl.rewards.docking_rewards import DockingFailureReward
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/DockingFailureReward_test_cases.yaml")
parameterized_fixture_keywords = [
    "platform_position",
    "platform_velocity",
    "sim_time",
    "timeout",
    "timeout_reward",
    "distance_reward",
    "crash_reward",
    "max_goal_distance",
    "docking_region_radius",
    "max_vel_constraint",
    "expected_value"
]
test_configs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='timeout')
def fixture_timeout(request):
    """
    Return 'timeout' value from the test config
    """
    return request.param


@pytest.fixture(name='timeout_reward')
def fixture_timeout_reward(request):
    """
    Return 'timeout_reward' value from the test config input
    """
    return request.param


@pytest.fixture(name='distance_reward')
def fixture_distance_reward(request):
    """
    Return 'distance_reward' value from the test config input
    """
    return request.param


@pytest.fixture(name='crash_reward')
def fixture_crash_reward(request):
    """
    Return 'crash_reward' value from the test config input
    """
    return request.param


@pytest.fixture(name='max_goal_distance')
def fixture_max_goal_distance(request):
    """
    Return 'max_goal_distance' value from the test config input
    """
    return request.param


@pytest.fixture(name='docking_region_radius')
def fixture_docking_region_radius(request):
    """
    Return 'docking_region_radius' value from the test config input
    """
    return request.param


@pytest.fixture(name='max_vel_constraint')
def fixture_max_vel_constraint(request):
    """
    Return 'max_vel_constraint' value from the test config input
    """
    return request.param


@pytest.fixture(name='sim_time')
def fixture_sim_time(request):
    """
    Return 'sim_time'  (time of the simulation)  value from the test config input
    """
    return request.param


@pytest.fixture(name='platform_velocity')
def fixture_platform_velocity(request):
    """
    Return 'platform_velocity'  value from the test config input
    """
    return request.param


@pytest.fixture(name='next_state')
def fixture_next_state(agent_name, cut_name):
    """
    A fixture for creating a StateDict populated with the structure expected by the DockingFailureRewardFunction

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


@pytest.fixture(name='cut')
def fixture_cut(
    cut_name,
    agent_name,
    timeout,
    timeout_reward,
    distance_reward,
    crash_reward,
    max_goal_distance,
    docking_region_radius,
    max_vel_constraint
):
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
    return DockingFailureReward(
        name=cut_name,
        agent_name=agent_name,
        timeout=timeout,
        timeout_reward=timeout_reward,
        distance_reward=distance_reward,
        crash_reward=crash_reward,
        max_goal_distance=max_goal_distance,
        docking_region_radius=docking_region_radius,
        max_vel_constraint=max_vel_constraint
    )


# @pytest.fixture(name='call_results')
# def fixture_call_results(
#     cut,
#     platform_position,
#     observation,
#     action,
#     next_observation,
#     state,
#     next_state,
#     observation_space,
#     observation_units,
#     platform,
#     sim_time,
#     platform_velocity,
# ):
#     """
#     A fixture responsible for calling the DockingFailureRewardFunction and returning the results.
#
#     Parameters
#     ----------
#     cut : DockingFailureRewardFunction
#         The component under test
#     observation : numpy.ndarray
#         The observation array
#     action : numpy.ndarray
#         The action array
#     next_observation : numpy.ndarray
#         The next_observation array
#     next_state : StateDict
#         The StateDict that the DockingFailureRewardFunction mutates
#     platform : MagicMock
#         The mock platform to be returned to the DockingFailureRewardFunction when it uses get_platform_by_name()
#
#     Returns
#     -------
#     results : RewardDict
#         The resulting RewardDict from calling the DockingFailureRewardFunction
#     """
#     with mock.patch("saferl.rewards.docking_rewards.get_platform_by_name") as func:
#         platform.position = platform_position
#         platform.platform_velocity = platform_velocity
#         platform.sim_time = sim_time
#         func.return_value = platform
#         results = cut(observation, action, next_observation, state, next_state, observation_space, observation_units)
#         return results


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True)
def test_reward_function(call_results, agent_name, expected_value):
    """
    A parameterized test to ensure that the DockingFailureRewardFunction behaves as intended.

    Parameters
    ----------
    call_results : RewardDict
        The resulting RewardDict from calling the DockingFailureRewardFunction
    agent_name : str
        The name of the agent
    expected_value : bool
        The expected bool corresponding to whether the agent's episode is done or not
    """
    assert call_results[agent_name] == expected_value
