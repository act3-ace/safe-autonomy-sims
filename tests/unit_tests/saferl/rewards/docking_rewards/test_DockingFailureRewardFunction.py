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

Unit tests for the DockingFailureReward function from the docking_rewards module
"""
import os

import pytest
from corl.libraries.state_dict import StateDict

from saferl.rewards.cwh.docking_rewards import DockingFailureReward
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../test_cases/rewards/docking/DockingFailureReward_test_cases.yaml"
)
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
    "velocity_threshold",
    "threshold_distance",
    "slope",
    "mean_motion",
    "lower_bound",
    "expected_value"
]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


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


@pytest.fixture(name='velocity_threshold')
def fixture_velocity_threshold(request):
    """
    Return 'velocity_threshold' value from the test config input
    """
    return request.param


@pytest.fixture(name='threshold_distance')
def fixture_threshold_distance(request):
    """
    Return 'threshold_distance' value from the test config input
    """
    return request.param


@pytest.fixture(name='slope')
def fixture_slope(request):
    """
    Return 'slope' value from the test config input
    """
    return request.param


@pytest.fixture(name='mean_motion')
def fixture_mean_motion(request):
    """
    Return 'mean_motion' value from the test config input
    """
    return request.param


@pytest.fixture(name='lower_bound')
def fixture_lower_bound(request):
    """
    Return 'lower_bound' value from the test config input
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


@pytest.fixture(name='platform_position')
def fixture_platform_position(request):
    """
    Return 'platform_position'  value from the test config input
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
    platform_name,
    timeout,
    timeout_reward,
    distance_reward,
    crash_reward,
    max_goal_distance,
    docking_region_radius,
    velocity_threshold,
    threshold_distance,
    slope,
    mean_motion,
    lower_bound,
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
        platform_names=[platform_name],
        timeout=timeout,
        timeout_reward=timeout_reward,
        distance_reward=distance_reward,
        crash_reward=crash_reward,
        max_goal_distance=max_goal_distance,
        docking_region_radius=docking_region_radius,
        velocity_threshold=velocity_threshold,
        threshold_distance=threshold_distance,
        slope=slope,
        mean_motion=mean_motion,
        lower_bound=lower_bound,
    )


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
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
