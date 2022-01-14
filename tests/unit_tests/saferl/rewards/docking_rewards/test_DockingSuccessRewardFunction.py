"""
Unit tests for the DockingSuccessReward function from the docking_rewards module
"""

import os
from unittest import mock

import pytest

from saferl.rewards.docking_rewards import DockingSuccessReward
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/DockingSuccessRewardFunction_test_cases.yaml")
parameterized_fixture_keywords = [
    "platform_position", "velocity", "scale", "sim_time", "timeout", "docking_region_radius", "max_vel_constraint", "expected_value"
]
test_configs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='platform_position')
def fixture_platform_position(request):
    """
    Parameterized fixture for returning platform position defined in test_configs.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D position
    """
    return request.param


@pytest.fixture(name='expected_value')
def fixture_expected_value(request):
    """
    Parameterized fixture for comparison to the expected value of reward to be found corresponding to the agent_name (the key)
    in the RewardDict returned by the DockingSuccessRewardFunction.

    Returns
    -------
    bool
        The expected value of the reward functino
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


@pytest.fixture(name='scale')
def fixture_scale(request):
    """
    Get 'scale' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='timeout')
def fixture_timeout(request):
    """
    Get the 'timeout' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='timeout_reward')
def fixture_timeout_reward(request):
    """
    Get the 'timeout_reward' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='distance_reward')
def fixture_distance_reward(request):
    """
    Get the 'distance_reward' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='crash_reward')
def fixture_crash_reward(request):
    """
    Get the 'crash_reward' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='max_goal_distance')
def fixture_max_goal_distance(request):
    """
    Get the 'max_goal_distance' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='docking_region_radius')
def fixture_docking_region_radius(request):
    """
    Get the 'docking_region_radius' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='max_vel_constraint')
def fixture_max_vel_constraint(request):
    """
    Get the 'max_vel_constraint' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='sim_time')
def fixture_sim_time(request):
    """
    Get the 'sim_time' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='velocity')
def fixture_velocity(request):
    """
    Get the 'velocity' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, scale, agent_name, timeout, docking_region_radius, max_vel_constraint):
    """
    A fixture that instantiates a DockingSuccessRewardFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent

    Returns
    -------
    DockingSuccessRewardFunction
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


@pytest.fixture(name='call_results')
def fixture_call_results(
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
    A fixture responsible for calling the DockingSuccessRewardFunction and returning the results.

    Parameters
    ----------
    cut : DockingSuccessRewardFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    next_state : StateDict
        The StateDict that the DockingSuccessRewardFunction mutates
    platform : MagicMock
        The mock platform to be returned to the DockingSuccessRewardFunction when it uses get_platform_by_name()

    Returns
    -------
    results : RewardDict
        The resulting RewardDict from calling the DockingSuccessRewardFunction
    """
    with mock.patch("saferl.rewards.docking_rewards.get_platform_by_name") as func:
        platform.position = platform_position
        platform.velocity = velocity
        platform.sim_time = sim_time
        func.return_value = platform
        results = cut(observation, action, next_observation, state, next_state, observation_space, observation_units)
        return results


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True)
def test_reward_function(call_results, agent_name, expected_value):
    """
    A parameterized test to ensure that the DockingSuccessRewardFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the DockingSuccessRewardFunction
    agent_name : str
        The name of the agent
    expected_value : float
        The expected value from the reward function
    """
    assert call_results[agent_name] == expected_value
