"""
Unit tests for the DockingSuccessReward function from the docking_rewards module
"""

import os

import pytest

from saferl.rewards.docking_rewards import DockingSuccessReward
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/DockingSuccessRewardFunction_test_cases.yaml")
parameterized_fixture_keywords = [
    "platform_position",
    "platform_velocity",
    "scale",
    "sim_time",
    "timeout",
    "docking_region_radius",
    "max_vel_constraint",
    "expected_value"
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
    """request.param
    Get the 'sim_time' parameter from the test config input
    """
    return request.param


@pytest.fixture(name='platform_velocity')
def fixture_platform_velocity(request):
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
