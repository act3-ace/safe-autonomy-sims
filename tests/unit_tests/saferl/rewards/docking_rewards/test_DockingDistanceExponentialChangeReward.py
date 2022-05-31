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

Unit tests for the DockingDistanceExponentialChangeReward function from the docking_rewards module
"""

import os
from unittest import mock

import pytest

from saferl.rewards.docking_rewards import DockingDistanceExponentialChangeReward
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../test_cases/rewards/docking/DockingDistanceExponentialChangeReward_test_cases.yaml"
)
parameterized_fixture_keywords = ["platform_position1", "platform_position2", "scale", "c", "a", "pivot", "pivot_ratio", "expected_value"]
test_configs, ids = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


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


@pytest.fixture(name='c')
def fixture_c(request):
    """
    Get the 'c' parameter from the test config
    """
    return request.param


@pytest.fixture(name='a')
def fixture_a(request):
    """
    Get the 'a' parameter from the test config
    """
    return request.param


@pytest.fixture(name='pivot')
def fixture_pivot(request):
    """
    Get the 'pivot' parameter from the test config
    """
    return request.param


@pytest.fixture(name='pivot_ratio')
def fixture_pivot_ratio(request):
    """
    Get the 'pivot_ratio' parameter from the test config
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, scale, c, a, pivot, pivot_ratio):
    """
    A fixture that instantiates a DockingDistanceExponentialChangeReward Function and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    scale : float
        value by which rewards can be scaled
    c : float
        TBD
    a : float
        TBD
    pivot : float
        TBD
    pivot_ratio : float
        TBD

    Returns
    -------
    DockingDistanceExponentialChangeReward
        An instantiated component under test
    """
    return DockingDistanceExponentialChangeReward(
        name=cut_name, scale=scale, c=c, a=a, pivot=pivot, pivot_ratio=pivot_ratio, agent_name=agent_name
    )


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
    A fixture responsible for calling the DockingDistanceExponentialChangeReward and returning the results.

    Parameters
    ----------
    cut : DockingDistanceExponentialChangeReward
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


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=ids)
def test_reward_function(call_results, agent_name, expected_value):
    """
    A parameterized test to ensure that the DockingDistanceExponentialChangeReward behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the DockingDistanceExponentialChangeReward
    agent_name : str
        The name of the agent
    expected_value : bool
        The expected bool corresponding to whether the agent's episode is done or not
    """
    assert call_results[agent_name] == expected_value
