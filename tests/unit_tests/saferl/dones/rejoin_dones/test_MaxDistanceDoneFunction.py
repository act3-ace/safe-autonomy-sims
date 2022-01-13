"""
This module holds unit tests and fixtures for the MaxDistanceDoneFunction.

Author: John McCarroll
"""

import os
from unittest import mock

import pytest

from saferl.dones.rejoin_dones import MaxDistanceDoneFunction
from tests.conftest import read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/MaxDistanceDoneFunction_test_cases.yaml")
parameterized_fixture_keywords = ["max_distance", "lead_position", "platform_position", "expected_value", "expected_status"]
delimiter = ","
test_configs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name="platform_position")
def fixture_platform_position(request):
    """
    Parameterized fixture for returning wingman position defined in test_MaxDistanceDoneFunction.yaml

    Returns
    -------
    numpy.ndarray
        Wingman 3D position vector

    """
    return request.param


@pytest.fixture(name='lead_position')
def fixture_lead_position(request):
    """
    Parameterized fixture for returning lead position defined in test_MaxDistanceDoneFunction.yaml

    Returns
    -------
    numpy.ndarray
        Three element array describing lead's 3D position vector
    """
    return request.param


@pytest.fixture(name="max_distance")
def fixture_max_distance(request):
    """
    Parameterized fixture for returning max_distance defined in test_MaxDistanceDoneFunction.yaml

    Returns
    -------
    float
        The defined max allowable distance of the rejoin region
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, lead_name, max_distance):
    """
    A fixture that instantiates a MaxDistanceDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    lead_name : str
        The name of the lead
    max_distance : float
        The maximum acceptable distance between wingman and lead before failure

    Returns
    -------
    MaxDistanceDoneFunction
        An instantiated component under test
    """

    return MaxDistanceDoneFunction(name=cut_name, agent_name=agent_name, max_distance=max_distance, lead=lead_name)


@pytest.fixture(name='call_results')
def fixture_call_results(cut, observation, action, next_observation, next_state, platform, lead):
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
        The mock wingman to be returned to the MaxDistanceDoneFunction when it uses get_platform_by_name()
    lead : MagicMock
        The mock lead to be returned to the MaxDistanceDoneFunction when it uses get_platform_by_name()


    Returns
    -------
    results : DoneDict
        The resulting DoneDict from calling the MaxDistanceDoneFunction
    """
    with mock.patch("saferl.dones.rejoin_dones.get_platform_by_name") as func:
        # construct iterable of return values (platforms)
        platforms = []
        for _ in test_configs:
            platforms.append(platform)
            platforms.append(lead)
        func.side_effect = platforms

        results = cut(observation, action, next_observation, next_state)
        return results


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True)
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
