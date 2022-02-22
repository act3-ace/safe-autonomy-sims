"""
This module holds unit tests and fixtures for the SuccessfulRejoinDoneFunction.

Author: John McCarroll
"""

import os
from unittest import mock

import pytest

from saferl.dones.rejoin_dones import SuccessfulRejoinDoneFunction
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../test_cases/dones/rejoin/SuccessfulRejoinDoneFunction_test_cases.yaml"
)
parameterized_fixture_keywords = [
    "rejoin_region_radius", "offset_values", "lead_orientation", "lead_position", "platform_position", "expected_value", "expected_status"
]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='lead_orientation')
def fixture_lead_orientation(request):
    """
    Parameterized fixture for returning platform velocity defined in test_SuccessfulRejoinDoneFunction.yaml

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D velocity
    """
    return request.param


@pytest.fixture(name='lead_position')
def fixture_lead_position(request):
    """
    Parameterized fixture for returning lead position defined in test_SuccessfulRejoinDoneFunction.yaml

    Returns
    -------
    numpy.ndarray
        Three element array describing lead's 3D position vector
    """
    return request.param


@pytest.fixture(name="rejoin_region_radius")
def fixture_rejoin_region_radius(request):
    """
    Parameterized fixture for returning offset_values defined in test_SuccessfulRejoinDoneFunction.yaml

    Returns
    -------
    float
        The defined radius of the rejoin region
    """
    return request.param


@pytest.fixture(name="offset_values")
def fixture_offset_values(request):
    """
    Parameterized fixture for returning offset_values defined in test_SuccessfulRejoinDoneFunction.yaml

    Returns
    -------
    list
        Three element array describing 3D position vector of xyz magnitudes for offset values
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, lead_name, rejoin_region_radius, offset_values):
    """
    A fixture that instantiates a SuccessfulRejoinDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    lead_name : str
        The name of the lead
    rejoin_region_radius : float
        The radius of the rejoin region
    offset_values : list
        A list of floats describing xyz magnitudes of the offset to the center of the rejoin region

    Returns
    -------
    SuccessfulRejoinDoneFunction
        An instantiated component under test
    """

    return SuccessfulRejoinDoneFunction(
        name=cut_name, agent_name=agent_name, rejoin_region_radius=rejoin_region_radius, lead=lead_name, offset_values=offset_values
    )


@pytest.fixture(name='call_results')
def fixture_call_results(cut, observation, action, next_observation, next_state, platform, lead):
    """
    A fixture responsible for calling the SuccessfulRejoinDoneFunction and returning the results.

    Parameters
    ----------
    cut : SuccessfulRejoinDoneFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    next_state : StateDict
        The StateDict that the SuccessfulRejoinDoneFunction mutates
    platform : MagicMock
        The mock wingman to be returned to the SuccessfulRejoinDoneFunction when it uses get_platform_by_name()
    lead : MagicMock
        The mock lead to be returned to the SuccessfulRejoinDoneFunction when it uses get_platform_by_name()


    Returns
    -------
    results : DoneDict
        The resulting DoneDict from calling the SuccessfulRejoinDoneFunction
    """
    with mock.patch("saferl.dones.rejoin_dones.get_platform_by_name") as func:
        # construct iterable of return values (platforms)
        platforms = []
        for _ in test_configs:
            platforms.append(lead)
            platforms.append(platform)
        func.side_effect = platforms

        results = cut(observation, action, next_observation, next_state)
        return results


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_call(call_results, next_state, agent_name, cut_name, expected_value, expected_status):
    """
    A parameterized test to ensure that the SuccessfulRejoinDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the SuccessfulRejoinDoneFunction
    next_state : StateDict
        The StateDict that may have been mutated by the SuccessfulRejoinDoneFunction
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
