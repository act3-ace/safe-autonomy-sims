"""
This module holds unit tests and fixtures for the SuccessfulRejoinDoneFunction.

Author: John McCarroll
"""

import os
from unittest import mock

import pytest

from saferl.core.dones.rejoin_dones import RejoinSuccessDone
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../test_cases/dones/rejoin/RejoinSuccessDoneFunction_test_cases.yaml"
)
parameterized_fixture_keywords = [
    "radius",
    "offset",
    "step_size",
    "success_time",
    "lead_orientation",
    "lead_position",
    "platform_position",
    "expected_value",
    "expected_status"
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


@pytest.fixture(name="radius")
def fixture_radius(request):
    """
    Parameterized fixture for returning radius defined in test_SuccessfulRejoinDoneFunction.yaml

    Returns
    -------
    float
        The defined radius of the rejoin region
    """
    return request.param


@pytest.fixture(name="offset")
def fixture_offset(request):
    """
    Parameterized fixture for returning offset defined in test_RejoinSuccessDoneFunction.yaml

    Returns
    -------
    list
        Three element array describing 3D position vector of xyz magnitudes for offset values
    """
    return request.param


@pytest.fixture(name="step_size")
def fixture_step_size(request):
    """
    Parameterized fixture for returning step_size defined in test_RejoinSuccessDoneFunction.yaml

    Returns
    -------
    float
        Simulation step size
    """
    return request.param


@pytest.fixture(name="success_time")
def fixture_success_time(request):
    """
    Parameterized fixture for returning success time defined in test_RejoinSuccessDoneFunction.yaml

    Returns
    -------
    float
        Time wingman is in rejoin to be determined successful
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, lead_name, radius, offset, step_size, success_time):
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
    radius : float
        The radius of the rejoin region
    offset : list
        A list of floats describing xyz magnitudes of the offset to the center of the rejoin region
    step_size : float
        Size of one single simulation step
    success_time : float
        Time wingman must spend in rejoin region for success to be true

    Returns
    -------
    RejoinSuccessDone
        An instantiated component under test
    """

    return RejoinSuccessDone(
        name=cut_name,
        agent_name=agent_name,
        platform_name=agent_name,
        radius=radius,
        lead=lead_name,
        offset=offset,
        step_size=step_size,
        success_time=success_time
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
    with mock.patch("saferl.core.dones.rejoin_dones.get_platform_by_name") as func:
        # construct iterable of return values (platforms)
        platforms = []
        for _ in test_configs:
            platforms.append(platform)
            platforms.append(lead)
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
