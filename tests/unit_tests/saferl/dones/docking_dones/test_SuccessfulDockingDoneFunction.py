"""
This module holds unit tests and fixtures for the SuccessfulDockingDoneFunction.

Author: John McCarroll
"""

import os

import pytest

from saferl.dones.docking_dones import SuccessfulDockingDoneFunction
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../test_cases/docking/dones/SuccessfulDockingDoneFunction_test_cases.yaml"
)
parameterized_fixture_keywords = [
    "platform_position", "platform_velocity", "docking_region_radius", "velocity_limit", "expected_value", "expected_status"
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


@pytest.fixture(name='platform_velocity')
def fixture_platform_velocity(request):
    """
    Parameterized fixture for returning platform velocity defined in test_configs.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's 3D velocity
    """
    return request.param


@pytest.fixture(name='docking_region_radius')
def fixture_docking_region_radius(request):
    """
    Parameterized fixture for returning the docking_region_radius passed to the SuccessfulDockingDoneFunction's constructor, as defined
    in test_configs.

    Returns
    -------
    int
        The max allowed distance in a docking episode
    """
    return request.param


@pytest.fixture(name='velocity_limit')
def fixture_velocity_limit(request):
    """
    Parameterized fixture for returning the velocity_limit passed to the SuccessfulDockingDoneFunction's constructor, as
    defined in test_configs.

    Returns
    -------
    int
        The max allowed velocity in a docking episode
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, docking_region_radius, velocity_limit):
    """
    A fixture that instantiates a SuccessfulDockingDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    docking_region_radius : int
        The radius of the docking region passed to the SuccessfulDockingDoneFunction constructor
    velocity_limit : int
        The velocity limit passed to the SuccessfulDockingDoneFunction constructor

    Returns
    -------
    SuccessfulDockingDoneFunction
        An instantiated component under test
    """
    return SuccessfulDockingDoneFunction(
        name=cut_name, agent_name=agent_name, docking_region_radius=docking_region_radius, velocity_limit=velocity_limit
    )


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True)
def test_call(call_results, next_state, agent_name, cut_name, expected_value, expected_status):
    """
    A parameterized test to ensure that the SuccessfulDockingDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the SuccessfulDockingDoneFunction
    next_state : StateDict
        The StateDict that may have been mutated by the SuccessfulDockingDoneFunction
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
