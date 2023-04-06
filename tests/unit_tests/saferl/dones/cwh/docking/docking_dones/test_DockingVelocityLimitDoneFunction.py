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

This module holds unit tests and fixtures for the DockingVelocityLimitDoneFunction.

Author: John McCarroll
"""

import os

import pytest

from saferl.dones.cwh.docking_dones import DockingVelocityLimitDoneFunction
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../../../test_cases/dones/cwh/docking/DockingVelocityLimitDoneFunction_test_cases.yaml"
)
parameterized_fixture_keywords = [
    "platform_velocity",
    "velocity_threshold",
    "threshold_distance",
    "slope",
    "mean_motion",
    "lower_bound",
    "expected_value",
    "expected_status"
]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='platform_velocity')
def fixture_platform_velocity(request):
    """
    Parameterized fixture for returning platform velocity defined in test_configs.

    Returns
    -------
    numpy.ndarray
        Three element array describing platform's velocity
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


@pytest.fixture(name='cut')
def cut(
    cut_name,
    agent_name,
    velocity_threshold,
    threshold_distance,
    slope,
    mean_motion,
    lower_bound,
):
    """
    A fixture that instantiates a DockingVelocityLimitDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    velocity_limit : int
        The velocity limit passed to the DockingVelocityLimitDoneFunction constructor

    Returns
    -------
    DockingVelocityLimitDoneFunction
        An instantiated component under test
    """
    return DockingVelocityLimitDoneFunction(
        name=cut_name,
        agent_name=agent_name,
        platform_name=agent_name,
        velocity_threshold=velocity_threshold,
        threshold_distance=threshold_distance,
        slope=slope,
        mean_motion=mean_motion,
        lower_bound=lower_bound,
    )


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_call(call_results, next_state, agent_name, cut_name, expected_value, expected_status):
    """
    A parameterized test to ensure that the DockingVelocityLimitDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the DockingVelocityLimitDoneFunction
    next_state : StateDict
        The StateDict that may have been mutated by the DockingVelocityLimitDoneFunction
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
