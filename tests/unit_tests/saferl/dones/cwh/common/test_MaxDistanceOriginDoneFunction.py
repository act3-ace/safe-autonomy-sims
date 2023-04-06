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

This module holds unit tests and fixtures for the MaxDistanceDoneFunction.

Author: John McCarroll
"""

import os

import pytest

from saferl.dones.cwh.common import MaxDistanceDoneFunction
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../../test_cases/dones/cwh/docking/common/MaxDistanceDoneFunction_test_cases.yaml"
)
parameterized_fixture_keywords = ["platform_position", "max_distance", "expected_value", "expected_status"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


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


@pytest.fixture(name='max_distance')
def fixture_max_distance(request):
    """
    Parameterized fixture for returning the max_distance passed to the MaxDistanceDoneFunction's constructor, as defined
    in test_configs.

    Returns
    -------
    int
        The max allowed distance in a docking episode
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, max_distance):
    """
    A fixture that instantiates a MaxDistanceDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    max_distance : int
        The max distance passed to the MaxDistanceDoneFunction constructor

    Returns
    -------
    MaxDistanceDoneFunction
        An instantiated component under test
    """
    return MaxDistanceDoneFunction(name=cut_name, platform_name=agent_name, max_distance=max_distance, agent_name=agent_name)


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
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
