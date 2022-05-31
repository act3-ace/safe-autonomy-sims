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

This module holds unit tests and fixtures for the TimeoutDoneFunction.

Author: John McCarroll
"""

import os

import pytest

from saferl.dones.common_dones import TimeoutDoneFunction
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/dones/common/TimeoutDoneFunction_test_cases.yaml")
parameterized_fixture_keywords = ["sim_time", "max_sim_time", "expected_value", "expected_status"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name='max_sim_time')
def fixture_max_sim_time(request):
    """
    Parameterized fixture for returning the, max_sim_time, max allowed time in the simulation,
    which is passed to the TimeoutDoneFunction's constructor, as defined in test_configs.

    Returns
    -------
    int
        The max allowed distance in a docking episode
    """
    return request.param


@pytest.fixture(name='sim_time')
def fixture_sim_time(request):
    """
    Parameterized fixture for returning the 'sim_time', i.e. current time of the simulation,
    passed to the TimeoutDoneFunction's constructor, as  defined in test_configs.

    Returns
    -------
    int
        The max allowed time in simulation in a docking episode
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, max_sim_time):
    """
    A fixture that instantiates a TimeoutDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    max_sim_time : float

    Returns
    -------
    TimeoutDoneFunction
        An instantiated component under test
    """
    return TimeoutDoneFunction(name=cut_name, agent_name=agent_name, platform_name=agent_name, max_sim_time=max_sim_time)


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_call(call_results, next_state, agent_name, cut_name, expected_value, expected_status):
    """
    A parameterized test to ensure that the TimeoutDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the TimeoutDoneFunction
    next_state : StateDict
        The StateDict that may have been mutated by the TimeoutDoneFunction
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
