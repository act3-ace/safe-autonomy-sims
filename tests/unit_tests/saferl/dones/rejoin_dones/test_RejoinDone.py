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

This module holds unit tests and fixtures for the CrashDoneFunction.

Author: John McCarroll
"""

import os

import numpy as np
import pytest
from corl.libraries.environment_dict import DoneDict

from saferl.dones.rejoin_dones import RejoinDone
from tests.conftest import delimiter, read_test_cases

# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/dones/rejoin/RejoinDone_test_cases.yaml")
parameterized_fixture_keywords = ["agent_name", "done_status", "expected_status"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture(name="agent_name")
def fixture_agent_name(request):
    """
    The name of the agent whose status is checked by RejoinDone.

    Returns
    -------
    str
        The agent's name
    """
    return request.param


@pytest.fixture(name="done_status")
def fixture_done_status(request):
    """
    The agent's DoneStatus

    Returns
    -------
    corl.dones.done_func_base.DoneStatusCodes
    """
    return request.param


@pytest.fixture(name="local_done_info")
def fixture_local_done_info():
    """
    Returns placeholder array for local_done_info

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """

    return np.array([1, 2, 3])


@pytest.fixture(name="local_dones")
def fixture_local_dones(agent_name, done_status):
    """
    Returns a DoneDict populated with dummy test agents and one defined agent name

    Returns
    -------
    corl.libraries.environment_dict.DoneDict
        The DoneDict of agent's DoneStatuses
    """

    dones = DoneDict()
    dones["test_agent_1"] = None
    dones["test_agent_2"] = None
    dones["test_agent_3"] = None

    dones[agent_name] = done_status

    return dones


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name):
    """
    A fixture that instantiates a CrashDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent

    Returns
    -------
    CrashDoneFunction
        An instantiated component under test
    """

    return RejoinDone(name=cut_name, agent_name=agent_name)


@pytest.fixture(name='call_results')
def fixture_call_results(cut, observation, action, next_observation, next_state, local_dones, local_done_info):
    """
    A fixture responsible for calling the CrashDoneFunction and returning the results.

    Parameters
    ----------
    cut : CrashDoneFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    next_state : StateDict
        The StateDict that the CrashDoneFunction mutates
    local_dones : DoneDict
        The DoneDict that holds names and DoneStatuses of each agent
    local_done_info : numpy.ndarray
        The local_dones_info placeholder array

    Returns
    -------
    results : DoneDict
        The resulting DoneDict from calling the CrashDoneFunction
    """

    results = cut(observation, action, next_observation, next_state, local_dones, local_done_info)
    return results


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_call(call_results, expected_status):
    """
    A parameterized test to ensure that the CrashDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the CrashDoneFunction
    expected_status : None or DoneStatusCodes
        The expected status corresponding to the status of the agent's episode
    """

    for name, status in call_results.items():
        assert status == expected_status, "Expected agent {} to have status {} but received {}".format(name, expected_status, status)
