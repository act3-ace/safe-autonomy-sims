"""
This module holds unit tests and fixtures for the DockingRelativeVelocityConstraintDoneFunction.
Author: John McCarroll
"""

from unittest import mock

import numpy as np
import pytest
from act3_rl_core.dones.done_func_base import DoneStatusCodes

from saferl.dones.docking_dones import DockingRelativeVelocityConstraintDoneFunction

test_configs = [
    # (platform_velocity, target_velocity, constraint_velocity, expected_value, expected_status),
    (np.array([19, 0, 0]), np.array([10, 0, 0]), 10, False, None),
    (np.array([20, 0, 0]), np.array([10, 0, 0]), 10, False, None),
    (np.array([21, 0, 0]), np.array([10, 0, 0]), 10, True, DoneStatusCodes.LOSE),
]


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


@pytest.fixture(name='target_name')
def fixture_target_name():
    """
    Fixture for returning target's name.

    Returns
    -------
    str
        name of the target platform
    """
    return "target"


@pytest.fixture(name='target_velocity')
def fixture_target_velocity(request):
    """
    Parameterized fixture for returning target velocity defined in test_configs.

    Returns
    -------
    numpy.ndarray
        Three element array describing target's 3D velocity vector
    """
    return request.param


@pytest.fixture(name='target_position')
def fixture_target_position():
    """
    Placeholder fixture for returning target position.

    Returns
    -------
    NoneType
        None
    """
    return None


@pytest.fixture(name='target')
def fixture_target(mocker, target_position, target_velocity, target_name):
    """
    A fixture to create a mock platform with a position property

    Parameters
    ----------
    mocker : fixture
        A pytest-mock fixture which exposes unittest.mock functions
    target_position : numpy.ndarray
        The platform's 3D positional vector
    target_velocity : numpy.ndarray
        The platform's 3D velocity vector
    target_name : str
        The name of the agent

    Returns
    -------
    test_target_platform : MagicMock
        A mock of a platform with a position property
    """
    test_target_platform = mocker.MagicMock(name=target_name)
    test_target_platform.position = target_position
    test_target_platform.velocity = target_velocity
    return test_target_platform


@pytest.fixture(name='constraint_velocity')
def fixture_constraint_velocity(request):
    """
    Parameterized fixture for returning the constraint_velocity (the maximum acceptable relative velocity between
    the deputy and the target) passed to the DockingRelativeVelocityConstraintDoneFunction's constructor, as
    defined in test_configs.

    Returns
    -------
    int
        The max allowed relative velocity in a docking episode
    """
    return request.param


@pytest.fixture(name='cut')
def fixture_cut(cut_name, agent_name, constraint_velocity):
    """
    A fixture that instantiates a DockingRelativeVelocityConstraintDoneFunction and returns it.

    Parameters
    ----------
    cut_name : str
        The name of the component under test
    agent_name : str
        The name of the agent
    constraint_velocity : int
        The velocity limit passed to the DockingRelativeVelocityConstraintDoneFunction constructor

    Returns
    -------
    DockingRelativeVelocityConstraintDoneFunction
        An instantiated component under test
    """
    target_name = "target"
    return DockingRelativeVelocityConstraintDoneFunction(
        name=cut_name, agent_name=agent_name, constraint_velocity=constraint_velocity, target=target_name
    )


@pytest.fixture(name='call_results')
def fixture_call_results(cut, observation, action, next_observation, next_state, platform, target):
    """
    A fixture responsible for calling the DockingVelocityLimitDoneFunction and returning the results.

    Parameters
    ----------
    cut : DockingVelocityLimitDoneFunction
        The component under test
    observation : numpy.ndarray
        The observation array
    action : numpy.ndarray
        The action array
    next_observation : numpy.ndarray
        The next_observation array
    next_state : StateDict
        The StateDict that the DockingVelocityLimitDoneFunction mutates
    platform : MagicMock
        The mock platform to be returned to the DockingVelocityLimitDoneFunction when it uses get_platform_by_name()

    Returns
    -------
    results : DoneDict
        The resulting DoneDict from calling the DockingVelocityLimitDoneFunction
    """
    with mock.patch("saferl.dones.docking_dones.get_platform_by_name") as func:
        # construct iterable of return values (platforms)
        platforms = []
        for _ in test_configs:
            platforms.append(platform)
            platforms.append(target)
        func.side_effect = platforms

        results = cut(observation, action, next_observation, next_state)
        return results


@pytest.mark.unit_test
@pytest.mark.parametrize(
    "platform_velocity,target_velocity,constraint_velocity,expected_value,expected_status", test_configs, indirect=True
)
def test_call(call_results, next_state, agent_name, cut_name, expected_value, expected_status):
    """
    A parameterized test to ensure that the DockingRelativeVelocityConstraintDoneFunction behaves as intended.

    Parameters
    ----------
    call_results : DoneDict
        The resulting DoneDict from calling the DockingRelativeVelocityConstraintDoneFunction
    next_state : StateDict
        The StateDict that may have been mutated by the DockingRelativeVelocityConstraintDoneFunction
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
