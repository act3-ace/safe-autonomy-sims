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

This module defines fixtures common to all rejoin DoneFunction tests.

Author: John McCarroll
"""

from unittest import mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture(name='lead')
def fixture_lead(lead_position, lead_orientation, lead_name):
    """
    A fixture to create a mock platform with a position property

    Parameters
    ----------
    mock : fixture
        A pytest-mock fixture which exposes unittest.mock functions
    lead_position : numpy.ndarray
        The platform's 3D positional vector
    lead_orientation : scipy.spatial.transform.Rotation
        The platform's 3D velocity vector
    lead_name : str
        The name of the agent

    Returns
    -------
    test_lead_platform : MagicMock
        A mock of a platform with a position property
    """
    test_lead_platform = mock.MagicMock(name=lead_name)
    test_lead_platform.position = lead_position
    test_lead_platform.orientation = lead_orientation
    return test_lead_platform


@pytest.fixture(name='lead_position')
def fixture_lead_position():
    """
    Default lead position.

    Returns
    -------
    numpy.ndarray
        Three element array describing lead's 3D position vector
    """
    return np.array([0.0, 0.0, 0.0])


@pytest.fixture(name='lead_orientation')
def fixture_lead_orientation():
    """
    Default lead orientation.

    Returns
    -------
    scipy.spatial.transform.Rotation
        The orientation of the lead aircraft in 3D space
    """
    return Rotation.identity()


@pytest.fixture(name='lead_name')
def fixture_lead_name():
    """
    Fixture for returning lead's name.

    Returns
    -------
    str
        name of the lead platform
    """
    return "test_lead"


@pytest.fixture()
def call_results(cut, observation, action, next_observation, next_state, observation_space, observation_units, platform):
    """
    A fixture responsible for calling the appropriate component under test and returns the results.
    This variation of the function is specific to docking_dones

    Parameters
    ----------
    cut : DoneFunction
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
    with mock.patch("saferl.core.dones.rejoin_dones.get_platform_by_name") as func:
        func.return_value = platform
        results = cut(observation, action, next_observation, next_state, observation_space, observation_units)
        return results
