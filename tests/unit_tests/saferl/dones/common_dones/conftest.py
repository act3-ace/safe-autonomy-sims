"""
This module defines fixtures common to common dones testing.
"""

from unittest import mock

import pytest


@pytest.fixture()
def call_results(cut, observation, action, next_observation, next_state, platform):
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
    with mock.patch("saferl.dones.common_dones.get_platform_by_name") as func:
        func.return_value = platform
        results = cut(observation, action, next_observation, next_state)
        return results
