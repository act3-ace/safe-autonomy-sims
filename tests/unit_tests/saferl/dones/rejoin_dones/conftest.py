"""
This module defines fixtures common to all rejoin DoneFunction tests.

Author: John McCarroll
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


@pytest.fixture(name='lead')
def fixture_lead(mocker, lead_position, lead_orientation, lead_name):
    """
    A fixture to create a mock platform with a position property

    Parameters
    ----------
    mocker : fixture
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
    test_lead_platform = mocker.MagicMock(name=lead_name)
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
