"""
This module holds fixtures common to the saferl package tests.

Author: John McCarroll
"""

import numpy as np
import pytest
from act3_rl_core.libraries.state_dict import StateDict


@pytest.fixture
def observation():
    """
    Generic fixture for creating a naive observation for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture
def action():
    """
    Generic fixture for creating a naive action for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture
def next_observation():
    """
    Generic fixture for creating a naive observation for running Done and Reward function tests.

    Returns
    -------
    numpy.ndarray
        Placeholder array
    """
    return np.array([0, 0, 0])


@pytest.fixture
def next_state():
    """
    Generic fixture for creating a naive state for running Done and Reward function tests.

    Returns
    -------
    StateDict
        Placeholder state
    """
    state = StateDict({})
    return state


@pytest.fixture
def agent_name():
    """
    Fixture to define a common agent name for tests.

    Returns
    -------
    str
        The common agent name
    """
    return "safety_buddy"


@pytest.fixture()
def cut_name():
    """
    Fixture to define a common name for the Component Under Test.

    Returns
    -------
    str
        The common cut name
    """
    return "cut"
