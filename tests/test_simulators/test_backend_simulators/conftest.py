"""
This module holds fixtures common to all backend simulator tests.

Author: John McCarroll
"""

from cv2 import exp
import pytest
import numpy as np
import math


@pytest.fixture
def angles(request):
    return request.param


@pytest.fixture
def attr_init(request):
    return request.param


@pytest.fixture
def initial_entity_state(attr_init):
    # common fixture that composes an entity's initial state array based on attr_init dict read in from yaml config file
    # (order of KVPs should be preserved from yaml file)
    if type(attr_init) is dict:
        initial_state = []
        for key, value in attr_init.items():
            initial_state += value if type(value) in [list, np.ndarray] else [value]
    return np.array(initial_state)


@pytest.fixture
def entity(initial_entity_state):
    # Placeholder fixture - each backend simulation test module should override this fixture to return a constructed
    # entity (specifically the CUT)
    raise NotImplementedError


@pytest.fixture
def acted_entity(num_steps, entity, control):
    for i in range(num_steps):
        entity.step(1, control)

    return entity


def evaluate(entity, attr_targets, angles=None, error_bound=None, proportional_error_bound=0):
    if angles is None:
        angles = {}

    error_message = ''

    for attr_name, expected in attr_targets.items():
        # get expected and actual results
        if type(expected) is list:
            expected = np.array(expected, dtype=np.float64)
        result = entity.__getattribute__(attr_name)

        if error_bound is None:
            # direct comparison
            if not np.array_equal(result, expected):
                error_message += "\t--Expected attribute {} values to be {} but instead received {}\n".format(
                    attr_name, expected, result)

        else:
            angle_wrap = angles.get(attr_name, False)
            if isinstance(angle_wrap, list):
                angle_wrap = np.array(angle_wrap, dtype=bool)
                assert isinstance(result, np.ndarray), "If angles are specified as a list, attribute must be a numpy ndarray"
                assert angle_wrap.shape == result.shape, "If angles are specified as a list, they must match the shape of the attr vector"

            in_bounds, diff, error_margin = bounded_compare(expected, result, error_bound, proportional_error_bound, angle_wrap)
            if not in_bounds:
                error_message += "\t--Expected attribute {} values to be {} +/- {} but instead received {} with an error of +/- {}\n".format(
                    attr_name, expected, error_margin, result, diff)

    failed = bool(error_message)
    assert not failed, '\n'+error_message


def bounded_compare(expected, result, error_bound, proportional_error_bound, angle_wrap):
    error_margin = np.abs(result) * proportional_error_bound + error_bound

    # compute error difference and apply angle wrapping
    diff = np.abs(result - expected)
    if not isinstance(diff, np.ndarray):
        diff = np.array(diff, dtype=np.float64)
    diff[angle_wrap] = (diff[angle_wrap] + np.pi) % (2*np.pi) - np.pi

    return np.all(diff <= error_margin), diff, error_margin
