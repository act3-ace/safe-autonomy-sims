"""
This module holds fixtures common to all backend simulator tests.

Author: John McCarroll
"""

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


def evaluate(entity, attr_targets, angles={}, error_bound=None, proportional_error_bound=0):
    # evaluation
    for key, value in attr_targets.items():
        # get expected and actual results
        if type(value) is list:
            value = np.array(value, dtype=np.float64)
        result = entity.__getattribute__(key)

        # angle wrapping
        if key in angles:
            if len(angles[key]) > 1:
                # variables compared are lists
                for index, is_angle in enumerate(angles[key]):
                    if is_angle:
                        diff = abs(value[index] - result[index]) % (2*np.pi)
                        error = min(wrapped_difference, 2 * np.pi - wrapped_difference)
                        result[index] = error
                        value[index] = 0.0          # replace value w/ expected difference after angle wrapping
            else:
                # variables compared are single values
                if angles[key][0]:
                    wrapped_difference = math.fmod(abs(value - result), 2 * np.pi)
                    error = min(wrapped_difference, 2*np.pi - wrapped_difference)
                    result = error
                    value = 0.0              # replace value w/ expected difference after angle wrapping

        if error_bound is None:
            # direct comparison
            if type(value) in [np.ndarray, list]:
                # handle array case
                assert np.array_equal(result, value), \
                    "Error of {}! Expected attribute {} values to be {} but instead received {}".format(
                        np.array(value) - np.array(result), key, value, result)
            else:
                # compare entity value and target value
                assert result == value, \
                    "Error of {}! Expected attribute {} value(s) to be {} but instead received {}".format(
                        value - result, key, value, result)
        else:
            # bounded comparison
            if type(value) in [np.ndarray, list]:
                # handle array case
                differences = np.abs(np.subtract(result, value))
                error_bounds = np.abs(result) * proportional_error_bound + error_bound
                in_bounds = differences <= error_bounds
                assert np.all(in_bounds), \
                    "Expected attribute {} values to be {} +/- {} but instead received {} with an error of +/- {}".format(
                        key, value, error_bound, result, differences)
            else:
                # compare entity value and target value
                error_bound = error_bound + proportional_error_bound
                difference = abs(value - result)
                assert abs(result - value) <= error_bound, \
                    "Expected attribute {} value to be {} +/- {} but instead received {} with an error of +/- {}".format(
                        key, value, error_bound, result, difference)
