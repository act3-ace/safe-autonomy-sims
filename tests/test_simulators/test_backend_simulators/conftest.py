"""
This module holds fixtures common to all backend simulator tests.

Author: John McCarroll
"""

import pytest
import numpy as np


@pytest.fixture
def initial_position(request):
    return request.param


@pytest.fixture
def initial_velocity(request):
    return request.param


@pytest.fixture
def initial_entity_state(initial_position, initial_velocity):
    return initial_position + initial_velocity


# @pytest.fixture
# def target_position(request):
#     return request.param
#
#
# @pytest.fixture
# def target_velocity(request):
#     return request.param
#
#
# @pytest.fixture
# def target_entity_state(target_position, target_velocity):
#     return target_position + target_velocity


@pytest.fixture
def entity(initial_entity_state):
    return None


@pytest.fixture
def acted_entity(num_steps, entity, action):
    for i in range(num_steps):
        entity.step(1, action)

    return entity


def evaluate(entity, attr_targets, error_bound=None):
    # evaluation
    if error_bound is None:
        # direct comparison
        for key, value in attr_targets.items():
            if type(value) in [np.ndarray, list]:
                # handle array case
                assert np.array_equal(entity.__getattribute__(key), value), \
                    "Expected attribute {} values to be {} but instead received {}".format(
                        key, value, entity.__getattribute__(key))
            else:
                # compare entity value and target value
                assert entity.__getattribute__(key) == value, \
                    "Expected attribute {} value(s) to be {} but instead received {}".format(
                        key, value, entity.__getattribute__(key))
    else:
        # bounded comparison
        for key, value in attr_targets.items():
            if type(value) in [np.ndarray, list]:
                # handle array case
                value = np.array(value)
                np.abs(np.subtract(entity.__getattribute__(key), value))
                in_bounds = value <= error_bound
                assert np.all(in_bounds), \
                    "Expected attribute {} values to be {} +/- {} but instead received {}".format(
                        key, value, error_bound, entity.__getattribute__(key))
            else:
                # compare entity value and target value
                assert abs(entity.__getattribute__(key) - value) <= error_bound, \
                    "Expected attribute {} value to be {} +/- {} but instead received {}".format(
                        key, value, error_bound, entity.__getattribute__(key))
