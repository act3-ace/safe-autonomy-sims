"""
This module holds fixtures common to all backend simulator tests.

Author: John McCarroll
"""
import pytest
# import numpy as np


@pytest.fixture
def initial_entity_state(request):
    return request.param


@pytest.fixture
def entity(initial_entity_state):
    return None


@pytest.fixture
def acted_entity(num_steps, entity, action):
    for i in range(num_steps):
        entity.step(1, action)

    return entity


# @pytest.fixture
# def evaluate(acted_entity, attr_targets, error_bound):
#     # evaluation
#     if error_bound is None:
#         # direct comparison
#         for key, value in attr_targets.items():
#             if type(value) in [np.ndarray, list]:
#                 # handle array case
#                 return np.array_equal(entity.__getattribute__(key), value), \
#                     "Resulting attribute values different from expected: {}".format(value)
#             else:
#                 # compare entity value and target value
#                 return entity.__getattribute__(key) == value, \
#                     "Resulting attribute value different from expected: {}".format(value)
#     else:
#         # bounded comparison
#         for key, value in attr_targets.items():
#             if type(value) in [np.ndarray, list]:
#                 # handle array case
#                 value = np.array(value)
#                 in_bounds = value <= error_bound
#                 return np.all(in_bounds), \
#                     "Resulting attribute values significantly different from expected: {}".format(value)
#             else:
#                 # compare entity value and target value
#                 return abs(entity.__getattribute__(key) - value) <= error_bound, \
#                     "Resulting attribute value significantly different from expected: {}".format(value)
