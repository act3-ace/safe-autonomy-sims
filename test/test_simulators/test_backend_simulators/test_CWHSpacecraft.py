"""
This module defines test for the CWHSpacecraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.cwh.cwh import CWHSpacecraft


# override entity fixture
@pytest.fixture
def entity(initial_entity_state):
    entity = CWHSpacecraft(name="test")

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


# Define test assay
test_configs = [
    (
        np.array([0, 0, 0, 0, 0, 0]),                           # initial_entity_state
        {'thrust_x': 0.5, 'thrust_y': 0.75, 'thrust_z': 1},     # action
        5,                                                      # num_steps
        {'state': np.array([0, 0, 0])},                         # attr_targets
        0.1                                                     # error_bound
    ),
]


@pytest.mark.parametrize("initial_entity_state,action,num_steps,attr_targets,error_bound", test_configs, indirect=True)
def test_CWHSpacecraft(initial_entity_state, action, num_steps, attr_targets, error_bound):
    # evaluation
    if error_bound is None:
        # direct comparison
        for key, value in attr_targets.items():
            if type(value) in [np.ndarray, list]:
                # handle array case
                assert np.array_equal(entity.__getattribute__(key), value), \
                    "Resulting attribute values different from expected: {}".format(value)
            else:
                # compare entity value and target value
                assert entity.__getattribute__(key) == value, \
                    "Resulting attribute value different from expected: {}".format(value)
    else:
        # bounded comparison
        for key, value in attr_targets.items():
            if type(value) in [np.ndarray, list]:
                # handle array case
                value = np.array(value)
                in_bounds = value <= error_bound
                assert np.all(in_bounds), \
                    "Resulting attribute values significantly different from expected: {}".format(value)
            else:
                # compare entity value and target value
                assert abs(entity.__getattribute__(key) - value) <= error_bound, \
                    "Resulting attribute value significantly different from expected: {}".format(value)
