"""
This module defines test for the Dubins3dAircraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.dubins.entities import Dubins3dAircraft


# override entity fixture
@pytest.fixture
def entity(initial_entity_state):
    entity = Dubins3dAircraft(name="test")

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


# Define test assay
test_configs = [
    (
        np.array([0, 0, 0, 0, 0, 0]),                                                                   # initial_state
        1,                                                                                              # num_steps
        {'gamma_rate': 0.1, 'roll_rate': -0.05, 'acceleration': 10},                                    # action
        {'position': [200, 0, 0], 'heading': 0, 'gamma': 0, 'roll': 0, 'v': 10, 'acceleration': 1},     # attr_targets
        0.1                                                                                             # error_bound
     ),
]


@pytest.mark.parametrize("initial_entity_state,action,num_steps,attr_targets,error_bound", test_configs, indirect=True)
def test_Dubins3dAircraft(initial_entity_state, action, num_steps, attr_targets, error_bound):
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
