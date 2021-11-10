"""
This module defines test for the Dubins2dAircraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.dubins.entities import Dubins2dAircraft


# override entity fixture
@pytest.fixture
def entity(initial_entity_state):
    entity = Dubins2dAircraft(name="test")

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


# Define test assay
test_configs = [
    (
        1,                                                                                  # num_steps
        {'heading_rate': 0.1, 'acceleration': 10},                                          # action
        {'position': [199.667, 9.992], 'heading': 0.1, 'v': 200, 'acceleration': 10}        # attr_targets                                                        # attr_targets
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
