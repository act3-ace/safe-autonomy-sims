"""
This module defines test for the Dubins3dAircraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.dubins.entities import Dubins3dAircraft


# define test params
num_steps = 1
action = {'gamma_rate': 0.1, 'roll_rate': -0.05, 'acceleration': 10}
attr_targets = {
    'position': [200, 0, 0],
    'heading': 0,
    'gamma': 0,
    'roll': 0,
    'v': 10,
    'acceleration': 1
}

# Define test assay
test_configs = [
    (num_steps, action, attr_targets),
]


@pytest.mark.parametrize("num_steps,action,attr_targets", test_configs, indirect=True)
def test_Dubins3dAircraft(num_steps, action, attr_targets):
    entity = Dubins3dAircraft(name="test")

    for i in range(num_steps):
        entity.step(1, action)

    for key, value in attr_targets.items():
        if type(value) in [np.ndarray, list]:
            # handle array case
            assert np.array_equal(entity.__getattribute__(key), value)
        else:
            # compare entity value and target value
            assert entity.__getattribute__(key) == value
