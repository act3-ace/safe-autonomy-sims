"""
This module defines test for the Dubins2dAircraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.dubins.entities import Dubins2dAircraft


# define test params
num_steps = 1
action = {'heading_rate': 0.1, 'acceleration': 10}
attr_targets = {
    'position': [199.667, 9.992],
    'heading': 0.1,
    'v': 200,
    'acceleration': 10
}

# Define test assay
test_configs = [
    (num_steps, action, attr_targets),
]


@pytest.mark.parametrize("num_steps,action,attr_targets", test_configs, indirect=True)
def test_Dubins2dAircraft(num_steps, action, attr_targets):
    entity = Dubins2dAircraft(name="test")

    for i in range(num_steps):
        entity.step(1, action)

    for key, value in attr_targets.items():
        if type(value) in [np.ndarray, list]:
            # handle array case
            assert np.array_equal(entity.__getattribute__(key), value)
        else:
            # compare entity value and target value
            assert entity.__getattribute__(key) == value
