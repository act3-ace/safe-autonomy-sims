"""
This module defines tests for the CWHSpacecraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.cwh.cwh import CWHSpacecraft


# define test params
num_steps = 5
action = {'thrust_x': 0.5, 'thrust_y': 0.75, 'thrust_z': 1}
attr_targets = {
    'state': np.array([1, 2, 3])
}

# Define test assay
test_configs = [
    (num_steps, action, attr_targets),
]


@pytest.mark.parametrize("num_steps,action,attr_targets", test_configs, indirect=True)
def test_CWHSpacecraft(num_steps, action, attr_targets):
    entity = CWHSpacecraft(name="test")

    for i in range(num_steps):
        entity.step(1, action)

    for key, value in attr_targets.items():
        if type(value) in [np.ndarray, list]:
            # handle array case
            assert np.array_equal(entity.__getattribute__(key), value)
        else:
            # compare entity value and target value
            assert entity.__getattribute__(key) == value
