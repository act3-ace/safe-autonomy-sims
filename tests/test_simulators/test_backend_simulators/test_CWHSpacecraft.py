"""
This module defines tests for the CWHSpacecraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.cwh.cwh import CWHSpacecraft
from tests.test_simulators.test_backend_simulators.conftest import evaluate


# override entity fixture
@pytest.fixture
def entity(initial_entity_state):
    entity = CWHSpacecraft(name="tests")

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


# Define tests assay
test_configs = [
    (
        np.array([0, 0, 0, 0, 0, 0]),                           # initial_entity_state
        {'thrust_x': 0.5, 'thrust_y': 0.75, 'thrust_z': 1},     # action
        5,                                                      # num_steps
        {'state': np.array([0, 0, 0, 0, 0, 0])},                # attr_targets
        0.1                                                     # error_bound
    ),
]


@pytest.mark.parametrize("initial_entity_state,action,num_steps,attr_targets,error_bound", test_configs, indirect=True)
def test_CWHSpacecraft(entity, action, num_steps, attr_targets, error_bound):
    evaluate(entity, attr_targets, error_bound=error_bound)
