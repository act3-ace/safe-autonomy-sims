"""
This module defines tests for the Dubins2dAircraft entity.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl_sim.dubins.entities import Dubins2dAircraft
from tests.test_simulators.test_backend_simulators.conftest import evaluate


# override entity fixture
@pytest.fixture
def entity(initial_entity_state):
    entity = Dubins2dAircraft(name="tests")

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


# Define tests assay
test_configs = [
    (
        np.array([0, 0, 0, 0, 0, 0]),                                                       # initial_state
        1,                                                                                  # num_steps
        {'heading_rate': 0.1, 'acceleration': 10},                                          # action
        {'position': [0, 0, 0], 'heading': 0.1, 'v': 0, 'acceleration': [0, 0, 0]},         # attr_targets
        0.1                                                                                 # error_bound
    ),
]


@pytest.mark.parametrize("initial_entity_state,action,num_steps,attr_targets,error_bound", test_configs, indirect=True)
def test_Dubins2dAircraft(entity, action, num_steps, attr_targets, error_bound):
    evaluate(entity, attr_targets, error_bound=error_bound)
