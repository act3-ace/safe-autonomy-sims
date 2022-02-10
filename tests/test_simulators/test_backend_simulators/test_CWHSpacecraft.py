"""
This module defines tests for the CWHSpacecraft entity.

Author: John McCarroll
"""

import pytest
import os

from saferl_sim.cwh.cwh import CWHSpacecraft
from tests.test_simulators.test_backend_simulators.conftest import evaluate
from tests.conftest import read_test_cases, delimiter


# Define test assay
test_cases_file_path = os.path.join(os.path.split(__file__)[0], "../../test_cases/CWHSpacecraft_test_cases.yaml")
parameterized_fixture_keywords = ["attr_init",
                                  "control",
                                  "num_steps",
                                  "attr_targets",
                                  "error_bound"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


# override entity fixture
@pytest.fixture
def entity(initial_entity_state):
    entity = CWHSpacecraft(name="tests")

    if initial_entity_state is not None:
        entity.state = initial_entity_state

    return entity


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, indirect=True, ids=IDs)
def test_CWHSpacecraft(acted_entity, control, num_steps, attr_targets, error_bound):
    evaluate(acted_entity, attr_targets, error_bound=error_bound)
