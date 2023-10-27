"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------

This module defines tests for the CWHSimulator class.

Author: John McCarroll
"""

import pytest
import numpy as np
import os
from unittest.mock import Mock
from test.conftest import delimiter, read_test_cases
from safe_autonomy_sims.simulators.inspection_simulator import InspectionPoints


# Define test assay
test_cases_dir = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/simulators/docking/CWHSimulator_test_cases/")
test_cases_file_path = os.path.join(test_cases_dir, "inspection_points_test_cases.yaml")
parameterized_fixture_keywords = ["position", "orientation", "expected_positions"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture
def expected_positions(request):
    return request.param


@pytest.fixture
def position(request):
    return request.param


@pytest.fixture
def orientation(request):
    return request.param


@pytest.fixture
def parent_entity(position, orientation):
    """Create a fixture for the parent_entity class."""
    mock_entity = Mock()
    mock_entity.position = position
    mock_entity.orientation = orientation
    return mock_entity


@pytest.fixture
def inspection_points_config():
    """Create a fixture for the parent_entity class."""
    return {
        "num_points": 100,
        "radius": 10,
    }


@pytest.fixture
def inspection_points(parent_entity, inspection_points_config):
    """Create a fixture for the InspectionPoints class."""
    return InspectionPoints(parent_entity, **inspection_points_config)

@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_inspection_point_movement(inspection_points, expected_positions):
    """Test that the InspectionPoints class can translate inspection points accurately."""
    inspection_points.update_points_position()

    points = inspection_points.points_position_dict

    for id, position in points.items():
        expected_position = expected_positions[id]
        assert np.array_equal(position, expected_position), f"Resulting position does not equal expected position for PointID {id}: {position} != {expected_position}"

