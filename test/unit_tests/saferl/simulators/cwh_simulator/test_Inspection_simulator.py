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
from safe_autonomy_simulation.sims.spacecraft import SixDOFSpacecraft


# Define test assay
test_cases_dir = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/simulators/docking/CWHSimulator_test_cases/")
test_cases_file_path = os.path.join(test_cases_dir, "inspection_points_test_cases.yaml")
parameterized_fixture_keywords = ["position", "orientation", "expected_positions"]
parameterized_fixture_keywords_insp = ["position", "orientation", "inspector_position", "inspector_orientation", "expected_inspection"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)
test_configs_insp, IDs_insp = read_test_cases(test_cases_file_path, parameterized_fixture_keywords_insp)


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
def inspector_position(request):
    return request.param


@pytest.fixture
def inspector_orientation(request):
    return request.param


@pytest.fixture
def expected_inspection(request):
    return request.param


@pytest.fixture
def parent_entity(position, orientation):
    """Create a fixture for the parent_entity class."""
    mock_entity = Mock()
    mock_entity.position = position
    mock_entity.orientation = orientation
    return mock_entity


@pytest.fixture
def inspector_entity(inspector_position, inspector_orientation):
    """Create a fixture for the parent_entity class."""
    mock_entity = SixDOFSpacecraft(
        name='blue0',
        position=np.array(inspector_position),
        orientation=np.array(inspector_orientation),
    )
    return mock_entity


@pytest.fixture
def inspection_points_config():
    """Create a fixture for the parent_entity class."""
    return {
        "num_points": 100,
        "radius": 10,
        "priority_vector": np.zeros(3),
        "sensor_fov": 1.0471975511965976,  # 60 degrees
        "illumination_params": {
            "mean_motion" : 0.001027,
            "avg_rad_Earth2Sun": 150000000000,
            "light_properties" : {'ambient': [1, 1, 1], 'diffuse': [1, 1, 1], 'specular': [1, 1, 1]},
            "chief_properties" : {'ambient': [.1, 0, 0], 'diffuse': [0.7, 0, 0], 'specular': [1, 1, 1], 'shininess': 100, 'reflection': 0.5}, # [.1, 0, 0] = red, [0.753, 0.753, 0.753] = silver
            "resolution" : [200, 200],
            "focal_length" : 9.6e-3,
            "pixel_pitch" : 5.6e-3,
            "bin_ray_flag": True,
            "render_flag_3d": False,
            "render_flag_subplots": False,
        },
    }


@pytest.fixture
def inspection_points(parent_entity, inspection_points_config):
    """Create a fixture for the InspectionPoints class."""
    return InspectionPoints(parent_entity, **inspection_points_config)

@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_inspection_point_movement(inspection_points, expected_positions):
    """Test that the InspectionPoints class can translate inspection points accurately."""
    inspection_points.update_points_position()

    points = inspection_points.points_position_dict

    for id, position in points.items():
        expected_position = expected_positions[id]
        assert np.array_equal(position, expected_position), f"Resulting position does not equal expected position for PointID {id}: {position} != {expected_position}"

@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords_insp), test_configs_insp, ids=IDs_insp, indirect=True)
def test_inspection_point_status(inspection_points, inspector_entity, expected_inspection):
    """Test that the InspectionPoints class can translate inspection points accurately."""
    inspection_points.update_points_inspection_status(inspector_entity)

    points = inspection_points.points_inspected_dict

    for id, inspected in points.items():
        expected = expected_inspection[id]
        assert expected==inspected, f"Resulting inspected point does not equal expected inspected point for PointID {id}: {expected} != {inspected}"
