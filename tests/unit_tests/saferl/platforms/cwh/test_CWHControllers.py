from unittest import mock

import numpy as np
import pytest

from saferl.platforms.cwh.cwh_controllers import CWHController, ThrustController, ThrustControllerValidator
from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl_sim.cwh.cwh import CWHSpacecraft
"""
Tests for the CHWController Interface
"""


@pytest.mark.unit_test
def test_CWHController_name():

    mock_platform = mock.MagicMock()
    mock_config = {'name': 'blue0'}
    mock_control_props = mock.MagicMock()

    obj = CWHController(mock_platform, mock_config, mock_control_props)
    dummy_np_arr = np.array([0., 0., 0.])

    name = obj.name
    expected_name = 'blue0CWHController'
    assert name == expected_name


@pytest.mark.unit_test
def test_CWHController_applycontrol():

    mock_platform = mock.MagicMock()
    mock_config = {'name': 'blue0'}
    mock_control_props = mock.MagicMock()

    obj = CWHController(mock_platform, mock_config, mock_control_props)
    dummy_np_arr = np.array([0., 0., 0.])

    with pytest.raises(NotImplementedError) as excinfo:
        obj.apply_control(dummy_np_arr)


@pytest.mark.unit_test
def test_CWHController_get_applied_control():

    mock_platform = mock.MagicMock()
    mock_config = {'name': 'blue0'}
    mock_control_props = mock.MagicMock()

    obj = CWHController(mock_platform, mock_config, mock_control_props)
    dummy_np_arr = np.array([0., 0., 0.])

    with pytest.raises(NotImplementedError) as excinfo:
        val = obj.get_applied_control()


"""
Tests for ThrustControllerValidator
"""
"""
Test the following : CWHController, ThrustControllerValidator, ThrustController

A little unsure of how to test CWHController
"""
"""
Tests for CWHController

parent_platform= <saferl.platforms.cwh.cwh_platform.CWHPlatform object at 0x7f5b6d1e0a90>
config= {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}
control_properties= <class 'saferl.platforms.cwh.cwh_properties.ThrustProp'>



"""
