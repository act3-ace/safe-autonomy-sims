from unittest import mock
from unittest.mock import MagicMock

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
Test the following : CWHController, ThrustControllerValidator, ThrustController
"""


"""
Tests for CWHController

parent_platform= <saferl.platforms.cwh.cwh_platform.CWHPlatform object at 0x7f5b6d1e0a90>
config= {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}
control_properties= <class 'saferl.platforms.cwh.cwh_properties.ThrustProp'>
"""
"""
Tests for ThrustController
"""


@pytest.fixture(name='cwh_spacecraft')
def setup_CWHSpacecraft():
    add_args = {'name':'CWH'}
    spcft = CWHSpacecraft(**add_args)
    return spcft

@pytest.fixture(name='cwh_platform')
def setup_cwhplatform(cwh_spacecraft):

    platform_name = 'blue0'
    platform_config = []

    platform_obj = CWHPlatform(platform_name,cwh_spacecraft,platform_config)
    return platform_obj

configs = [
            ({'name': 'X Thrust', "axis": 0, 'properties': {'name': 'x_thrust'}}),
            ({"name": "Y Thrust", "axis": 1,'properties': {'name': "y_thrust"}}),
            ({"name": "Z Thrust", "axis": 2, 'properties': {'name': "z_thrust"}}),
          ]

@pytest.fixture(name='config')
def get_config(request):
    return request.param


@pytest.fixture(name='thrust_controller')
def setup_thrustcontroller(cwh_platform,config):
    controller = ThrustController(cwh_platform,config)
    return controller


@pytest.mark.unit_test
@pytest.mark.parametrize('config',configs,indirect=True)
def test_apply_control(thrust_controller,config):
    action = 10.
    thrust_controller.apply_control(action)
    assert thrust_controller.parent_platform._last_applied_action[config['axis']] == action

@pytest.mark.unit_test
@pytest.mark.parametrize('config',configs,indirect=True)
def test_get_applied_action(thrust_controller,config):
    dummy_arr = np.array([10.,20.,30.])
    thrust_controller.parent_platform._last_applied_action = dummy_arr

    assert thrust_controller.get_applied_control() == thrust_controller.parent_platform._last_applied_action[config['axis']]
