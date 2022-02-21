"""
Tests for the CWHControllers module of the cwh platform.
"""

import os
from unittest import mock

import numpy as np
import pytest

from saferl.platforms.cwh.cwh_controllers import CWHController, ThrustController
from tests.conftest import read_test_cases

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../test_cases/cwh_platform_test_cases/thrust_controller_test_cases.yaml"
)
parameterized_fixture_keywords = ["config"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


# CWHController Interface Tests
@pytest.mark.unit_test
def test_CWHController_name():
    """
    Of the CWHController interface test the name attribute
    """

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
    """
    Test CWHController interface method - applycontrol
    """

    mock_platform = mock.MagicMock()
    mock_config = {'name': 'blue0'}
    mock_control_props = mock.MagicMock()

    obj = CWHController(mock_platform, mock_config, mock_control_props)
    dummy_np_arr = np.array([0., 0., 0.])

    with pytest.raises(NotImplementedError):
        obj.apply_control(dummy_np_arr)


@pytest.mark.unit_test
def test_CWHController_get_applied_control():
    """
    Test the CWHController interface - get_applied_control method
    """

    mock_platform = mock.MagicMock()
    mock_config = {'name': 'blue0'}
    mock_control_props = mock.MagicMock()

    obj = CWHController(mock_platform, mock_config, mock_control_props)
    dummy_np_arr = np.array([0., 0., 0.])

    with pytest.raises(NotImplementedError):
        obj.get_applied_control()


@pytest.fixture(name='config')
def get_config(request):
    """
    retrieve the parameter 'config' from a list of parameters
    """
    return request.param


@pytest.fixture(name='thrust_controller')
def setup_thrustcontroller(cwh_platform, config):
    """
    A method to create a ThrustController.
    """
    # TODO: remove list wrap issue
    if isinstance(config, list):
        config = config[0]

    controller = ThrustController(cwh_platform, config)
    return controller


@pytest.mark.unit_test
@pytest.mark.parametrize('config', test_configs, indirect=True, ids=IDs)
def test_apply_control(thrust_controller, config):
    """
    Parametrized test for the apply_control method of ThrustController
    """
    # TODO: remove list wrap issue
    if isinstance(config, list):
        config = config[0]

    action = 10.
    thrust_controller.apply_control(action)
    assert thrust_controller.parent_platform._last_applied_action[config['axis']] == action  # pylint: disable=W0212


@pytest.mark.unit_test
@pytest.mark.parametrize('config', test_configs, indirect=True, ids=IDs)
def test_get_applied_action(thrust_controller, config):
    """
    Parametrized test for the get_applied_control method of ThrustController
    """
    # TODO: remove list wrap issue
    if isinstance(config, list):
        config = config[0]

    dummy_arr = np.array([10., 20., 30.])
    thrust_controller.parent_platform._last_applied_action = dummy_arr  # pylint: disable=W0212

    assert thrust_controller.get_applied_control() == thrust_controller.parent_platform._last_applied_action[config['axis']]  # pylint: disable=W0212
