from unittest import mock

import numpy as np
import pytest

from saferl.platforms.cwh.cwh_controllers import CWHController, ThrustController, ThrustControllerValidator
from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl_sim.cwh.cwh import CWHSpacecraft

# how would you test CWHController ?


@pytest.fixture(name='arb_CWHPlatform')
def setup_arb_CWHPlatform():
    platform_name = 'blue0'
    platform_config = {}
    # below should be a saferl_sim.cwh.cwh.CWHSpacecraft
    platform_obj = CWHSpacecraft()
    platform = CWHPlatform(platform_name, platform_obj, platform_config)
    return platform


@pytest.fixture(name='env_config')
def setup_env_config():
    config = ("saferl.platforms.cwh.cwh_controllers.ThrustController", {"name": "X Thrust", "axis": 0})

    return config


"""
mock the cwh_props, into a random prop
mock CWHPlatform into a arbitrary obj for the platform
mock
"""


# currently failing - need appropriate args to constructor
@pytest.mark.unit_test
def test_CWHController_applycontrol():

    #parent_platform = arb_CWHPlatform
    #config = {}
    #control_properties =
    mock_platform = mock.MagicMock()
    mock_config = {'name': 'blue0'}
    mock_control_props = mock.MagicMock()

    obj = CWHController()
    dummy_np_arr = np.array([0., 0., 0.])

    with pytest.raises(NotImplementedError) as excinfo:
        obj.apply_control(dummy_np_arr)


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
"""
# currently failing - need appropriate args to constructor
def test_CWHController_applycontrol(env_config):
    obj = ThrustController()
    dummy_np_arr = np.array([0.,0.,0.])

    with pytest.raises(NotImplementedError) as excinfo:
        obj.apply_control(dummy_np_arr)
"""
