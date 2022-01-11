import pytest
from unittest import mock
import numpy as np
from saferl.platforms.cwh.cwh_controllers import CWHController, ThrustControllerValidator, ThrustController

# how would you test CWHController ?
"""
Fixtures for parameter setup
"""
@pytest.fixture(name='env_config')
def setup_env_config():

    config = ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                        "name": "X Thrust", "axis": 0 })

    return config


"""
Test the following : CWHController, ThrustControllerValidator, ThrustController

A little unsure of how to test CWHController
"""

"""
Tests for CWHController
"""


# currently failing - need appropriate args to constructor
def test_CWHController_applycontrol(env_config):
    obj = CWHController()
    dummy_np_arr = np.array([0.,0.,0.])

    with pytest.raises(NotImplementedError) as excinfo:
        obj.apply_control(dummy_np_arr)
