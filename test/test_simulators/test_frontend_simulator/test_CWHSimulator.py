"""
This module defines tests for the CWHSimulator class.

Author: John McCarroll
"""

import pytest
import numpy as np

from saferl.simulators.cwh_simulator import CWHSimulator


@pytest.fixture
def tmp_config(entity_config):
    tmp_config = {
        "step_size": 1,
        "agent_configs": entity_config,
    }
    return tmp_config


# define test params
entity_config = {
    "blue0": {
        "sim_config": {},
        "platform_config": [
            ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                "name": "X Thrust", "axis": 0
            }),
            ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                "name": "Y Thrust", "axis": 1
            }),
            ("saferl.platforms.cwh.cwh_controllers.ThrustController", {
                "name": "Z Thrust", "axis": 2
            }),
            ("saferl.platforms.cwh.cwh_sensors.PositionSensor", {}),
            ("saferl.platforms.cwh.cwh_sensors.VelocitySensor", {}),
        ],
    }
}
num_steps = 5
action = [1, 2, 3]
attr_targets = {
    'x': 0,
    'y': 1,
    'state': np.array([1, 2, 3])
}

# Define test assay
test_configs = [
    (5, entity_config, 1, attr_targets),
]


@pytest.mark.parametrize("num_steps,entity_config,action,attr_targets", test_configs, indirect=True)
def test_CWHSimulator(tmp_config, num_steps, action, attr_targets):

    reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 2], "velocity": [0, 0, 0]}}}

    tmp = CWHSimulator(**tmp_config)
    state = tmp.reset(reset_config)

    for i in range(num_steps):
        state.sim_platforms[0]._controllers[0].apply_control(action[0])
        state.sim_platforms[0]._controllers[1].apply_control(action[1])
        state.sim_platforms[0]._controllers[2].apply_control(action[2])
        state = tmp.step()

    for key, value in attr_targets.items():
        assert state.sim_platforms[0].__getattribute__(key) == value
