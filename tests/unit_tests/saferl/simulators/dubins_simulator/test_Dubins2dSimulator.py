"""
This module defines tests for the Dubins2dSimulator class.

Author: John McCarroll
"""

import numpy as np
import pytest

from saferl.simulators.dubins_simulator import Dubins2dSimulator


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
            # ("saferl.platforms.dubins.dubins_controllers.CombinedTurnRateAccelerationController", {
            #     "name": "YawAccControl"
            # }),
            ("saferl.platforms.dubins.dubins_controllers.YawRateController", {
                "name": "YawRateControl", "axis": 0
            }),
            ("saferl.platforms.dubins.dubins_controllers.AccelerationController", {
                "name": "AccelerationControl", "axis": 1
            }),
            ("saferl.platforms.dubins.dubins_sensors.PositionSensor", {}),
            ("saferl.platforms.dubins.dubins_sensors.VelocitySensor", {}),
            ("saferl.platforms.dubins.dubins_sensors.HeadingSensor", {}),
        ],
    }
}
num_steps = 5
action = [1, 2, 3]
attr_targets = {'x': 0, 'y': 1, 'state': np.array([1, 2, 3])}

# Define test assay
test_configs = [
    (5, entity_config, 1, attr_targets),
]


@pytest.mark.parametrize("num_steps,entity_config,action,attr_targets", test_configs, indirect=True)
def test_Dubins2dSimulator(tmp_config, num_steps, action, attr_targets):

    reset_config = {"agent_initialization": {"blue0": {"position": [0, 1, 2], "velocity": [0, 0, 0]}}}

    tmp = Dubins2dSimulator(**tmp_config)
    state = tmp.reset(reset_config)

    for i in range(num_steps):
        state.sim_platforms[0]._controllers[0].apply_control(action[0])
        state.sim_platforms[0]._controllers[1].apply_control(action[1])
        state.sim_platforms[0]._controllers[2].apply_control(action[2])

        state = tmp.step()

    for key, value in attr_targets:
        assert state.sim_platforms[0].getattr(key) == value


@pytest.fixture(name='cut')
def fixture_cut(step_size, agent_configs):
    return Dubins2dSimulator(step_size=step_size, agent_configs=agent_configs)


@pytest.mark.unit_test
def test_reset(cut, reset_config, expected_state):
    state = cut.reset(reset_config)
    assert state == expected_state


@pytest.mark.unit_test
def test_construct_sim_entities(cut, expected_sim_entities):
    sim_entities = cut.construct_sim_entities()
    assert sim_entities == expected_sim_entities


@pytest.mark.unit_test
def test_construct_platforms(cut, expected_sim_platforms):
    sim_platforms = cut.construct_platforms()
    assert sim_platforms == expected_sim_platforms


@pytest.mark.unit_test
def test_step(cut, expected_state):
    state = cut.step()
    assert state == expected_state
