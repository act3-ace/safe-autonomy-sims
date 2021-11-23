"""
This module holds unit tests for the MaxDistanceDoneFunction.

Author: John McCarroll
"""

import pytest
import numpy as np
from saferl.dones.docking_dones import MaxDistanceDoneFunction
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.simulators.base_platform import BasePlatform, BaseSensor


class WrappedBasePlatform(BasePlatform):
    def operable(self) -> bool:
        pass


@pytest.fixture
def platform():
    return WrappedBasePlatform("test_platform", "platform", [(BaseSensor, {})])


@pytest.fixture
def agent_name():
    return "safety_buddy"


@pytest.fixture
def cut(agent_name):
    return MaxDistanceDoneFunction(name='cut', max_distance=500, agent_name=agent_name)


@pytest.fixture
def observation():
    return np.array([0, 0, 0])


@pytest.fixture
def action():
    return np.array([0, 0, 0])


@pytest.fixture
def next_observation():
    return np.array([0, 0, 0])


@pytest.fixture
def next_state(BasePlatform):
    state_dict = {
        "sim_platforms": [BasePlatform]
    }
    state = StateDict(state_dict)
    return state


@pytest.fixture
def call_results(cut, observation, action, next_observation, next_state):
    results = cut(observation, action, next_observation, next_state)
    return results


@pytest.fixture
def expected_value():
    return False


@pytest.mark.unit_test
def test_call(call_results, agent_name, expected_value):
    assert call_results[agent_name] == expected_value

# I'm assuming core is friendly
# need to mock Platform or state to avoid constructor complications