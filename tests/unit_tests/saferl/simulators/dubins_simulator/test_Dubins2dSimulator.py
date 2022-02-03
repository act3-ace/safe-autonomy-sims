"""
This module defines tests for the Dubins2dSimulator class.

Author: John McCarroll
"""
import os
from collections import OrderedDict, defaultdict, deque

import numpy as np
import pytest
from act3_rl_core.libraries.state_dict import StateDict

from saferl.platforms.dubins.dubins_platform import Dubins2dPlatform
from saferl.simulators.dubins_simulator import Dubins2dSimulator
from saferl_sim.dubins.entities import Dubins2dAircraft
from tests.conftest import delimiter, read_test_cases
from tests.factories.dubins.dubins_platform import Dubins2dPlatformFactory

# Define test assay
test_cases_file_path = os.path.join(
    os.path.split(__file__)[0], "../../../../test_cases/DockingRelativeVelocityConstraintDoneFunction_test_cases.yaml"
)
parameterized_fixture_keywords = ["platform_velocity", "target_velocity", "constraint_velocity", "expected_value", "expected_status"]
test_configs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.fixture
def tmp_config(entity_config):
    tmp_config = {
        "step_size": 1,
        "agent_configs": entity_config,
    }
    return tmp_config


@pytest.fixture(name='expected_state')
def fixture_expected_state(expected_sim_platforms):
    state = StateDict(
        {
            "sim_platforms": expected_sim_platforms,
            "episode_history": defaultdict(),
            "episode_state": OrderedDict(),
            "step_state": OrderedDict()
        }
    )
    return state


@pytest.fixture(name='expected_sim_entities')
def fixture_expected_sim_entities(expected_sim_platforms):
    entities = {plat.name: plat._platform for plat in expected_sim_platforms}
    return entities


@pytest.fixture(name='expected_sim_platforms')
def fixture_expected_sim_platforms(platform_configs):
    platforms = (Dubins2dPlatformFactory(**plat_cfg) for plat_cfg in platform_configs)
    return platforms


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
