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

Author: Jamie Cunningham
"""
import os

import pytest
from corl.libraries.state_dict import StateDict

from saferl.core.simulators.cwh_simulator import CWHSimulator
from tests.conftest import delimiter, read_test_cases
from tests.factories.cwh.cwh_platform import CWHPlatformFactory

# Define test assay
test_cases_dir = os.path.join(os.path.split(__file__)[0], "../../../../test_cases/simulators/docking/CWHSimulator_test_cases/")


@pytest.fixture(name='step_size')
def fixture_step_size(request):
    """Returns simulator step size for test"""
    return request.param


@pytest.fixture(name='agent_configs')
def fixture_agent_configs():
    """Returns valid agent configuration for testing"""
    configs = {
        "blue0": {
            "sim_config": {
                "name": 'CWH'
            }, "platform_config": CWHPlatformFactory.platform_config
        },
    }
    return configs


@pytest.fixture(name='reset_config')
def fixture_reset_config(request):
    """Returns reset configuration from test assay"""
    return request.param


@pytest.fixture(name='expected_state')
def fixture_expected_state(expected_sim_platforms):
    """Returns valid expected state dict build from test assay settings"""
    state = StateDict({
        "sim_platforms": expected_sim_platforms,
    })
    return state


@pytest.fixture(name='expected_sim_entities')
def fixture_expected_sim_entities(expected_sim_platforms):
    """Returns dict of valid sim entities built from test assay platforms"""
    entities = {plat.name: plat._platform for plat in expected_sim_platforms}  # pylint: disable=W0212
    return entities


@pytest.fixture(name='expected_platform_configs')
def fixture_expected_platform_configs(request):
    """Returns platform configs for expected platforms in test assay"""
    return request.param


@pytest.fixture(name='expected_sim_platforms')
def fixture_expected_sim_platforms(expected_platform_configs):
    """Returns iterable of expected platforms built from test assay platform configs"""
    platforms = tuple([CWHPlatformFactory(**(expected_platform_configs[0]))])
    return platforms


@pytest.fixture(name='cut')
def fixture_cut(step_size, agent_configs):
    """Returns an initialized CWHSimulator"""
    return CWHSimulator(step_size=step_size, agent_configs=agent_configs)


test_cases_file_path = os.path.join(test_cases_dir, "reset_test_cases.yaml")
parameterized_fixture_keywords = ["reset_config", "expected_platform_configs", "step_size"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_reset(cut, reset_config, expected_state):
    """Tests the reset method of the CWHSimulator"""
    state = cut.reset(reset_config)
    assert state == expected_state


test_cases_file_path = os.path.join(test_cases_dir, "construct_sim_entities_test_cases.yaml")
parameterized_fixture_keywords = ["expected_platform_configs", "step_size"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_construct_sim_entities(cut, expected_sim_entities):
    """Tests the construct_sim_entities method of the CWHSimulator"""
    sim_entities = cut.construct_sim_entities()
    assert sim_entities == expected_sim_entities


test_cases_file_path = os.path.join(test_cases_dir, "construct_platforms_test_cases.yaml")
parameterized_fixture_keywords = ["expected_platform_configs", "step_size"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_construct_platforms(cut, expected_sim_platforms):
    """Tests the construct_platforms method of the CWHSimulator"""
    sim_platforms = cut.construct_platforms()
    assert sim_platforms == expected_sim_platforms


test_cases_file_path = os.path.join(test_cases_dir, "step_test_cases.yaml")
parameterized_fixture_keywords = ["reset_config", "expected_platform_configs", "step_size"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.mark.unit_test
@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_step(cut, reset_config, expected_state):
    """Tests the step method of the CWHSimulator"""
    cut.reset(reset_config)
    state = cut.step()
    assert state == expected_state
