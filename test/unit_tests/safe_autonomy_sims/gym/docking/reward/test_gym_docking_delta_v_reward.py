import pytest
import os
from safe_autonomy_sims.gym.docking.reward import delta_v_reward
from test.conftest import delimiter, read_test_cases


@pytest.fixture(name='prev_state')
def fixture_prev_state(request):
    prev_state = request.param
    assert isinstance(prev_state, dict)
    assert 'deputy' in prev_state
    assert prev_state['deputy'].shape == (6,)
    return prev_state

@pytest.fixture(name='state')
def fixture_state(request):
    state = request.param
    assert isinstance(state, dict)
    assert 'deputy' in state
    assert state['deputy'].shape == (6,)
    return state

@pytest.fixture(name='expected_value')
def fixture_expected_value(request):
    v = request.param
    return v

# test delta v function
test_cases_dir = os.path.split(__file__)[0]
test_cases_file_path = os.path.join(test_cases_dir, "delta_v_reward_test_cases.yml")
parameterized_fixture_keywords = ["prev_state", "state", "expected_value"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)

@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_delta_v_reward(prev_state, state, expected_value):
    dv = delta_v_reward(state, prev_state)
    assert dv == expected_value
