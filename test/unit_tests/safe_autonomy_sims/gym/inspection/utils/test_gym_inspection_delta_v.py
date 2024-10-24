import pytest
import numpy as np
import os
from safe_autonomy_sims.gym.inspection.utils import delta_v
from test.unit_tests.safe_autonomy_sims.conftest import delimiter, read_test_cases


@pytest.fixture(name='prev_v')
def fixture_prev_v(request):
    prev_v = request.param
    assert isinstance(prev_v, np.ndarray)
    assert prev_v.shape == (3,)
    return prev_v

@pytest.fixture(name='v')
def fixture_v(request):
    v = request.param
    assert isinstance(v, np.ndarray)
    assert v.shape == (3,)
    return v

@pytest.fixture(name='expected_value')
def fixture_expected_value(request):
    v = request.param
    return v

# test delta v function
test_cases_dir = os.path.split(__file__)[0]
test_cases_file_path = os.path.join(test_cases_dir, "delta_v_test_cases.yml")
parameterized_fixture_keywords = ["prev_v", "v", "expected_value"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_delta_v(prev_v, v, expected_value):
    dv = delta_v(v, prev_v)
    assert dv == expected_value
