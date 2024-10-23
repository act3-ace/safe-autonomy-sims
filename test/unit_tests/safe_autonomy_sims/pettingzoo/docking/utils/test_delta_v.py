import pytest
import numpy as np
import os
import yaml
from safe_autonomy_sims.pettingzoo.docking.utils import delta_v


# test case parsing (TODO: import from saferl.conftest)
delimiter = ","

def execute_strings(value):
    """
    A recursive helper function to convert string expressions (read from a file) into numerical floats.

    Parameters
    ----------
    value
        A value to be converted or modified based on string format.

    Returns
    -------
    value
        Either the original value or the result of an expression encoded in the string
    """

    if type(value) is str:
        # convert numbers + expressions into floats + execute python library calls
        try:
            value = eval(value)
        except:
            pass
    elif type(value) is dict:
        for k, v in value.items():
            value[k] = execute_strings(v)
    elif type(value) is list:
        for i in range(0, len(value)):
            value[i] = execute_strings(value[i])
    return value

def preprocess_values(value, keyword, functions):
    """
    A recursive helper function to

    Parameters
    ----------
    value
        A value to be converted or modified based on string format.

    Returns
    -------
    value
        Either the original value or the result of an expression encoded in the string
    """

    if type(value) is dict:
        for k, v in value.items():
            # default = defaults[k] if k in defaults else None
            value[k] = preprocess_values(v, k, functions)

    # default processing
    if keyword in functions:
        function_string = functions[keyword].split()[0]  # clean input
        function_string += "({})".format(value)
        try:
            value = eval(function_string)
        except Exception as e:
            print(e)

    return value

def read_test_cases(file_path, parameter_keywords):
    """
    A util function for parameterized tests which searches a YAML test case config file and constructs a list of
    positionally organized test cases.

    Parameters
    ----------
    file_path : str
        The path to the yaml file which defines the values to be used in each test case
    parameter_keywords : list
        The list of keywords to be searched for in the given YAML file, in the position they are to be presented to the
        parameterized test function.

    Returns
    -------
    test_cases : list
        A list of tuples with positionally organized test case values found in the YAML file.
    IDs : list
        A list of strings describing individual test cases
    """

    test_cases = []
    IDs = []
    defaults = {}
    functions = {}

    with open(file_path, 'r') as file:
        file_contents = yaml.safe_load(file)

        if file_contents[0].get("ID") == "defaults":
            # if defaults defined for test assay
            constants = file_contents.pop(0)
            defaults = constants.get("defaults", {})
            functions = constants.get("functions", {})

        for test_case in file_contents:
            # iterate through read test cases
            values = []
            for keyword in parameter_keywords:
                # iteratively search for given keywords
                if keyword in test_case:
                    value = execute_strings(test_case[keyword])
                    value = preprocess_values(value, keyword, functions)
                    values.append(value)
                elif keyword in defaults:
                    values.append(defaults[keyword])
                else:
                    # TODO: raise error if invalid keyword
                    values.append(None)

            # collect test case IDs
            IDs.append(test_case.get("ID", "unnamed test"))

            # add test case to list of cases
            test_cases.append(values)

    return test_cases, IDs

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
    # assert isinstance(v, np.ndarray)
    # assert v.shape == (3,)
    return v

# test delta v function
test_cases_dir = os.path.split(__file__)[0]
test_cases_file_path = os.path.join(test_cases_dir, "delta_v_test_cases.yml")
parameterized_fixture_keywords = ["prev_v", "v", "expected_value"]
test_configs, IDs = read_test_cases(test_cases_file_path, parameterized_fixture_keywords)


@pytest.mark.parametrize(delimiter.join(parameterized_fixture_keywords), test_configs, ids=IDs, indirect=True)
def test_delta_v(prev_v, v, expected_value):
    dv = delta_v(v, prev_v)
    print(dv)
    print(expected_value)
    assert dv == expected_value
