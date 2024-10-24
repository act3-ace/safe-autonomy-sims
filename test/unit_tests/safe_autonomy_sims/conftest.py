import pytest
import numpy as np
import yaml


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
