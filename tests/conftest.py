"""
This module defines fixtures and functions common to the entire test suite.
Python packages used in test case configs must be imported to this module for error free value loading.

Author: John McCarroll
"""

import yaml
import numpy as np

from scipy.spatial.transform import Rotation
from act3_rl_core.dones.done_func_base import DoneStatusCodes


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
        # convert numbers + expressions into floats + instantiate python library calls
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
    """

    test_cases = []
    with open(file_path, 'r') as file:
        file_contents = yaml.safe_load(file)
        for test_case in file_contents:
            # iterate through read test cases
            values = []
            for keyword in parameter_keywords:
                # iteratively search for given keywords
                if keyword in test_case:
                    value = execute_strings(test_case[keyword])
                    values.append(value)
                else:
                    # TODO: raise error if invalid keyword
                    values.append(None)
            # add test case to list of cases
            test_cases.append(values)

    return test_cases
