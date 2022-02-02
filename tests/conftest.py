"""
This module defines fixtures, functions, and constants common to the entire test suite.
Python packages used in test case configs must be imported to this module for error free value loading.

Author: John McCarroll
"""

import yaml
import numpy as np
from scipy.spatial.transform import Rotation
from act3_rl_core.dones.done_func_base import DoneStatusCodes


# Define constants
delimiter = ","


# Define functions
def wrap_angle(angle, max=np.pi):
    """
    Wraps angle between -pi and pi by default.

    Parameters
    ----------
    angle: float
        Given angle in radians
    max: float
        The max positive angle allowed

    Returns
    -------
        The wrapped angle.
    """
    if abs(angle) > max:
        shifted_angle = angle + max
        wrapped_angle = shifted_angle % (2 * max)
        angle = wrapped_angle - max if wrapped_angle != 0.0 else wrapped_angle
    elif angle == -max:
        # handle -pi discontinuity
        angle *= -1
    return angle


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
    """

    test_cases = []
    IDs = []
    defaults = {}
    functions = {}

    with open(file_path, 'r') as file:
        file_contents = yaml.safe_load(file)

        if file_contents[0]["ID"] == "defaults":
            # if defaults defined for test assay
            constants = file_contents.pop(0)
            defaults = constants["defaults"]
            functions = constants["functions"]

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
            if "ID" in test_case:
                IDs.append(test_case["ID"])
            else:
                IDs.append("unnamed test")                # TODO: how to skip an ID?

            # add test case to list of cases
            test_cases.append(values)

    return test_cases, IDs


print(read_test_cases("test_cases/Dubins3dAircraft_test_cases.yaml", ['attr_targets']))


# ["attr_init",
# "control",
# "num_steps",
# "attr_targets",
# "error_bound",
# "proportional_error_bound"]
