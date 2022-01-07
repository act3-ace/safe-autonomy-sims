"""
This module defines fixtures and functions common to the entire test suite.
"""

import yaml


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
                    values.append(test_case[keyword])
                else:
                    # TODO: raise error if invalid keyword
                    values.append(None)
            # add test case to list of cases
            test_cases.append(values)

    return test_cases
