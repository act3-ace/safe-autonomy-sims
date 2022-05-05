"""
This module holds fixtures common to simulator tests.

Author: John McCarroll
"""

import pytest


@pytest.fixture
def entity_config(request):
    """
    Get configuration for simulator entity from test case.

    Parameters
    ----------
    request

    Returns
    -------


    """
    return request.param
