"""
This module holds fixtures common to simulator tests.

Author: John McCarroll
"""

import pytest


@pytest.fixture
def entity_config(request):
    return request.param
