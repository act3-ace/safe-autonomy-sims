"""
This module defines fixtures common to dubins controller unit tests

Author: John McCarroll
"""

from unittest import mock

import pytest


@pytest.fixture(name="platform_name")
def get_platform_name():
    """
    Returns string of platform's name
    """
    return "cut"


@pytest.fixture(name="platform")
def get_platform():
    """
    Returns a mock platform object
    """
    return mock.MagicMock(name="platform")
