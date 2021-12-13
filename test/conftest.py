"""
This module holds fixtures common to all simulator tests.

Author: John McCarroll
"""


import pytest


@pytest.fixture
def num_steps(request):
    return request.param


@pytest.fixture
def action(request):
    return request.param


@pytest.fixture
def attr_targets(request):
    return request.param
