"""
# -------------------------------------------------------------------------------
# Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
# Reinforcement Learning Core (CoRL) Runtime Assurance Extensions
#
# This is a US Government Work not subject to copyright protection in the US.
#
# The use, dissemination or disclosure of data in this file is subject to
# limitation or restriction. See accompanying README and LICENSE for details.
# -------------------------------------------------------------------------------

This module holds fixtures common to all simulator tests.

Author: John McCarroll
"""


import pytest


@pytest.fixture
def num_steps(request):
    return request.param


@pytest.fixture
def control(request):
    return request.param


@pytest.fixture
def attr_targets(request):
    return request.param


@pytest.fixture
def error_bound(request):
    return request.param


@pytest.fixture
def proportional_error_bound(request):
    return request.param
