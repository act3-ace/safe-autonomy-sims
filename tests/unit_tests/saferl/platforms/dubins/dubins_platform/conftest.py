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
