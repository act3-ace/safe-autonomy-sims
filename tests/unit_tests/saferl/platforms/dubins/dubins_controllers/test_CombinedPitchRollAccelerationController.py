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

Tests for the CombinedPitchRollAccelerationController module of the dubins platform.

Author: John McCarroll
"""

from unittest import mock

import numpy as np
import pytest

from saferl.platforms.dubins.dubins_controllers import CombinedPitchRollAccelerationController
from saferl.platforms.dubins.dubins_properties import PitchRollAndAccelerationProp


@pytest.fixture(name="control_properties")
def get_control_properties():
    """
    Returns a friendly prop class
    """
    return PitchRollAndAccelerationProp


@pytest.fixture(name="controller")
def setup_controller(dubins_platform):
    """
    Set up CombinedPitchRollAccelerationController with default values
    """
    config = {}
    return CombinedPitchRollAccelerationController(parent_platform=dubins_platform, config=config)


# Unit Tests
@pytest.mark.unit_test
def test_constructor(control_properties):
    """
    Simple test to ensure construction and attribute assignment functions appropriately
    """
    parent_platform = mock.MagicMock()

    config = {}
    cut = CombinedPitchRollAccelerationController(parent_platform=parent_platform, config=config)

    assert isinstance(cut._properties, control_properties)  # pylint: disable=W0212
    assert cut._parent_platform == parent_platform  # pylint: disable=W0212


@pytest.mark.unit_test
def test_apply_control(controller, control):
    """
    Simple test for the apply_control method of CombinedPitchRollAccelerationController
    """
    controller.apply_control(control)
    assert controller.parent_platform._last_applied_action == control  # pylint: disable=W0212


@pytest.mark.unit_test
def test_get_applied_action(controller):
    """
    Test for the get_applied_control method of CombinedPitchRollAccelerationController
    """

    dummy_arr = np.array([0.05, 0.1, 66.6])
    controller.parent_platform._last_applied_action = dummy_arr  # pylint: disable=W0212

    assert np.array_equal(controller.get_applied_control(), dummy_arr)  # pylint: disable=W0212
