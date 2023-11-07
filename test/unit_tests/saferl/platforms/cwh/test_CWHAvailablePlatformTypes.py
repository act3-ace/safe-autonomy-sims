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

Tests for the CWHAvailablePlatformTypes module
"""
import pytest

from safe_autonomy_sims.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes


@pytest.mark.unit_test
def test_match_model():
    """
    Test for the method - parse, checking empty config case
    """
    config = {"name": "CWH"}
    assert CWHAvailablePlatformTypes.match_model(config)
    config = {"name": "CWHSixDOF"}
    assert CWHAvailablePlatformTypes.match_model(config)
