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
def test_parseMethod_emptyconfig():
    """
    Test for the method - parse, checking empty config case
    """
    config = {}
    with pytest.raises(RuntimeError) as excinfo:
        CWHAvailablePlatformTypes.ParseFromNameModel(config)

    assert str(excinfo.value) == "Attempting to parse a PlatformType from name/model config, but both are not given!"


@pytest.mark.unit_test
def test_parseMethod_wrong_name():
    """
    Test for the method - parse, checking case where wrong value provided for
    'name property '
    """

    # the model key gets injected in here test and see what goes wrong here.
    config = {'name': 'H'}
    with pytest.raises(RuntimeError) as excinfo:
        CWHAvailablePlatformTypes.ParseFromNameModel(config)

    assert 'did not match a known platform type' in str(excinfo.value)


@pytest.mark.unit_test
def test_parseMethod_correct_config():
    """
    Test for the method - parse, checking case for basic case ,
    """

    config = {'name': 'CWH'}
    ret_val = CWHAvailablePlatformTypes.ParseFromNameModel(config)
    assert ret_val.value == (1, )


@pytest.mark.unit_test
def test_CWHtuple():
    """
    Check that the tuple value is correct for CWH enum
    """

    # remember you are dealing with a enum here
    tup = CWHAvailablePlatformTypes.CWH.value
    assert tup == (1, )
