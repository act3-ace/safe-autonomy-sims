from unittest import mock

import pytest
from frozendict import frozendict

from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes

# define 3-6 separate config as a variables
# parametrize the main function to take in the vars and test against them
"""
1. Make a config no name -- should get RuntimeError
2. Make a config with name NOT 'CWH' -- Runtime Error
3. Make a config with name = 'CWH' -- get a variable back
"""
"""
Issue with test :

no 'model' key in config in the first place
"""


@pytest.mark.unit_test
def test_parseMethod_emptyconfig():
    config = {}
    with pytest.raises(RuntimeError) as excinfo:
        CWHAvailablePlatformTypes.ParseFromNameModel(config)

    assert str(excinfo.value) == "Attempting to parse a PlatformType from name/model config, but both are not given!"


@pytest.mark.unit_test
def test_parseMethod_wrong_name():
    # the model key gets injected in here test and see what goes wrong here.
    config = {'name': 'H'}
    with pytest.raises(RuntimeError) as excinfo:
        CWHAvailablePlatformTypes.ParseFromNameModel(config)

    assert 'did not match a known platform type' in str(excinfo.value)


@pytest.mark.unit_test
def test_parseMethod_correct_config():
    config = {'name': 'CWH'}
    ret_val = CWHAvailablePlatformTypes.ParseFromNameModel(config)
    assert ret_val.value == (1, )


@pytest.mark.unit_test
def test_CWHtuple():
    # remember you are dealing with a enum here
    tup = CWHAvailablePlatformTypes.CWH.value
    print('tup=', tup)
    assert tup == (1, )
