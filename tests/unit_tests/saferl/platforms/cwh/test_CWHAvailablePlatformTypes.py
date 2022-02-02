from unittest import mock

import pytest
from frozendict import frozendict

from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes


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
    assert tup == (1, )
