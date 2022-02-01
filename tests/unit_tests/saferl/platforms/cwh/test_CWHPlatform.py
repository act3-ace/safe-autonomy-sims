import pytest
from unittest import mock
from unittest.mock import MagicMock
import numpy as np
from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl_sim.cwh.cwh import CWHSpacecraft


"""
platform_name = blue0
platform = <saferl_sim.cwh.cwh.CWHSpacecraft object at 0x7f2e4cf26f70>
platform_config= [(<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Y Thrust', 'axis': 1, 'properties': {'name': 'y_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Z Thrust', 'axis': 2, 'properties': {'name': 'z_thrust'}}), (<class 'saferl.platforms.cwh.cwh_sensors.PositionSensor'>, {}), (<class 'saferl.platforms.cwh.cwh_sensors.VelocitySensor'>, {})]
"""


"""
Setup methods
"""


@pytest.fixture(name='cwh_spacecraft')
def setup_CWHSpacecraft():
    add_args = {'name':'CWH'}
    spcft = CWHSpacecraft(**add_args)

    return spcft

@pytest.fixture(name='cwh_platform')
def setup_cwhplatform(cwh_spacecraft):
    # platform_name= blue0
    # platform= <saferl_sim.cwh.cwh.CWHSpacecraft object at 0x7f2178c6df70>
    # platform_config= [(<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Y Thrust', 'axis': 1, 'properties': {'name': 'y_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Z Thrust', 'axis': 2, 'properties': {'name': 'z_thrust'}}), (<class 'saferl.platforms.cwh.cwh_sensors.PositionSensor'>, {}), (<class 'saferl.platforms.cwh.cwh_sensors.VelocitySensor'>, {})]
    # type(platform_config) = list

    platform_name = 'blue0'
    platform_config = []

    platform_obj = CWHPlatform(platform_name,cwh_spacecraft,platform_config)
    return platform_obj

"""
Tests for get_applied_action
"""


@pytest.mark.unit_test
def test_CWHPlatform_getAppliedAction_default(cwh_platform):
    platform_obj = cwh_platform
    expected = np.array([0.,0.,0.])
    assert np.array_equiv(platform_obj.get_applied_action(),expected)


@pytest.mark.unit_test
def test_CWHPlatform_getAppliedAction_default(cwh_platform):
    platform_obj = cwh_platform
    expected = np.array([0.,0.,0.])
    assert np.array_equiv(platform_obj.get_applied_action(),expected)


get_applied_tests = [ (np.array([15.,16.,17.])),
                      (np.array([100.,100.,100.])),
                      (np.array([5000.,5000.,5000.])) ]


@pytest.fixture(name='applied_action')
def applied_action(request):
    return request.param




@pytest.mark.unit_test
@pytest.mark.parametrize('applied_action',get_applied_tests,indirect=True)
def test_CWHPlatform_getAppliedAction(cwh_platform,applied_action):
    cwh_platform._last_applied_action = applied_action

    result = cwh_platform.get_applied_action()
    assert np.array_equiv(result,applied_action)


"""
Tests for save_action_to_platform
"""



action_to_platform_tests = [
                            (np.array([5.,6.,7.])),
                            (np.array([10.,10.,10.])),
                            (np.array([1000.,1000.,1000.])) ]

@pytest.mark.unit_test
@pytest.mark.parametrize('applied_action',action_to_platform_tests,indirect=True)
def test_CWHPlatform_saveActionToPlatform(cwh_platform,applied_action):
    cwh_platform.save_action_to_platform(applied_action[0],0)
    cwh_platform.save_action_to_platform(applied_action[1],1)
    cwh_platform.save_action_to_platform(applied_action[2],2)

    assert cwh_platform._last_applied_action[0] == applied_action[0]
    assert cwh_platform._last_applied_action[1] == applied_action[1]
    assert cwh_platform._last_applied_action[2] == applied_action[2]



"""
Test position attribute
"""


@pytest.fixture(name='pos_input')
def fixture_pos_expected(request):
    return request.param

@pytest.fixture(name='pos_expected')
def fixture_pos_actual(request):
    return request.param


@pytest.fixture(name='cwh_spacecraft_pos')
def setup_CWHSpacecraft_pos(pos_input):
    add_args = {'name':'CWH','x':pos_input[0], 'y':pos_input[1],'z':pos_input[2]}
    spcft = CWHSpacecraft(**add_args)

    return spcft

@pytest.fixture(name='cwh_platform_pos')
def setup_cwhplatform_pos(cwh_spacecraft_pos):
    # platform_name= blue0
    # platform= <saferl_sim.cwh.cwh.CWHSpacecraft object at 0x7f2178c6df70>
    # platform_config= [(<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Y Thrust', 'axis': 1, 'properties': {'name': 'y_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Z Thrust', 'axis': 2, 'properties': {'name': 'z_thrust'}}), (<class 'saferl.platforms.cwh.cwh_sensors.PositionSensor'>, {}), (<class 'saferl.platforms.cwh.cwh_sensors.VelocitySensor'>, {})]
    # type(platform_config) = list

    platform_name = 'blue0'
    platform_config = []

    platform_obj = CWHPlatform(platform_name,cwh_spacecraft_pos,platform_config)
    return platform_obj

pos_attr_tests = [  (np.array([0.,0.,0.]),np.array([0.,0.,0.])),
                    (np.array([50.,100.,150.]),np.array([50.,100.,150.])),
                    (np.array([10000.,10000.,10000.]),np.array([10000.,10000.,10000.]))
                 ]

@pytest.mark.unit_test
@pytest.mark.parametrize("pos_input,pos_expected",pos_attr_tests,indirect=True)
def test_position_attribute(cwh_platform_pos,pos_expected):
    assert np.array_equiv(cwh_platform_pos.position,pos_expected)

"""
Tests for velocity attribute
"""


@pytest.fixture(name='vel_input')
def fixture_vel_input(request):
    return request.param

@pytest.fixture(name='vel_expected')
def fixture_vel_expected(request):
    return request.param


@pytest.fixture(name='cwh_spacecraft_vel')
def setup_CWHSpacecraft_vel(vel_input):
    add_args = {'name':'CWH','xdot':vel_input[0], 'ydot':vel_input[1],'zdot':vel_input[2]}
    spcft = CWHSpacecraft(**add_args)

    return spcft

@pytest.fixture(name='cwh_platform_vel')
def setup_cwhplatform_vel(cwh_spacecraft_vel):

    platform_name = 'blue0'
    platform_config = []

    platform_obj = CWHPlatform(platform_name,cwh_spacecraft_vel,platform_config)
    return platform_obj


# pos,
vel_attr_tests = [ ([0., 0., 0.], np.array([0., 0., 0.])),
                    ([1., 2., 3.],np.array([1., 2., 3.])),
                    ([10., 10., 10.],np.array([10., 10., 10.])),
                    ([1000., 1000., 1000.],np.array([1000., 1000., 1000.]))
                 ]
@pytest.mark.unit_test
@pytest.mark.parametrize("vel_input,vel_expected",vel_attr_tests,indirect=True)
def test_velocity_attrbute(cwh_platform_vel,vel_expected):
    calced = cwh_platform_vel.velocity
    assert np.array_equiv(calced,vel_expected)


"""
Tests for sim_time attribute , accessor and mutator
"""

@pytest.mark.unit_test
def test_sim_time_default(cwh_platform):
    simulation_time = cwh_platform.sim_time
    assert simulation_time == 0.0

@pytest.fixture(name='set_to_time')
def setup_set_to_time(request):
    return request.param

#@pytest.mark.unit_test
#@pytest.mark.parametrize("vel_input,vel_expected",vel_attr_tests,indirect=True)
@pytest.fixture(name='simtime_platform')
def setup_sim_time_platform(cwh_platform,set_to_time):
    cwh_platform._sim_time = set_to_time
    return cwh_platform


time_prop_tests = [(10.5),
                    (100.),
                    (200.)]

@pytest.mark.unit_test
@pytest.mark.parametrize("set_to_time",time_prop_tests,indirect=True)
def test_simtime_property(simtime_platform,set_to_time):
    assert simtime_platform.sim_time == set_to_time

time_accsr_tests = [(10.5),
                    (100.),
                    (200.)]



@pytest.mark.unit_test
@pytest.mark.parametrize("set_to_time",time_accsr_tests,indirect=True)
def test_simtime_setter(cwh_platform,set_to_time):
    cwh_platform.sim_time = set_to_time
    assert cwh_platform._sim_time == set_to_time


"""
Tests for 'operable' property
"""
@pytest.mark.unit_test
def test_is_operable(cwh_platform):
    assert cwh_platform.operable == True
