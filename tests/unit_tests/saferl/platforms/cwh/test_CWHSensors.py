from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
import saferl.platforms.cwh.cwh_properties as cwh_props
from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.platforms.cwh.cwh_sensors import CWHSensor, PositionSensor, VelocitySensor
from saferl_sim.cwh.cwh import CWHSpacecraft


@pytest.mark.unit_test
def test_CWHSensor_interface():
    platform = MagicMock()
    sensor = CWHSensor(platform,{},cwh_props.ThrustProp)
    dummy_state = np.array([0.,0.,0.])
    with pytest.raises(NotImplementedError) as excinfo:
        sensor._calculate_measurement(dummy_state)



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


@pytest.fixture(name='pos_sensor')
def setup_pos_sensor(cwh_platform_pos):
    #config = {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}
    config = {}
    measurement_props = cwh_props.PositionProp
    pos_sensor = PositionSensor(cwh_platform_pos,config,measurement_props)
    return pos_sensor

# pos,
pos_sensor_tests = [  ([0.,0.,0.],np.array([0.,0.,0.])),
                    ([1.,2.,3.],np.array([1.,2.,3.])),
                    ([10000.,10000.,10000.],np.array([10000.,10000.,10000.]))
                 ]
@pytest.mark.unit_test
@pytest.mark.parametrize("pos_input,pos_expected",pos_sensor_tests,indirect=True)
def test_PositionSensor_calc_msmt(pos_sensor,pos_input,pos_expected):
    # need to instantiate a CWHPlatform,
    # args needed parent_platform, config
    state = np.array([0.,0.,0.])
    # measurement_properties= <class 'saferl.platforms.cwh.cwh_properties.PositionProp'>
    # parent_platform= <saferl.platforms.cwh.cwh_platform.CWHPlatform object at 0x7f686bc60b50>
    # config= {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}
    calced = pos_sensor._calculate_measurement(state)
    assert np.array_equiv(calced,pos_expected)


"""
Tests for velocity sensor
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
    # platform_name= blue0
    # platform= <saferl_sim.cwh.cwh.CWHSpacecraft object at 0x7f2178c6df70>
    # platform_config= [(<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Y Thrust', 'axis': 1, 'properties': {'name': 'y_thrust'}}), (<class 'saferl.platforms.cwh.cwh_controllers.ThrustController'>, {'name': 'Z Thrust', 'axis': 2, 'properties': {'name': 'z_thrust'}}), (<class 'saferl.platforms.cwh.cwh_sensors.PositionSensor'>, {}), (<class 'saferl.platforms.cwh.cwh_sensors.VelocitySensor'>, {})]
    # type(platform_config) = list

    platform_name = 'blue0'
    platform_config = []

    platform_obj = CWHPlatform(platform_name,cwh_spacecraft_vel,platform_config)
    return platform_obj


@pytest.fixture(name='vel_sensor')
def setup_vel_sensor(cwh_platform_vel):
    #config = {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}
    config = {}
    measurement_props = cwh_props.VelocityProp
    vel_sensor = VelocitySensor(cwh_platform_vel,config,measurement_props)
    return vel_sensor

# pos,
vel_sensor_tests = [ ([0.,0.,0.],np.array([0.,0.,0.])),
                    ([1.,2.,3.],np.array([1.,2.,3.])),
                    ([10.,10.,10.],np.array([10.,10.,10.])),
                    ([10000.,10000.,10000.],np.array([10000.,10000.,10000.]))
                 ]
@pytest.mark.unit_test
@pytest.mark.parametrize("vel_input,vel_expected",vel_sensor_tests,indirect=True)
def test_VelocitySensor_calc_msmt(vel_sensor,vel_input,vel_expected):
    # need to instantiate a CWHPlatform,
    # args needed parent_platform, config
    state = np.array([0.,0.,0.])
    # measurement_properties= <class 'saferl.platforms.cwh.cwh_properties.PositionProp'>
    # parent_platform= <saferl.platforms.cwh.cwh_platform.CWHPlatform object at 0x7f686bc60b50>
    # config= {'name': 'X Thrust', 'axis': 0, 'properties': {'name': 'x_thrust'}}
    calced = vel_sensor._calculate_measurement(state)
    assert np.array_equiv(calced,vel_expected)
