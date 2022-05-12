"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains implementations of sensors that reside on the Dubins platform
"""
import typing

import numpy as np
from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_parts import BasePlatformPartValidator, BaseTimeSensor

import saferl.core.platforms.dubins.dubins_properties as dubins_props
from saferl.core.platforms.common.sensors import TransformSensor
from saferl.core.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.core.simulators.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator


class DubinsSensorValidator(BasePlatformPartValidator):
    """
    oriented: Flag to orient sensor value from platform's partner orientation. Default: False.
    """
    oriented: bool = False


class DubinsSensor(TransformSensor):
    """
    Interface for a basic sensor of the Dubins platform
    """

    @property
    def get_validator(self) -> typing.Type[BasePlatformPartValidator]:
        return DubinsSensorValidator

    def _raw_measurement(self, state):
        raise NotImplementedError

    def _transform(self, measurement, state):
        if self.config.oriented:
            ref_rotation = self.parent_platform.partner_orientation.inv()
            measurement = ref_rotation.apply(measurement)
        return measurement


class PositionSensor(DubinsSensor):
    """
    Implementation of a sensor designed to give the position at any time
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.PositionProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _raw_measurement(self, state):
        """
        Calculate the position

        Parameters
        ----------
        state: np.ndarray
            current state

        Returns
        -------
        list of ints
            position of spacecraft
        """
        return self.parent_platform.position


PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class VelocitySensor(DubinsSensor):
    """
    Implementation of a sensor to give velocity at any time
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.VelocityProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _raw_measurement(self, state):
        """
        Calculate the measurement - velocity

        Parameters
        ----------
        state: np.ndarray
            current state

        Returns
        -------
        list of floats
            velocity of aircraft
        """
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class HeadingSensor(DubinsSensor):
    """
    Implementation of a sensor to give heading at any point in time.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.HeadingProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _raw_measurement(self, state):
        """
        Calculate the measurement - heading

        Parameters
        ----------
        state: np.ndarray
            current state

        Returns
        -------
        list of floats
            heading of aircraft
        """
        return np.array([self.parent_platform.heading], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    HeadingSensor, "Sensor_Heading", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    HeadingSensor, "Sensor_Heading", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class FlightPathSensor(DubinsSensor):
    """
    Implementation of a sensor to give flight path angle at any time.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.FlightPathProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _raw_measurement(self, state):
        """
        Calculate the measurement - flight path angle

        Parameters
        ----------
        state: np.ndarray
            current state

        Returns
        -------
        float
            flight path angle
        """
        return np.array([np.deg2rad(self.parent_platform.flight_path_angle)], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    FlightPathSensor, "Sensor_Flight_Path_Angle", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    FlightPathSensor, "Sensor_Flight_Path_Angle", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class RollSensor(DubinsSensor):
    """
    Implementation of a sensor to give roll angle in radians at any time.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.RollProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _raw_measurement(self, state):
        """
        Calculate the measurement - roll angle

        Parameters
        ----------
        state: np.ndarray
            current state

        Returns
        -------
        float
            roll angle
        """
        return np.array([np.deg2rad(self.parent_platform.roll)], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    RollSensor, "Sensor_Roll", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    RollSensor, "Sensor_Roll", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class QuaternionSensor(DubinsSensor):
    """
    Implementation of a sensor to give a quaternion view of the orientation at any time.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.QuaternionProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _raw_measurement(self, state):
        """
        Calculate the measurement - quaternion

        Parameters
        ----------
        state: np.ndarray
            current state

        Returns
        -------
        np.ndarray
            quaternion
        """
        return self.parent_platform.orientation.as_quat()


PluginLibrary.AddClassToGroup(
    QuaternionSensor, "Sensor_Orientation", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    QuaternionSensor, "Sensor_Orientation", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class DubinsTimeSensor(BaseTimeSensor):
    """
    Implementation of a sensor to give the time since episode start
    """

    def _calculate_measurement(self, state):
        return np.array([self.parent_platform.sim_time], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    DubinsTimeSensor, "Sensor_Time", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    DubinsTimeSensor, "Sensor_Time", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class SpeedSensor(DubinsSensor):
    """
    Implementation of a sensor to give speed in ft/s at any time.
    """

    def __init__(self, parent_platform, config, property_class=dubins_props.SpeedProp):
        super().__init__(property_class=property_class, parent_platform=parent_platform, config=config)

    def _raw_measurement(self, state):
        """
        Calculate the measurement - speed

        Parameters
        ------
        state: np.ndarray
            current state

        Returns
        -------
        np.ndarray
            speed
        """
        return np.array([self.parent_platform.v], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    SpeedSensor, "Sensor_Speed", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    SpeedSensor, "Sensor_Speed", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)
