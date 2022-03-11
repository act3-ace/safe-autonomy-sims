"""
This module contains implementations of sensors that reside on the Dubins platform
"""
import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseSensor, BaseTimeSensor

import saferl.platforms.dubins.dubins_properties as dubins_props
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator


class DubinsSensor(BaseSensor):
    """
    Interface for a basic sensor of the CWH platform
    """

    def _calculate_measurement(self, state):
        """
        get measurements from the sensor

        Raises
        ------
        NotImplementedError
            If the method has not been implemented
        """
        raise NotImplementedError


class PositionSensor(DubinsSensor):
    """
    Implementation of a sensor designed to give the position at any time
    """

    def __init__(self, parent_platform, config, measurement_property_class=dubins_props.PositionProp):
        super().__init__(measurement_property_class=measurement_property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the position

        Params
        ------
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

    def __init__(self, parent_platform, config, measurement_property_class=dubins_props.VelocityProp):
        super().__init__(measurement_property_class=measurement_property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - velocity

        Params
        ------
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

    def __init__(self, parent_platform, config, measurement_property_class=dubins_props.HeadingProp):
        super().__init__(measurement_property_class=measurement_property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - heading

        Params
        ------
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

    def __init__(self, parent_platform, config, measurement_property_class=dubins_props.FlightPathProp):
        super().__init__(measurement_property_class=measurement_property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - flight path angle

        Params
        ------
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
    Implementation of a sensor to give roll angle at any time.
    """

    def __init__(self, parent_platform, config, measurement_property_class=dubins_props.RollProp):
        super().__init__(measurement_property_class=measurement_property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - roll angle

        Params
        ------
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

    def __init__(self, parent_platform, config, measurement_property_class=dubins_props.QuaternionProp):
        super().__init__(measurement_property_class=measurement_property_class, parent_platform=parent_platform, config=config)

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - quaternion

        Params
        ------
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
        return self.parent_platform.sim_time


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

# class TimeSensor(DubinsSensor):
#     """
#     Implementation of a sensor to give flight path angle at any time.
#     """
#
#     @property
#     def measurement_property_class(self):
#         """
#         Retreive the measurement properies.
#         Specifically here return the bounds and units of the flight path angle.
#
#         Returns
#         -------
#         fp_properties : MultiBoxProp
#             bounds and units of the flight path angle
#         """
#         velocity_properties = MultiBoxProp(
#             name="time", low=[-math.inf], high=[math.inf], unit=["sec"], description="time since beginning of simulation"
#         )
#         return velocity_properties
#
#     def _calculate_measurement(self, state):
#         """
#         Calculate the measurement - current time
#
#         Params
#         ------
#         state: np.ndarray
#             current state
#
#         Returns
#         -------
#         float
#             current time of the simulation
#         """
#         return self.parent_platform.sim_time
