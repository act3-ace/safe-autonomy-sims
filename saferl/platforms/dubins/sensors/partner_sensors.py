"""
This module contains Dubins platform sensors which read values sent from the platform's partner.

Author: Jamie Cunningham
"""
import numpy as np
from act3_rl_core.libraries.plugin_library import PluginLibrary

import saferl.platforms.dubins.sensors.dubins_sensors as d
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator


class PartnerPositionSensor(d.PositionSensor):
    """
    Implementation of a sensor designed to give the position of the platform's partner at any time
    """

    def _raw_measurement(self, state):
        """
        Calculate the position of the platform's partner

        Params
        ------
        state: np.ndarray
            current state

        Returns
        -------
        list of ints
            position of spacecraft
        """
        return self.parent_platform.partner_position


PluginLibrary.AddClassToGroup(
    PartnerPositionSensor,
    "Sensor_Partner_Position", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    PartnerPositionSensor,
    "Sensor_Partner_Position", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class PartnerVelocitySensor(d.VelocitySensor):
    """
    Implementation of a sensor to give velocity of the platform's partner at any time
    """

    def _raw_measurement(self, state):
        """
        Calculate the measurement - partner velocity

        Params
        ------
        state: np.ndarray
            current state


        Returns
        -------
        list of floats
            velocity of aircraft
        """
        return self.parent_platform.partner_velocity


PluginLibrary.AddClassToGroup(
    PartnerVelocitySensor,
    "Sensor_Partner_Velocity", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    PartnerVelocitySensor,
    "Sensor_Partner_Velocity", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class PartnerHeadingSensor(d.HeadingSensor):
    """
    Implementation of a sensor to give the heading of the platform's partner at any point in time.
    """

    def _raw_measurement(self, state):
        """
        Calculate the measurement - partner heading

        Params
        ------
        state: np.ndarray
            current state

        Returns
        -------
        list of floats
            heading of aircraft
        """
        return np.array([self.parent_platform.partner_heading], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    PartnerHeadingSensor,
    "Sensor_Partner_Heading", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    PartnerHeadingSensor,
    "Sensor_Partner_Heading", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class PartnerFlightPathSensor(d.FlightPathSensor):
    """
    Implementation of a sensor to give flight path angle of the platform's partner at any time.
    """

    def _raw_measurement(self, state):
        """
        Calculate the measurement - partner flight path angle

        Params
        ------
        state: np.ndarray
            current state

        Returns
        -------
        float
            flight path angle
        """
        return np.array([np.deg2rad(self.parent_platform.partner_flight_path_angle)], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    PartnerFlightPathSensor,
    "Sensor_Partner_Flight_Path_Angle", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    PartnerFlightPathSensor,
    "Sensor_Partner_Flight_Path_Angle", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class PartnerRollSensor(d.RollSensor):
    """
    Implementation of a sensor to give the roll angle of the platform's partner in radians at any time.
    """

    def _raw_measurement(self, state):
        """
        Calculate the measurement - partner roll angle

        Params
        ------
        state: np.ndarray
            current state

        Returns
        -------
        float
            roll angle
        """
        return np.array([np.deg2rad(self.parent_platform.partner_roll)], dtype=np.float32)


PluginLibrary.AddClassToGroup(
    PartnerRollSensor, "Sensor_Partner_Roll", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    PartnerRollSensor, "Sensor_Partner_Roll", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)


class PartnerQuaternionSensor(d.QuaternionSensor):
    """
    Implementation of a sensor to give a quaternion view of the platform partner's orientation at any time.
    """

    def _raw_measurement(self, state):
        """
        Calculate the measurement - partner quaternion orientation

        Params
        ------
        state: np.ndarray
            current state

        Returns
        -------
        np.ndarray
            quaternion
        """
        return self.parent_platform.partner_orientation.as_quat()


PluginLibrary.AddClassToGroup(
    PartnerQuaternionSensor,
    "Sensor_Partner_Orientation", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)
PluginLibrary.AddClassToGroup(
    PartnerQuaternionSensor,
    "Sensor_Partner_Orientation", {
        "simulator": Dubins3dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
    }
)
