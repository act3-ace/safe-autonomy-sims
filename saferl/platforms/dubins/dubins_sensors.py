"""
This module contains implementations of sensors that reside on the Dubins platform
"""

import math

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.libraries.property import MultiBoxProp
from act3_rl_core.simulators.base_parts import BaseSensor

from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator


class DubinsSensor(BaseSensor):
    """
    Interface for a basic sensor of the CWH platform
    """

    @property
    def name(self) -> str:
        """
        Returns
        -------
        str
            returns name of the sensor
        """
        return self.__class__.__name__

    @property
    def measurement_properties(self):
        """
        gives the measurement properties of a sensors -  units and bounds

        Raises
        ------
        NotImplementedError
            If the method has not been implemented
        """
        raise NotImplementedError

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

    @property
    def measurement_properties(self):
        """
        retreive the measurement properies

        Returns
        -------
        position_properties : MultiBoxProp
        """

        position_properties = MultiBoxProp(
            name="position", low=[-math.inf] * 3, high=[math.inf] * 3, unit=["meters"] * 3, description="position of the aircraft"
        )
        return position_properties

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


class VelocitySensor(DubinsSensor):
    """
    Implementation of a sensor to give velocity at any time
    """

    @property
    def measurement_properties(self):
        """
        Retreive the measurement properies.
        Specifically here return the bounds and units of the velocity of spacecraft.

        Returns
        -------
        velocity_properties : MultiBoxProp
            bounds and units of the velocity measurement
        """
        velocity_properties = MultiBoxProp(
            name="velocity", low=[-math.inf] * 3, high=[math.inf] * 3, unit=["m/s"] * 3, description="velocity of the aircraft"
        )
        return velocity_properties

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


class HeadingSensor(DubinsSensor):
    """
    Implementation of a sensor to give heading at any point in time.
    """

    @property
    def measurement_properties(self):
        """
        Retreive the measurement properies.
        Specifically here return the bounds and units of the heading of aircraft.

        Returns
        -------
        heading_properties : MultiBoxProp
            bounds and units of the heading measurement
        """

        heading_properties = MultiBoxProp(
            name="heading", low=[-2 * math.pi], high=[2 * math.pi], unit=["rad"], description="heading of the aircraft"
        )
        return heading_properties

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

        return self.parent_platform.heading


PluginLibrary.AddClassToGroup(
    HeadingSensor, "Sensor_Heading", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class FlightPathSensor(DubinsSensor):
    """
    Implementation of a sensor to give flight path angle at any time.
    """

    @property
    def measurement_properties(self):
        """
        Retreive the measurement properies.
        Specifically here return the bounds and units of the flight path angle.

        Returns
        -------
        fp_properties : MultiBoxProp
            bounds and units of the flight path angle
        """
        fp_properties = MultiBoxProp(
            name="flight_path_angle", low=[-2 * math.pi], high=[2 * math.pi], unit=["rad"], description="flight path angle of the aircraft"
        )
        return fp_properties

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
        return self.parent_platform.gamma


# PluginLibrary.AddClassToGroup(
#     FlightPathSensor, "Sensor_Flight_Path_Angle", {
#         "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
#     }
# )

# TODO: Add to plugin group


class TimeSensor(DubinsSensor):
    """
    Implementation of a sensor to give flight path angle at any time.
    """

    @property
    def measurement_properties(self):
        """
        Retreive the measurement properies.
        Specifically here return the bounds and units of the flight path angle.

        Returns
        -------
        fp_properties : MultiBoxProp
            bounds and units of the flight path angle
        """
        velocity_properties = MultiBoxProp(
            name="time", low=[-math.inf], high=[math.inf], unit=["sec"], description="time since beginning of simulation"
        )
        return velocity_properties

    def _calculate_measurement(self, state):
        """
        Calculate the measurement - current time

        Params
        ------
        state: np.ndarray
            current state

        Returns
        -------
        float
            current time of the simulation
        """
        return self.parent_platform.sim_time
