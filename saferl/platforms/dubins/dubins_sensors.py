"""
This module contains implementations of sensors that reside on the Dubins platform
"""
from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseSensor

import saferl.platforms.dubins.dubins_properties as dubins_props
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator


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

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.PositionProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

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

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.VelocityProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

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

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.HeadingProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

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

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.FlightPathProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

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

# class TimeSensor(DubinsSensor):
#     """
#     Implementation of a sensor to give flight path angle at any time.
#     """
#
#     @property
#     def measurement_properties(self):
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
