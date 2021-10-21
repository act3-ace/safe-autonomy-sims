from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_parts import BaseSensor

import saferl.platforms.dubins.dubins_properties as dubins_props
from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator


class DubinsSensor(BaseSensor):

    def _calculate_measurement(self, state):
        raise NotImplementedError


class PositionSensor(DubinsSensor):

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.PositionProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

    def _calculate_measurement(self, state):
        return self.parent_platform.position


PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class VelocitySensor(DubinsSensor):

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.VelocityProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

    def _calculate_measurement(self, state):
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class HeadingSensor(DubinsSensor):

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.HeadingProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

    def _calculate_measurement(self, state):
        return self.parent_platform.heading


PluginLibrary.AddClassToGroup(
    HeadingSensor, "Sensor_Heading", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class FlightPathSensor(DubinsSensor):

    def __init__(self, parent_platform, config, measurement_properties=dubins_props.FlightPathProp, exclusiveness=set()):  # pylint: disable=W0102
        super().__init__(
            measurement_properties=measurement_properties, parent_platform=parent_platform, config=config, exclusiveness=exclusiveness
        )

    def _calculate_measurement(self, state):
        return self.parent_platform.gamma


# PluginLibrary.AddClassToGroup(
#     FlightPathSensor, "Sensor_Flight_Path_Angle", {
#         "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
#     }
# )

# TODO: Add to plugin group

# class TimeSensor(DubinsSensor):
#
#     def measurement_properties(self):
#         velocity_properties = MultiBoxProp(
#             name="time", low=[-math.inf], high=[math.inf], unit=["sec"],
#             description="time since beginning of simulation"
#         )
#         return velocity_properties
#
#     def _calculate_measurement(self, state):
#         return self.parent_platform.sim_time
