import math

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.libraries.property import MultiBoxProp
from act3_rl_core.simulators.base_parts import BaseSensor

from saferl.platforms.dubins.dubins_available_platforms import DubinsAvailablePlatformTypes
from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator


class DubinsSensor(BaseSensor):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def measurement_properties(self):
        raise NotImplementedError

    def _calculate_measurement(self, state):
        raise NotImplementedError


class PositionSensor(DubinsSensor):

    @property
    def measurement_properties(self):
        position_properties = MultiBoxProp(
            name="position", low=[-math.inf] * 3, high=[math.inf] * 3, unit=["meters"] * 3, description="position of the aircraft"
        )
        return position_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.position


PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class VelocitySensor(DubinsSensor):

    @property
    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="velocity", low=[-math.inf] * 3, high=[math.inf] * 3, unit=["m/s"] * 3, description="velocity of the aircraft"
        )
        return velocity_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class HeadingSensor(DubinsSensor):

    @property
    def measurement_properties(self):
        heading_properties = MultiBoxProp(
            name="heading", low=[-2 * math.pi], high=[2 * math.pi], unit=["rad"], description="heading of the aircraft"
        )
        return heading_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.heading


PluginLibrary.AddClassToGroup(
    HeadingSensor, "Sensor_Heading", {
        "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS2D
    }
)


class FlightPathSensor(DubinsSensor):

    @property
    def measurement_properties(self):
        fp_properties = MultiBoxProp(
            name="flight_path_angle", low=[-2 * math.pi], high=[2 * math.pi], unit=["rad"], description="flight path angle of the aircraft"
        )
        return fp_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.gamma


# PluginLibrary.AddClassToGroup(
#     FlightPathSensor, "Sensor_Flight_Path_Angle", {
#         "simulator": Dubins2dSimulator, "platform_type": DubinsAvailablePlatformTypes.DUBINS3D
#     }
# )

# TODO: Add to plugin group


class TimeSensor(DubinsSensor):

    @property
    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="time", low=[-math.inf], high=[math.inf], unit=["sec"], description="time since beginning of simulation"
        )
        return velocity_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.sim_time
