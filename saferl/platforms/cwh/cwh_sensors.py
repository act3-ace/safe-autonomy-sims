from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.libraries.property import MultiBoxProp
from act3_rl_core.simulators.base_parts import BaseSensor

from saferl.platforms.cwh.cwh_available_platforms import CWHAvailablePlatformTypes
from saferl.simulators.cwh.cwh_simulator import CWHSimulator


class CWHSensor(BaseSensor):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def measurement_properties(self):
        raise NotImplementedError

    def _calculate_measurement(self, state):
        raise NotImplementedError


class PositionSensor(CWHSensor):

    @property
    def measurement_properties(self):
        position_properties = MultiBoxProp(
            name="position", low=[-80000] * 3, high=[80000] * 3, unit=["meters"] * 3, description="position of the spacecraft"
        )
        return position_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.position


PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)


class VelocitySensor(CWHSensor):

    @property
    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="velocity", low=[-10000] * 3, high=[10000] * 3, unit=["m/s"] * 3, description="velocity of the spacecraft"
        )
        return velocity_properties

    # state - tuple
    def _calculate_measurement(self, state):
        return self.parent_platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)
