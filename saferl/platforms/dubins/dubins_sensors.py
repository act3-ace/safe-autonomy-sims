import math

from act3_rl_core.libraries.property import MultiBoxProp
from act3_rl_core.simulators.base_parts import BaseSensor


class DubinsSensor(BaseSensor):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def measurement_properties(self):
        raise NotImplementedError

    def _calculate_measurement(self, state):
        raise NotImplementedError


class PositionSensor(DubinsSensor):

    def measurement_properties(self):
        position_properties = MultiBoxProp(
            name="position", low=[-math.inf] * 3, high=[math.inf] * 3, unit=["meters"] * 3, description="position of the spacecraft"
        )
        return position_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.position


class VelocitySensor(DubinsSensor):

    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="velocity", low=[-math.inf] * 3, high=[math.inf] * 3, unit=["m/s"] * 3, description="velocity of the spacecraft"
        )
        return velocity_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.velocity


class TimeSensor(DubinsSensor):

    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="time", low=[-math.inf], high=[math.inf], unit=["sec"], description="time since beginning of simulation"
        )
        return velocity_properties

    def _calculate_measurement(self, state):
        return self.parent_platform.sim_time
