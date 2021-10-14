from act3_rl_core.simulators.base_parts import BaseSensor
from act3_rl_core.simulators.base_platform import BasePlatform
from act3_rl_core.libraries.property import MultiBoxProp
import math


class PositionSensor(BaseSensor):

    def __init__(self, parent_platform, config):
        super().__init__()
        self._platform = parent_platform
        self._config = config

    def name(self) -> str:
        return self.__class__.__name__

    def parent_platform(self) -> 'BasePlatform':
        return self._platform

    def measurement_properties(self):
        position_properties = MultiBoxProp(
            name="position",
            low=[-math.inf]*3,
            high=[math.inf]*3,
            unit=["meters"]*3,
            description="position of the spacecraft")
        return position_properties

    def _calculate_measurement(self,state):
        return self._platform.position


class VelocitySensor(BaseSensor):

    def __init__(self, parent_platform, config):
        super().__init__()
        self._platform = parent_platform
        self._config = config

    def name(self) -> str:
        return self.__class__.__name__

    def parent_platform(self) -> 'BasePlatform':
        return self._platform

    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="velocity",
            low=[-math.inf]*3,
            high=[math.inf]*3,
            unit=["m/s"]*3,
            description="velocity of the spacecraft")
        return velocity_properties

    # state - tuple
    def _calculate_measurement(self,state):
        return self._platform.velocity


class TimeSensor(CWHSensor):

    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="time",
            low=[-math.inf],
            high=[math.inf],
            unit=["sec"],
            description="time since beginning of simulation")
        return velocity_properties

    def _calculate_measurement(self, state):
        return self._platform.sim_time
