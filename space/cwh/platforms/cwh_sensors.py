from act3_rl_core.simulators.base_parts import BaseSensor
from act3_rl_core.simulators.base_platform import BasePlatform
from act3_rl_core.libraries.property import MultiBoxProp
import math
from act3_rl_core.libraries.plugin_library import PluginLibrary
from simulators.cwh_available_platforms import CWHAvailablePlatformTypes
from simulators.cwh_simulator import CWHSimulator
import numpy as np


class CWHSensor(BaseSensor):
    def __init__(self, parent_platform, config):
        super().__init__()
        self._platform = parent_platform
        self._config = config

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def parent_platform(self) -> 'BasePlatform':
        return self._platform

    def measurement_properties(self):
        raise NotImplementedError

    def _calculate_measurement(self, state):
        raise NotImplementedError


class PositionSensor(CWHSensor):

    @property
    def measurement_properties(self):
        position_properties = MultiBoxProp(
            name="position",
            low=[-80000]*3,
            high=[80000]*3,
            unit=["meters"]*3,
            description="position of the spacecraft")
        return position_properties

    def _calculate_measurement(self,state):
        return self._platform.position

PluginLibrary.AddClassToGroup(
    PositionSensor, "Sensor_Position", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)


class VelocitySensor(CWHSensor):

    @property
    def measurement_properties(self):
        velocity_properties = MultiBoxProp(
            name="velocity",
            low=[-10000]*3,
            high=[10000]*3,
            unit=["m/s"]*3,
            description="velocity of the spacecraft")
        return velocity_properties

    # state - tuple
    def _calculate_measurement(self, state):
        return self._platform.velocity


PluginLibrary.AddClassToGroup(
    VelocitySensor, "Sensor_Velocity", {
        "simulator": CWHSimulator, "platform_type": CWHAvailablePlatformTypes.CWH
    }
)