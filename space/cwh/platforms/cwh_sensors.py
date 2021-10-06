from act3_rl_core.simulators.base_parts import BaseSensor
from act3_rl_core.libraries.property import MultiBoxProp
import math

class PositionSensor(BaseSensor):

    def __init__(self,parent_platform,config):
         super.__init__()
         self._platform = parent_platform
         self._config = config

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

    def __init__(self,parent_platform,config):
         super.__init__()
         self._platform = parent_platform
         self._config = config

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
