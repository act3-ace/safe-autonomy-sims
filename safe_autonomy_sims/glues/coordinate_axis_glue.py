"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glue which returns the dot product between the deputy orientation unit
vector and the unit vector pointing from the deputy to the chief.

Author: Kochise Bennett
"""
import typing
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.base_glue import BaseAgentGlue, BaseAgentGlueValidator

from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator


class CoordinateAxisGlueValidator(BaseAgentGlueValidator):
    """
    """
    axis: str = 'x'

class CoordinateAxisGlue(BaseAgentGlue):
    """Glue that simply outputs a simple 16x16 image for testing purposes"""

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"
        
    @property
    def get_validator(self) -> typing.Type[CoordinateAxisGlueValidator]:
        return CoordinateAxisGlueValidator

    def get_unique_name(self) -> str:
        return "Coordinate_Axis_Glue_" + self.config.axis.upper() + "-Axis"

    def observation_space(self) -> gym.spaces.Space:
        d = gym.spaces.dict.Dict()
        low = np.array([-1.0, -1.0, -1.0])
        high = np.array([1.0, 1.0, 1.0])
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        if self.config.axis.lower() == 'x':
            out = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif self.config.axis.lower() == 'y':
            out = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif self.config.axis.lower() == 'z':
            out = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = out

        return d
