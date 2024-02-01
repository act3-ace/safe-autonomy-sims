"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a glue which passes a coordinate axis as an observation to an agent.

Author: Kochise Bennett
"""
import typing
from collections import OrderedDict
from functools import cached_property

import gymnasium
import numpy as np
from corl.glues.base_glue import BaseAgentGlue, BaseAgentGlueValidator
from corl.libraries.property import BoxProp, DictProp
from corl.libraries.units import corl_get_ureg
from pydantic import validator


class CoordinateAxisGlueValidator(BaseAgentGlueValidator):
    """
    A configuration validator for CoordinateAxisGlue

    Attributes
    ----------
    axis : str
        name of the axis to observe
    """
    axis: str = 'x'

    @validator('axis')
    def axis_must_be_in_xyz(cls, v):
        """Validate axis"""
        if v.lower() not in ['x', 'y', 'z']:
            raise ValueError('Axis must be x, y, or z')
        return v


class CoordinateAxisGlue(BaseAgentGlue):
    """Glue that simply outputs a simple 16x16 image for testing purposes"""

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    @staticmethod
    def get_validator() -> typing.Type[CoordinateAxisGlueValidator]:
        return CoordinateAxisGlueValidator

    def get_unique_name(self) -> str:
        return "Coordinate_Axis_Glue_" + self.config.axis.upper() + "-Axis"

    @cached_property
    def observation_prop(self):
        prop = BoxProp(low=[-1, -1, -1], high=[1, 1, 1], unit="dimensionless")
        return DictProp(spaces={self.Fields.DIRECT_OBSERVATION: prop})

    @cached_property
    def normalized_observation_space(self) -> typing.Optional[gymnasium.spaces.Space]:
        """
        passthrough property
        """
        return self.observation_space

    @cached_property
    def observation_space(self) -> gymnasium.spaces.Space:
        d = gymnasium.spaces.dict.Dict()
        low = np.array([-1.0, -1.0, -1.0])
        high = np.array([1.0, 1.0, 1.0])
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        if self.config.axis.lower() == 'x':
            out = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif self.config.axis.lower() == 'y':
            out = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif self.config.axis.lower() == 'z':
            out = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = corl_get_ureg().Quantity(out, "dimensionless")

        return d
