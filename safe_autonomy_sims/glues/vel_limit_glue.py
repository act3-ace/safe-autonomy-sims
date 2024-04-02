"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Wrapper glue which returns a velocity constraint based on position and velocity sensor data as an observation.
"""
import typing
from collections import OrderedDict
from functools import cached_property

import gymnasium
import numpy as np
from corl.glues.base_glue import BaseAgentGlueNormalizationValidator
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator
from corl.libraries.normalization import StandardNormalNormalizer
from corl.libraries.property import BoxProp, DictProp
from corl.libraries.units import corl_get_ureg


class VelocityLimitGlueValidator(ObserveSensorValidator):
    """
    A configuration validator for the VelocityLimitGlue.

    Attributes
    ----------
    velocity_threshold: float
        The maximum tolerated velocity within docking region without crashing
    threshold_distance: float
        The radius of the docking region
    mean_motion: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    slope: float
        The slope of the linear velocity limit as a function of distance from docking region
    normalization: BaseAgentGlueNormalizationValidator
        Default normalization
    """
    velocity_threshold: float = 0.2
    threshold_distance: float = 0.5
    mean_motion: float = 0.001027
    slope: float = 2.0
    normalization: BaseAgentGlueNormalizationValidator = BaseAgentGlueNormalizationValidator(normalizer=StandardNormalNormalizer)


class VelocityLimitGlue(ObserveSensor):
    """
    Computes a velocity constraint from position and velocity sensor data.
    """

    @staticmethod
    def get_validator():
        return VelocityLimitGlueValidator

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_VelocityLimit"

    def _vel_limit(self, dist):
        return self.config.velocity_threshold + (self.config.slope * self.config.mean_motion * (dist - self.config.threshold_distance))

    @cached_property
    def observation_prop(self):
        prop = BoxProp(low=[-10000], high=[10000], unit='m/s')
        return DictProp(spaces={self.Fields.DIRECT_OBSERVATION: prop})

    @cached_property
    def observation_space(self) -> typing.Optional[gymnasium.spaces.Space]:
        pos_obs_space = super().observation_space[self.Fields.DIRECT_OBSERVATION]
        high = self._vel_limit(np.linalg.norm(pos_obs_space.high)) * np.sign(pos_obs_space.high)[0]
        low = self._vel_limit(np.linalg.norm(pos_obs_space.low)) * np.sign(pos_obs_space.low)[0]
        d = gymnasium.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gymnasium.spaces.Box(low, high, shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        pos_obs = super().get_observation(other_obs, obs_space, obs_units)[self.Fields.DIRECT_OBSERVATION]
        obs = corl_get_ureg().Quantity(np.array([self._vel_limit(np.linalg.norm(pos_obs.m))], dtype=np.float32), str(pos_obs.units))
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = obs
        return d

    @cached_property
    def normalized_action_space(self) -> typing.Optional[gymnasium.spaces.Space]:
        return self.action_space

    @cached_property
    def normalized_observation_space(self) -> typing.Optional[gymnasium.spaces.Space]:
        return self.observation_space
