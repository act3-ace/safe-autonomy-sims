"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Wrapper glue which returns a velocity constraint based on position and velocity sensor data as an observation.

Author: Jamie Cunningham
"""
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.base_glue import BaseAgentGlueNormalizationValidator
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator
from corl.libraries.normalization import StandardNormalNormalizer


class VelocityLimitGlueValidator(ObserveSensorValidator):
    """
    Validator for the VelocityLimitGlue.

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

    @property
    def get_validator(self):
        return VelocityLimitGlueValidator

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_VelocityLimit"

    def _vel_limit(self, dist):
        return self.config.velocity_threshold + (self.config.slope * self.config.mean_motion * (dist - self.config.threshold_distance))

    def observation_space(self):
        pos_obs_space = super().observation_space()[self.Fields.DIRECT_OBSERVATION]
        high = self._vel_limit(np.linalg.norm(pos_obs_space.high)) * np.sign(pos_obs_space.high)[0]
        low = self._vel_limit(np.linalg.norm(pos_obs_space.low)) * np.sign(pos_obs_space.low)[0]
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low, high, shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        pos_obs = super().get_observation(other_obs, obs_space, obs_units)[self.Fields.DIRECT_OBSERVATION]
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([self._vel_limit(np.linalg.norm(pos_obs))], dtype=np.float32)
        return d
