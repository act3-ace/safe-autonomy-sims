"""
Wrapper glue which returns a velocity constraint based on position and velocity sensor data as an observation.

Author: Jamie Cunningham
"""
from collections import OrderedDict

import gym
import numpy as np

from saferl.core.glues.normal.normal_observe_glue import NormalObserveSensorGlue, NormalObserveSensorGlueValidator


class VelocityLimitGlueValidator(NormalObserveSensorGlueValidator):
    """
    Validator for the VelocityLimitGlue
    velocity_threshold: float
        The maximum tolerated velocity within docking region without crashing
    threshold_distance: float
        The radius of the docking region
    mean_motion: float
        TODO: define
    slope: float
        The slope of the linear velocity limit as a function of distance from docking region
    """
    velocity_threshold: float = 0.2
    threshold_distance: float = 0.5
    mean_motion: float = 0.001027
    slope: float = 2.0


class VelocityLimitGlue(NormalObserveSensorGlue):
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

    def get_observation(self):
        pos_obs = super().get_observation()[self.Fields.DIRECT_OBSERVATION]
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([self._vel_limit(np.linalg.norm(pos_obs))], dtype=np.float32)
        return d
