"""
Wrapper glue which returns a velocity constraint based on position and velocity sensor data as an observation.

Author: Jamie Cunningham
"""
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np
from act3_rl_core.libraries.env_space_util import EnvSpaceUtil

from saferl.glues.common import CustomNormalizationWrapperGlue, CustomNormalizationWrapperGlueValidator


class VelocityLimitGlueValidator(CustomNormalizationWrapperGlueValidator):
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


class VelocityLimitGlue(CustomNormalizationWrapperGlue):
    """
    Computes a velocity constraint from position and velocity sensor data.
    """

    class Fields:
        """
        Field data
        """
        OBSERVATION = "direct_observation"
        VELOCITY_LIMIT = "velocity_limit"

    @property
    def get_validator(self):
        return VelocityLimitGlueValidator

    @lru_cache(maxsize=1)
    def get_unique_name(self):
        return "VelocityLimit"

    def _vel_limit(self, dist):
        return self.config.velocity_threshold + (self.config.slope * self.config.mean_motion * (dist - self.config.threshold_distance))

    @lru_cache(maxsize=1)
    def observation_space(self) -> gym.spaces.Space:
        pos_obs_space = self.glue().observation_space()[self.Fields.OBSERVATION]
        high = self._vel_limit(np.linalg.norm(pos_obs_space.high)) * np.sign(pos_obs_space.high)[0]
        low = self._vel_limit(np.linalg.norm(pos_obs_space.low)) * np.sign(pos_obs_space.low)[0]
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.VELOCITY_LIMIT] = gym.spaces.Box(low, high, shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self) -> EnvSpaceUtil.sample_type:
        pos_obs = self.glue().get_observation()[self.Fields.OBSERVATION]
        d = OrderedDict()
        d[self.Fields.VELOCITY_LIMIT] = np.array([self._vel_limit(np.linalg.norm(pos_obs))], dtype=np.float32)
        return d

    @lru_cache(maxsize=1)
    def action_space(self):
        """Action space"""
        return None

    def apply_action(self, action, observation):  # pylint: disable=unused-argument
        """Apply action"""
        return None
