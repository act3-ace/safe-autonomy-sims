"""
Wrapper glue which returns the magnitude of its wrapped glue as an observation.

Author: Jamie Cunningham
"""
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np
from act3_rl_core.libraries.env_space_util import EnvSpaceUtil

from saferl.glues.common import CustomNormalizationWrapperGlue


class MagnitudeGlue(CustomNormalizationWrapperGlue):
    """
    Returns the magnitude of the wrapped glue's returned observation.
    """

    class Fields:
        """
        Field data
        """
        OBSERVATION = "direct_observation"
        MAGNITUDE = "magnitude"

    @lru_cache(maxsize=1)
    def get_unique_name(self):
        wrapped_glue_name = self.glue().get_unique_name()
        if wrapped_glue_name is None:
            return None
        return wrapped_glue_name + "Mag"

    @lru_cache(maxsize=1)
    def observation_space(self) -> gym.spaces.Space:
        obs_space = self.glue().observation_space()[self.Fields.OBSERVATION]
        d = gym.spaces.dict.Dict()
        d.spaces[
            self.Fields.MAGNITUDE
        ] = gym.spaces.Box(0, np.maximum(np.linalg.norm(obs_space.low), np.linalg.norm(obs_space.high)), shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self) -> EnvSpaceUtil.sample_type:
        obs = self.glue().get_observation()[self.Fields.OBSERVATION]
        d = OrderedDict()
        d[self.Fields.MAGNITUDE] = np.array([np.linalg.norm(obs)], dtype=np.float32)
        return d

    @lru_cache(maxsize=1)
    def action_space(self):
        """Action space"""
        return None

    def apply_action(self, action, observation):  # pylint: disable=unused-argument
        """Apply action"""
        return None
