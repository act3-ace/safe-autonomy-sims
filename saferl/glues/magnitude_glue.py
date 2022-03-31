"""
Wrapper glue which returns the magnitude of its wrapped glue as an observation.

Author: Jamie Cunningham
"""
from collections import OrderedDict

import gym
import numpy as np

from saferl.glues.normal.normal_observe_glue import NormalObserveSensorGlue


class MagnitudeGlue(NormalObserveSensorGlue):
    """
    Returns the magnitude of the wrapped glue's returned observation.
    """

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_Magnitude"

    def observation_space(self) -> gym.spaces.Space:
        obs_space = super().observation_space()[self.Fields.DIRECT_OBSERVATION]
        d = gym.spaces.dict.Dict()
        d.spaces[
            self.Fields.DIRECT_OBSERVATION
        ] = gym.spaces.Box(0, np.maximum(np.linalg.norm(obs_space.low), np.linalg.norm(obs_space.high)), shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self):
        obs = super().get_observation()[self.Fields.DIRECT_OBSERVATION]
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([np.linalg.norm(obs)], dtype=np.float32)
        return d
