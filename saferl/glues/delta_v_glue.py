"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator


class DeltaVGlueValidator(ObserveSensorValidator):
    """
    Validator for the DeltaVGlueValidator.
    """
    ...


class DeltaVGlue(ObserveSensor):
    """
    Computes delta-v reward scale (for logging)
    """

    @property
    def get_validator(self):
        return DeltaVGlueValidator

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_DeltaV"

    def observation_space(self):
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(-10000, 10000, shape=(1, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([self._platform.delta_v_scale], dtype=np.float32)
        return d

    def get_info_dict(self):
        return {
            "delta_v_scale": self._platform.delta_v_scale,
            "total_steps_counter": self._platform.total_steps_counter,
        }
