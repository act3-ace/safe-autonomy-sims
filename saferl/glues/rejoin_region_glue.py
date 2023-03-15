"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glues related to the rejoin region.

Author: Jamie Cunningham
"""
from collections import OrderedDict

import numpy as np
from corl.glues.base_glue import BaseAgentGlueNormalizationValidator
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator
from corl.libraries.normalization import StandardNormalNormalizer


class InRejoinGlueValidator(ObserveSensorValidator):
    """
    Validator for InRejoinGlue.

    radius: float
        The radius of the rejoin region.
    normalization: BaseAgentGlueNormalizationValidator
        Default normalization
    """
    radius: float
    normalization: BaseAgentGlueNormalizationValidator = BaseAgentGlueNormalizationValidator(normalizer=StandardNormalNormalizer)


class InRejoinGlue(ObserveSensor):
    """
    Reports if sensor measurements indicate that the agent is inside a rejoin region.
    """

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "InRejoin"

    @property
    def get_validator(self):
        return InRejoinGlueValidator

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict) -> OrderedDict:
        sensor_obs = super().get_observation(other_obs, obs_space, obs_units)
        rel_position = sensor_obs[self.Fields.DIRECT_OBSERVATION]

        d = OrderedDict()
        in_rejoin = rel_position < self.config.radius
        d[self.Fields.DIRECT_OBSERVATION] = np.array([in_rejoin], dtype=np.float32)
        return d
