"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glue which transforms observation into a unit vector and allows custom normalization.

Author: Jamie Cunningham
"""
from collections import OrderedDict

import numpy as np

from saferl.glues.normal.normal_observe_glue import NormalObserveSensorGlue


class UnitVectorGlue(NormalObserveSensorGlue):
    """
    Transforms an observation into a unit vector.
    """

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_UnitVector"

    def get_observation(self) -> OrderedDict:
        # Get observation vector
        sensor_obs = super().get_observation()
        direct_obs = sensor_obs[self.Fields.DIRECT_OBSERVATION]

        # Compute unit vector
        norm = np.linalg.norm(direct_obs)
        unit_vec = direct_obs / norm

        # Return observation
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array(unit_vec, dtype=np.float32)
        return d
