"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glue which returns the dot product between the deputy position unit vector
and the uninspected points vector.  This tells if the uninspected points cluster
is occluded by the chief.

Author: Kochise Bennett
"""
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator


class UninspectedPointsOccludedGlueValidator(ObserveSensorValidator):
    """
    Validator for the UninspectedPointsOccludedGlue.

    position_obs_name: str
        Name of the observation space entry corresponding to the position of the
        deputy in Hill's frame or deputy frame.
    orientation_obs_name: str
        Name of the observation space entry corresponding to the unit vector
        pointing in the direction of a cluster of uninspected points in Hill's
        frame or deputy frame.
    """
    position_obs_name = 'ObserveSensor_Sensor_Position'
    uninspected_points_obs_name = 'ObserveSensor_Sensor_UninspectedPoints'


class UninspectedPointsOccludedGlue(ObserveSensor):
    """
    Computes the normalized dot product between the uninspected points vector
    and the deputy position vector.  In Hill's frame, this quantity positive
    if the uninspected point cluster an the deputy are on the same side of the
    Chief satellite (implying that the uninspected point is not occluded by
    the chief).  If this quantity is negative, the chief occludes the
    uninspected point from the deputy's current position.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        return UninspectedPointsOccludedGlueValidator

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_UninspectedPointsOccluded"

    def observation_units(self):
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = ['N/A']
        return d

    def observation_space(self):
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        position_obs = other_obs[self.config.position_obs_name][self.Fields.DIRECT_OBSERVATION]
        uninspected_points_obs = other_obs[self.config.uninspected_points_obs_name][self.Fields.DIRECT_OBSERVATION]

        mag1 = np.linalg.norm(position_obs)
        mag2 = np.linalg.norm(uninspected_points_obs)

        occluded_measure = np.dot(position_obs, uninspected_points_obs) / (mag1 * mag2 + 1e-5)

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([occluded_measure], dtype=np.float32)
        return d
