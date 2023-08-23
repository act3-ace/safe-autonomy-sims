"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glue which returns the dot product between the deputy orientation unit
vector and the unit vector pointing from the deputy to the chief.

Author: Kochise Bennett
"""
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator


class FacingChiefGlueValidator(ObserveSensorValidator):
    """
    Validator for the FacingChiefGlue.

    position_obs_name: str
        Name of the observation space entry corresponding to the position of the
        deputy in Hill's frame.
    orientation_obs_name: str
        Name of the observation space entry corresponding to the unit vector 
        pointing in the direction of the deputy's orientation in Hill's frame.
    """
    position_obs_name = 'ObserveSensor_Sensor_Position'
    orientation_obs_name = 'ObserveSensor_Sensor_OrientationUnitVector'


class FacingChiefGlue(ObserveSensor):
    """
    Computes dot product between the deputy orientation unit vector and the 
    unit vector pointing from the deputy to the chief
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        return FacingChiefGlueValidator

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_FacingChief"
    
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
        orientation_obs = other_obs[self.config.orientation_obs_name][self.Fields.DIRECT_OBSERVATION]

        deputy_to_chief = -position_obs / (np.linalg.norm(position_obs) + 1e-5)
        
        facing_chief = np.dot(deputy_to_chief, orientation_obs)
        facing_chief = np.clip(facing_chief, -1.0, 1.0)
        
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([facing_chief], dtype=np.float32)
        return d
