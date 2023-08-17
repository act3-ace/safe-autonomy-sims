"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Wrapper glue which returns the dot product between the deputy orientation unit
vector and the unit vector pointing from the deputy to the chief.

Author: Kochise Bennett
"""
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.base_glue import BaseAgentGlueNormalizationValidator
from corl.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator
from corl.libraries.normalization import StandardNormalNormalizer
from corl.simulators.common_platform_utils import get_sensor_by_name
from corl.libraries.units import Convert, GetStrFromUnit, GetUnitFromStr



class FacingChiefGlueValidator(ObserveSensorValidator):
    """
    Validator for the FacingChiefGlue.

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
    # key_prefix = 'ObserveSensor_'
    # position_sensor = 'Sensor_Position'
    position_obs_name = 'ObserveSensor_Sensor_Position'
    orientation_obs_name = 'ObserveSensor_Sensor_OrientationUnitVector'


class FacingChiefGlue(ObserveSensor):
    """
    Computes dot product between the deputy orientation unit vector and the 
    unit vector pointing from the deputy to the chief
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._position_sensor = get_sensor_by_name(self._platform, self.config.position_sensor)

    @property
    def get_validator(self):
        return FacingChiefGlueValidator

    def get_unique_name(self) -> str:
        return super().get_unique_name() + "_FacingChief"
    
    def observation_units(self):
        d = gym.spaces.dict.Dict()
        # unit = GetUnitFromStr('N/A')
        d.spaces[self.Fields.DIRECT_OBSERVATION] = ['N/A']
        return d

    def observation_space(self):
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        # sensed_orientation = self._sensor.get_measurement()
        # sensed_position = self._position_sensor.get_measurement()
        position_obs = other_obs[self.config.position_obs_name][self.Fields.DIRECT_OBSERVATION]
        orientation_obs = other_obs[self.config.orientation_obs_name][self.Fields.DIRECT_OBSERVATION]

        deputy_to_chief = -position_obs / (np.linalg.norm(position_obs) + 1e-5)
        
        facing_chief = np.dot(deputy_to_chief, orientation_obs)
        facing_chief = np.clip(facing_chief, -1.0, 1.0)
        
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([facing_chief], dtype=np.float32)
        return d
