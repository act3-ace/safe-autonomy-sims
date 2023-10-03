"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glues for the osam environments
"""
import typing
from pydantic import validator
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue, BaseMultiWrapperGlueValidator
from corl.glues.base_wrapper import BaseWrapperGlue
from scipy.spatial.transform import Rotation


class MagNorm3DGlue(BaseWrapperGlue):
    """Wrapper glue that converts output vector of wrapped glue into the form [magnitude, normalized unit vector]"""

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    def get_unique_name(self) -> str:
        return self.glue().get_unique_name() + "_MagNorm3D"

    def observation_space(self) -> gym.spaces.Space:
        wrapped_space = self.glue().observation_space()[self.glue().Fields.DIRECT_OBSERVATION]

        mag = np.linalg.norm(np.maximum(np.abs(wrapped_space.low), np.abs(wrapped_space.high)))

        low = np.concatenate([[0], -1 * np.ones(3)], dtype=np.float32)  # pylint: disable=unexpected-keyword-arg
        high = np.concatenate([[mag], np.ones(3)], dtype=np.float32)  # pylint: disable=unexpected-keyword-arg

        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        obs = self.glue().get_observation(other_obs, obs_space, obs_units)[self.glue().Fields.DIRECT_OBSERVATION]

        mag = np.linalg.norm(obs)

        # if mag == 0:
        if mag < 1e-5:
            output = np.concatenate([[0], np.zeros_like(obs)], dtype=np.float32)  # pylint: disable=unexpected-keyword-arg
        else:
            output = np.concatenate([[mag], obs / mag], dtype=np.float32)  # pylint: disable=unexpected-keyword-arg

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = output

        return d


class RotateVectorToLocalRef3dGlueValidator(BaseMultiWrapperGlueValidator):
    """_summary_
    """
    mode: str = 'quaternion'
    apply_inv: bool = True
    
    @validator('mode')
    def mode_is_recognized(cls, v):
        if v not in ['quaternion', 'euler']:
            raise ValueError('mode indicates an unrecognized format for the rotation; accepts "euler" or "quaternion"')
        return v


class RotateVectorToLocalRef3d(BaseMultiWrapperGlue):
    """Multiwrapped glue that transforms a 3d vector to a local reference frame

    First wrapped glue should be the orientation observable (angle or quaternion)
    Second wrapped glue will be rotated into the local reference frame defined by the orientation
    """

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    @property
    def get_validator(self) -> typing.Type[RotateVectorToLocalRef3dGlueValidator]:
        return RotateVectorToLocalRef3dGlueValidator

    def get_unique_name(self) -> str:
        return self.glues()[1].get_unique_name() + "_Local_Ref"

    def observation_space(self) -> gym.spaces.Space:
        source_glue = self.glues()[1]
        source_obs_space = source_glue.observation_space()[source_glue.Fields.DIRECT_OBSERVATION]
        mag = np.linalg.norm(np.maximum(np.abs(source_obs_space.low), np.abs(source_obs_space.high)))
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low=-mag, high=mag, shape=(3, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        orientation_wrapped = self.glues()[0]
        input_vector_wrapped = self.glues()[1]

        orientation_obs = orientation_wrapped.get_observation(other_obs, obs_space, obs_units)[
            orientation_wrapped.Fields.DIRECT_OBSERVATION]
        input_vector = input_vector_wrapped.get_observation(other_obs, obs_space, obs_units)[
            input_vector_wrapped.Fields.DIRECT_OBSERVATION]

        if self.config.mode == 'euler':
            orientation = Rotation.from_euler('z', angles=orientation_obs[0])
        elif self.config.mode == 'quaternion':
            orientation = Rotation.from_quat(orientation_obs)

        if self.config.apply_inv:
            rotated_vector = orientation.inv().apply(input_vector).astype(np.float32)
        else:
            rotated_vector = orientation.apply(input_vector).astype(np.float32)

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = rotated_vector

        return d
