"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Glues for the osam environments
"""
import typing
from collections import OrderedDict

import gym
import numpy as np
from corl.glues.base_glue import BaseAgentGlue
from corl.glues.base_multi_wrapper import BaseMultiWrapperGlue
from corl.glues.base_wrapper import BaseWrapperGlue, BaseWrapperGlueValidator
from corl.glues.common.observe_sensor import ObserveSensor
from scipy.spatial.transform import Rotation


class VectorIndexingGlueValidator(BaseWrapperGlueValidator):
    """Validator for VectorIndexingGlue"""
    index: typing.List[int]


class VectorIndexingGlue(BaseWrapperGlue):
    """Wrapper glue that performs indexing of the vector output of its wrapped glue"""

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    @property
    def get_validator(self) -> typing.Type[VectorIndexingGlueValidator]:
        return VectorIndexingGlueValidator

    def get_unique_name(self) -> str:
        return self.glue().get_unique_name() + "_Indexed"

    def observation_space(self) -> gym.spaces.Space:
        wrapped_space = self.glue().observation_space()[self.glue().Fields.DIRECT_OBSERVATION]

        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION
                 ] = gym.spaces.Box(low=wrapped_space.low[self.config.index], high=wrapped_space.high[self.config.index], dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        obs = self.glue().get_observation(other_obs, obs_space, obs_units)[self.glue().Fields.DIRECT_OBSERVATION]
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = obs[self.config.index]
        return d


class MagNormGlue(BaseWrapperGlue):
    """Wrapper glue that converts output vector of wrapped glue into the form [magnitude, normalized unit vector]"""

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    def get_unique_name(self) -> str:
        return self.glue().get_unique_name() + "_MagNorm"

    def observation_space(self) -> gym.spaces.Space:
        wrapped_space = self.glue().observation_space()[self.glue().Fields.DIRECT_OBSERVATION]

        mag = np.linalg.norm(np.maximum(np.abs(wrapped_space.low), np.abs(wrapped_space.high))[0:2])

        low = np.concatenate(([0], -1 * np.ones_like(wrapped_space.low)), dtype=np.float32)  # pylint: disable=unexpected-keyword-arg
        high = np.concatenate(([mag], np.ones_like(wrapped_space.low)), dtype=np.float32)  # pylint: disable=unexpected-keyword-arg

        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        obs = self.glue().get_observation(other_obs, obs_space, obs_units)[self.glue().Fields.DIRECT_OBSERVATION]

        mag = np.linalg.norm(obs)

        if mag == 0:
            output = np.concatenate(([0], np.zeros_like(obs)), dtype=np.float32)  # pylint: disable=unexpected-keyword-arg
        else:
            output = np.concatenate(([mag], obs / mag), dtype=np.float32)  # pylint: disable=unexpected-keyword-arg

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = output

        return d


class Angle2UnitCircleGlueValidator(BaseWrapperGlueValidator):
    """
    Validator for Angle2UnitCircleGlue
    """
    wrapped: ObserveSensor


class Angle2UnitCircleGlue(BaseWrapperGlue):
    """Wrapped glue that transforms sensor glues reporting angles to unit circle x,y position"""

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    @property
    def get_validator(self) -> typing.Type[Angle2UnitCircleGlueValidator]:
        return Angle2UnitCircleGlueValidator

    def get_unique_name(self) -> str:
        return self.glue().get_unique_name() + "_Unit_Circle"

    def observation_space(self) -> gym.spaces.Space:
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        obs = self.glue().get_observation(other_obs, obs_space, obs_units)[self.glue().Fields.DIRECT_OBSERVATION]
        assert obs.shape == (1, )
        angle = obs[0]
        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        return d


class RotateVectorToLocalRef2d(BaseMultiWrapperGlue):
    """Multiwrapped glue that transforms a 2d vector to a local reference frame

    First wrapped glue should be the orientation angle
    Second wrapped glue will be rotated into the local reference frame defined by the orientation
    """

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    def get_unique_name(self) -> str:
        return self.glues()[1].get_unique_name() + "_Local_Ref"

    def observation_space(self) -> gym.spaces.Space:
        source_glue = self.glues()[1]
        source_obs_space = source_glue.observation_space()[source_glue.Fields.DIRECT_OBSERVATION]
        mag = np.linalg.norm(np.maximum(np.abs(source_obs_space.low), np.abs(source_obs_space.high))[0:2])
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low=-mag, high=mag, shape=(2, ), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        orientation_wrapped = self.glues()[0]
        input_vector_wrapped = self.glues()[1]

        theta = orientation_wrapped.get_observation(other_obs, obs_space, obs_units)[orientation_wrapped.Fields.DIRECT_OBSERVATION][0]
        input_vector = input_vector_wrapped.get_observation(other_obs, obs_space, obs_units)[input_vector_wrapped.Fields.DIRECT_OBSERVATION]

        if input_vector.shape == (3, ):
            assert input_vector[2] == 0, "rotated vector must be 2d, got a 3d vector with non-zero z element"
        elif input_vector.shape != (2, ):
            raise ValueError(f"rotated vector must be 2d, got a vector with shape {input_vector.shape}")
        else:
            input_vector = np.append(input_vector, 0.0)

        orientation = Rotation.from_euler('z', angles=theta)

        rotated_vector = orientation.inv().apply(input_vector).astype(np.float32)

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = rotated_vector[:2]

        return d


class TestImageObsGlue(BaseAgentGlue):
    """Glue that simply outputs a simple 16x16 image for testing purposes"""

    class Fields:
        """
        Fields in this glue
        """
        DIRECT_OBSERVATION = "direct_observation"

    def get_unique_name(self) -> str:
        return "Test_Images"

    def observation_space(self) -> gym.spaces.Space:
        d = gym.spaces.dict.Dict()
        d.spaces[self.Fields.DIRECT_OBSERVATION] = gym.spaces.Box(low=-1, high=1, shape=(16, 16), dtype=np.float32)
        return d

    def get_observation(self, other_obs: OrderedDict, obs_space: OrderedDict, obs_units: OrderedDict):
        img = np.zeros((16, 16), dtype=np.float32)

        img[4:10, 5:7] = 1

        d = OrderedDict()
        d[self.Fields.DIRECT_OBSERVATION] = img

        return d
