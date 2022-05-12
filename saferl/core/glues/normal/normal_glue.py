"""
Glue which allows custom definition of mu and sigma for normalization.

Author: Jamie Cunningham
"""

import copy
import typing
from collections import OrderedDict
from functools import lru_cache

import gym
import numpy as np
from corl.glues.base_glue import BaseAgentGlue, BaseAgentGlueNormalizationValidator, BaseAgentGlueValidator
from corl.libraries.env_space_util import EnvSpaceUtil
from pydantic import validator


def normalize_space_from_mu_sigma(
    space_likes: typing.Tuple[gym.spaces.Space],
    mu: float = 0.0,
    sigma: float = 1.0,
) -> gym.spaces.Space:
    """
    Normalizes a given gym box using the provided mu and sigma.

    Parameters
    ----------
    space_likes: typing.Tuple[gym.spaces.Space]
        The gym space to turn all boxes into the scaled space.
    mu: float = 0.0
        Mu for normalization.
    sigma: float = 1.0
        Sigma for normalization.

    Returns
    -------
    gym.spaces.Space:
        The new gym spaces where all boxes have had their bounds changed.
    """
    space_arg = space_likes[0]
    if isinstance(space_arg, gym.spaces.Box):
        low = np.divide(np.subtract(space_arg.low, mu), sigma)
        high = np.divide(np.subtract(space_arg.high, mu), sigma)
        return gym.spaces.Box(low=low, high=high, shape=space_arg.shape, dtype=np.float32)
    return copy.deepcopy(space_arg)


def normalize_sample_from_mu_sigma(
    space_likes: typing.Tuple[gym.spaces.Space, typing.Union[OrderedDict, dict, tuple, np.ndarray, list]],
    mu: float = 0.0,
    sigma: float = 1,
) -> typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
    """
    This normalizes a sample from a box space using the mu and sigma arguments.

    Parameters
    ----------
    space_likes: typing.Tuple[gym.spaces.Space, sample_type]
        The first element is the gym space.
        The second element is the sample of this space to scale.
    mu: float
        The mu used for normalizing the sample.
    sigma: float
        The sigma used for normalizing the sample.

    Returns
    -------
    typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
        The normalized sample.
    """
    (space_arg, space_sample_arg) = space_likes
    if isinstance(space_arg, gym.spaces.Box):
        val = np.array(space_sample_arg)
        norm_value = np.subtract(val, mu)
        norm_value = np.divide(norm_value, sigma)
        return norm_value.astype(np.float32)
    return copy.deepcopy(space_sample_arg)


def unnormalize_sample_from_mu_sigma(
    space_likes: typing.Tuple[gym.spaces.Space, typing.Union[OrderedDict, dict, tuple, np.ndarray, list]],
    mu: float = 0.0,
    sigma: float = 1,
) -> typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
    """
    This unnormalizes a sample from a box space using the mu and sigma arguments.

    Parameters
    ----------
    space_likes: typing.Tuple[gym.spaces.Space, sample_type]
        The first element is the gym space.
        The second element is the sample of this space to scale.
    mu: float
        The mu used for unnormalizing the sample.
    sigma: float
        The sigma used for unnormalizing the sample.

    Returns
    -------
    typing.Union[OrderedDict, dict, tuple, np.ndarray, list]:
        The unnormalized sample.
    """
    (space_arg, space_sample_arg) = space_likes
    if isinstance(space_arg, gym.spaces.Box):
        val = np.array(space_sample_arg)
        norm_value = np.add(np.multiply(val, sigma), mu)
        return norm_value.astype(np.float32)
    return copy.deepcopy(space_sample_arg)


class NormalGlueNormalizationValidator(BaseAgentGlueNormalizationValidator):
    """
    Allows custom definition of mu and sigma for normalization. Both or neither mu and sigma must be defined.
    mu: custom mu value to normalize by.
    sigma: custom sigma value to normalize by.
    """
    mu: typing.Union[float, typing.List[float], None] = None
    sigma: typing.Union[float, typing.List[float], None] = None

    @validator('mu', 'sigma', always=True)
    def check_iterable(cls, v):
        """
        Check if mu is iterable.
        """
        if isinstance(v, float):
            v = [v]
        return v


class NormalGlueValidator(BaseAgentGlueValidator):
    """
    Validates NormalGlue config.

    normalization: NormalGlueNormalizationValidator
        Enable normalization and set mu and sigma or max and min.
    """
    normalization: NormalGlueNormalizationValidator = NormalGlueNormalizationValidator()


class NormalGlue(BaseAgentGlue):
    """
    Returns the magnitude of the wrapped glue's returned observation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._align_norm_parameters()
        self.custom_defined = self.config.normalization.mu is not None and self.config.normalization.sigma is not None
        self.custom_not_defined = self.config.normalization.mu is None and self.config.normalization.sigma is None
        assert self.custom_defined or self.custom_not_defined, "Either both mu and sigma must be defined or neither can be defined."

    @property
    def get_validator(self) -> typing.Type[NormalGlueValidator]:
        return NormalGlueValidator

    def _align_norm_parameters(self):
        """
        Align the provided normalization parameters to match the shape of the observation space.
        Unspecified mu values are filled in with zeros.
        Unspecified sigma values are filled in with ones.

        Returns
        -------
        None
        """
        zeros = np.zeros(self.observation_space()[self.Fields.DIRECT_OBSERVATION].shape)
        ones = np.ones(self.observation_space()[self.Fields.DIRECT_OBSERVATION].shape)
        if self.config.normalization.mu is not None:
            self.config.normalization.mu = zeros + self.config.normalization.mu
        if self.config.normalization.sigma is not None:
            ones[:len(self.config.normalization.sigma)] = self.config.normalization.sigma
            self.config.normalization.sigma = ones

    @lru_cache(maxsize=1)
    def normalized_action_space(self) -> typing.Optional[gym.spaces.Space]:

        action_space = self.action_space()
        if action_space and self.config.normalization.enabled:
            if not self.custom_defined:
                return super().normalized_observation_space()
            return EnvSpaceUtil.iterate_over_space_likes(
                func=normalize_space_from_mu_sigma,
                space_likes=(self.action_space(), ),
                return_space=True,
                mu=self.config.normalization.mu,
                sigma=self.config.normalization.sigma,
            )
        if action_space:
            return action_space
        return None

    def unnormalize_action(self, action: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        if self.config.normalization.enabled:
            if self.custom_not_defined:
                ret = super().unnormalize_action(action)
            else:
                ret = EnvSpaceUtil.iterate_over_space_likes(
                    func=unnormalize_sample_from_mu_sigma,
                    space_likes=(self.action_space(), action),
                    return_space=False,
                    mu=self.config.normalization.mu,
                    sigma=self.config.normalization.sigma,
                )
        else:
            ret = action
        return ret

    @lru_cache(maxsize=1)
    def normalized_observation_space(self) -> typing.Optional[gym.spaces.Space]:
        observation_space = self.observation_space()
        if observation_space and self.config.normalization.enabled:
            if not self.custom_defined:
                return super().normalized_observation_space()
            return EnvSpaceUtil.iterate_over_space_likes(
                func=normalize_space_from_mu_sigma,
                space_likes=(self.observation_space(), ),
                return_space=True,
                mu=self.config.normalization.mu,
                sigma=self.config.normalization.sigma,
            )
        if observation_space:
            return observation_space
        return None

    def normalize_observation(self, observation: EnvSpaceUtil.sample_type) -> EnvSpaceUtil.sample_type:
        if not self.config.normalization.enabled:
            ret = observation
        else:
            if self.custom_not_defined:
                ret = super().normalize_observation(observation)
            else:
                ret = EnvSpaceUtil.iterate_over_space_likes(
                    func=normalize_sample_from_mu_sigma,
                    space_likes=(self.observation_space(), observation),
                    return_space=False,
                    mu=self.config.normalization.mu,
                    sigma=self.config.normalization.sigma,
                )
        return ret
