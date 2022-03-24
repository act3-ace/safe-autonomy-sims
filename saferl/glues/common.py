"""
Set of common glues used in all benchmarks.

Author: Jamie Cunningham
"""
import typing
from functools import lru_cache

import gym
import numpy as np
from act3_rl_core.glues.base_glue import BaseAgentGlue, BaseAgentGlueNormalizationValidator, BaseAgentGlueValidator
from act3_rl_core.glues.base_wrapper import BaseWrapperGlue
from act3_rl_core.glues.common.observe_sensor import ObserveSensor, ObserveSensorValidator
from act3_rl_core.libraries.env_space_util import EnvSpaceUtil

from saferl.utils import normalize_sample_from_mu_sigma, normalize_space_from_mu_sigma, unnormalize_sample_from_mu_sigma


class CustomNormalizationGlueNormalizationValidator(BaseAgentGlueNormalizationValidator):
    """
    Allows custom definition of mu and sigma for normalization. Both or neither mu and sigma must be defined.
    mu: custom mu value to normalize by
    sigma: custom sigma value to normalize by
    """
    mu: typing.Union[float, typing.List[float]] = np.inf
    sigma: typing.Union[float, typing.List[float]] = np.inf


class CustomNormalizationGlueValidator(BaseAgentGlueValidator):
    """
    Validate CustomNormalizationGlue
    normalization: enable normalization and set mu and sigma or max and min
    """
    normalization = CustomNormalizationGlueNormalizationValidator = CustomNormalizationGlueNormalizationValidator()


class CustomNormalizationGlue(BaseAgentGlue):
    """
    Returns the magnitude of the wrapped glue's returned observation.
    """

    @property
    def get_validator(self) -> typing.Type[BaseAgentGlueValidator]:
        return CustomNormalizationGlueValidator

    @lru_cache(maxsize=1)
    def normalized_action_space(self) -> typing.Optional[gym.spaces.Space]:
        custom_defined = self.config.normalization.mu != np.inf and self.config.normalization.sigma != np.inf
        custom_not_defined = self.config.normalization.mu == np.inf and self.config.normalization.sigma == np.inf

        action_space = self.action_space()
        if action_space and self.config.normalization.enabled:
            assert custom_defined or custom_not_defined, "Either both mu and sigma must be defined or neither can be defined."
            if not custom_defined:
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
        custom_defined = self.config.normalization.mu != np.inf and self.config.normalization.sigma != np.inf
        custom_not_defined = self.config.normalization.mu == np.inf and self.config.normalization.sigma == np.inf

        if self.config.normalization.enabled:
            assert custom_defined or custom_not_defined, "Either both mu and sigma must be defined or neither can be defined."
            if custom_not_defined:
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
        custom_defined = self.config.normalization.mu != np.inf and self.config.normalization.sigma != np.inf
        custom_not_defined = self.config.normalization.mu == np.inf and self.config.normalization.sigma == np.inf

        observation_space = self.observation_space()
        if observation_space and self.config.normalization.enabled:
            assert custom_defined or custom_not_defined, "Either both mu and sigma must be defined or neither can be defined."
            if not custom_defined:
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
        custom_defined = self.config.normalization.mu != np.inf and self.config.normalization.sigma != np.inf
        custom_not_defined = self.config.normalization.mu == np.inf and self.config.normalization.sigma == np.inf

        if not self.config.normalization.enabled:
            ret = observation
        else:
            assert custom_defined or custom_not_defined, "Either both mu and sigma must be defined or neither can be defined."
            if custom_not_defined:
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


class CustomNormalizationWrapperGlueValidator(CustomNormalizationGlueValidator):
    """
    wrapped - the wrapped glue instance
    """
    wrapped: BaseAgentGlue

    class Config:  # pylint: disable=C0115, R0903
        arbitrary_types_allowed = True


class CustomNormalizationWrapperGlue(CustomNormalizationGlue, BaseWrapperGlue):
    """
    Wrapper glue which allows normalization of wrapped glue actions and observations using custom mu and sigma.
    """

    @property
    def get_validator(self) -> typing.Type[BaseAgentGlueValidator]:
        return CustomNormalizationWrapperGlueValidator


class CustomNormalizationObserveSensorGlueValidator(ObserveSensorValidator):
    """
    Validate CustomNormalizationObserveSensorGlue
    normalization: enable normalization and set mu and sigma or max and min
    """
    normalization = CustomNormalizationGlueNormalizationValidator = CustomNormalizationGlueNormalizationValidator()


class CustomNormalizationObserveSensorGlue(CustomNormalizationGlue, ObserveSensor):
    """
    ObserveSensor glue which allows normalization of glue observations using custom mu and sigma.
    """

    @property
    def get_validator(self) -> typing.Type[BaseAgentGlueValidator]:
        return CustomNormalizationObserveSensorGlueValidator
