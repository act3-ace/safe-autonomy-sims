"""
---------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a gaussian decay reward function.
"""

import logging
import typing

import numpy as np
from corl.libraries.utils import get_wrap_diff
from corl.rewards.base_measurement_operation import BaseMeasurementOperation, BaseMeasurementOperationValidator


class GaussianDecayFromTargetValueValidator(BaseMeasurementOperationValidator):
    """
    A configuration validator for GaussianDecayFromTargetValue

    Attributes
    ----------
    reward scale: scale of this reward, this would be the maximum reward value
                  for a given time step
    eps: the length of the reward curve for the exponential decay, would
         recommend playing with this value in a plotting software to determine
         the value you need
    target_value: the value with which to take the difference from the wrapped
                  observation value
    index: the index with which to pull data out of the observation extractor,
           useful if len(observation) > 1
    is_wrap: if the obs difference needs to wrap around 0/360
    is_rad: if the obs difference is in terms of radians
    closer: to only reward with this reward if the difference to the target
            value is less than the last time step
    closer_tolerance: the difference tolerance at which point the agent is close
                      enough to the target value that "closer" is not a concern
    max_diff: reward is zero if absolute difference from target is greater than
              this quantity. Infinite by default.
    """
    reward_scale: float
    eps: float
    target_value: typing.Optional[float] = 0
    index: typing.Optional[int] = 0
    is_wrap: typing.Optional[bool] = False
    is_rad: typing.Optional[bool] = False
    method: typing.Optional[bool] = False
    closer: typing.Optional[bool] = False
    closer_tolerance: typing.Optional[float] = 0.0
    max_diff: float = np.inf


class GaussianDecayFromTargetValue(BaseMeasurementOperation):
    """
    Gaussian Decay from Target Value wraps some sort of observation and takes
    in a target value, the reward based on the difference between the target
    value and the observation. the reward Gaussian decays in value the further
    you are away from the target value.
    """

    @staticmethod
    def get_validator() -> typing.Type[GaussianDecayFromTargetValueValidator]:
        return GaussianDecayFromTargetValueValidator

    def __init__(self, **kwargs) -> None:
        self.config: GaussianDecayFromTargetValueValidator
        super().__init__(**kwargs)
        self._last_value = None
        self._logger = logging.getLogger(self.name)
        self._reward = 0.0

    def __call__(self, observation, action, next_observation, state, next_state, observation_space, observation_units) -> float:
        self._reward = 0.0

        if self.config.agent_name not in next_observation:
            return self._reward

        obs = self.extractor.value(next_observation[self.config.agent_name])[self.config.index]

        if self.config.is_wrap:
            # Note: this always returns the min angle diff and is positive
            diff = get_wrap_diff(obs, self.config.target_value, self.config.is_rad, self.config.method)
        else:
            diff = obs.m - self.config.target_value

        abs_diff = abs(diff)
        if self._last_value is None:
            self._last_value = abs_diff  # type: ignore

        func_applied = 0
        if not self.config.closer or ((self._last_value >= abs_diff) or abs_diff < self.config.closer_tolerance):  # type: ignore
            if not abs_diff > self.config.max_diff:
                func_applied = np.exp(-np.abs(diff**2 / self.config.eps))

        self._reward = self.config.reward_scale * func_applied

        self._last_value = abs_diff  # type: ignore
        return self._reward
