"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Safe Autonomy Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a Reward function that rewards the agent for ever live
timestep up to some maximum
"""
import logging
import typing

import numpy as np
from corl.libraries.environment_dict import RewardDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class LiveTimestepRewardValidator(RewardFuncBaseValidator):
    """
    step_reward: the maximum reward value for each timestep
    max_time_rewarded: the maximum simulation time.  Reward is zero for all
        timesteps for which sim_time > max_time_rewarded [Note use of sim_time
        instead of timestep count!]
    """
    step_reward: float = 0.01
    max_time_rewarded: float = np.inf


class LiveTimestepReward(RewardFuncBase):
    """
    Gives reward for a discrete action matching its target
    """

    @property
    def get_validator(self) -> typing.Type[LiveTimestepRewardValidator]:
        return LiveTimestepRewardValidator

    def __init__(self, **kwargs) -> None:
        self.config: LiveTimestepRewardValidator
        super().__init__(**kwargs)
        self._logger = logging.getLogger(self.name)

    def __call__(self, observation, action, next_observation, state, next_state, observation_space, observation_units) -> RewardDict:

        reward = RewardDict()
        reward[self.config.agent_name] = 0

        if self.config.agent_name not in next_observation:
            return reward

        sim_time = next_state.sim_time

        if sim_time <= self.config.max_time_rewarded:
            reward[self.config.agent_name] = self.config.step_reward

        return reward