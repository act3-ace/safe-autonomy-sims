"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements the Reward Functions and Reward Validators specific to RTA.
"""
from collections import OrderedDict

from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator


class RTAInterveningRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the RTAInterveningReward Reward Function.

    Attributes
    ----------
    scale : float
        Scalar value to adjust magnitude of the reward.
    """
    scale: float


class RTAInterveningReward(RewardFuncBase):
    """
    This reward function allocates reward based on if RTA is intervening or not.
    """

    def __init__(self, **kwargs):
        self.config: RTAInterveningRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        return RTAInterveningRewardValidator

    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> float:

        reward = 0.
        if next_observation[self.config.agent_name]['RTAModule']['intervening']:
            reward = self.config.scale

        return reward
