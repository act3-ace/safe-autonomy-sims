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
    Validator for the RTAInterveningReward Reward Function.

    scale : float
        Scalar value to adjust magnitude of the reward.
    """
    scale: float


class RTAInterveningReward(RewardFuncBase):
    """
    This reward function allocates reward based on if RTA is intervening or not.

    Parameters
    ----------
    observation : OrderedDict
        The observations available to the agent from the previous state.
    action
        The last action performed by the agent.
    next_observation : OrderedDict
        The observations available to the agent from the current state.
    state : StateDict
        The previous state of the simulation.
    next_state : StateDict
        The current state of the simulation.
    observation_space : StateDict
        The agent's observation space.
    observation_units : StateDict
        The units corresponding to values in the observation_space?

    Returns
    -------
    reward : float
        The agent's reward for the change in time.
    """

    def __init__(self, **kwargs):
        self.config: RTAInterveningRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        """
        Method to return class's Validator.
        """
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
