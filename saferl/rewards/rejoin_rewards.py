"""
This module implements the Reward Functions and Reward Validators specific to the docking task.
"""
from collections import OrderedDict

import numpy as np
from act3_rl_core.libraries.environment_dict import RewardDict
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from numpy_ringbuffer import RingBuffer


class RejoinDistanceChangeRewardValidator(RewardFuncBaseValidator):
    """
    scale: Scalar value to adjust magnitude of the reward
    """

    scale: float


class RejoinDistanceChangeReward(RewardFuncBase):
    """
    This RewardFuncBase extension is responsible for calculating the reward associated with a change in agent position.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)

    @classmethod
    def get_validator(cls):
        """
        Method to return class's Validator.
        """
        return RejoinDistanceChangeRewardValidator

    def __call__(
        self,
        observation: OrderedDict,
        action,
        next_observation: OrderedDict,
        state: StateDict,
        next_state: StateDict,
        observation_space: StateDict,
        observation_units: StateDict,
    ) -> RewardDict:
        """
        This method calculates the current position of the agent and compares it to the previous position. The
        difference is used to return a proportional reward.

        Parameters
        ----------
        observation : OrderedDict
            The observations available to the agent from the previous state.
        action :
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
            The agent's reward for their change in distance.
        """

        reward = RewardDict()
        val = 0

        # question, less brittle way to refer to platforms?
        position = next_state.sim_platforms[0].position
        distance = np.linalg.norm(position)
        self._dist_buffer.append(distance)

        # TODO intialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            val = self.config.scale * (self._dist_buffer[1] - self._dist_buffer[0])

        reward[self.config.agent_name] = val

        return reward
