"""
This module implements the Reward Functions and Reward Validators specific to the docking task.
"""
from collections import OrderedDict

import numpy as np
from act3_rl_core.libraries.environment_dict import RewardDict
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from numpy_ringbuffer import RingBuffer


class CWHDistanceChangeRewardValidator(RewardFuncBaseValidator):
    """
    scale: Scalar value to adjust magnitude of the reward
    """

    scale: float


class CWHDistanceChangeReward(RewardFuncBase):
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
        return CWHDistanceChangeRewardValidator

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


########################################################################################################################
class TimeRewardValidator(RewardFuncBaseValidator):
    """
    scale: Scalar value to adjust magnitude of the reward
    """
    scale: float


class TimeRewardFunction(RewardFuncBase):
    def __init__(self, name=None, reward=None):
        super().__init__(name=name, reward=reward)

    @classmethod
    def get_validator(cls):
        """
        Method to return class's Validator.
        """
        return TimeRewardValidator

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




    def reset(self, sim_state):
        super().reset(sim_state)
        self.previous_step_size = 0

    def _increment(self, sim_state, step_size):
        # update state
        self.previous_step_size = step_size

    def _process(self, sim_state):
        step_reward = self.previous_step_size * self.reward
        return step_reward






class DistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, name=None, deputy=None, docking_region=None, reward=None):
        super().__init__(name=name, reward=reward)
        self.deputy = deputy
        self.docking_region = docking_region

    def reset(self, sim_state):
        super().reset(sim_state)
        self.cur_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])
        self.prev_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])

    def _increment(self, sim_state, step_size):
        self.prev_distance = self.cur_distance
        self.cur_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])

    def _process(self, sim_state):
        dist_change = self.cur_distance - self.prev_distance
        step_reward = dist_change * self.reward
        return step_reward


class DistanceChangeZRewardProcessor(RewardProcessor):
    def __init__(self, name=None, deputy=None, docking_region=None, reward=None):
        super().__init__(name=name, reward=reward)
        self.deputy = deputy
        self.docking_region = docking_region

    def reset(self, sim_state):
        super().reset(sim_state)
        self.prev_z_distance = 0
        self.cur_z_distance = abs(sim_state.env_objs[self.deputy].z - sim_state.env_objs[self.docking_region].z)

    def _increment(self, sim_state, step_size):
        self.prev_z_distance = self.cur_z_distance
        self.cur_z_distance = abs(sim_state.env_objs[self.deputy].z - sim_state.env_objs[self.docking_region].z)

    def _process(self, sim_state):
        dist_z_change = self.cur_z_distance - self.prev_z_distance
        step_reward = dist_z_change * self.reward
        return step_reward


class SuccessRewardProcessor(RewardProcessor):
    def __init__(self, name=None, success_status=None, reward=None, timeout=None):
        super().__init__(name=name, reward=reward)
        self.success_status = success_status
        self.timeout = timeout

    def _increment(self, sim_state, step_size):
        # reward derived straight from status dict, therefore no state machine necessary
        pass

    def _process(self, sim_state):
        step_reward = 0
        if sim_state.status[self.success_status]:
            step_reward = self.reward
            if self.timeout is not None:
                step_reward += 1 - (sim_state.time_elapsed / self.timeout)
        return step_reward


class FailureRewardProcessor(RewardProcessor):
    def __init__(self, name=None, failure_status=None, reward=None):
        super().__init__(name=name, reward=reward)
        self.failure_status = failure_status

    def _increment(self, sim_state, step_size):
        # reward derived straight from status dict, therefore no state machine necessary
        pass

    def _process(self, sim_state):
        step_reward = 0
        if sim_state.status[self.failure_status]:
            step_reward = self.reward[sim_state.status[self.failure_status]]
        return step_reward

