"""
This module implements the Reward Functions and Reward Validators specific to the docking task.
"""
from collections import OrderedDict

import numpy as np
from act3_rl_core.libraries.environment_dict import RewardDict
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name
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
        reward : RewardDict
            The agent's reward for their change in distance.
        """

        reward = RewardDict()
        val = 0

        deputy = get_platform_by_name(next_state, self.config.agent_name)
        position = deputy.position

        distance = np.linalg.norm(position)
        self._dist_buffer.append(distance)

        # TODO intialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            val = self.config.scale * (self._dist_buffer[1] - self._dist_buffer[0])

        reward[self.config.agent_name] = val

        return reward


class DockingSuccessRewardValidator(RewardFuncBaseValidator):
    """
    scale: Scalar value to adjust magnitude of the reward
    timeout: The max time for an episode         TODO: [optional]
    docking_region_radius: The radius of the docking region in meters
    """
    scale: float
    timeout: float
    docking_region_radius: float
    max_vel_constraint: float


class DockingSuccessReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward associated with a successful docking.
    """

    @classmethod
    def get_validator(cls):
        """
        Method to return class's Validator.
        """
        return DockingSuccessRewardValidator

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
        This method determines if the agent has succeeded and returns an appropriate reward.

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
        reward : RewardDict
            The agent's reward for their change in distance.
        """

        reward = RewardDict()
        value = 0

        deputy = get_platform_by_name(next_state, self.config.agent_name)

        position = deputy.position
        sim_time = deputy.sim_time
        velocity_vector = deputy.velocity

        origin = np.array([0, 0, 0])
        docking_region_radius = self.config.docking_region_radius
        velocity = np.linalg.norm(velocity_vector)

        radial_distance = np.linalg.norm(np.array(position) - origin)
        in_docking = radial_distance <= docking_region_radius

        if in_docking and velocity < self.config.max_vel_constraint:
            value = self.config.scale
            if self.config.timeout:
                # Add time reward component, if timeout specified
                value += 1 - (sim_time / self.config.timeout)

        reward[self.config.agent_name] = value
        return reward


class DockingFailureRewardValidator(RewardFuncBaseValidator):
    """
    timeout_reward: Reward (penalty) associated with a failure by exceeding max episode time
    distance_reward: Reward (penalty) associated with a failure by exceeding max distance
    crash_reward: Reward (penalty) associated with a failure by crashing
    timeout: Max episode time
    max_goal_distance: Max distance from the goal
    docking_region_radius: Radius of the docking region in meters
    max_vel_constraint: The max velocity allowed for a successful dock in the docking region in meters per second
    """
    timeout_reward: float
    distance_reward: float
    crash_reward: float
    timeout: float
    max_goal_distance: float
    docking_region_radius: float
    max_vel_constraint: float


class DockingFailureReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward (penalty) associated with a failed episode.
    """

    @classmethod
    def get_validator(cls):
        """
        Method to return class's Validator.
        """
        return DockingFailureRewardValidator

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
        This method determines if the agent had failed the task and allocates an appropriate reward.

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
        reward : RewardDict
            The agent's reward for their change in distance.
        """

        reward = RewardDict()
        value = 0

        deputy = get_platform_by_name(next_state, self.config.agent_name)

        position = deputy.position
        sim_time = deputy.sim_time
        velocity_vector = deputy.velocity

        distance = np.linalg.norm(position)
        velocity = np.linalg.norm(velocity_vector)

        # TODO: update to chief location when multiple platforms enabled
        origin = np.array([0, 0, 0])
        radial_distance = np.linalg.norm(np.array(position) - origin)
        in_docking = radial_distance <= self.config.docking_region_radius

        if sim_time > self.config.timeout:
            # episode reached max time
            value = self.config.timeout_reward
        elif distance >= self.config.max_goal_distance:
            # agent exceeded max distance from goal
            value = self.config.distance_reward
        elif in_docking and velocity >= self.config.max_vel_constraint:
            # agent exceeded velocity constraint within docking region
            value = self.config.crash_reward

        reward[self.config.agent_name] = value
        return reward
