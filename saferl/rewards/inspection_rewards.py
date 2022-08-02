"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements the Reward Functions and Reward Validators specific to the docking task.
"""
import math
from collections import OrderedDict

import numpy as np
from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.common_platform_utils import get_platform_by_name
from numpy_ringbuffer import RingBuffer

from saferl.utils import max_vel_violation


class ObservedPointsExponentialChangeRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the DockingDistanceExponentialChangeReward Reward Function.

    c : float
        Scale factor of exponential distance function
    a : float
        Exponential coefficient of exponential distance function. Do not specify if `pivot` is defined.
    pivot : float
        Exponential scaling coefficient of exponential distance function. Do not specify if `a` is defined.
    pivot_ratio : float
        Exponential scaling coefficient of exponential distance function. Do not specify if `a` is defined.
    scale : float
        Reward scaling value.
    """


class ObservedPointsExponentialChangeReward(RewardFuncBase):
    """
    Calculates an exponential reward based on the change in distance of the agent.
    Reward is based on the multiplicative scale factor to the exponential potential function:
        reward = ce^(ln(pivot_ratio)/pivot * x)


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

    This method calculates the current position of the agent and compares it to the previous position. The
    difference is used to return an exponential reward.

    Parameters
    ----------
    observation : OrderedDict
        The observations available to the agent from the previous state.
    action : np.ndarray
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
        The units corresponding to keys in the observation_space.

    Returns
    -------
    reward : RewardDict
        The agent's reward for their change in distance.
    """

    def __init__(self, **kwargs):
        self.config: ObservedPointsExponentialChangeRewardValidator
        super().__init__(**kwargs)

        self.num_points_inspected = 0

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return ObservedPointsExponentialChangeRewardValidator


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

        reward = RewardDict()

        total_points_found = 0

        new_points = next_state.points
        for point in new_points:
            if new_points[point]:
                total_points_found +=1
        num_new_points = total_points_found - self.num_points_inspected
        reward_value = 0.0

        if num_new_points:
            self.num_points_inspected = total_points_found
            reward_value = total_points_found / len(new_points)

        reward[self.config.agent_name] = num_new_points #+ num_new_points * (total_points_found / len(old_points))
        return reward


class InspectionDeltaVRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the DockingDeltaVReward Reward Function.

    scale : float
        A scalar value applied to reward.
    bias : float
        A bias value added to the reward.
    step_size : float
        Size of a single simulation step.
    mass : float
        The mass (kg) of the agent's spacecraft.
    """
    scale: float
    bias: float = 0.0
    step_size: float = 1.0
    mass: float


class InspectionDeltaVReward(RewardFuncBase):
    """
    Calculates reward based on the agent's fuel consumption measured in delta-v.


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

    This method retrieves the current thrust control applied by the agent (delta v), which is used to calculate and
    return a proportional reward.

    Parameters
    ----------
    observation : OrderedDict
        The observations available to the agent from the previous state.
    action : np.ndarray
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
        The units corresponding to keys in the observation_space.

    Returns
    -------
    reward : RewardDict
        The agent's reward for their change in distance.
    """

    def __init__(self, **kwargs):
        self.config: InspectionDeltaVRewardValidator
        super().__init__(**kwargs)
        self.bias = self.config.bias
        self.scale = self.config.scale
        self.mass = self.config.mass
        self.step_size = self.config.step_size

    def delta_v(self, state):
        """
        Get change in agent's velocity from the current state.

        Parameters
        ----------
        state: StateDict
            The current state of the system.

        Returns
        -------
        d_v: float
            The agent's change in velocity
        """
        deputy = get_platform_by_name(state, self.config.agent_name)
        control_vec = deputy.get_applied_action()
        d_v = np.sum(np.abs(control_vec)) / self.mass * self.step_size
        return d_v

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return InspectionDeltaVRewardValidator

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
        reward = RewardDict()
        val = self.scale * self.delta_v(next_state) + self.bias
        reward[self.config.agent_name] = val
        return reward


class InspectionSuccessRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the DockingSuccessRewardValidator Reward Function.

    scale : float
        Scalar value to adjust magnitude of the reward.
    timeout : float
        The max time for an episode.
    docking_region_radius : float
        The radius of the docking region in meters.
    velocity_threshold : float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the docking region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    lower_bound : bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    """
    scale: float
    timeout: float
    docking_region_radius: float

class InspectionSuccessReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward associated with a successful docking.


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

    This method determines if the agent has succeeded and returns an appropriate reward.

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
        The units corresponding to keys in the observation_space?

    Returns
    -------
    reward : RewardDict
        The agent's reward for their change in distance.
    """

    def __init__(self, **kwargs) -> None:
        self.config: InspectionSuccessRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return InspectionSuccessRewardValidator

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

        reward = RewardDict()
        value = 0.0

        deputy = get_platform_by_name(next_state, self.config.agent_name)

        sim_time = deputy.sim_time

        all_inspected = not (False in next_state.points.values())

        if all_inspected:
            value = self.config.scale
            if self.config.timeout:
                # Add time reward component, if timeout specified
                value += 1 - (sim_time / self.config.timeout)

        reward[self.config.agent_name] = value
        return reward


class InspectionFailureRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the DockingFailureRewardValidator Reward Function.

    timeout_reward : float
        Reward (penalty) associated with a failure by exceeding max episode time.
    distance_reward : float
        Reward (penalty) associated with a failure by exceeding max distance.
    crash_reward : float
        Reward (penalty) associated with a failure by crashing.
    timeout : float
        Max episode time.
    max_goal_distance : float
        Max distance from the goal.
    docking_region_radius : float
        Radius of the docking region in meters.
    velocity_threshold : float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the docking region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s, by default 0.001027.
    lower_bound : bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    """
    timeout_reward: float
    distance_reward: float
    crash_reward: float
    timeout: float
    min_goal_points: int


class InspectionFailureReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward (penalty) associated with a failed episode.


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

    This method determines if the agent had failed the task and allocates an appropriate reward.

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
        The units corresponding to keys in the observation_space?

    Returns
    -------
    reward : RewardDict
        The agent's reward for their change in distance.
    """

    def __init__(self, **kwargs) -> None:
        self.config: InspectionFailureRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return InspectionFailureRewardValidator

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

        reward = RewardDict()
        value = 0.0

        deputy = get_platform_by_name(next_state, self.config.agent_name)

        sim_time = deputy.sim_time

        if sim_time >= self.config.timeout:
            # episode reached max time
            value = self.config.timeout_reward

        #elif  violated:
            # agent exceeded velocity constraint within docking region
        #    value = self.config.crash_reward

        reward[self.config.agent_name] = value
        return reward
