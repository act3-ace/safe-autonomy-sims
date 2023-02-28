"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Safe Autonomy Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements the Reward Functions and Reward Validators specific to the inspection task.
"""
from collections import OrderedDict

import numpy as np
from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.common_platform_utils import get_platform_by_name


class ObservedPointsRewardValidator(RewardFuncBaseValidator):
    """
    Configuration validator for ObservedPointsReward.

    scale : float
        A scalar value applied to reward.
    """
    scale: float


class ObservedPointsReward(RewardFuncBase):
    """
    Calculates reward based on the number of new
    points inspected by the agent.

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
        The agent's reward for the number of new points inspected.
    """

    def __init__(self, **kwargs):
        self.config: ObservedPointsRewardValidator
        super().__init__(**kwargs)
        self.num_points_inspected = 0

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return ObservedPointsRewardValidator

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
                total_points_found += 1
        num_new_points = total_points_found - self.num_points_inspected

        self.num_points_inspected += num_new_points

        reward[self.config.agent_name] = self.config.scale * num_new_points
        return reward


class ChiefDistanceRewardValidator(RewardFuncBaseValidator):
    """
    Configuration validator for ChiefDistanceReward.

    scale : float
        A scalar value applied to reward.
    punishment_reward : float
        Negative scalar associated with the distance done condition.
    max_dist : float
        Maximum allowable distance to chief to recieve reward.
    threshold_dist : float
        Threshold distance
    """

    scale: float
    punishment_reward: float
    threshold_dist: float
    max_dist: float


class ChiefDistanceReward(RewardFuncBase):
    """
    Calculates reward based on the distance from chief.

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
        The agent's reward for the number of new points inspected.
    """

    def __init__(self, **kwargs):
        self.config: ChiefDistanceRewardValidator
        super().__init__(**kwargs)
        self.dist_prev = 0.

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return ChiefDistanceRewardValidator

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
        value = 0.

        deputy = get_platform_by_name(next_state, self.config.agent_name)
        position = deputy.position
        dist = np.linalg.norm(np.array(position))

        # Soft constraint
        if dist >= self.config.threshold_dist:
            value = -self.config.scale * (np.sign(dist - self.dist_prev))

        if dist >= self.config.max_dist:
            value = self.config.punishment_reward

        self.dist_prev = dist
        reward[self.config.agent_name] = value
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
    mode : str
        Type of delta-v penalty, either "scale" or "linear_increasing"
    rate : float
        rate at which penalty increases for linear_increasing
    """
    scale: float
    bias: float = 0.0
    step_size: float = 1.0
    mass: float
    mode: str = 'scale'
    rate: float = 0.1


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
    action : OrderedDict
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
        self.mode = self.config.mode
        self.rate = self.config.rate

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

    def linear_scalar(self, time):
        """
        Delta-v penalty increases linearly with training iteration
        """
        inc_scalar = time * self.rate * self.scale
        return inc_scalar

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
        if self.mode == "scale":
            val = self.scale * self.delta_v(next_state) + self.bias
        elif self.mode == "linear_increasing":
            val = self.linear_scalar(state.total_steps) * self.delta_v(next_state) + self.bias
        else:
            raise ValueError('mode must be either "scale" or "linear_increasing"')
        reward[self.config.agent_name] = val
        return reward


class InspectionSuccessRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the InspectionSuccessReward Reward Function.

    scale : float
        Scalar value to adjust magnitude of the reward.
    """
    scale: float


class InspectionSuccessReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward associated with a successful inspection.


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

        all_inspected = False not in next_state.points.values()

        if all_inspected:
            value = self.config.scale

        reward[self.config.agent_name] = value
        return reward


class InspectionCrashOriginRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the InspectionCollisionRewardValidator Reward Function.

    scale : float
        Scalar value to adjust magnitude of the reward.
    crash_region_radius : float
        The radius of the crashing region in meters.
    """
    scale: float
    crash_region_radius: float


class InspectionCrashOriginReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward (penalty) associated with a collision.


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
        self.config: InspectionCrashOriginRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return InspectionCrashOriginRewardValidator

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
        position = deputy.position
        in_crash_region = np.linalg.norm(np.array(position)) <= self.config.crash_region_radius

        if in_crash_region:
            value = self.config.scale

        reward[self.config.agent_name] = value
        return reward
