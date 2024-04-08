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
import typing
from collections import OrderedDict

import numpy as np
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.common_platform_utils import get_platform_by_name

from safe_autonomy_sims.utils import get_closest_fft_distance, get_relative_position


class ObservedPointsRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for ObservedPointsReward.

    Attributes
    ----------
    scale : float
        A scalar value applied to reward.
    inspection_entity_name: str
        The name of the entity under inspection.
    weighted_priority : bool
        True to base reward off of point weights, False to base reward off of number of points
    """
    scale: float
    inspection_entity_name: str = "chief"
    weighted_priority: bool = False


class ObservedPointsReward(RewardFuncBase):
    """
    Calculates reward based on the number of new points inspected by the agent.
    """

    def __init__(self, **kwargs):
        self.config: ObservedPointsRewardValidator
        super().__init__(**kwargs)
        self.previous_num_points_inspected = 0
        if self.config.weighted_priority:
            self.previous_weight_inspected = 0.

    @staticmethod
    def get_validator():
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
    ) -> float:

        reward = 0.0

        inspection_points = next_state.inspection_points_map[self.config.inspection_entity_name]
        current_num_points_inspected = inspection_points.get_num_points_inspected()
        num_new_points = current_num_points_inspected - self.previous_num_points_inspected
        self.previous_num_points_inspected = current_num_points_inspected

        if self.config.weighted_priority:
            current_weight_inspected = inspection_points.get_total_weight_inspected()
            new_weight = current_weight_inspected - self.previous_weight_inspected
            self.previous_weight_inspected = current_weight_inspected
            reward = self.config.scale * new_weight
        else:
            reward = self.config.scale * num_new_points

        return reward


class ChiefDistanceRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for ChiefDistanceReward.

    Attributes
    ----------
    scale : float
        A scalar value applied to reward.
    punishment_reward : float
        Negative scalar associated with the distance done condition.
    max_dist : float
        Maximum allowable distance to chief to recieve reward.
    threshold_dist : float
        Threshold distance
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    """

    scale: float
    punishment_reward: float
    threshold_dist: float
    max_dist: float
    reference_position_sensor_name: str = "reference_position"


class ChiefDistanceReward(RewardFuncBase):
    """
    Calculates reward based on the distance from chief.
    """

    def __init__(self, **kwargs):
        self.config: ChiefDistanceRewardValidator
        super().__init__(**kwargs)
        self.dist_prev = 0.

    @staticmethod
    def get_validator():
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
    ) -> float:

        reward = 0.0

        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        dist = np.linalg.norm(relative_position)

        # Soft constraint
        if dist >= self.config.threshold_dist:
            reward = -self.config.scale * (np.sign(dist - self.dist_prev))

        if dist >= self.config.max_dist:
            reward = self.config.punishment_reward

        self.dist_prev = dist
        return reward


class InspectionDeltaVRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for DockingDeltaVReward

    Attributes
    ----------
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
    constant_scale : float
        scalar penalty multiplied by delta-v.
        If None, increasing scale value is taken from simulator
    """
    bias: float = 0.0
    step_size: float
    mass: float
    mode: str = 'scale'
    rate: float = 0.0005
    constant_scale: typing.Union[None, float] = None


class InspectionDeltaVReward(RewardFuncBase):
    """
    Calculates reward based on the agent's fuel consumption measured in delta-v.
    """

    def __init__(self, **kwargs):
        self.config: InspectionDeltaVRewardValidator
        super().__init__(**kwargs)
        self.bias = self.config.bias
        self.mass = self.config.mass
        self.step_size = self.config.step_size
        self.mode = self.config.mode
        self.rate = self.config.rate
        self.constant_scale = self.config.constant_scale
        self.scale = 0.0

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
        deputy = get_platform_by_name(state, self.config.platform_names[0])  # TODO: assuming 1:1 agent:platform
        control_vec = deputy.get_applied_action().m
        d_v = np.sum(np.abs(control_vec)) / self.mass * self.step_size
        return d_v

    def linear_scalar(self, time):
        """
        Delta-v penalty increases linearly with training iteration
        """
        inc_scalar = time * self.rate * self.scale
        return inc_scalar

    @staticmethod
    def get_validator():
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
    ) -> float:

        if self.constant_scale is None:
            self.scale = state.delta_v_scale
        else:
            self.scale = self.constant_scale
        reward = 0.0
        if self.mode == "scale":
            reward = self.scale * self.delta_v(next_state) + self.bias
        elif self.mode == "linear_increasing":
            reward = self.linear_scalar(state.sim_time) * self.delta_v(next_state) + self.bias
        else:
            raise ValueError('mode must be either "scale" or "linear_increasing"')

        return reward


class InspectionSuccessRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the InspectionSuccessReward Reward Function.

    Attributes
    ----------
    scale : float
        Scalar value to adjust magnitude of the reward.
    inspection_entity_name: str
        The name of the entity under inspection.
    weight_threshold : float
        Points score value indicating success.
        By default None, so success occurs when all points are inspected
    """
    scale: float
    inspection_entity_name: str = "chief"
    weight_threshold: typing.Union[float, None] = None


class InspectionSuccessReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward associated with a successful inspection.
    """

    def __init__(self, **kwargs) -> None:
        self.config: InspectionSuccessRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
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
    ) -> float:

        reward = 0.0

        if self.config.weight_threshold is not None:
            weight = next_state.inspection_points_map[self.config.inspection_entity_name].get_total_weight_inspected()
            if weight >= self.config.weight_threshold:
                reward = self.config.scale

        else:
            inspection_points = next_state.inspection_points_map[self.config.inspection_entity_name]
            all_inspected = all(inspection_points.points_inspected_dict.values())

            if all_inspected:
                reward = self.config.scale

        return reward


class SafeInspectionSuccessRewardValidator(InspectionSuccessRewardValidator):
    """
    A configuration validator for SafeInspectionSuccessReward

    Attributes
    ----------
    mean_motion : float
        orbital mean motion in rad/s of current Hill's reference frame
    crash_region_radius : float
        The radius of the crashing region in meters.
    fft_time_step : float
        Time step to compute the FFT trajectory. FFT is computed for 1 orbit.
    crash_scale : float
        Scalar reward value in the event of a future crash
    """
    mean_motion: float
    crash_region_radius: float
    fft_time_step: float = 1
    crash_scale: float


class SafeInspectionSuccessReward(InspectionSuccessReward):
    """
    This Reward Function is responsible for calculating the reward associated with a successful inspection.
    Considers if a Free Flight Trajectory once the episode ends would result in a collision.
    """

    def __init__(self, **kwargs) -> None:
        self.config: SafeInspectionSuccessRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        return SafeInspectionSuccessRewardValidator

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

        reward = super().__call__(observation, action, next_observation, state, next_state, observation_space, observation_units)

        if reward != 0.0:
            name = [p for p in self.config.platform_names if p in self.config.agent_name][0]
            pos = next_state.sim_platforms[name].position
            vel = next_state.sim_platforms[name].velocity
            state = np.concatenate((pos, vel))
            n = self.config.mean_motion
            times = np.arange(0, 2 * np.pi / n, self.config.fft_time_step)
            dist = get_closest_fft_distance(state, self.config.mean_motion, times)
            if dist < self.config.crash_region_radius:
                reward = self.config.crash_scale

        return reward


class InspectionCrashRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the InspectionCollisionRewardValidator Reward Function.

    Attributes
    ----------
    scale : float
        Scalar value to adjust magnitude of the reward.
    crash_region_radius : float
        The radius of the crashing region in meters.
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    """
    scale: float
    crash_region_radius: float
    reference_position_sensor_name: str = "reference_position"


class InspectionCrashReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward (penalty) associated with a collision.
    """

    def __init__(self, **kwargs) -> None:
        self.config: InspectionCrashRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        return InspectionCrashRewardValidator

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

        reward = 0.0

        # Get relatative position + velocity between platform and docking region
        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)

        in_crash_region = distance <= self.config.crash_region_radius

        if in_crash_region:
            reward = self.config.scale

        return reward


class MaxDistanceRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the InspectionCollisionRewardValidator Reward Function.

    Attributes
    ----------
    scale : float
        Scalar value to adjust magnitude of the reward.
    crash_region_radius : float
        The radius of the crashing region in meters.
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    """
    scale: float = 1.0
    max_distance: float
    reference_position_sensor_name: str = "reference_position"


class MaxDistanceReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward (penalty) associated with a collision.
    """

    def __init__(self, **kwargs) -> None:
        self.config: InspectionCrashRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        return MaxDistanceRewardValidator

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

        reward = 0.0

        # Get relatative position + velocity between platform and docking region
        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)

        out_of_bounds = distance > self.config.max_distance

        if out_of_bounds:
            reward = -self.config.scale

        return reward
