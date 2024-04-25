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
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.common_platform_utils import get_platform_by_name
from numpy_ringbuffer import RingBuffer

from safe_autonomy_sims.utils import get_relative_position, get_relative_velocity, max_vel_violation


class DockingTimeRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the DockingTimeRewardValidator Reward Function.

    Attributes
    ----------
    scale : float
        Scalar value to adjust magnitude of the reward.
    step_size : float
        The size of one simulation step (sec).
    """
    scale: float
    step_size: float


# Change name to TimeReward?
class DockingTimeReward(RewardFuncBase):
    """
    This reward function allocates reward based on the number of time steps (BaseSimulator.step() method calls) that
    have passed in a single episode.
    """

    def __init__(self, **kwargs):
        self.config: DockingTimeRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        return DockingTimeRewardValidator

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

        reward = self.config.scale * self.config.step_size

        return reward


class DockingDistanceChangeRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the DockingTimeRewardValidator Reward Function.

    Attributes
    ----------
    scale : float
        Scalar value to adjust magnitude of the reward.
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    """
    scale: float
    reference_position_sensor_name: str = "reference_position"


class DockingDistanceChangeReward(RewardFuncBase):
    """
    This RewardFuncBase extension is responsible for calculating the reward associated with a change in agent position.

    """

    def __init__(self, **kwargs):
        self.config: DockingDistanceChangeRewardValidator
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)

    @staticmethod
    def get_validator():
        return DockingDistanceChangeRewardValidator

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

        # get relative dist
        # Assumes one platfrom per agent!
        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)

        self._dist_buffer.append(distance)

        reward = 0.0

        # TODO intialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            reward = self.config.scale * (self._dist_buffer[1] - self._dist_buffer[0])

        return reward


class DockingDistanceExponentialChangeRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the DockingDistanceExponentialChangeReward Reward Function.

    Attributes
    ----------
    c : float
        Scale factor of exponential distance function
    a : float
        Exponential coefficient of exponential distance function. Do not specify if `pivot` is specified.
    pivot : float
        Divisor of the exponential coefficient setting the scale of input values. Do not specify if `a` is specified.
    pivot_ratio : float
        logarithmic scale of exponential scaling coefficient. Do not specify if `a` is specified.
        determines portion of cumulative reward between distances <pivot and >pivot.
    scale : float
        Reward scaling value.
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    """
    c: float = 2.0
    a: float = math.inf
    pivot: float = math.inf
    pivot_ratio: float = 2.0
    scale: float = 1.0
    reference_position_sensor_name: str = "reference_position"


class DockingDistanceExponentialChangeReward(RewardFuncBase):
    """
    Calculates an exponential reward based on the change in distance of the agent.
    Reward is based on the multiplicative scale factor to the exponential potential function:
        reward = psi(||r(t+1)||) - psi(||r(t)||)
        where psi(x) = ce^(-a*x)

    We choose a to be
        a = ln(pivot_ratio)/pivot
    This means that the reward accumulated going from distance r=pivot to r=0 is R=c(1 - 1/pivot_ratio)
        and the reward accumulated going from distance r=inf to r=pivot is c(1/pivot_ratio)

    Alternatively a can be set directly instead of using pivot and pivot ratio
    """

    def __init__(self, **kwargs):
        self.config: DockingDistanceExponentialChangeRewardValidator
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)

        assert not (self.config.a == math.inf and self.config.pivot == math.inf), "Both 'a' and 'pivot' cannot be specified."
        assert self.config.a != math.inf or self.config.pivot != math.inf, "Either 'a' or 'pivot' must be specified."

        if self.config.a != math.inf:
            self.a = self.config.a
        else:
            self.a = math.log(self.config.pivot_ratio) / self.config.pivot

        self.c = self.config.c
        self.scale = self.config.scale

    @staticmethod
    def get_validator():
        return DockingDistanceExponentialChangeRewardValidator

    @property
    def prev_dist(self):
        """
        The previous distance of the agent from the target.

        Returns
        -------
        float
            The previous distance of the agent from the target.
        """
        return self._dist_buffer[0]

    @property
    def curr_dist(self):
        """
        The current distance of the agent from the target.

        Returns
        -------
        float
            The current distance of the agent from the target.
        """
        return self._dist_buffer[1]

    def update_dist(self, dist):
        """
        Store the current distance from the agent to the target.

        Parameters
        ----------
        dist: float
            The current distance of the agent from the target.
        """
        self._dist_buffer.append(dist)

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

        # get relative dist
        # assumes one platform per agent!
        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)

        self.update_dist(distance)

        reward = 0.0

        # TODO initialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            reward = self.c * (math.exp(-self.a * self.curr_dist) - math.exp(-self.a * self.prev_dist))
            reward = self.scale * reward

        return reward


class DockingDeltaVRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the DockingDeltaVReward Reward Function.

    Attributes
    ----------
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


class DockingDeltaVReward(RewardFuncBase):
    """
    Calculates reward based on the agent's fuel consumption measured in delta-v.
    """

    def __init__(self, **kwargs):
        self.config: DockingDeltaVRewardValidator
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
        deputy = get_platform_by_name(state, self.config.platform_names[0])
        control_vec = deputy.get_applied_action()
        d_v = np.sum(np.abs(control_vec.m)) / self.mass * self.step_size
        return d_v

    @staticmethod
    def get_validator():
        return DockingDeltaVRewardValidator

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

        reward = self.scale * self.delta_v(next_state) + self.bias
        return reward


class DockingVelocityConstraintRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the DockingVelocityConstraintReward Reward function.

    Attributes
    ----------
    scale : float
        Scalar value to adjust magnitude of the reward.
    bias : float
        Y intercept of the linear region of the velocity constraint function.
    step_size : float
        The size of one simulation step (sec).
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
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    reference_velocity_sensor_name: str
        The name of the sensor responsible for returning the relative velocity of a reference entity.
    """
    scale: float
    bias: float = 0.0
    step_size: float = 1.0
    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027
    lower_bound: bool = False
    reference_position_sensor_name: str = "reference_position"
    reference_velocity_sensor_name: str = "reference_velocity"


class DockingVelocityConstraintReward(RewardFuncBase):
    """
    Calculates reward based on agent's violation of the velocity constraint.
    """

    def __init__(self, **kwargs):
        self.config: DockingVelocityConstraintRewardValidator
        super().__init__(**kwargs)
        self.bias = self.config.bias
        self.scale = self.config.scale
        self.step_size = self.config.step_size
        self.velocity_threshold = self.config.velocity_threshold
        self.threshold_distance = self.config.threshold_distance
        self.slope = self.config.slope
        self.mean_motion = self.config.mean_motion
        self.lower_bound = self.config.lower_bound

        self.agent_name = self.config.agent_name

    @staticmethod
    def get_validator():
        return DockingVelocityConstraintRewardValidator

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

        # Get relative position and velocity
        # Assumes one platfrom per agent!
        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        relative_velocity = get_relative_velocity(platform, self.config.reference_velocity_sensor_name)

        violated, violation = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        reward = 0.0

        if violated:
            reward = self.scale * violation + self.bias

        return reward


class DockingSuccessRewardValidator(RewardFuncBaseValidator):
    """
    A configuration validator for the DockingSuccessRewardValidator Reward Function.

    Attributes
    ----------
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
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    reference_velocity_sensor_name: str
        The name of the sensor responsible for returning the relative velocity of a reference entity.
    """
    scale: float
    timeout: float
    docking_region_radius: float
    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027
    lower_bound: bool = False
    reference_position_sensor_name: str = "reference_position"
    reference_velocity_sensor_name: str = "reference_velocity"


class DockingSuccessReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward associated with a successful docking.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingSuccessRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
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
    ) -> float:

        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        sim_time = platform.sim_time

        # Get relative position and velocity
        # Assumes one platfrom per agent!
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        relative_velocity = get_relative_velocity(platform, self.config.reference_velocity_sensor_name)

        distance = np.linalg.norm(relative_position)

        in_docking = distance <= self.config.docking_region_radius

        violated, _ = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        reward = 0.0

        if in_docking and not violated:
            reward = self.config.scale
            if self.config.timeout:
                # Add time reward component, if timeout specified
                reward += 1 - (sim_time / self.config.timeout)

        return reward


class DockingFailureRewardValidator(RewardFuncBaseValidator):
    """
    A configuraion validator for the DockingFailureRewardValidator Reward Function.

    Attributes
    ----------
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
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    reference_velocity_sensor_name: str
        The name of the sensor responsible for returning the relative velocity of a reference entity.
    """
    timeout_reward: float
    distance_reward: float
    crash_reward: float
    timeout: float
    max_goal_distance: float
    docking_region_radius: float
    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027
    lower_bound: bool = False
    reference_position_sensor_name: str = "reference_position"
    reference_velocity_sensor_name: str = "reference_velocity"


class DockingFailureReward(RewardFuncBase):
    """
    This Reward Function is responsible for calculating the reward (penalty) associated with a failed episode.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingFailureRewardValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
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
    ) -> float:

        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        sim_time = platform.sim_time

        # TODO: update to chief location when multiple platforms enabled
        # Get relative position and velocity
        # Assumes one platfrom per agent!
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        relative_velocity = get_relative_velocity(platform, self.config.reference_velocity_sensor_name)
        distance = np.linalg.norm(relative_position)

        in_docking = distance <= self.config.docking_region_radius

        violated, _ = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        reward = 0.0

        if sim_time >= self.config.timeout:
            # episode reached max time
            reward = self.config.timeout_reward
        elif distance >= self.config.max_goal_distance:
            # agent exceeded max distance from goal
            reward = self.config.distance_reward
        elif in_docking and violated:
            # agent exceeded velocity constraint within docking region
            reward = self.config.crash_reward

        return reward
