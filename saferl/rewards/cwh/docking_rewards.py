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

from saferl.utils import get_relative_position, max_vel_violation


class DockingTimeRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the DockingTimeRewardValidator Reward Function.

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

    This method allocates reward based on the step size of the simulation.

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
    reward : RewardDict
        The agent's reward for the change in time.
    """

    def __init__(self, **kwargs):
        self.config: DockingTimeRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
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
    ) -> RewardDict:

        reward = RewardDict()
        reward[self.config.agent_name] = self.config.scale * self.config.step_size

        return reward


class DockingDistanceChangeRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the DockingTimeRewardValidator Reward Function.

    scale : float
        Scalar value to adjust magnitude of the reward.
    """
    scale: float
    reference_position_sensor_name: str = "reference_position"


class DockingDistanceChangeReward(RewardFuncBase):
    """
    This RewardFuncBase extension is responsible for calculating the reward associated with a change in agent position.


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
    difference is used to return a proportional reward.

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
        The units corresponding to keys in the observation_space.

    Returns
    -------
    reward : RewardDict
        The agent's reward for their change in distance.
    """

    def __init__(self, **kwargs):
        self.config: DockingDistanceChangeRewardValidator
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
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
    ) -> RewardDict:

        reward = RewardDict()
        val = 0

        # get relative dist
        # Assumes one platfrom per agent!
        relative_position = get_relative_position(next_state, self.config.platform_names[0], self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)

        self._dist_buffer.append(distance)

        # TODO intialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            val = self.config.scale * (self._dist_buffer[1] - self._dist_buffer[0])

        reward[self.config.agent_name] = val

        return reward


class DockingDistanceExponentialChangeRewardValidator(RewardFuncBaseValidator):
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

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
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

        Returns
        -------
        None
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
    ) -> RewardDict:

        reward = RewardDict()
        val = 0

        # get relative dist
        # assumes one platform per agent!
        relative_position = get_relative_position(next_state, self.config.platform_names[0], self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)

        self.update_dist(distance)

        # TODO initialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            val = self.c * (math.exp(-self.a * self.curr_dist) - math.exp(-self.a * self.prev_dist))
            val = self.scale * val

        reward[self.config.agent_name] = val
        return reward


class DockingDeltaVRewardValidator(RewardFuncBaseValidator):
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


class DockingDeltaVReward(RewardFuncBase):
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
        d_v = np.sum(np.abs(control_vec)) / self.mass * self.step_size
        return d_v

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
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
    ) -> RewardDict:
        reward = RewardDict()
        val = self.scale * self.delta_v(next_state) + self.bias
        reward[self.config.agent_name] = val
        return reward


class DockingVelocityConstraintRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the DockingVelocityConstraintReward Reward function.

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


class DockingVelocityConstraintReward(RewardFuncBase):
    """
    Calculates reward based on agent's violation of the velocity constraint.


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

    This method calculates the current velocity constraint based on the relative distance from the chief.
    It compares the velocity constraint with the deputy's current velocity. If the velocity constraint is
    exceeded, the function returns a penalty for the agent.

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

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
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
    ) -> RewardDict:
        reward = RewardDict()

        # Get relative position and velocity
        # Assumes one platfrom per agent!
        relative_position = get_relative_position(next_state, self.config.platform_names[0], self.config.reference_position_sensor_name)
        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_velocity = platform.velocity  # docking region assumed stationary

        violated, violation = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        if violated:
            val = self.scale * violation + self.bias
        else:
            val = 0
        reward[self.config.agent_name] = val
        return reward


class DockingSuccessRewardValidator(RewardFuncBaseValidator):
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
    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027
    lower_bound: bool = False
    reference_position_sensor_name: str = "reference_position"


class DockingSuccessReward(RewardFuncBase):
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
        self.config: DockingSuccessRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
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

        reward = RewardDict()
        value = 0.0

        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        relative_velocity = platform.velocity  # docking region assumed stationary
        sim_time = platform.sim_time

        # Get relative position and velocity
        # Assumes one platfrom per agent!
        relative_position = get_relative_position(next_state, self.config.platform_names[0], self.config.reference_position_sensor_name)
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

        if in_docking and not violated:
            value = self.config.scale
            if self.config.timeout:
                # Add time reward component, if timeout specified
                value += 1 - (sim_time / self.config.timeout)

        reward[self.config.agent_name] = value
        return reward


class DockingFailureRewardValidator(RewardFuncBaseValidator):
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
    max_goal_distance: float
    docking_region_radius: float
    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027
    lower_bound: bool = False
    reference_position_sensor_name: str = "reference_position"


class DockingFailureReward(RewardFuncBase):
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
        self.config: DockingFailureRewardValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
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

        reward = RewardDict()
        value = 0.0

        platform = get_platform_by_name(next_state, self.config.platform_names[0])
        sim_time = platform.sim_time

        # TODO: update to chief location when multiple platforms enabled
        # Get relative position and velocity
        # Assumes one platfrom per agent!
        relative_position = get_relative_position(next_state, self.config.platform_names[0], self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)
        relative_velocity = platform.velocity  # docking region assumed stationary

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

        if sim_time >= self.config.timeout:
            # episode reached max time
            value = self.config.timeout_reward
        elif distance >= self.config.max_goal_distance:
            # agent exceeded max distance from goal
            value = self.config.distance_reward
        elif in_docking and violated:
            # agent exceeded velocity constraint within docking region
            value = self.config.crash_reward

        reward[self.config.agent_name] = value
        return reward
