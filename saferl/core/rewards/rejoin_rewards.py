"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements the Reward Functions and Reward Validators specific to the rejoin task.
"""
import math
import typing
from collections import OrderedDict

import numpy as np
from corl.libraries.environment_dict import RewardDict
from corl.libraries.state_dict import StateDict
from corl.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from corl.simulators.common_platform_utils import get_platform_by_name
from numpy_ringbuffer import RingBuffer

from saferl.core.utils import get_rejoin_region_center, in_rejoin


class RejoinDistanceChangeRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the RejoinDistanceChangeReward Reward Function
    Attributes
    ----------
    radius : float
        size of the radius of the region region
    offset : [float,float,float]
        vector detailing the location of the center of the rejoin region from the aircraft
    lead : str
        name of the lead platform, for later lookup
    reward : float
        reward for accomplishing the task
    """
    radius: float
    offset: typing.List[float]
    lead: str
    reward: float


class RejoinDistanceChangeReward(RewardFuncBase):
    """
    A reward function that provides a reward proportional to the change in distance from the rejoin distance.
    """

    def __init__(self, **kwargs):
        self.config: RejoinDistanceChangeRewardValidator
        super().__init__(**kwargs)
        self._dist_buffer = RingBuffer(capacity=2, dtype=float)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return RejoinDistanceChangeRewardValidator

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
        """
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
            The units corresponding to values in the observation_space?

        Returns
        -------
        reward : RewardDict
            The agent's reward for their change in distance.
        """

        reward = RewardDict()
        value = 0.0

        wingman = get_platform_by_name(next_state, self.config.agent_name)
        lead = get_platform_by_name(next_state, self.config.lead)

        in_rejoin_region, distance = in_rejoin(wingman=wingman, lead=lead, radius=self.config.radius, offset=self.config.offset)
        self.update_dist(distance)

        if not in_rejoin_region and len(self._dist_buffer) == 2:
            distance_change = self.curr_dist - self.prev_dist
            value = self.config.reward * distance_change

        reward[self.config.agent_name] = value
        return reward


class RejoinDistanceExponentialChangeRewardValidator(RewardFuncBaseValidator):
    """
    TODO: Get the descriptions of these values
    """
    lead: str
    offset: typing.List[float]
    c: float = 2.0
    a: float = math.inf
    pivot: typing.Union[float, int] = math.inf
    pivot_ratio: typing.Union[float, int] = 2.0
    scale: typing.Union[float, int] = 1.0


class RejoinDistanceExponentialChangeReward(RewardFuncBase):
    """
    Calculates an exponential reward based on the change in distance of the agent.
    """

    def __init__(self, **kwargs):
        self.config: RejoinDistanceExponentialChangeRewardValidator
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
        return RejoinDistanceExponentialChangeRewardValidator

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
        val = 0.0

        wingman = get_platform_by_name(next_state, self.config.agent_name)
        wingman_position = wingman.position

        lead = get_platform_by_name(next_state, self.config.lead)
        rejoin_center = get_rejoin_region_center(lead, self.config.offset)

        distance = np.linalg.norm(wingman_position - rejoin_center)
        self.update_dist(distance)

        # TODO initialize distance buffer from initial state
        if len(self._dist_buffer) == 2:
            val = self.c * (math.exp(-self.a * self.curr_dist) - math.exp(-self.a * self.prev_dist))
            val = self.scale * val

        reward[self.config.agent_name] = val
        return reward


class RejoinRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the RejoinReward Reward Function

    Attributes
    ----------
    reward : typing.Union[float, int]
        reward for accomplishing the task
    radius : float
        size of the radius of the region region
    offset : [float,float,float]
        vector detailing the offset of the center of the rejoin region from the lead platform
    lead : str
        name of the lead platform, for later lookup
    refund : bool, optional
        Flag which if true refunds reward if the rejoin region is exited. Default: False.
    """
    reward: typing.Union[float, int]
    radius: typing.Union[float, int]
    offset: typing.List[typing.Union[float, int]]
    lead: str
    refund: bool = False


class RejoinReward(RewardFuncBase):
    """
    A reward function that provides a reward for time spent in the rejoin region.
    """

    def __init__(self, **kwargs) -> None:
        self.config: RejoinRewardValidator
        self.rejoin_prev = False
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return RejoinRewardValidator

    def reset(self):
        self.rejoin_prev = False
        super().reset()

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
        This method returns the reward specified in its configuration.

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
            The agent's reward.
        """

        reward = RewardDict()
        value = 0.0

        wingman = get_platform_by_name(next_state, self.config.agent_name)
        lead = get_platform_by_name(next_state, self.config.lead)

        in_rejoin_region, _ = in_rejoin(wingman=wingman, lead=lead, radius=self.config.radius, offset=self.config.offset)

        if in_rejoin_region and self.rejoin_prev:
            value = self.config.reward
        elif not in_rejoin_region and self.rejoin_prev and self.config.refund:
            value = -1 * self.config.reward

        reward[self.config.agent_name] = value
        self.rejoin_prev = in_rejoin_region

        return reward


class RejoinFirstTimeRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the RejoinReward Reward Function

    Attributes
    ----------
    reward : float
        reward for accomplishing the task
    """
    reward: typing.Union[float, int]
    radius: typing.Union[float, int]
    offset: typing.List[typing.Union[float, int]]
    lead: str


class RejoinFirstTimeReward(RewardFuncBase):
    """
    A reward function that provides a reward for the first time the platform enters the rejoin region.
    """

    def __init__(self, **kwargs) -> None:
        self.config: RejoinFirstTimeRewardValidator
        self.rejoin_first_time = True
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return RejoinFirstTimeRewardValidator

    def reset(self):
        self.rejoin_first_time = True
        super().reset()

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
        This method returns the reward specified in its configuration.

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
            The agent's reward.
        """

        reward = RewardDict()
        value = 0.0

        wingman = get_platform_by_name(next_state, self.config.agent_name)
        lead = get_platform_by_name(next_state, self.config.lead)

        in_rejoin_region, _ = in_rejoin(wingman=wingman, lead=lead, radius=self.config.radius, offset=self.config.offset)

        if in_rejoin_region and self.rejoin_first_time:
            value = self.config.reward
            self.rejoin_first_time = False

        reward[self.config.agent_name] = value

        return reward


class RejoinSuccessRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the SuccessfulRejoinReward function
    Attributes
    ----------
    radius : float
        size of the radius of the region region
    offset : [float,float,float]
        vector detailing the location of the center of the rejoin region from the aircraft
    lead : str
        name of the lead platform, for later lookup
    reward : float
        reward for accomplishing the task
    step_size : float
        size of one single simulation step
    success_time : float
        time wingman must remain in rejoin region to obtain reward
    """
    radius: typing.Union[float, int]
    offset: typing.List[typing.Union[float, int]]
    lead: str
    reward: typing.Union[float, int]
    step_size: typing.Union[float, int]
    success_time: typing.Union[float, int]


class RejoinSuccessReward(RewardFuncBase):
    """
    This function determines the reward for when the wingman successfully stays in the rejoin region for the given duration
    """

    def __init__(self, **kwargs) -> None:
        self.config: RejoinSuccessRewardValidator
        self.rejoin_time = 0.0
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return RejoinSuccessRewardValidator

    def _update_rejoin_time(self, state):
        wingman = get_platform_by_name(state, self.config.agent_name)
        lead = get_platform_by_name(state, self.config.lead)

        in_rejoin_region, _ = in_rejoin(wingman=wingman, lead=lead, radius=self.config.radius, offset=self.config.offset)

        if in_rejoin_region:
            self.rejoin_time += self.config.step_size
        else:
            self.rejoin_time = 0.0

    def reset(self):
        self.rejoin_time = 0.0

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
        This method calculates the agent's reward for succeeding in the rejoin task.

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
            The agent's reward for succeeding in the rejoin task.
        """

        reward = RewardDict()
        value = 0.0

        self._update_rejoin_time(next_state)

        if self.rejoin_time > self.config.success_time:
            value = self.config.reward

        reward[self.config.agent_name] = value
        return reward


class RejoinFailureRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the RejoinFailureReward function
    Attributes
    ----------
    radius : float
        size of the radius of the region region
    offset : [float,float,float]
        vector detailing the location of the center of the rejoin region from the aircraft
    lead : str
        name of the lead platform, for later lookup
    crash_reward : float
        reward for violating the lead's safety margin
    distance_reward : float
        reward for exceeding the max allowable distance from the lead
    timeout_reward : float
        reward for exceeding the max allowable completion time
    leave_rejoin_reward : float
        reward for leaving the rejoin region
    max_time : float
        maximum allowable time
    max_distance : float
        maximum allowable distance from lead
    safety_margin : float
        minimum allowable distance to lead
    """
    radius: typing.Union[float, int]
    offset: typing.List[typing.Union[float, int]]
    lead: str
    crash_reward: typing.Union[float, int]
    distance_reward: typing.Union[float, int]
    timeout_reward: typing.Union[float, int]
    leave_rejoin_reward: typing.Union[float, int]
    max_time: typing.Union[float, int]
    max_distance: typing.Union[float, int]
    safety_margin: typing.Union[float, int]


class RejoinFailureReward(RewardFuncBase):
    """
    This function determines the reward for when the wingman reaches a failure condition.
    TODO: Consider breaking into constituent rewards?
    """

    def __init__(self, **kwargs) -> None:
        self.config: RejoinFailureRewardValidator
        self._rejoin_prev = False
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return RejoinFailureRewardValidator

    def crash(self, state):
        """
        Determine if wingman violated lead safety margin.

        Parameters
        ----------
        state : StateDict
            Current state of the simulation.

        Returns
        -------
        bool
            True if wingman violated lead safety margin. False otherwise.
        """
        lead = get_platform_by_name(state, self.config.lead)
        wingman = get_platform_by_name(state, self.config.agent_name)
        distance = np.linalg.norm(lead.position - wingman.position)
        return distance < self.config.safety_margin

    def timeout(self, state):
        """
        Determine if wingman exceeded max time allocated for task.

        Parameters
        ----------
        state : StateDict
            Current state of the simulation.

        Returns
        -------
        bool
            True if wingman exceeded max time. False otherwise.
        """
        wingman = get_platform_by_name(state, self.config.agent_name)
        return wingman.sim_time >= self.config.max_time

    def oob(self, state):
        """
        Determine if wingman exceeded max distance from lead.

        Parameters
        ----------
        state : StateDict
            Current state of the simulation.

        Returns
        -------
        bool
            True if wingman exceeded max distance. False otherwise.
        """
        lead = get_platform_by_name(state, self.config.lead)
        wingman = get_platform_by_name(state, self.config.agent_name)
        distance = np.linalg.norm(lead.position - wingman.position)
        return distance >= self.config.max_distance

    def reset(self):
        self._rejoin_prev = False

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
        This method calculates reward for failing the rejoin task.

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
            The agent's reward for failing the rejoin task.
        """

        reward = RewardDict()
        value = 0.0

        lead = get_platform_by_name(state, self.config.lead)
        wingman = get_platform_by_name(state, self.config.agent_name)
        in_rejoin_region, _ = in_rejoin(wingman=wingman, lead=lead, radius=self.config.radius, offset=self.config.offset)

        if self.crash(state):
            value = self.config.crash_reward
        elif self.timeout(state):
            value = self.config.timeout_reward
        elif self.oob(state):
            value = self.config.distance_reward
        elif not in_rejoin_region and self._rejoin_prev:
            value = self.config.leave_rejoin_reward

        self._rejoin_prev = in_rejoin_region

        reward[self.config.agent_name] = value
        return reward
