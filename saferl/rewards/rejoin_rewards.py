"""
This module implements the Reward Functions and Reward Validators specific to the rejoin task.
"""
import typing
from collections import OrderedDict

import numpy as np
from act3_rl_core.libraries.environment_dict import RewardDict
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.rewards.reward_func_base import RewardFuncBase, RewardFuncBaseValidator
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name


class DubinsRejoinSuccessRewardValidator(RewardFuncBaseValidator):
    """
    Validator for the SuccessfulRejoinDoneFunction
    Attributes
    ----------
        rejoin_region_radius : float
            size of the radius of the region region
        offset_values : [float,float,float]
            vector detailing the location of the center of the rejoin region from the aircraft
        lead : str
            name of the lead platform, for later lookup
        reward : float
            reward for accomplishing the task
    """
    rejoin_region_radius: float
    offset_values: typing.List[float]
    lead: str
    reward: float


class DubinsRejoinSuccessReward(RewardFuncBase):
    """
    This function determines the reward for when the wingman successfully enters the rejoin region
    """

    @classmethod
    def get_validator(cls):
        """
        Method to return class's Validator.
        """
        return DubinsRejoinSuccessRewardValidator

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

        # get necessary platforms
        lead_aircraft_platform = get_platform_by_name(next_state, self.config.lead)
        wingman_agent_platform = get_platform_by_name(next_state, self.agent)

        # all 3 pieces
        rejoin_region_radius = self.config.rejoin_region_radius
        lead_orientation = lead_aircraft_platform.lead_orientation
        offset_vector = np.array(self.config.offset_values)

        # rotate vector then add it to the lead center
        rotated_vector = lead_orientation.apply(offset_vector)
        rejoin_region_center = lead_aircraft_platform.position + rotated_vector

        radial_distance = np.linalg.norm(np.array(wingman_agent_platform.position) - rejoin_region_center)
        done = radial_distance <= rejoin_region_radius

        if done:
            reward[self.config.agent_name] = self.config.reward

        return reward


class RejoinDistanceChangeRewardValidator(RewardFuncBaseValidator):
    """
        Validator for the RejoinDistanceChangeReward Reward Function
        Attributes
        ----------
        rejoin_region_radius : float
            size of the radius of the region region
        offset_values : [float,float,float]
            vector detailing the location of the center of the rejoin region from the aircraft
        lead : str
            name of the lead platform, for later lookup
        reward : float
            reward for accomplishing the task
            """
    rejoin_region_radius: float
    offset_values: typing.List[float]
    lead: str
    reward: float


class RejoinDistanceChangeReward(RewardFuncBase):
    """
    A reward function that provides a reward proportional to the change in distance from the rejoin distance.
    """

    def __init__(self, prev_dist, **kwargs):
        super().__init__(**kwargs)
        self.prev_dist = prev_dist

    @classmethod
    def get_validator(cls):
        """
            Method to return class's Validator.
            """
        return DubinsRejoinSuccessRewardValidator

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

        lead_aircraft_platform = get_platform_by_name(next_state, self.config.lead)
        wingman_agent_platform = get_platform_by_name(next_state, self.agent)

        # all 3 pieces
        # rejoin_region_radius = self.config.rejoin_region_radius
        lead_orientation = lead_aircraft_platform.lead_orientation
        offset_vector = np.array(self.config.offset_values)

        # rotate vector then add it to the lead center
        rotated_vector = lead_orientation.apply(offset_vector)
        rejoin_region_center = lead_aircraft_platform.position + rotated_vector

        radial_distance = np.linalg.norm(np.array(wingman_agent_platform.position) - rejoin_region_center)
        diff_distance = self.prev_dist - radial_distance

        reward[self.config.agent_name] = self.config.reward * diff_distance

        return reward
