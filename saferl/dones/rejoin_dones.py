"""
Functions that define the terminal conditions for the Docking Environment.
This in turn defines whether the end is episode is reached or not.
"""

import numpy as np
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict

# need to import get_platform_name, WIP


class MaxDistanceDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the max_distance data needed for
    computations in the MaxDistanceDoneFucntion.
    """

    max_distance: float


class MaxDistanceDoneFunction(DoneFuncBase):
    """
    A done function that determines if the max distance has been traveled or not.
    """

    @classmethod
    def get_validator(cls):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        MaxDistanceDoneValidator
            config validator for the MaxDistanceDoneFucntion

        """
        return MaxDistanceDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Params
        ------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
            done : DoneDict
                dictionary containing the condition condition for the current agent

        """

        done = DoneDict()

        # compute distance to origin
        # platform = get_platform_name(next_state,self.agent)
        # pos = platform.position

        position = next_state.sim_platforms[0].position

        # compute to origin
        origin = np.array([0, 0, 0])
        dist = np.linalg.norm(origin - np.array(position))

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done


class SuccessfulRejoinDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the docking_region_radius data needed for
    computations in the SuccessfulDockingDoneFunction.
    """

    rejoin_region_radius: float


class SuccessfulRejoinDoneFunction(DoneFuncBase):
    """
    A done function that determines if deputy has successfully docked with the cheif or not.
    """

    @classmethod
    def get_validator(cls):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        SuccessfulDockingDoneValidator
            config validator for the SuccessfulDockingDoneFunction

        """
        return SuccessfulRejoinDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Params
        ------
        observation : np.ndarray
            np.ndarray describing the current observation
        action : np.ndarray
            np.ndarray describing the current action
        next_observation : np.ndarray
            np.ndarray describing the incoming observation
        next_state : np.ndarray
            np.ndarray describing the incoming state

        Returns
        -------
            done : DoneDict
                dictionary containing the condition condition for the current agent

        """
        # eventually will include velocity constraint
        done = DoneDict()
        # platform = get_platform_name(next_state,self.agent)

        # pos = platform.position
        position = next_state.sim_platforms[0].position

        origin = np.array([0, 0, 0])
        rejoin_region_radius = self.config.rejoin_region_radius

        radial_distance = np.linalg.norm(np.array(position) - origin)
        done[self.agent] = radial_distance <= rejoin_region_radius

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN

        return done
