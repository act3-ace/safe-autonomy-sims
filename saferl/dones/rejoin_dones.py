"""
Contains implementations of the necessary done functions for the rejoin environment.
Namely three done funcitons : SuccessfulRejoinFunction, MaxDistanceDoneFunction, CrashDoneFunction
"""
import numpy as np
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name



class SuccessfulRejoinDoneValidator(DoneFuncBaseValidator):
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
    """
    rejoin_region_radius: float
    offset_values: [float, float, float]
    lead: str


class SuccessfulRejoinDoneFunction(DoneFuncBase):
    """
    Done function that details whether a successful rejoin has been made or not.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        """
        Returns the validator for this done function.

        Params
        ------
        cls : class constructor

        Returns
        -------
        SuccessfulRejoinDoneValidator
            done function validator

        """

        return SuccessfulRejoinDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Logic that returns the done condition given the current environment conditions

        Params
        ------
        observation : np.ndarray
             current observation from environment
        action : np.ndarray
             current action to be applied
        next_observation : np.ndarray
             incoming observation from environment
        next_state : np.ndarray
             incoming state from environment

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent

        """

        # eventually will include velocity constraint
        done = DoneDict()

        lead_aircraft_platform = get_platform_by_name(next_state, self.config.lead)
        wingman_agent_platform = get_platform_by_name(next_state, self.agent)

        # compute the rejoin region , using all three pieces of info

        # all 3 pieces
        rejoin_region_radius = self.config.rejoin_region_radius
        lead_orientation = lead_aircraft_platform.lead_orientation
        offset_vector = np.array(self.config.offset_values)

        # rotate vector then add it to the lead center
        rotated_vector = lead_orientation.apply(offset_vector)
        rejoin_region_center = lead_aircraft_platform.position + rotated_vector

        radial_distance = np.linalg.norm(np.array(position) - rejoin_region_center)
        done[self.agent] = radial_distance <= rejoin_region_center

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN

        return done


class MaxDistanceDoneValidator(DoneFuncBaseValidator):
    """
    Validator for the MaxDistanceDoneFunction
    Attributes
    ----------
        max_distance : float
            max distance the wingman can be away from the lead, exceeding this stops simulation
        lead : str
            name of the lead platform, for later lookup
    """
    max_distance: float
    lead: str


class MaxDistanceDoneFunction(DoneFuncBase):
    """
    Done function that determines if the wingman  has exceeded the max distance threshold and has exited the bounds of the simulation.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        """
        Returns the validator for this done function.

        Params
        ------
        cls : class constructor

        Returns
        -------
        SuccessfulRejoinDoneValidator
            done function validator

        """
        return MaxDistanceDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Logic that returns the done condition given the current environment conditions

        Params
        ------
        observation : np.ndarray
             current observation from environment
        action : np.ndarray
             current action to be applied
        next_observation : np.ndarray
             incoming observation from environment
        next_state : np.ndarray
             incoming state from environment

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent

        """

        done = DoneDict()

        wingman_agent_platform = get_platform_by_name(next_state, self.agent)

        dist = np.linalg.norm(wingman_agent_platform.position - lead_aircraft_platform.position)

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done


class CrashDoneValidator(DoneFuncBaseValidator):
    """
    Validator for the CrashDoneFunction
    Attributes
    ----------
        safety_margin : float
            the distance between the lead and wingman that needs to be maintained
        lead : str
            name of the lead platform, for later lookup
    """
    safety_margin: float
    lead: str


class CrashDoneFunction(DoneFuncBase):
    """
    Done function that determines whether a crash occured or not.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def get_validator(cls):
        """
        Returns the validator for this done function.

        Params
        ------
        cls : class constructor

        Returns
        -------
        SuccessfulRejoinDoneValidator
            done function validator

        """
        return CrashDoneValidator

    def __call__(self, observation, action, next_observation, next_state):

        """
        Logic that returns the done condition given the current environment conditions

        Params
        ------
        observation : np.ndarray
             current observation from environment
        action : np.ndarray
             current action to be applied
        next_observation : np.ndarray
             incoming observation from environment
        next_state : np.ndarray
             incoming state from environment

        Returns
        -------
        done : DoneDict
            dictionary containing the condition condition for the current agent

        """

        done = DoneDict()

        wingman_agent_platform = get_platform_by_name(next_state, self.agent)

        lead_aircraft_platform = get_platform_by_name(next_state, self.config.lead)

        dist = np.linalg.norm(wingman_agent_platform.position - lead_aircraft_platform.position)

        done[self.agent] = dist <= self.config.safety_margin

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        return done
