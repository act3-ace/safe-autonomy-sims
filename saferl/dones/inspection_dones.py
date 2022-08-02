"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Functions that define the terminal conditions for the Inspection Environment.
This in turn defines whether the end of an episode has been reached.
"""
import typing
from collections import OrderedDict

import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes, SharedDoneFuncBaseValidator, \
    SharedDoneFuncBase
from corl.libraries.environment_dict import DoneDict
from corl.libraries.state_dict import StateDict
from corl.simulators.common_platform_utils import get_platform_by_name

from saferl.utils import max_vel_violation


class SuccessfulInspectionDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the Inspection_region_radius data needed for
    computations in the SuccessfulInspectionDoneFunction.

    Inspection_region_radius : float
        The radius of the Inspection region in meters.
    velocity_threshold : float
        The maximum tolerated velocity within Inspection region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the Inspection region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    lower_bound : bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    """
    Inspection_region_radius: float

#TODO
class SuccessfulInspectionDoneFunction(DoneFuncBase):
    """
    A done function that determines if the deputy has successfully docked with the chief.


    def __call__(self, observation, action, next_observation, next_state):

    Parameters
    ----------
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
        Dictionary containing the done condition for the current agent.
    """

    def __init__(self, **kwargs) -> None:
        self.config: SuccessfulInspectionDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        SuccessfulInspectionDoneValidator
            Config validator for the SuccessfulInspectionDoneFunction.
        """
        return SuccessfulInspectionDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        # eventually will include velocity constraint
        done = DoneDict()

        #all_inspected = not (False in next_state.points.values())
        all_inspected = True
        for point in next_state.points:
            if next_state.points[point] == False:
                all_inspected = False


        done[self.agent] = bool(all_inspected)
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN
        self._set_all_done(done)
        return done

class CollisionDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    spacecraft_safety_constraint : float
        The minimum radial distance between spacecrafts that must be maintained in order to avoid a collision.
    """
    spacecraft_safety_constraint: float = 0.5  # meters


class CollisionDoneFunction(SharedDoneFuncBase):
    """
    A done function that determines if an agent's spacecraft has collided with another
    agent's spacecraft in the environemt.


    def __call__(self, observation, action, next_observation, next_state, local_dones, local_done_info):

    Parameters
    ----------
    observation : np.ndarray
        np.ndarray describing the current observation
    action : np.ndarray
        np.ndarray describing the current action
    next_observation : np.ndarray
        np.ndarray describing the incoming observation
    next_state : np.ndarray
        np.ndarray describing the incoming state
    local_dones: DoneDict
        DoneDict containing name to boolean KVPs representing done statuses of each agent
    local_done_info: OrderedDict
        An OrderedDict containing nested OrderedDicts of done function to done status KVPs for each agent

    Returns
    -------
    done : DoneDict
        Dictionary containing the done condition for each agent.
    """

    @property
    def get_validator(self) -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Returns
        -------
        CollisionDoneFunctionValidator
            done function validator
        """
        return CollisionDoneFunctionValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        local_dones: DoneDict,
        local_done_info: OrderedDict
    ) -> DoneDict:

        # get list of spacecrafts
        agent_names = list(local_dones.keys())
        try:
            agent_names.remove("__all__")
        except ValueError as err:
            print(err)

        # populate DoneDict
        done = DoneDict()
        for name in local_dones.keys():
            done[name] = False

        # check if any spacecraft violates boundaries of other spacecrafts
        while len(agent_names) > 1:
            agent_name = agent_names.pop()
            agent_platform = get_platform_by_name(next_state, agent_name)
            agent_position = agent_platform.position

            for other_agent_name in agent_names:

                if local_dones[other_agent_name]:
                    # skip if other_agent is done
                    continue

                # check location against location of other agents for boundary violation
                other_agent_platform = get_platform_by_name(next_state, other_agent_name)
                other_agent_position = other_agent_platform.position

                radial_distance = np.linalg.norm(np.array(agent_position) - np.array(other_agent_position))

                if radial_distance < self.config.spacecraft_safety_constraint:
                    # collision detected. stop loop and end episode
                    for k in local_dones.keys():
                        done[k] = True
                    # break
                    return done
        return done


class MultiagentSuccessfulInspectionDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    The validator for the MultiagentSuccessfulDockingDoneFunction.

    success_function_name : str
        The name of the successful docking function, which this function will reference to ensure all agents have reached a
        DoneStatusCodes.WIN before ending the episode.
    """
    success_function_name: str = "SuccessfulInspectionDoneFunction"


class MultiagentSuccessfulInspectionDoneFunction(SharedDoneFuncBase):
    """
    This done function determines whether every agent in the environment has reached a specified successful done condition.


    def __call__(self, observation, action, next_observation, next_state, local_dones, local_done_info):

    Parameters
    ----------
    observation : np.ndarray
        np.ndarray describing the current observation
    action : np.ndarray
        np.ndarray describing the current action
    next_observation : np.ndarray
        np.ndarray describing the incoming observation
    next_state : np.ndarray
        np.ndarray describing the incoming state
    local_dones: DoneDict
        DoneDict containing name to boolean KVPs representing done statuses of each agent
    local_done_info: OrderedDict
        An OrderedDict containing nested OrderedDicts of done function to done status KVPs for each agent

    Returns
    -------
    done : DoneDict
        Dictionary containing the done condition for each agent.
    """

    @property
    def get_validator(self) -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Returns
        -------
        MultiagentSuccessfulDockingDoneFunctionValidator
            done function validator

        """
        return MultiagentSuccessfulInspectionDoneFunctionValidator

    def __call__(
        self,
        observation: OrderedDict,
        action: OrderedDict,
        next_observation: OrderedDict,
        next_state: StateDict,
        local_dones: DoneDict,
        local_done_info: OrderedDict
    ) -> DoneDict:

        done = DoneDict()

        all_inspected = not (False in next_state.points.values())

        if all_inspected:
            for k in local_dones.keys():
                done[k] = True

        return done

