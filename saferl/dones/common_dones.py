"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains functions that define common terminal conditions across environments.
"""

import typing
from collections import OrderedDict

import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes, SharedDoneFuncBase, SharedDoneFuncBaseValidator
from corl.libraries.environment_dict import DoneDict
from corl.libraries.state_dict import StateDict
from corl.simulators.common_platform_utils import get_platform_by_name


class TimeoutDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the TimeoutDoneFunction config contains the max_sim_time value.
    """
    max_sim_time: float


# TODO: remove redundant done func
class TimeoutDoneFunction(DoneFuncBase):
    """
    A done function that determines if the max episode time has been reached.
    """

    def __init__(self, **kwargs) -> None:
        self.config: TimeoutDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(cls):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        TimeoutDoneValidator
            config validator for the TimeoutDoneValidator.
        """
        return TimeoutDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
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
            dictionary containing the done condition for the current agent.
        """

        done = DoneDict()

        # get sim time
        platform = get_platform_by_name(next_state, self.agent)
        sim_time = platform.sim_time

        done[self.agent] = sim_time >= self.config.max_sim_time

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        self._set_all_done(done)
        return done


class CollisionDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    Validator for the CollisionDoneFunction.

    spacecraft_safety_constraint : float
        The minimum radial distance between spacecrafts that must be maintained in order to avoid a collision.
    """
    safety_constraint: float = 0.5  # meters


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

                if radial_distance < self.config.safety_constraint:
                    # collision detected. stop loop and end episode
                    for k in local_dones.keys():
                        done[k] = True
                    # break
                    return done
        return done


class MultiagentSuccessDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    The validator for the MultiagentSuccessfulDockingDoneFunction.

    success_function_name : str
        The name of the successful docking function, which this function will reference to ensure all agents have reached a
        DoneStatusCodes.WIN before ending the episode.
    """
    success_function_name: str = "RejoinSuccessDone"


class MultiagentSuccessDoneFunction(SharedDoneFuncBase):
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
        MultiagentSuccessDoneFunctionValidator
            done function validator

        """
        return MultiagentSuccessDoneFunctionValidator

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

        for agent_name in local_done_info.keys():
            if self.config.success_function_name in next_state.episode_state[agent_name]:
                # docking kvp exists
                if next_state.episode_state[agent_name][self.config.success_function_name] != DoneStatusCodes.WIN:
                    # agent failed to dock
                    return done
            else:
                # agent has not reached done condition
                return done

        # all agents have succeeded, set all dones to True
        for k in local_dones.keys():
            done[k] = True
        return done
