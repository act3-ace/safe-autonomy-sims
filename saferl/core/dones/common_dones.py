"""
This module contains functions that define common terminal conditions across environments.
"""

import typing
from collections import OrderedDict

import numpy as np
from act3_rl_core.dones.done_func_base import (
    DoneFuncBase,
    DoneFuncBaseValidator,
    DoneStatusCodes,
    SharedDoneFuncBase,
    SharedDoneFuncBaseValidator,
)
from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.libraries.state_dict import StateDict
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name


class TimeoutDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the TimeoutDoneFunction config contains the max_sim_time value.
    """
    max_sim_time: float


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
        Params
        ------
        cls : constructor function

        Returns
        -------
        TimeoutDoneValidator
            config validator for the TimeoutDoneValidator.
        """
        return TimeoutDoneValidator

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
    name : str
        The name of this done condition
    """
    safety_constraint: float = 0.5  # meters


class CollisionDoneFunction(SharedDoneFuncBase):
    """
    Done function that determines whether the other agent is done.
    """

    @property
    def get_validator(self) -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Params
        ------
        cls : class constructor

        Returns
        -------
        RejoinDoneValidator
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
            dictionary containing the condition for the current agent
        """

        # get list of agents
        agent_names = list(local_done_info.keys())

        # populate DoneDict
        done = DoneDict()
        for name in local_dones.keys():
            done[name] = False

        # check if any agent violates safety_constraint of other agents
        while len(agent_names) > 1:
            agent_name = agent_names.pop()
            agent_platform = get_platform_by_name(next_state, agent_name)  # TODO: better way to do this
            agent_position = agent_platform.position

            for other_agent_name in agent_names:

                if local_dones.get(other_agent_name, True):
                    # skip if other_agent is done or inoperable
                    continue

                # check location against location of other agents for boundary violation
                other_agent_platform = get_platform_by_name(next_state, other_agent_name)  # TODO: better way to do this
                other_agent_position = other_agent_platform.position
                radial_distance = np.linalg.norm(np.array(agent_position) - np.array(other_agent_position))

                if radial_distance < self.config.safety_constraint:
                    # collision detected. stop loop and end episode
                    for k in local_dones.keys():
                        done[k] = True
                    break
        return done


class MultiagentSuccessDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    name : str
        The name of this done condition
    """
    success_function_name: str = "RejoinSuccessDone"


class MultiagentSuccessDoneFunction(SharedDoneFuncBase):
    """
    Done function that determines whether the other agent is done.
    """

    @property
    def get_validator(self) -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Params
        ------
        cls : class constructor

        Returns
        -------
        RejoinDoneValidator
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
            dictionary containing the condition for the current agent

        """

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

        # all agents have suceeded, set all dones to True
        for k in local_dones.keys():
            done[k] = True
        return done
