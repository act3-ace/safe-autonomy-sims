"""
Contains implementations of the necessary done functions for the rejoin environment.
Namely, three done functions : SuccessfulRejoinFunction, MaxDistanceDoneFunction, CrashDoneFunction
"""
import typing
from collections import OrderedDict

import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes, SharedDoneFuncBase, SharedDoneFuncBaseValidator
from corl.libraries.environment_dict import DoneDict
from corl.libraries.state_dict import StateDict
from corl.simulators.common_platform_utils import get_platform_by_name

from saferl.core.utils import in_rejoin


class RejoinSuccessDoneValidator(DoneFuncBaseValidator):
    """
    Validator for the SuccessfulRejoinDoneFunction.

    Attributes
    ----------
    radius : float
        Size of the radius of the rejoin region.
    offset : [float,float,float]
        Vector detailing the location of the center of the rejoin region from the aircraft.
    lead : str
        Name of the lead platform, for later lookup.
    step_size : float
        Size of one single simulation step.
    success_time : float
        Time wingman must remain in rejoin region to obtain reward.
    """
    radius: typing.Union[float, int]
    offset: typing.List[typing.Union[float, int]]
    lead: str
    step_size: typing.Union[float, int]
    success_time: typing.Union[float, int]


class RejoinSuccessDone(DoneFuncBase):
    """
    This function determines the reward for when the wingman successfully stays in the rejoin region for the
    duration of the given rejoin_time.
    """

    def __init__(self, **kwargs) -> None:
        self.config: RejoinSuccessDoneValidator
        self.rejoin_time = 0.0
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Method to return class's Validator.
        """
        return RejoinSuccessDoneValidator

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
        next_state: StateDict,
    ) -> DoneDict:
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
        next_state : StateDict
            The current state of the simulation.

        Returns
        -------
        reward : DoneDict
            Dictionary containing the done condition for the current agent.
        """

        done = DoneDict()

        self._update_rejoin_time(next_state)

        done[self.agent] = self.rejoin_time > self.config.success_time

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN

        self._set_all_done(done)
        return done


class MaxDistanceDoneValidator(DoneFuncBaseValidator):
    """
    Validator for the MaxDistanceDoneFunction.

    Attributes
    ----------
        max_distance : float
            Max distance the wingman can be away from the lead, exceeding this stops simulation.
        lead : str
            Name of the lead platform, for later lookup.
    """
    max_distance: float
    lead: str


class MaxDistanceDoneFunction(DoneFuncBase):
    """
    Done function that determines if the wingman has exceeded the max distance threshold and has exited the bounds of
    the simulation.
    """

    def __init__(self, **kwargs) -> None:
        self.config: MaxDistanceDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[DoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Parameters
        ----------
        cls : class constructor

        Returns
        -------
        SuccessfulRejoinDoneValidator
            Done function validator.
        """
        return MaxDistanceDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Returns the done condition of the agent based on if the relative distance between the lead and the wingman has
        exceeded the value passed by the max_distance argument.

        Parameters
        ----------
        observation : np.ndarray
            Current observation from environment.
        action : np.ndarray
            Current action to be applied.
        next_observation : np.ndarray
            Incoming observation from environment.
        next_state : np.ndarray
            Incoming state from environment.

        Returns
        -------
        done : DoneDict
            Dictionary containing the condition for the current agent.

        """

        done = DoneDict()

        wingman_agent_platform = get_platform_by_name(next_state, self.agent)
        lead_aircraft_platform = get_platform_by_name(next_state, self.config.lead)

        dist = np.linalg.norm(wingman_agent_platform.position - lead_aircraft_platform.position)

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        self._set_all_done(done)
        return done


class CrashDoneValidator(DoneFuncBaseValidator):
    """
    Validator for the CrashDoneFunction.

    Attributes
    ----------
    safety_margin : float
        The minimum distance between the lead and the wingman that needs to be maintained.
    lead : str
        Name of the lead platform.
    """
    safety_margin: float
    lead: str


class CrashDoneFunction(DoneFuncBase):
    """
    Done function that determines whether a crash occurred or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: CrashDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self) -> typing.Type[DoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Parameters
        ----------
        cls : class constructor

        Returns
        -------
        SuccessfulRejoinDoneValidator
            Done function validator.

        """
        return CrashDoneValidator

    def __call__(self, observation, action, next_observation, next_state):
        """
        Returns the done condition of the agent based on if the relative distance between the lead and the wingman is 
        below the safety_margin.

        Parameters
        ----------
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
            dictionary containing the condition  for the current agent

        """

        done = DoneDict()

        wingman_agent_platform = get_platform_by_name(next_state, self.agent)
        lead_aircraft_platform = get_platform_by_name(next_state, self.config.lead)

        dist = np.linalg.norm(wingman_agent_platform.position - lead_aircraft_platform.position)

        done[self.agent] = dist <= self.config.safety_margin

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE

        self._set_all_done(done)
        return done


class RejoinDoneValidator(SharedDoneFuncBaseValidator):
    """
    agent_name : str
        The name of the agent whom will determine the done status of the episode.
    """
    agent_name: str


class RejoinDone(SharedDoneFuncBase):
    """
    Done function that determines whether the other agent is done.
    """

    @property
    def get_validator(self) -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Parameters
        ----------
        cls : class constructor

        Returns
        -------
        RejoinDoneValidator
            Done function validator.

        """
        return RejoinDoneValidator

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
        Logic that returns the done condition based on the agent's done condition, whose agent_name was provided.

        Parameters
        ------
        observation : np.ndarray
             Current observation from environment.
        action : np.ndarray
             Current action to be applied.
        next_observation : np.ndarray
             Incoming observation from environment.
        next_state : np.ndarray
             Incoming state from environment.

        Returns
        -------
        done : DoneDict
            Dictionary containing the condition for the current agent.
        """

        done = DoneDict()

        all_done = local_dones[self.config.agent_name]

        for k in local_dones.keys():
            done[k] = all_done
        return done
