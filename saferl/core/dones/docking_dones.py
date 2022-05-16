"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Functions that define the terminal conditions for the Docking Environment.
This in turn defines whether the end is episode is reached or not.
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
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name, get_sensor_by_name

from saferl.core.utils import max_vel_violation


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

    def __init__(self, **kwargs) -> None:
        self.config: MaxDistanceDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        MaxDistanceDoneValidator
            config validator for the MaxDistanceDoneFucntion

        """
        return MaxDistanceDoneValidator

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
            dictionary containing the condition condition for the current agent

        """

        done = DoneDict()

        # compute distance to origin
        platform = get_platform_by_name(next_state, self.agent)
        position = platform.position

        # compute to origin
        origin = np.array([0, 0, 0])
        dist = np.linalg.norm(origin - np.array(position))

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class SuccessfulDockingDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the docking_region_radius data needed for
    computations in the SuccessfulDockingDoneFunction.
    """

    docking_region_radius: float
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0


class SuccessfulDockingDoneFunction(DoneFuncBase):
    """
    A done function that determines if deputy has successfully docked with the cheif or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: SuccessfulDockingDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        SuccessfulDockingDoneValidator
            config validator for the SuccessfulDockingDoneFunction

        """
        return SuccessfulDockingDoneValidator

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
            dictionary containing the condition condition for the current agent

        """
        # eventually will include velocity constraint
        done = DoneDict()
        deputy = get_platform_by_name(next_state, self.config.agent_name)

        position = deputy.position

        origin = np.array([0, 0, 0])
        docking_region_radius = self.config.docking_region_radius

        radial_distance = np.linalg.norm(np.array(position) - origin)
        in_docking = radial_distance <= docking_region_radius

        violated, _ = max_vel_violation(
            next_state,
            self.config.agent_name,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope
        )

        # main:
        # done[self.agent] = in_docking and not violated
        # multi
        done[self.agent] = bool(in_docking and not violated)
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN
        self._set_all_done(done)
        return done


class DockingVelocityLimitDoneFunctionValidator(DoneFuncBaseValidator):
    """
    Validator for the DockingVelocityLimitDoneFunction
    """
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0


class DockingVelocityLimitDoneFunction(DoneFuncBase):
    """
    This done fucntion determines whether the velocity limit has been exceeded or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingVelocityLimitDoneFunctionValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        DockingVelocityLimitDoneFunctionValidator : Done Function
            done function for the DockingVelocityLimitDoneFunction
        """
        return DockingVelocityLimitDoneFunctionValidator

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
            dictionary containing the condition condition for the current agent
        """

        done = DoneDict()

        violated, _ = max_vel_violation(
            next_state,
            self.config.agent_name,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope
        )

        done[self.agent] = violated
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class DockingRelativeVelocityConstraintDoneFunctionValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains essential peices of data for the done function
    """
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0


# needs a reference object
class DockingRelativeVelocityConstraintDoneFunction(DoneFuncBase):
    """
    A done function that checks if the docking velocity relative to a target object has exceeded a certain specified threshold velocity.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingRelativeVelocityConstraintDoneFunctionValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        DockingRelativeVelocityConstraintDoneFunctionValidator : DoneFunctionValidator
        """

        return DockingRelativeVelocityConstraintDoneFunctionValidator

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
            dictionary containing the condition condition for the current agent
        """
        # eventually will include velocity constraint
        done = DoneDict()

        violated, _ = max_vel_violation(
            next_state,
            self.config.agent_name,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope
        )

        done[self.agent] = violated

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class CrashDockingDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the docking_region_radius data needed for
    computations in the SuccessfulDockingDoneFunction.
    """

    docking_region_radius: float
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0


class CrashDockingDoneFunction(DoneFuncBase):
    """
    A done function that determines if deputy has successfully docked with the cheif or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: CrashDockingDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        SuccessfulDockingDoneValidator
            config validator for the SuccessfulDockingDoneFunction

        """
        return CrashDockingDoneValidator

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
            dictionary containing the condition condition for the current agent

        """
        done = DoneDict()
        deputy = get_platform_by_name(next_state, self.config.agent_name)

        position = deputy.position

        origin = np.array([0, 0, 0])
        docking_region_radius = self.config.docking_region_radius

        radial_distance = np.linalg.norm(np.array(position) - origin)
        in_docking = radial_distance <= docking_region_radius

        violated, _ = max_vel_violation(
            next_state,
            self.config.agent_name,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope
        )

        done[self.agent] = in_docking and violated
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class CollisionDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    name : str
        The name of this done condition
    """
    spacecraft_safety_constraint: float = 0.5  # meters
    position_sensor_name: str = "Sensor_Position"


class CollisionDoneFunction(SharedDoneFuncBase):
    """
    Done function that determines whether the other agent is done.
    """

    @property
    def get_validator(self) -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

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

        observation : np.ndarray
             current observation from environment
        action : np.ndarray
             current action to be applied
        next_observation : np.ndarray
             incoming observation from environment
        next_state : np.ndarray
             incoming state from environment
        local_dones: DoneDict
            TODO
        local_done_info: OrderedDict
            TODO

        Returns
        -------
        done : DoneDict
            dictionary containing the condition for the current agent

        """

        # get list of spacecrafts
        agent_names = list(local_done_info.keys())

        # populate DoneDict
        done = DoneDict()
        for name in local_dones.keys():
            done[name] = False

        # check if any spacecraft violates boundaries of other spacecrafts
        while len(agent_names) > 1:
            agent_name = agent_names.pop()
            agent_platform = get_platform_by_name(next_state, agent_name)
            agent_sensor = get_sensor_by_name(agent_platform, self.config.position_sensor_name)
            agent_position = agent_sensor.get_measurement()

            for other_agent_name in agent_names:

                if local_dones[other_agent_name]:
                    # skip if other_agent is done
                    continue

                # check location against location of other agents for boundary violation
                other_agent_platform = get_platform_by_name(next_state, other_agent_name)
                other_agent_sensor = get_sensor_by_name(other_agent_platform, self.config.position_sensor_name)
                other_agent_position = other_agent_sensor.get_measurement()

                radial_distance = np.linalg.norm(np.array(agent_position) - np.array(other_agent_position))

                if radial_distance < self.config.spacecraft_safety_constraint:
                    # collision detected. stop loop and end episode
                    for k in local_dones.keys():
                        done[k] = True
                    break
        return done


class MultiagentSuccessfulDockingDoneFunctionValidator(SharedDoneFuncBaseValidator):
    """
    name : str
        The name of this done condition
    """
    success_function_name: str = "SuccessfulDockingDoneFunction"


class MultiagentSuccessfulDockingDoneFunction(SharedDoneFuncBase):
    """
    Done function that determines whether the other agent is done.
    """

    @property
    def get_validator(self) -> typing.Type[SharedDoneFuncBaseValidator]:
        """
        Returns the validator for this done function.

        Returns
        -------
        RejoinDoneValidator
            done function validator

        """
        return MultiagentSuccessfulDockingDoneFunctionValidator

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
        local_dones: DoneDict
            TODO ****
        local_done_info: OrderedDict
            TODO****

        Returns
        -------
        done : DoneDict
            dictionary containing the condition for the current agent

        """

        # get list of spacecrafts
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

        # all agents have docked, set all dones to True
        for k in local_dones.keys():
            done[k] = True
        return done
