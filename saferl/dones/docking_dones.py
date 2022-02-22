"""
Functions that define the terminal conditions for the Docking Environment.
This in turn defines whether the end is episode is reached or not.
"""

import numpy as np
from act3_rl_core.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from act3_rl_core.libraries.environment_dict import DoneDict
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name

from saferl.utils import VelocityHandler, VelocityHandlerValidator


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


class SuccessfulDockingDoneValidator(DoneFuncBaseValidator, VelocityHandlerValidator):
    """
    This class validates that the config contains the docking_region_radius data needed for
    computations in the SuccessfulDockingDoneFunction.
    """

    docking_region_radius: float


class SuccessfulDockingDoneFunction(DoneFuncBase, VelocityHandler):
    """
    A done function that determines if deputy has successfully docked with the cheif or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: SuccessfulDockingDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        SuccessfulDockingDoneValidator
            config validator for the SuccessfulDockingDoneFunction

        """
        return SuccessfulDockingDoneValidator

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
        deputy = get_platform_by_name(next_state, self.config.agent_name)

        position = deputy.position

        origin = np.array([0, 0, 0])
        docking_region_radius = self.config.docking_region_radius

        radial_distance = np.linalg.norm(np.array(position) - origin)
        in_docking = radial_distance <= docking_region_radius

        violated, _ = self.max_vel_violation(next_state)

        done[self.agent] = in_docking and not violated
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN
        self._set_all_done(done)
        return done


class DockingVelocityLimitDoneFunctionValidator(DoneFuncBaseValidator, VelocityHandlerValidator):
    """
    Validator for the DockingVelocityLimitDoneFunction
    """
    ...


class DockingVelocityLimitDoneFunction(DoneFuncBase, VelocityHandler):
    """
    This done fucntion determines whether the velocity limit has been exceeded or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingVelocityLimitDoneFunctionValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        DockingVelocityLimitDoneFunctionValidator : Done Function
            done function for the DockingVelocityLimitDoneFunction
        """
        return DockingVelocityLimitDoneFunctionValidator

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
        violated, _ = self.max_vel_violation(next_state)
        done[self.agent] = violated
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class DockingRelativeVelocityConstraintDoneFunctionValidator(DoneFuncBaseValidator, VelocityHandlerValidator):
    """
    This class validates that the config contains essential peices of data for the done function
    """
    ...


# needs a reference object
class DockingRelativeVelocityConstraintDoneFunction(DoneFuncBase, VelocityHandler):
    """
    A done function that checks if the docking velocity relative to a target object has exceeded a certain specified threshold velocity.
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingRelativeVelocityConstraintDoneFunctionValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        DockingRelativeVelocityConstraintDoneFunctionValidator : DoneFunctionValidator
        """

        return DockingRelativeVelocityConstraintDoneFunctionValidator

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

        violated, _ = self.max_vel_violation(next_state)

        done[self.agent] = violated

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class CrashDockingDoneValidator(DoneFuncBaseValidator, VelocityHandlerValidator):
    """
    This class validates that the config contains the docking_region_radius data needed for
    computations in the SuccessfulDockingDoneFunction.
    """

    docking_region_radius: float


class CrashDockingDoneFunction(DoneFuncBase, VelocityHandler):
    """
    A done function that determines if deputy has successfully docked with the cheif or not.
    """

    def __init__(self, **kwargs) -> None:
        self.config: CrashDockingDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        SuccessfulDockingDoneValidator
            config validator for the SuccessfulDockingDoneFunction

        """
        return CrashDockingDoneValidator

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
        deputy = get_platform_by_name(next_state, self.config.agent_name)

        position = deputy.position

        origin = np.array([0, 0, 0])
        docking_region_radius = self.config.docking_region_radius

        radial_distance = np.linalg.norm(np.array(position) - origin)
        in_docking = radial_distance <= docking_region_radius

        violated, _ = self.max_vel_violation(next_state)

        done[self.agent] = in_docking and violated
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class TimeoutDoneValidator(DoneFuncBaseValidator):
    """
    Validator for the TimeoutDoneFunction.

    timeout: float
        Number of simulation seconds which must pass before episode is over.
    """

    timeout: float


class TimeoutDoneFunction(DoneFuncBase):
    """
    A done function that determines if the max sim time has been reached.
    """

    def __init__(self, **kwargs) -> None:
        self.config: TimeoutDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Params
        ------
        cls : constructor function

        Returns
        -------
        MaxDistanceDoneValidator
            config validator for the MaxDistanceDoneFucntion

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
                dictionary containing the condition for the current agent

        """

        done = DoneDict()

        # compute distance to origin
        platform = get_platform_by_name(next_state, self.agent)
        sim_time = platform.sim_time

        done[self.agent] = sim_time > self.config.timeout

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done
