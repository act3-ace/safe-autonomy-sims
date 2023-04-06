"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Functions that define the terminal conditions for the Docking Environment.
This in turn defines whether the end of an episode has been reached.
"""

import gym
import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.environment_dict import DoneDict
from corl.simulators.common_platform_utils import get_platform_by_name

from saferl.utils import get_relative_position, max_vel_violation


class SuccessfulDockingDoneValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains the docking_region_radius data needed for
    computations in the SuccessfulDockingDoneFunction.

    docking_region_radius : float
        The radius of the docking region in meters.
    velocity_threshold : float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the docking region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    lower_bound : bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    """

    docking_region_radius: float
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0
    reference_position_sensor_name: str = "reference_position"


class SuccessfulDockingDoneFunction(DoneFuncBase):
    """
    A done function that determines if the deputy has successfully docked with the chief.

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space,
        observation_units
    ) -> DoneDict:

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
    observation_space : gym.spaces.dict.Dict
        The agent observation space.
    observation_units : gym.spaces.dict.Dict
        The units of the observations in the observation space. This may be None.

    Returns
    -------
    done : DoneDict
        Dictionary containing the done condition for the current agent.
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
            Config validator for the SuccessfulDockingDoneFunction.
        """
        return SuccessfulDockingDoneValidator

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space: gym.spaces.dict.Dict,
        observation_units: gym.spaces.dict.Dict,
    ) -> DoneDict:

        # eventually will include velocity constraint
        done = DoneDict()

        # Get relatative position + velocity between platform and docking region
        relative_position = get_relative_position(next_state, self.config.platform_name, self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)
        platform = get_platform_by_name(next_state, self.config.platform_name)
        relative_velocity = platform.velocity  # docking region assumed stationary

        in_docking = distance <= self.config.docking_region_radius

        violated, _ = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope
        )

        done[self.config.platform_name] = bool(in_docking and not violated)

        if done[self.config.platform_name]:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.WIN
        self._set_all_done(done)
        return done


class DockingVelocityLimitDoneFunctionValidator(DoneFuncBaseValidator):
    """
    Validator for the DockingVelocityLimitDoneFunction.

    docking_region_radius : float
        The radius of the docking region in meters.
    velocity_threshold : float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the docking region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    lower_bound : bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    """
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0
    reference_position_sensor_name: str = "reference_position"


class DockingVelocityLimitDoneFunction(DoneFuncBase):
    """
    This done function determines whether the velocity limit has been exceeded or not.


    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space,
        observation_units
    ) -> DoneDict:

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
    observation_space : gym.spaces.dict.Dict
        The agent observation space.
    observation_units : gym.spaces.dict.Dict
        The units of the observations in the observation space. This may be None.


    Returns
    -------
    done : DoneDict
        Dictionary containing the done condition for the current agent.
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
        DockingVelocityLimitDoneFunctionValidator : done function
            Done function for the DockingVelocityLimitDoneFunction.
        """
        return DockingVelocityLimitDoneFunctionValidator

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space: gym.spaces.dict.Dict,
        observation_units: gym.spaces.dict.Dict,
    ) -> DoneDict:

        done = DoneDict()

        # Get relatative position + velocity between platform and docking region
        relative_position = get_relative_position(next_state, self.config.platform_name, self.config.reference_position_sensor_name)
        platform = get_platform_by_name(next_state, self.config.platform_name)
        relative_velocity = platform.velocity  # docking region assumed stationary

        violated, _ = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        done[self.config.platform_name] = violated
        if done[self.config.platform_name]:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class DockingRelativeVelocityConstraintDoneFunctionValidator(DoneFuncBaseValidator):
    """
    This class validates that the config contains essential data for the done function.

    velocity_threshold : float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the docking region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    lower_bound : bool
        If True, the function enforces a minimum velocity constraint on the agent's platform.
    """
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0
    reference_position_sensor_name: str = "reference_position"


# needs a reference object
class DockingRelativeVelocityConstraintDoneFunction(DoneFuncBase):
    """
    A done function that checks if the docking velocity relative to a target object has exceeded a certain specified
    threshold velocity.


    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space,
        observation_units
    ) -> DoneDict:

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
    observation_space : gym.spaces.dict.Dict
        The agent observation space.
    observation_units : gym.spaces.dict.Dict
        The units of the observations in the observation space. This may be None.

    Returns
    -------
    done : DoneDict
        Dictionary containing the done condition for the current agent.
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

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space: gym.spaces.dict.Dict,
        observation_units: gym.spaces.dict.Dict,
    ) -> DoneDict:

        # eventually will include velocity constraint
        done = DoneDict()

        # Get relatative position + velocity between platform and docking region
        relative_position = get_relative_position(next_state, self.config.platform_name, self.config.reference_position_sensor_name)
        platform = get_platform_by_name(next_state, self.config.platform_name)
        relative_velocity = platform.velocity  # docking region assumed stationary

        violated, _ = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        done[self.config.platform_name] = violated

        if done[self.config.platform_name]:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done
