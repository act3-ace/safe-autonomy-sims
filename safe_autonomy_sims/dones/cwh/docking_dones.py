"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements functions that define the terminal conditions
for the Docking Environment.
"""

import gymnasium
import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.simulators.common_platform_utils import get_platform_by_name

from safe_autonomy_sims.utils import get_relative_position, get_relative_velocity, max_vel_violation


class SuccessfulDockingDoneValidator(DoneFuncBaseValidator):
    """
    A configuration validator for the SuccessfulDockingDoneFunction.

    Attributes
    ----------
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
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    reference_velocity_sensor_name: str
        The name of the sensor responsible for returning the relative velocity of a reference entity.
    """

    docking_region_radius: float
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0
    reference_position_sensor_name: str = "reference_position"
    reference_velocity_sensor_name: str = "reference_velocity"


class SuccessfulDockingDoneFunction(DoneFuncBase):
    """
    A done function that determines if the deputy has successfully
    docked with the chief.

    Attributes
    ----------
    config: SuccessfulDockingDoneValidator
        The function's validated configuration parameters
    """

    def __init__(self, **kwargs) -> None:
        self.config: SuccessfulDockingDoneValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        """
        Returns configuration validator object for this done
        function.

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
        observation_space: gymnasium.spaces.dict.Dict,
        observation_units: gymnasium.spaces.dict.Dict,
    ) -> bool:
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
        observation_space : gymnasium.spaces.dict.Dict
            The agent observation space.
        observation_units : gymnasium.spaces.dict.Dict
            The units of the observations in the observation space. This may be None.

        Returns
        -------
        done : bool
            Dictionary containing the done condition for the current agent.
        """

        # eventually will include velocity constraint

        # Get relatative position + velocity between platform and docking region
        platform = get_platform_by_name(next_state, self.config.platform_name)
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        relative_velocity = get_relative_velocity(platform, self.config.reference_velocity_sensor_name)
        distance = np.linalg.norm(relative_position)

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

        done = bool(in_docking and not violated)

        if done:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.WIN

        return done


class DockingVelocityLimitDoneFunctionValidator(DoneFuncBaseValidator):
    """
    A configuration validator for the DockingVelocityLimitDoneFunction.

    Attributes
    ----------
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
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    reference_velocity_sensor_name: str
        The name of the sensor responsible for returning the relative velocity of a reference entity.
    """
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0
    reference_position_sensor_name: str = "reference_position"
    reference_velocity_sensor_name: str = "reference_velocity"


class DockingVelocityLimitDoneFunction(DoneFuncBase):
    """
    This done function determines whether the velocity limit
    has been exceeded.

    Attributes
    ----------
    config: DockingVelocityLimitDoneFunctionValidator
        The function's validated configuration parameters
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingVelocityLimitDoneFunctionValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        """
        Returns configuration validator object for this done
        function.

        Returns
        -------
        DockingVelocityLimitDoneFunctionValidator
            Config validator for the DockingVelocityLimitDoneFunction.
        """
        return DockingVelocityLimitDoneFunctionValidator

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space: gymnasium.spaces.dict.Dict,
        observation_units: gymnasium.spaces.dict.Dict,
    ) -> bool:
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
        observation_space : gymnasium.spaces.dict.Dict
            The agent observation space.
        observation_units : gymnasium.spaces.dict.Dict
            The units of the observations in the observation space. This may be None.

        Returns
        -------
        done : bool
            Dictionary containing the done condition for the current agent.
        """

        # Get relatative position + velocity between platform and docking region
        platform = get_platform_by_name(next_state, self.config.platform_name)
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        relative_velocity = get_relative_velocity(platform, self.config.reference_velocity_sensor_name)

        violated, _ = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        done = bool(violated)
        if done:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE

        return done


class DockingRelativeVelocityConstraintDoneFunctionValidator(DoneFuncBaseValidator):
    """
    A configuration validator for the DockingRelativeVelocityConstraintDoneFunction.

    Attributes
    ----------
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
    reference_position_sensor_name: str
        The name of the sensor responsible for returning the relative position of a reference entity.
    reference_velocity_sensor_name: str
        The name of the sensor responsible for returning the relative velocity of a reference entity.
    """
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0
    reference_position_sensor_name: str = "reference_position"
    reference_velocity_sensor_name: str = "reference_velocity"


# needs a reference object
class DockingRelativeVelocityConstraintDoneFunction(DoneFuncBase):
    """
    A done function that checks if the docking velocity relative
    to a target object has exceeded a certain specified threshold
    velocity.

    Attributes
    ----------
    config: DockingRelativeVelocityConstraintDoneFunctionValidator
        The function's validated configuration parameters
    """

    def __init__(self, **kwargs) -> None:
        self.config: DockingRelativeVelocityConstraintDoneFunctionValidator
        super().__init__(**kwargs)

    @staticmethod
    def get_validator():
        """
        Returns configuration validator object for this done
        function.

        Returns
        -------
        DockingRelativeVelocityConstraintDoneFunctionValidator
            Configuration validator for the DockingRelativeVelocityConstraintDoneFunction
        """

        return DockingRelativeVelocityConstraintDoneFunctionValidator

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space: gymnasium.spaces.dict.Dict,
        observation_units: gymnasium.spaces.dict.Dict,
    ) -> bool:
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
        observation_space : gymnasium.spaces.dict.Dict
            The agent observation space.
        observation_units : gymnasium.spaces.dict.Dict
            The units of the observations in the observation space. This may be None.

        Returns
        -------
        done : bool
            Dictionary containing the done condition for the current agent.
        """
        # eventually will include velocity constraint

        # Get relatative position + velocity between platform and docking region
        platform = get_platform_by_name(next_state, self.config.platform_name)
        relative_position = get_relative_position(platform, self.config.reference_position_sensor_name)
        relative_velocity = get_relative_velocity(platform, self.config.reference_velocity_sensor_name)

        violated, _ = max_vel_violation(
            relative_position,
            relative_velocity,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope,
        )

        done = bool(violated)

        if done:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE

        return done
