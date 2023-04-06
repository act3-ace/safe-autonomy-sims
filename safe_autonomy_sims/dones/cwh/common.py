"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning Core (CoRL) Safe Autonomy Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Functions that define the terminal conditions for CWH Spacecraft Environments.
"""
import typing

import gym
import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.environment_dict import DoneDict
from corl.simulators.common_platform_utils import get_platform_by_name
from pydantic import PyObject

from safe_autonomy_sims.utils import VelocityConstraintValidator, get_relative_position, max_vel_violation


class MaxDistanceDoneValidator(DoneFuncBaseValidator):
    """
    Configuration validator for the MaxDistanceDoneFunction

    max_distance: float
        The maximum tolerated relative distance between deputy and origin before episode termination.
    """

    max_distance: float
    reference_position_sensor_name: str = "reference_position"


class MaxDistanceDoneFunction(DoneFuncBase):
    """
    A done function that determines if the agent is
    beyond a maximum distance from the origin.


    def __call__(
        self,
        observation,
        action,relavtive_distance = np.linalg.norm(np.array(position) - chief_position)

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
            Config validator for the MaxDistanceDoneFunction.
        """
        return MaxDistanceDoneValidator

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

        # compute distance to reference entity
        relative_position = get_relative_position(next_state, self.config.platform_name, self.config.reference_position_sensor_name)
        distance = np.linalg.norm(relative_position)

        done[self.config.platform_name] = distance > self.config.max_distance

        if done[self.config.platform_name]:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class CrashDoneValidator(DoneFuncBaseValidator):
    """
    Configuration validator for CrashDoneFunction

    crash_region_radius : float
        The radius of the crashing region in meters.
    velocity_constraint : VelocityConstraintValidator
        Velocity constraint parameters.
    """
    crash_region_radius: float
    velocity_constraint: typing.Union[VelocityConstraintValidator, None] = None
    reference_position_sensor_name: str = "reference_position"


class CrashDoneFunction(DoneFuncBase):
    """
    A done function that determines if deputy has crashed with the chief (at origin).

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
        self.config: CrashDoneValidator
        super().__init__(**kwargs)

    @property
    def get_validator(self):
        """
        Parameters
        ----------
        cls : constructor function

        Returns
        -------
        CrashDockingDoneValidator
            Config validator for the CrashDockingDoneFunction.

        """
        return CrashDoneValidator

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
        distance = np.linalg.norm(relative_position)

        platform = get_platform_by_name(next_state, self.config.platform_name)
        relative_velocity = platform.velocity  # docking region assumed stationary

        in_crash_region = distance <= self.config.crash_region_radius

        done[self.config.platform_name] = in_crash_region

        if self.config.velocity_constraint is not None:
            # check velocity constraint
            violated, _ = max_vel_violation(
                relative_position,
                relative_velocity,
                self.config.velocity_constraint.velocity_threshold,
                self.config.velocity_constraint.threshold_distance,
                self.config.velocity_constraint.mean_motion,
                self.config.velocity_constraint.lower_bound,
                slope=self.config.velocity_constraint.slope,
            )

            done[self.config.platform_name] = done[self.config.platform_name] and violated

        if done[self.config.platform_name]:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class TerminalRewardSaturationDoneFunctionValidator(DoneFuncBaseValidator):
    """Validator for TerminalRewardSaturationDoneFunction

    limit : float
        cumulative reward value limit. Done triggers when this limit value is reached
    bound: string
        One of 'upper' or 'lower'. Defines whether the limit value should be an upper bound or a lower bound
    reward_functor: string
        Python path to reward function class. Is resolved into a PyObject
    reward_config: dict
        configuration args for the reward function
    """
    limit: float
    bound: typing.Literal["upper", "lower"]
    reward_functor: PyObject
    reward_config: typing.Dict


class TerminalRewardSaturationDoneFunction(DoneFuncBase):
    """Triggers done condition when wrapped cumulative reward limit reached"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config: TerminalRewardSaturationDoneFunctionValidator

        self.reward_total = 0
        self.reward_function = self.config.reward_functor(
            **self.config.reward_config, agent_name=self.config.agent_name, platform_names=[self.config.platform_name]
        )

    def reset(self):
        self.reward_total = 0
        self.reward_function.reset()

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

        return TerminalRewardSaturationDoneFunctionValidator

    def __call__(
        self,
        observation,
        action,
        next_observation,
        next_state,
        observation_space: gym.spaces.dict.Dict,
        observation_units: gym.spaces.dict.Dict,
    ) -> DoneDict:

        reward_dict = self.reward_function(
            observation, action, next_observation, next_state, next_state, observation_space, observation_units
        )

        reward_val = reward_dict[self.config.agent_name]

        self.reward_total += reward_val

        done = DoneDict()

        if self.config.bound == 'upper' and self.reward_total >= self.config.limit:
            limit_reached = True
        elif self.config.bound == 'lower' and self.reward_total <= self.config.limit:
            limit_reached = True
        else:
            limit_reached = False

        done[self.config.platform_name] = limit_reached
        if done[self.config.platform_name]:
            next_state.episode_state[self.config.platform_name][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done
