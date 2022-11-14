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

from saferl.utils import VelocityConstraintValidator, max_vel_violation


class MaxDistanceOriginDoneValidator(DoneFuncBaseValidator):
    """
    Configuration validator for the MaxDistanceOriginDoneFunction

    max_distance: float
        The maximum tolerated relative distance between deputy and origin before episode termination.
    """

    max_distance: float


class MaxDistanceOriginDoneFunction(DoneFuncBase):
    """
    A done function that determines if the agent is
    beyond a maximum distance from the origin.


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
        self.config: MaxDistanceOriginDoneValidator
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
        return MaxDistanceOriginDoneValidator

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

        # compute distance to origin
        platform = get_platform_by_name(next_state, self.agent)
        position = platform.position

        # compute distance to origin
        dist = np.linalg.norm(np.array(position))

        done[self.agent] = dist > self.config.max_distance

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done


class CrashOriginDoneValidator(DoneFuncBaseValidator):
    """
    Configuration validator for CrashOriginDoneFunction

    crash_region_radius : float
        The radius of the crashing region in meters.
    velocity_constraint : VelocityConstraintValidator
        Velocity constraint parameters.
    """
    crash_region_radius: float
    velocity_constraint: typing.Union[VelocityConstraintValidator, None] = None


class CrashOriginDoneFunction(DoneFuncBase):
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
        self.config: CrashOriginDoneValidator
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
        return CrashOriginDoneValidator

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

        # check if in crashing region
        deputy = get_platform_by_name(next_state, self.config.agent_name)
        position = deputy.position
        in_crash_region = np.linalg.norm(np.array(position)) <= self.config.crash_region_radius

        done[self.agent] = in_crash_region

        if self.config.velocity_constraint is not None:
            # check velocity constraint
            violated, _ = max_vel_violation(
                next_state,
                self.config.agent_name,
                self.config.velocity_constraint.velocity_threshold,
                self.config.velocity_constraint.threshold_distance,
                self.config.velocity_constraint.mean_motion,
                self.config.velocity_constraint.lower_bound,
                slope=self.config.velocity_constraint.slope
            )

            done[self.agent] = done[self.agent] and violated

        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.LOSE
        self._set_all_done(done)
        return done
