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

import numpy as np
from corl.dones.done_func_base import DoneFuncBase, DoneFuncBaseValidator, DoneStatusCodes
from corl.libraries.environment_dict import DoneDict
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
    velocity_threshold: float
    threshold_distance: float
    mean_motion: float = 0.001027
    lower_bound: bool = False
    slope: float = 2.0

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

        all_inspected = not (False in next_state.points.values())

        violated, _ = max_vel_violation(
            next_state,
            self.config.agent_name,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.lower_bound,
            slope=self.config.slope
        )

        done[self.agent] = bool(all_inspected and not violated)
        if done[self.agent]:
            next_state.episode_state[self.agent][self.name] = DoneStatusCodes.WIN
        self._set_all_done(done)
        return done


