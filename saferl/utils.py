"""
This module contains various utility functions.

Author: Jamie Cunningham
"""

import numpy as np
from act3_rl_core.simulators.common_platform_utils import get_platform_by_name
from pydantic import BaseModel


class VelocityHandlerValidator(BaseModel):
    """
    Validator for VelocityHandler
    TODO: get description of these values
    """
    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float
    lower_bound: bool = False


class VelocityHandler:
    """
    Container class with common functions for calculating velocity constraints. Consider
    unwrapping this class and just using static functions.
    """

    def velocity_limit(self, state):
        """
        Get the velocity limit from the agent's current position.

        Parameters
        ----------
        state: StateDict
            The current state of the system.

        Returns
        -------
        float
            The velocity limit given the agent's position.
        """
        deputy = get_platform_by_name(state, self.config.agent_name)
        dist = np.linalg.norm(deputy.position)
        vel_limit = self.config.velocity_threshold
        if dist > self.config.threshold_distance:
            vel_limit += self.config.slope * self.config.mean_motion * (dist - self.config.threshold_distance)
        return vel_limit

    def max_vel_violation(self, state):
        """
        Get the magnitude of a velocity limit violation if one has occurred.

        Parameters
        ----------
        state: StateDict
            The current state of the system.

        Returns
        -------
        violated: bool
            Boolean value indicating if the velocity limit has been violated
        violation: float
            The magnitude of the velocity limit violation.
        """
        deputy = get_platform_by_name(state, self.config.agent_name)
        rel_vel = deputy.velocity
        rel_vel_mag = np.linalg.norm(rel_vel)

        vel_limit = self.velocity_limit(state)

        violation = rel_vel_mag - vel_limit
        violated = rel_vel_mag > vel_limit
        if self.config.lower_bound:
            violation *= -1
            violated = rel_vel_mag < vel_limit

        return violated, violation
