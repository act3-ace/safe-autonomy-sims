"""
This module defines the docking initializer class.

Author: John McCarroll
"""

import math
import random

import numpy as np
from pydantic import BaseModel

from saferl.simulators.initializers.initializer import BaseInitializer


def velocity_limit(position, velocity_threshold, threshold_distance, mean_motion=0.001027, slope=2.0):
    """
    Get the velocity limit from the agent's current position, assuming the chief is at the origin.

    Parameters
    ----------
    velocity_threshold: float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance: float
        The radius of the docking region.
    mean_motion: float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    slope: float
        The slope of the linear velocity limit as a function of distance from docking region.

    Returns
    -------
    float
        The velocity limit given the agent's position.
    """
    dist = np.linalg.norm(position)
    vel_limit = velocity_threshold
    if dist > threshold_distance:
        vel_limit += slope * mean_motion * (dist - threshold_distance)
    return vel_limit


class Docking3DInitializerValidator(BaseModel):
    """
    Validator for Docking3DInitializer.

    Parameters
    ----------
    velocity_threshold : float
        The maximum tolerated velocity within docking region without crashing.
    threshold_distance : float
        The distance at which the velocity constraint reaches a minimum (typically the docking region radius).
    slope : float
        The slope of the linear region of the velocity constraint function.
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    x: float
        The reset value for the x position of the agent
    y: float
        The reset value for the y position of the agent
    z: float
        The reset value for the z position of the agent
    xdot: float
        The reset value for the x velocity of the agent
    ydot: float
        The reset value for the y velocity of the agent
    zdot: float
        The reset value for the z velocity of the agent
    """

    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027
    x: float = 100
    y: float = 100
    z: float = 100
    xdot: float = 0
    ydot: float = 0
    zdot: float = 0


class Docking3DInitializer(BaseInitializer):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D docking environment.
    It ensures that the initial velocity of the deputy does not violate the maximum velocity safety constraint.

    def __call__(self, reset_config):

    Parameters
    ----------
    reset_config: dict
        A dictionary containing the reset values for each agent. Agent names are the keys and initialization config dicts
        are the values

    Returns
    -------
    reset_config: dict
        The modified reset config of agent name to initialization values KVPs.
    """

    def __init__(self, config):
        self.config = self.get_validator(**config)

    @property
    def get_validator(self):
        """
        Returns
        -------
        Docking3DInitializerValidator
            Config validator for the Docking3DInitializerValidator.
        """
        return Docking3DInitializerValidator

    def __call__(self, reset_config):

        for agent_name, agent_reset_config in reset_config.items():

            if "x" not in agent_reset_config or "y" not in agent_reset_config or "z" not in agent_reset_config:
                raise ValueError("{} agent_reset_config missing one or more positional keys".format(agent_name))

            # constrained rng velocity
            vel_limit = velocity_limit(
                [agent_reset_config["x"], agent_reset_config["y"], agent_reset_config["z"]],
                self.config.velocity_threshold,
                self.config.threshold_distance,
                self.config.mean_motion,
                self.config.slope
            )

            # find magnitude of x,y,z components of max vel limit given position
            axis_vel_limit = math.sqrt(vel_limit**2 / 3)

            # initialize from boundary
            agent_reset_config.update({"xdot": random.uniform(-axis_vel_limit, axis_vel_limit)})
            agent_reset_config.update({"ydot": random.uniform(-axis_vel_limit, axis_vel_limit)})
            agent_reset_config.update({"zdot": random.uniform(-axis_vel_limit, axis_vel_limit)})

        return reset_config
