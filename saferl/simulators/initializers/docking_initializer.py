"""
This module defines the docking initializer class.

Author: John McCarroll
"""

import math
import random

import numpy as np

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
    position_values = [elem.value for elem in position]
    dist = np.linalg.norm(position_values)
    vel_limit = velocity_threshold
    if dist > threshold_distance:
        vel_limit += slope * mean_motion * (dist - threshold_distance)
    return vel_limit


class Docking3DInitializer(BaseInitializer):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D docking problem.
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

    def __call__(self, agent_reset_config):
        if "x" not in agent_reset_config or "y" not in agent_reset_config or "z" not in agent_reset_config:
            raise ValueError("agent_reset_config missing one or more positional keys")

        # constrained rng velocity
        vel_limit = velocity_limit(
            [agent_reset_config["x"], agent_reset_config["y"], agent_reset_config["z"]],
            self.velocity_threshold,
            self.threshold_distance,
            self.mean_motion,
            self.slope
        )

        # find magnitude of x,y,z components of max vel limit given position
        axis_vel_limit = math.sqrt(vel_limit**2 / 3)

        # # compare to range
        # upper_x_dot_bound = axis_vel_limit if axis_vel_limit < self.x_dot[1] else self.x_dot[1]
        # lower_x_dot_bound = -axis_vel_limit if -axis_vel_limit > self.x_dot[0] else self.x_dot[0]
        # upper_y_dot_bound = axis_vel_limit if axis_vel_limit < self.y_dot[1] else self.y_dot[1]
        # lower_y_dot_bound = -axis_vel_limit if -axis_vel_limit > self.y_dot[0] else self.y_dot[0]
        # upper_z_dot_bound = axis_vel_limit if axis_vel_limit < self.z_dot[1] else self.z_dot[1]
        # lower_z_dot_bound = -axis_vel_limit if -axis_vel_limit > self.z_dot[0] else self.z_dot[0]

        # initialize from boundary
        agent_reset_config.update({"xdot": random.uniform(-axis_vel_limit, axis_vel_limit)})
        agent_reset_config.update({"ydot": random.uniform(-axis_vel_limit, axis_vel_limit)})
        agent_reset_config.update({"zdot": random.uniform(-axis_vel_limit, axis_vel_limit)})

        return agent_reset_config
