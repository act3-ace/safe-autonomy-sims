"""
This module defines the docking initializer class.

Author: John McCarroll
"""

import math
import random
import typing

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

    dist = np.linalg.norm(position)
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
    x_range: typing.List[float] = [-100, 100]
    y_range: typing.List[float] = [0, 100]
    z_range: typing.List[float] = [0, 100]
    x_dot_range: typing.List[float] = [-20, 20]
    y_dot_range: typing.List[float] = [-20, 20]
    z_dot_range: typing.List[float] = [-20, 20]

    def __call__(self):
        agent_reset_config = {}

        # rng position
        agent_reset_config.update({"x": random.uniform(self.x_range[0], self.x_range[1])})
        agent_reset_config.update({"y": random.uniform(self.y_range[0], self.y_range[1])})
        agent_reset_config.update({"z": random.uniform(self.z_range[0], self.z_range[1])})

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

        # compare to range
        upper_x_dot_bound = axis_vel_limit if axis_vel_limit < self.x_dot_range[1] else self.x_dot_range[1]
        lower_x_dot_bound = -axis_vel_limit if -axis_vel_limit > self.x_dot_range[0] else self.x_dot_range[0]
        upper_y_dot_bound = axis_vel_limit if axis_vel_limit < self.y_dot_range[1] else self.y_dot_range[1]
        lower_y_dot_bound = -axis_vel_limit if -axis_vel_limit > self.y_dot_range[0] else self.y_dot_range[0]
        upper_z_dot_bound = axis_vel_limit if axis_vel_limit < self.z_dot_range[1] else self.z_dot_range[1]
        lower_z_dot_bound = -axis_vel_limit if -axis_vel_limit > self.z_dot_range[0] else self.z_dot_range[0]

        # initialize from tighter boundary
        agent_reset_config.update({"x_dot": random.uniform(lower_x_dot_bound, upper_x_dot_bound)})
        agent_reset_config.update({"y_dot": random.uniform(lower_y_dot_bound, upper_y_dot_bound)})
        agent_reset_config.update({"z_dot": random.uniform(lower_z_dot_bound, upper_z_dot_bound)})

        return agent_reset_config
