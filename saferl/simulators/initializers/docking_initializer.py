"""
This module defines the docking initializer class.

Author: John McCarroll, Umberto Ravaioli
"""
import math
import random
import typing

import numpy as np
from pydantic import BaseModel

from saferl.simulators.initializers.initializer import BaseInitializer, BaseInitializerWithUnits
from saferl.utils import velocity_limit


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

            # TODO: get position relative to chief/reference entity!
            # assumes that the docking region is at origin
            relative_position = np.array([agent_reset_config["x"].value, agent_reset_config["y"].value, agent_reset_config["z"].value])
            distance = np.linalg.norm(relative_position)

            # constrained rng velocity
            vel_limit = velocity_limit(
                distance, self.config.velocity_threshold, self.config.threshold_distance, self.config.mean_motion, self.config.slope
            )

            # find magnitude of x,y,z components of max vel limit given position
            axis_vel_limit = math.sqrt(vel_limit**2 / 3)

            # initialize from boundary
            agent_reset_config.update({"xdot": random.uniform(-axis_vel_limit, axis_vel_limit)})
            agent_reset_config.update({"ydot": random.uniform(-axis_vel_limit, axis_vel_limit)})
            agent_reset_config.update({"zdot": random.uniform(-axis_vel_limit, axis_vel_limit)})

        return reset_config


class Docking3DRadialInitializerValidator(BaseModel):
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


class Docking3DRadialInitializer(BaseInitializerWithUnits):
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

    @property
    def get_validator(self):
        """
        Returns
        -------
        Docking3DInitializerValidator
            Config validator for the Docking3DInitializerValidator.
        """
        return Docking3DRadialInitializerValidator

    def compute_initial_conds(self, **kwargs) -> typing.Dict:
        return self._compute_initial_conds(**kwargs)

    def _compute_initial_conds(
        self,
        radius: float,
        azimuth_angle: float,
        elevation_angle: float,
        vel_max_ratio: float,
        vel_azimuth_angle: float,
        vel_elevation_angle: float
    ) -> typing.Dict:
        """Computes radial initial conditions for 3d docking problem

        Parameters
        ----------
        radius : float
            radius from origin. meters
        azimuth_angle : float
            location azimuthal angle in spherical coordinates (right hand convention). rad
        elevation_angle : float
            location elevation angle from x-y plane. Positive angles = positive z. rad
        vel_max_ratio : float
            Ratio of max safe velocity to assign to initial velocity magnitde
        vel_azimuth_angle : float
            velocity vector azimuthal angle in spherical coordinates (right hand convention). rad
        vel_elevation_angle : float
            velocity vector elevation angle from x-y plane. Positive angles = positive z. rad

        Returns
        -------
        typing.Dict
            initial conditions of platform
        """
        x = radius * np.cos(azimuth_angle) * np.cos(elevation_angle)
        y = radius * np.sin(azimuth_angle) * np.cos(elevation_angle)
        z = radius * np.sin(elevation_angle)

        distance = np.linalg.norm([x, y, z])

        vel_limit = velocity_limit(
            distance, self.config.velocity_threshold, self.config.threshold_distance, self.config.mean_motion, self.config.slope
        )

        vel_mag = vel_max_ratio * vel_limit

        x_dot = vel_mag * np.cos(vel_azimuth_angle) * np.cos(vel_elevation_angle)
        y_dot = vel_mag * np.sin(vel_azimuth_angle) * np.cos(vel_elevation_angle)
        z_dot = vel_mag * np.sin(vel_elevation_angle)

        return {
            'x': x,
            'y': y,
            'z': z,
            'x_dot': x_dot,
            'y_dot': y_dot,
            'z_dot': z_dot,
        }
