"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains CWH Initializers
"""

import typing

import numpy as np
from scipy.spatial.transform import Rotation

from safe_autonomy_sims.simulators.initializers.initializer import InitializerValidator, PintUnitConversionInitializer
from safe_autonomy_sims.utils import velocity_limit


class CWH3DRadialInitializer(PintUnitConversionInitializer):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    Both position and velocity are initialized radially (with radius and angles) to allow for control over magnitude and direction
        of the resulting vectors.

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

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'vel_mag': 'meters/second',
        'vel_azimuth_angle': 'radians',
        'vel_elevation_angle': 'radians',
    }

    def compute_with_units(self, kwargs_with_converted_units, kwargs_with_stripped_units):
        return self._compute_with_units(**kwargs_with_stripped_units)

    def _compute_with_units(
        self,
        radius: float,
        azimuth_angle: float,
        elevation_angle: float,
        vel_mag: float,
        vel_azimuth_angle: float,
        vel_elevation_angle: float,
    ) -> typing.Dict:
        """Computes radial initial conditions for cwh 3d

        Parameters
        ----------
        radius : float
            radius from origin. meters
        azimuth_angle : float
            location azimuthal angle in spherical coordinates (right hand convention). rad
        elevation_angle : float
            location elevation angle from x-y plane. Positive angles = positive z. rad
        vel_mag : float
            magnitude of velocity vector. meters/second
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


class CWH3DENMTInitializerValidator(InitializerValidator):
    """
    Validator for CWH3DENMTInitializer.

    Parameters
    ----------
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    """

    mean_motion: float = 0.001027


class CWH3DENMTInitializer(PintUnitConversionInitializer):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    It ensures that each agent starts on an elliptical Natural Motion Trajectory (eNMT)

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

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'z_dot': 'meters/second',
    }

    @property
    def get_validator(self):
        """
        Returns
        -------
        CWH3DENMTInitializerValidator
            Config validator for the CWH3DENMTInitializerValidator.
        """
        return CWH3DENMTInitializerValidator

    def compute_with_units(self, kwargs_with_converted_units, kwargs_with_stripped_units):
        return self._compute_with_units(**kwargs_with_stripped_units)

    def _compute_with_units(
        self,
        radius: float,
        azimuth_angle: float,
        elevation_angle: float,
        z_dot: float,
    ) -> typing.Dict:
        """Computes eNMT initial conditions for cwh 3d

        Parameters
        ----------
        radius : float
            radius from origin. meters
        azimuth_angle : float
            location azimuthal angle in spherical coordinates (right hand convention). rad
        elevation_angle : float
            location elevation angle from x-y plane. Positive angles = positive z. rad
        z_dot : float
            The reset value for the z velocity of the agent

        Returns
        -------
        typing.Dict
            initial conditions of platform
        """
        x = radius * np.cos(azimuth_angle) * np.cos(elevation_angle)
        y = radius * np.sin(azimuth_angle) * np.cos(elevation_angle)
        z = radius * np.sin(elevation_angle)

        x_dot = self.config.mean_motion * y / 2
        y_dot = -2 * self.config.mean_motion * x

        return {
            'x': x,
            'y': y,
            'z': z,
            'x_dot': x_dot,
            'y_dot': y_dot,
            'z_dot': z_dot,
        }


class Docking3DRadialInitializerValidator(InitializerValidator):
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
    """

    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027


class Docking3DRadialInitializer(PintUnitConversionInitializer):
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

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'vel_max_ratio': None,
        'vel_azimuth_angle': 'radians',
        'vel_elevation_angle': 'radians',
    }

    def __init__(self, config):
        super().__init__(config)
        self.config: Docking3DRadialInitializerValidator

    @property
    def get_validator(self):
        """
        Returns
        -------
        Docking3DInitializerValidator
            Config validator for the Docking3DInitializerValidator.
        """
        return Docking3DRadialInitializerValidator

    def compute_with_units(self, kwargs_with_converted_units, kwargs_with_stripped_units):
        return self._compute_with_units(**kwargs_with_stripped_units)

    def _compute_with_units(
        self,
        radius: float,
        azimuth_angle: float,
        elevation_angle: float,
        vel_max_ratio: float,
        vel_azimuth_angle: float,
        vel_elevation_angle: float,
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


class CWHSixDOFRadialInitializer(PintUnitConversionInitializer):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    Both position and velocity are initialized radially (with radius and angles) to allow for control over magnitude and direction
        of the resulting vectors.

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

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'vel_mag': 'meters/second',
        'vel_azimuth_angle': 'radians',
        'vel_elevation_angle': 'radians',
        'wx': 'radians/second',
        'wy': 'radians/second',
        'wz': 'radians/second',
    }

    def compute_with_units(self, kwargs_with_converted_units, kwargs_with_stripped_units):
        return self._compute_with_units(**kwargs_with_stripped_units)

    def _compute_with_units(
        self,
        radius: float,
        azimuth_angle: float,
        elevation_angle: float,
        vel_mag: float,
        vel_azimuth_angle: float,
        vel_elevation_angle: float,
        wx: float,
        wy: float,
        wz: float,
    ) -> typing.Dict:
        """Computes radial initial conditions for cwh 3d

        Parameters
        ----------
        radius : float
            radius from origin. meters
        azimuth_angle : float
            location azimuthal angle in spherical coordinates (right hand convention). rad
        elevation_angle : float
            location elevation angle from x-y plane. Positive angles = positive z. rad
        vel_mag : float
            magnitude of velocity vector. meters/second
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

        x_dot = vel_mag * np.cos(vel_azimuth_angle) * np.cos(vel_elevation_angle)
        y_dot = vel_mag * np.sin(vel_azimuth_angle) * np.cos(vel_elevation_angle)
        z_dot = vel_mag * np.sin(vel_elevation_angle)

        q = Rotation.random().as_quat()

        return {
            'x': x,
            'y': y,
            'z': z,
            'x_dot': x_dot,
            'y_dot': y_dot,
            'z_dot': z_dot,
            'q1': q[0],
            'q2': q[1],
            'q3': q[2],
            'q4': q[3],
            'wx': wx,
            'wy': wy,
            'wz': wz,
        }
