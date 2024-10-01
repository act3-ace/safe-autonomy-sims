"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements initializers for CWH platforms.
"""

import typing

import numpy as np
from corl.libraries.units import Quantity
from scipy.spatial.transform import Rotation

from safe_autonomy_sims.simulators.initializers.initializer import BaseInitializer, BaseInitializerWithPint, InitializerValidator, strip_units_from_dict
from safe_autonomy_sims.utils import velocity_limit


class CWH3DRadialInitializer(BaseInitializerWithPint):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    Both position and velocity are initialized radially (with radius and angles) to allow for control over magnitude and direction
    of the resulting vectors.
    """

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'vel_mag': 'meter/second',
        'vel_azimuth_angle': 'radians',
        'vel_elevation_angle': 'radians',
    }

    def compute(self, **kwargs):
        return self._compute_with_units(**strip_units_from_dict(kwargs))

    # TODO: change name or function to reflect unit use. Currently, no units used.
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
            magnitude of velocity vector. meter/second
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
            'position': np.array([x, y, z]),
            'velocity': np.array([x_dot, y_dot, z_dot])
        }


class CWH3DENMTInitializerValidator(InitializerValidator):
    """
    A configuration validator for CWH3DENMTInitializer.

    Attributes
    ----------
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    """

    mean_motion: float = 0.001027


class CWH3DENMTInitializer(BaseInitializerWithPint):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    It ensures that each agent starts on an elliptical Natural Motion Trajectory (eNMT)
    """

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'z_dot': 'meter/second',
    }

    @staticmethod
    def get_validator():
        """
        Returns
        -------
        CWH3DENMTInitializerValidator
            Config validator for the CWH3DENMTInitializerValidator.
        """
        return CWH3DENMTInitializerValidator

    def compute(self, **kwargs):
        return self._compute_with_units(**strip_units_from_dict(kwargs))

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
            'position': np.array([x, y, z]),
            'velocity': np.array([x_dot, y_dot, z_dot])
        }
    
class PositionVelocityInitializer(BaseInitializer):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    Both position and velocity are initialized by specifying the individual components of position
    velocity..
    """

    def compute(self, **kwargs):
        return self._compute(**kwargs)
    
    def _compute(self,
                x: float,
                y: float,
                z: float,
                x_dot: float,
                y_dot: float,
                z_dot: float
    ):
        """Computes initial conditions for cwh 3d

        Parameters
        ----------
        x : float
            The x component of the position, meters
        y : float
            The y component of the position, meters
        z : float
            The z component of the position, meters
        x_dot : float
            The reset value for the x velocity of the agent
        y_dot : float
            The reset value for the y velocity of the agent
        z_dot : float
            The reset value for the z velocity of the agent

        Returns
        -------
        typing.Dict
            initial conditions of platform
        """

        return {
            "position": np.array([x, y, z]),
            "velocity": np.array([x_dot, y_dot, z_dot])
        }

class Docking3DRadialInitializerValidator(InitializerValidator):
    """
    A configuration validator for Docking3DInitializer.

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
    """

    velocity_threshold: float
    threshold_distance: float
    slope: float = 2.0
    mean_motion: float = 0.001027


class Docking3DRadialInitializer(BaseInitializerWithPint):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D docking environment.
    It ensures that the initial velocity of the deputy does not violate the maximum velocity safety constraint.
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

    @staticmethod
    def get_validator():
        """
        Returns
        -------
        Docking3DInitializerValidator
            Config validator for the Docking3DInitializerValidator.
        """
        return Docking3DRadialInitializerValidator

    def compute(self, **kwargs):
        return self._compute_with_units(**strip_units_from_dict(kwargs))

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
        # contend with corl Quantities
        radius = radius.value if isinstance(radius, Quantity) else radius
        azimuth_angle = azimuth_angle.value if isinstance(azimuth_angle, Quantity) else azimuth_angle
        elevation_angle = elevation_angle.value if isinstance(elevation_angle, Quantity) else elevation_angle
        vel_max_ratio = vel_max_ratio.value if isinstance(vel_max_ratio, Quantity) else vel_max_ratio
        vel_azimuth_angle = vel_azimuth_angle.value if isinstance(vel_azimuth_angle, Quantity) else vel_azimuth_angle
        vel_elevation_angle = vel_elevation_angle.value if isinstance(vel_elevation_angle, Quantity) else vel_elevation_angle

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
            'position': np.array([x, y, z]),
            'velocity': np.array([x_dot, y_dot, z_dot])
        }


class CWHSixDOFRadialInitializer(BaseInitializerWithPint):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    Both position and velocity are initialized radially (with radius and angles) to allow for control over magnitude and direction
        of the resulting vectors.
    """

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'vel_mag': 'meter/second',
        'vel_azimuth_angle': 'radians',
        'vel_elevation_angle': 'radians',
        'wx': 'radian/second',
        'wy': 'radian/second',
        'wz': 'radian/second',
    }

    def compute(self, **kwargs):
        return self._compute_with_units(**strip_units_from_dict(kwargs))

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
            magnitude of velocity vector. meter/second
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
            'position': np.array([x, y, z]),
            'velocity': np.array([x_dot, y_dot, z_dot]),
            'orientation': np.array([q[0], q[1], q[2], q[3]]),
            'angular_velocity': np.array([wx, wy, wz])
        }


class CWH3DRadialWithSunInitializer(BaseInitializerWithPint):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
    Both position and velocity are initialized radially (with radius and angles) to allow for
    control over magnitude and direction of the resulting vectors.
    The sun angle is also passed through.
    """

    param_units = {
        'radius': 'meters',
        'azimuth_angle': 'radians',
        'elevation_angle': 'radians',
        'vel_mag': 'meter/second',
        'vel_azimuth_angle': 'radians',
        'vel_elevation_angle': 'radians',
        'sun_angle': 'radians',
    }

    def compute(self, **kwargs):
        return self._compute_with_units(**strip_units_from_dict(kwargs))

    def _compute_with_units(
        self,
        radius: float,
        azimuth_angle: float,
        elevation_angle: float,
        vel_mag: float,
        vel_azimuth_angle: float,
        vel_elevation_angle: float,
        sun_angle: float,
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
            magnitude of velocity vector. meter/second
        vel_azimuth_angle : float
            velocity vector azimuthal angle in spherical coordinates (right hand convention). rad
        vel_elevation_angle : float
            velocity vector elevation angle from x-y plane. Positive angles = positive z. rad
        sun_angle : float
            Initial angle of the sun. rad

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
            'position': np.array([x, y, z]),
            'velocity': np.array([x_dot, y_dot, z_dot]),
            'sun_angle': sun_angle,
        }
