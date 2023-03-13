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
from corl.libraries.parameters import ConstantParameter, Parameter
from pydantic import BaseModel

from saferl.simulators.initializers.docking_initializer import velocity_limit
from saferl.simulators.initializers.initializer import BaseInitializer
from saferl.utils import VelocityConstraintValidator


class CWH3DRadialInitializer(BaseInitializer):
    """
    This class handles the initialization of agent reset parameters for the cwh 3D environment.
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
        VelocityConstraintValidator
            Config validator for the VelocityConstraintValidator.
        """
        return VelocityConstraintValidator

    def __call__(self, reset_config):

        for agent_name, agent_reset_config in reset_config.items():

            reset_config[agent_name] = self._compute_initial_conds(**agent_reset_config)

        return reset_config

    def _compute_initial_conds(
        self,
        radius: Parameter,
        azimuth_angle: Parameter,
        elevation_angle: Parameter,
        vel_max_ratio: Parameter,
        vel_azimuth_angle: Parameter,
        vel_elevation_angle: Parameter
    ) -> typing.Dict:
        """Computes radial initial conditions for cwh 3d

        Parameters
        ----------
        radius : Parameter
            radius from origin. meters
        azimuth_angle : Parameter
            location azimuthal angle in spherical coordinates (right hand convention). rad
        elevation_angle : Parameter
            location elevation angle from x-y plane. Positive angles = positive z. rad
        vel_max_ratio : Parameter
            Ratio of max safe velocity to assign to initial velocity magnitude
        vel_azimuth_angle : Parameter
            velocity vector azimuthal angle in spherical coordinates (right hand convention). rad
        vel_elevation_angle : Parameter
            velocity vector elevation angle from x-y plane. Positive angles = positive z. rad

        Returns
        -------
        typing.Dict
            initial conditions of platform
        """
        x = ConstantParameter(
            value=radius.value * np.cos(azimuth_angle.value) * np.cos(elevation_angle.value), units=radius.units.value[1][0]
        )
        y = ConstantParameter(
            value=radius.value * np.sin(azimuth_angle.value) * np.cos(elevation_angle.value), units=radius.units.value[1][0]
        )
        z = ConstantParameter(value=radius.value * np.sin(elevation_angle.value), units=radius.units.value[1][0])

        # assumes that the docking region is at origin
        relative_position = [x.get_value(0, None).value, y.get_value(0, None).value, z.get_value(0, None).value]
        distance = np.linalg.norm(relative_position)

        vel_limit = velocity_limit(
            distance,
            self.config.velocity_threshold,
            self.config.threshold_distance,
            self.config.mean_motion,
            self.config.slope
        )

        vel_mag = vel_max_ratio.value * vel_limit

        x_dot = vel_mag * np.cos(vel_azimuth_angle.value) * np.cos(vel_elevation_angle.value)
        y_dot = vel_mag * np.sin(vel_azimuth_angle.value) * np.cos(vel_elevation_angle.value)
        z_dot = vel_mag * np.sin(vel_elevation_angle.value)

        return {
            'x': x.get_value(0, None),
            'y': y.get_value(0, None),
            'z': z.get_value(0, None),
            'x_dot': x_dot,
            'y_dot': y_dot,
            'z_dot': z_dot,
        }


class CWH3DENMTInitializerValidator(BaseModel):
    """
    Validator for CWH3DENMTInitializer.

    Parameters
    ----------
    mean_motion : float
        Orbital mean motion of Hill's reference frame's circular orbit in rad/s
    """

    mean_motion: float = 0.001027


class CWH3DENMTInitializer(BaseInitializer):
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

    def __init__(self, config):
        self.config = self.get_validator(**config)

    @property
    def get_validator(self):
        """
        Returns
        -------
        CWH3DENMTInitializerValidator
            Config validator for the CWH3DENMTInitializerValidator.
        """
        return CWH3DENMTInitializerValidator

    def __call__(self, reset_config):

        for agent_name, agent_reset_config in reset_config.items():

            reset_config[agent_name] = self._compute_initial_conds(**agent_reset_config)

        return reset_config

    def _compute_initial_conds(
        self,
        radius: Parameter,
        azimuth_angle: Parameter,
        elevation_angle: Parameter,
        z_dot: Parameter,
    ) -> typing.Dict:
        """Computes eNMT initial conditions for cwh 3d

        Parameters
        ----------
        radius : Parameter
            radius from origin. meters
        azimuth_angle : Parameter
            location azimuthal angle in spherical coordinates (right hand convention). rad
        elevation_angle : Parameter
            location elevation angle from x-y plane. Positive angles = positive z. rad
        z_dot : Parameter
            The reset value for the z velocity of the agent

        Returns
        -------
        typing.Dict
            initial conditions of platform
        """
        x = ConstantParameter(
            value=radius.value * np.cos(azimuth_angle.value) * np.cos(elevation_angle.value), units=radius.units.value[1][0]
        )
        y = ConstantParameter(
            value=radius.value * np.sin(azimuth_angle.value) * np.cos(elevation_angle.value), units=radius.units.value[1][0]
        )
        z = ConstantParameter(value=radius.value * np.sin(elevation_angle.value), units=radius.units.value[1][0])

        x_dot = self.config.mean_motion * y.get_value(0, None).value / 2
        y_dot = -2 * self.config.mean_motion * x.get_value(0, None).value

        return {
            'x': x.get_value(0, None),
            'y': y.get_value(0, None),
            'z': z.get_value(0, None),
            'x_dot': x_dot,
            'y_dot': y_dot,
            'z_dot': z_dot,
        }
