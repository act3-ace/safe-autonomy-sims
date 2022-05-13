"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module defines the platforms used with saferl Dubins2dSimulator and Dubins3dSimulator classes. It represents an
aircraft operating under the Dubins dynamics model.
"""

import numpy as np
from corl.simulators.base_platform import BasePlatform


class DubinsPlatform(BasePlatform):
    """
    A platform representing an aircraft operating under Dubins dynamics.
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function.

    Parameters
    ----------
    platform_name : str
        Name of the platform.
    platform : sim_entity
        Backend simulation entity associated with the platform.
    platform_config : dict
        Platform-specific configuration dictionary.
    """

    def __init__(self, platform_name, platform, platform_config):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, parts_list=platform_config)
        self._last_applied_action = None
        self._sim_time = 0.0

    def __eq__(self, other):
        if isinstance(other, DubinsPlatform):
            eq = np.allclose(self.velocity, other.velocity)
            eq = eq and np.allclose(self.position, other.position)
            eq = eq and np.allclose(self.orientation.as_euler("zyx"), other.orientation.as_euler("zyx"))
            eq = eq and self.heading == other.heading
            eq = eq and self.sim_time == other.sim_time
            return eq
        return False

    def get_applied_action(self):
        """
        Returns the action stored in this platform.

        Returns:
            typing.Any -- Any sort of stored action.
        """
        return self._last_applied_action

    def save_action_to_platform(self, action, axis=None):
        """
        Saves an action to the platform if it matches the action space.

        Arguments:
            action typing.Any -- The action to store in the platform.
        """
        if axis is not None:
            self._last_applied_action[axis] = action
        else:
            self._last_applied_action = action

    @property
    def position(self):
        """
        The position of the platform.

        Returns
        -------
        np.ndarray
            The position vector of the platform.
        """
        return self._platform.position

    @property
    def velocity(self):
        """
        The velocity of the platform.

        Returns
        -------
        np.ndarray
            The velocity vector of the platform.
        """
        return self._platform.velocity

    @property
    def heading(self):
        """
        The heading of the platform.

        Returns
        -------
        float
            The heading angle of the platform in radians.
        """
        return self._platform.heading

    @property
    def orientation(self):
        """
        The orientation of the platform.

        Returns
        -------
        scipy.Rotation
            The scipy rotation of the platform.
        """
        return self._platform.orientation

    # TODO: don't require this property
    @property
    def partner(self):
        """
        The platform's partner entity.

        Returns
        -------
        BaseEntity
            The platform's partner or None if partner does not exist
        """
        return self._platform.partner

    @property
    def partner_position(self):
        """
        The position of the platform's partner.

        Returns
        -------
        np.ndarray
            The position vector of the platform's partner or None if partner is not specified.
        """
        return self._platform.partner.position if self._platform.partner is not None else None

    @property
    def partner_velocity(self):
        """
        The velocity of the platform's partner.

        Returns
        -------
        np.ndarray
            The velocity vector of the platform's partner.
        """
        return self._platform.partner.velocity if self._platform.partner is not None else None

    @property
    def partner_heading(self):
        """
        The heading of the platform's partner.

        Returns
        -------
        float
            The heading angle of the platform's partner in radians.
        """
        return self._platform.partner.heading if self._platform.partner is not None else None

    @property
    def partner_orientation(self):
        """
        The orientation of the platform's partner.

        Returns
        -------
        scipy.Rotation
            The scipy rotation of the platform's partner.
        """
        return self._platform.partner.orientation if self._platform.partner is not None else None

    @property
    def sim_time(self):
        """
        The current simulation time in seconds.

        Returns
        -------
        float
            Current simulation time.
        """
        return self._sim_time

    @sim_time.setter
    def sim_time(self, time):
        self._sim_time = time

    @property
    def operable(self):
        return True


class Dubins2dPlatform(DubinsPlatform):
    """
    A platform representing an aircraft operating under 2D Dubins dynamics.
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function.

    Parameters
    ----------
    platform_name : str
        Name of the platform.
    platform : sim_entity
        Backend simulation entity associated with the platform.
    platform_config : dict
        Platform-specific configuration dictionary.
    """

    def __init__(self, platform_name, platform, platform_config):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, platform_config=platform_config)
        self._last_applied_action = np.array([0, 0], dtype=np.float32)  # turn rate, acceleration


class Dubins3dPlatform(DubinsPlatform):
    """
    A platform representing an aircraft operating under 3D Dubins dynamics.
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function

    Parameters
    ----------
    platform_name : str
        Name of the platform.
    platform : sim_entity
        Backend simulation entity associated with the platform.
    platform_config : dict
        Platform-specific configuration dictionary.
    """

    def __init__(self, platform_name, platform, platform_config):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, platform_config=platform_config)
        self._last_applied_action = np.array([0, 0, 0], dtype=np.float32)  # elevator, ailerons, throttle

    @property
    def flight_path_angle(self):
        """
        The flight path angle of the platform.

        Returns
        -------
        float
            The flight path angle of the platform in radians.
        """
        return self._platform.gamma

    @property
    def roll(self):
        """
        The roll of the platform.

        Returns
        -------
        float
            The roll angle of the platform in radians.
        """
        return self._platform.roll

    @property
    def partner_flight_path_angle(self):
        """
        The flight path angle of the platform's partner.

        Returns
        -------
        float
            The flight path angle of the platform's partner in radians.
        """
        return self._platform.partner.gamma if self._platform.partner is not None else None

    @property
    def partner_roll(self):
        """
        The roll of the platform's partner.

        Returns
        -------
        float
            The roll angle of the platform's partner in radians.
        """
        return self._platform.partner.roll if self._platform.partner is not None else None
