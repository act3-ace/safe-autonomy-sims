"""
This module defines the platforms used with saferl Dubins2dSimulator and Subins3dSimulator classes. It represents an
aircraft operating under the Dubins dynamics model.
"""

import numpy as np
from act3_rl_core.simulators.base_platform import BasePlatform


class DubinsPlatform(BasePlatform):
    """
    A platform representing an aircraft operating under Dubins dynamics.
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function

    Parameters
    ----------
    platform_name : str
        Name of the platform
    platform : sim_entity
        Backend simulation entity associated with the platform
    platform_config : dict
        Platform-specific configuration dictionary
    """

    def __init__(self, platform_name, platform, platform_config):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, parts_list=platform_config)
        self._last_applied_action = None
        self._sim_time = 0.0

    def get_applied_action(self):
        """returns the action stored in this platform

        Returns:
            typing.Any -- any sort of stored action
        """
        return self._last_applied_action

    def save_action_to_platform(self, action, axis=None):
        """
        saves an action to the platform if it matches
        the action space

        Arguments:
            action typing.Any -- The action to store in the platform
        """
        if axis is not None:
            self._last_applied_action[axis] = action
        else:
            self._last_applied_action = action

    @property
    def position(self):
        """
        The position of the platform

        Returns
        -------
        np.ndarray
            The position vector of the platform
        """
        return self._platform.position

    @property
    def velocity(self):
        """
        The velocity of the platform

        Returns
        -------
        np.ndarray
            The velocity vector of the platform
        """
        return self._platform.velocity

    @property
    def heading(self):
        """
        The heading of the platform

        Returns
        -------
        float
            The heading angle of the platform in radians
        """
        return self._platform.heading

    @property
    def sim_time(self):
        """
        The current simulation time in seconds.

        Returns
        -------
        float
            Current simulation time
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
    to give an action to the environment during the environment step function

    Parameters
    ----------
    platform_name : str
        Name of the platform
    platform : sim_entity
        Backend simulation entity associated with the platform
    platform_config : dict
        Platform-specific configuration dictionary
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
        Name of the platform
    platform : sim_entity
        Backend simulation entity associated with the platform
    platform_config : dict
        Platform-specific configuration dictionary
    """

    def __init__(self, platform_name, platform, platform_config):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, platform_config=platform_config)
        self._last_applied_action = np.array([0, 0, 0], dtype=np.float32)  # elevator, ailerons, throttle

    @property
    def flight_path_angle(self):
        """
        The flight path angle of the platform

        Returns
        -------
        float
            The flight path angle of the platform in radians
        """
        return self._platform.gamma

    @property
    def roll(self):
        """
        The roll of the platform

        Returns
        -------
        float
            The roll angle of the platform in radians
        """
        return self._platform.roll


# class Dubins6DOFPlatform(Base6DOFPlatform):
#     """
#     The __________________ as it's platform and
#     allows for saving an action to the platform for when the platform needs
#     to give an action to the environment during the environment step function
#     """
#
#     def __init__(self, platform, platform_config, seed=None):  # pylint: disable=W0613
#         self._platform = platform
#         self._controllers: typing.Tuple[BaseController, ...] = ()
#         self._sensors: typing.Tuple[BaseSensor, ...] = ()
#         self.next_action = [0, 0, 0]
#         self._controllers = tuple(part[0](self, part[1]) for part in platform_config if issubclass(part[0], BaseController))
#         self._sensors = tuple(part[0](self, part[1]) for part in platform_config if issubclass(part[0], BaseSensor))
#
#     @property
#     def name(self) -> str:
#         return self._platform.name
#
#     @property
#     def position(self) -> np.ndarray:
#         # TODO: find appropriate observer values (this is the Statue of Liberty)
#         obs_lat, obs_lon, obs_a = 40.6892, 74.0445, 10000
#         n, e, d = self._platform.position
#         lla = np.array(pm.ned.ned2geodetic(n, e, d, lat0=obs_lat, lon0=obs_lon, h0=obs_a))
#         return lla
#
#     @property
#     def orientation(self) -> np.ndarray:
#         yaw = self._platform.yaw
#         pitch = self._platform.pitch
#         roll = self._platform.roll
#         return np.array([yaw, pitch, roll])
#
#     @property
#     def velocity_ned(self) -> np.ndarray:
#         return self._platform.velocity
#
#     @property
#     def acceleration_ned(self) -> np.ndarray:
#         return self._platform.acceleration
#
#     @property
#     def speed(self) -> np.ndarray:
#         return self._platform.v
#
#     @property
#     def sensors(self) -> typing.Tuple[BaseSensor, ...]:
#         return self._sensors
#
#     @property
#     def controllers(self) -> typing.Tuple[BaseController, ...]:
#         return self._controllers
#
#     def get_applied_action(self):
#         return self.next_action
#
#     @property
#     def operable(self):
#         return True
