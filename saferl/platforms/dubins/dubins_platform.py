import typing

import numpy as np
import pymap3d as pm
from act3_rl_core.simulators.base_parts import BaseController, BaseSensor
from act3_rl_core.simulators.six_dof.base_six_dof_platform import BasePlatform, Base6DOFPlatform


class DubinsPlatform(BasePlatform):
    """
    The __________________ as it's platform and
    allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function
    """

    def __init__(self, platform, platform_config, seed=None):  # pylint: disable=W0613
        self._platform = platform
        self._controllers: typing.Tuple[BaseController, ...] = ()
        self._sensors: typing.Tuple[BaseSensor, ...] = ()
        self.next_action = np.array([0, 0, 0], dtype=np.float32)
        self._controllers = tuple(part[0](self, part[1]) for part in platform_config if issubclass(part[0], BaseController))
        self._sensors = tuple(part[0](self, part[1]) for part in platform_config if issubclass(part[0], BaseSensor))
        self._sim_time = 0.0

    @property
    def name(self) -> str:
        return self._platform.name

    @property
    def position(self):
        return self._platform.position

    @property
    def velocity(self):
        return self._platform.velocity

    @property
    def sensors(self):
        return self._sensors

    @property
    def controllers(self):
        return self._controllers

    @property
    def sim_time(self):
        return self._sim_time

    @sim_time.setter
    def sim_time(self, time):
        self._sim_time = time

    def get_applied_action(self):
        return self.next_action

    @property
    def operable(self) -> bool:
        return True


class Dubins6DOFPlatform(Base6DOFPlatform):
    """
    The __________________ as it's platform and
    allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function
    """

    def __init__(self, platform, platform_config, seed=None):  # pylint: disable=W0613
        self._platform = platform
        self._controllers: typing.Tuple[BaseController, ...] = ()
        self._sensors: typing.Tuple[BaseSensor, ...] = ()
        self.next_action = [0, 0, 0]
        self._controllers = tuple(part[0](self, part[1]) for part in platform_config if issubclass(part[0], BaseController))
        self._sensors = tuple(part[0](self, part[1]) for part in platform_config if issubclass(part[0], BaseSensor))

    @property
    def name(self) -> str:
        return self._platform.name

    @property
    def position(self) -> np.ndarray:
        # TODO: find appropriate observer values (this is the Statue of Liberty)
        obs_lat, obs_lon, obs_a = 40.6892, 74.0445, 10000
        n, e, d = self._platform.position
        lla = np.array(pm.ned.ned2geodetic(n, e, d, lat0=obs_lat, lon0=obs_lon, h0=obs_a))
        return lla

    @property
    def orientation(self) -> np.ndarray:
        yaw = self._platform.yaw
        pitch = self._platform.pitch
        roll = self._platform.roll
        return np.array([yaw, pitch, roll])

    @property
    def velocity_ned(self) -> np.ndarray:
        return self._platform.velocity

    @property
    def acceleration_ned(self) -> np.ndarray:
        return self._platform.acceleration

    @property
    def speed(self) -> np.ndarray:
        return self._platform.v

    @property
    def sensors(self) -> typing.Tuple[BaseSensor, ...]:
        return self._sensors

    @property
    def controllers(self) -> typing.Tuple[BaseController, ...]:
        return self._controllers

    def get_applied_action(self):
        return self.next_action

    @property
    def operable(self):
        return True
