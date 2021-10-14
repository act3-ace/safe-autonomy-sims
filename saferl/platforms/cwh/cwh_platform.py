import typing

import numpy as np
from act3_rl_core.simulators.base_parts import BaseController, BaseSensor
from act3_rl_core.simulators.base_platform import BasePlatform


class CWHPlatform(BasePlatform):
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
        self.next_action = [0, 0, 0]
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

    def get_applied_action(self):
        return self.next_action

    @property
    def operable(self):
        return True
