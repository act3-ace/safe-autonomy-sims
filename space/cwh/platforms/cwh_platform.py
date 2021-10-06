import typing

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

    @property
    def name(self) -> str:
        return self._platform.name

    @property
    def position(self):
        return self._platform.position

    @property
    def velocity(self):
        return self._platform.velocity

    def get_applied_action(self):
        ...
