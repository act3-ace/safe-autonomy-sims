import numpy as np
from act3_rl_core.simulators.base_platform import BasePlatform


class CWHPlatform(BasePlatform):
    """
    The __________________ as it's platform and
    allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function
    """

    def __init__(self, platform_name, platform, platform_config):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, parts_list=platform_config)
        self._last_applied_action = np.array([0, 0, 0], dtype=np.float32)
        self._sim_time = 0.0

    def get_applied_action(self):
        """returns the action stored in this platform

        Returns:
            typing.Any -- any sort of stored action
        """
        return self._last_applied_action

    def save_action_to_platform(self, action, axis):
        """
        saves an action to the platform if it matches
        the action space

        Arguments:
            action typing.Any -- The action to store in the platform
        """
        self._last_applied_action[axis] = action

    @property
    def position(self):
        return self._platform.position

    @property
    def velocity(self):
        return self._platform.velocity

    @property
    def sim_time(self):
        return self._sim_time

    @sim_time.setter
    def sim_time(self, time):
        self._sim_time = time

    @property
    def operable(self):
        return True
