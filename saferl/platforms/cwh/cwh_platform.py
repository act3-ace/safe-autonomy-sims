"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module defines the platform used with saferl CWHSimulator class. It represents a spacecraft
operating under the Clohessy-Wiltshire dynamics model.
"""

import numpy as np

from saferl.platforms.common.platform import BaseSafeRLPlatform


class CWHPlatform(BaseSafeRLPlatform):
    """
    A platform representing a spacecraft operating under CWH dynamics.
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
    sim_time : float
        simulation time at platform creation
    """

    def __init__(self, platform_name, platform, platform_config, sim_time=0.0):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, parts_list=platform_config, sim_time=sim_time)
        self._last_applied_action = np.array([0, 0, 0], dtype=np.float32)

    def get_applied_action(self):
        """
        Returns the action stored in this platform.

        Returns
        -------
        typing.Any -- Any sort of stored action.
        """
        return self._last_applied_action

    def __eq__(self, other):
        if isinstance(other, CWHPlatform):
            eq = (self.velocity == other.velocity).all()
            eq = eq and (self.position == other.position).all()
            eq = eq and self.sim_time == other.sim_time
            return eq
        return False

    def save_action_to_platform(self, action, axis):
        """
        Saves an action to the platform if it matches the action space.

        Parameters
        ----------
        action: typing.Any
            The action to store in the platform
        axis: int
            The index of the action space where the action shall be saved.
        """
        self._last_applied_action[axis] = action

    @property
    def position(self) -> np.ndarray:
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
