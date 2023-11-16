"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module implements a common platform object for the safe autonomy environments.
"""

from corl.simulators.base_platform import BasePlatform


class BaseSafeRLPlatform(BasePlatform):
    """
    Base platform for safe autonomy sims
    Allows for saving an action to the platform for when the platform needs
    to give an action to the environment during the environment step function.

    Parameters
    ----------
    platform_name : str
        Name of the platform.
    platform : sim_entity
        Backend simulation entity associated with the platform.
    parts_list : dict
        Platform-specific parts configuration dictionary.
    sim_time : float
        simulation time at platform creation
    """

    def __init__(self, platform_name, platform, parts_list, sim_time=0.0):  # pylint: disable=W0613
        super().__init__(platform_name=platform_name, platform=platform, parts_list=parts_list)
        self._sim_time = sim_time

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
