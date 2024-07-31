"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module contains the CWH Simulator for interacting with the CWH Docking task simulator
"""

from corl.libraries.plugin_library import PluginLibrary
from safe_autonomy_simulation.sims.spacecraft import CWHSpacecraft

from safe_autonomy_sims.platforms.cwh.cwh_platform import CWHPlatform
from safe_autonomy_sims.simulators.saferl_simulator import SafeRLSimulator


class CWHSimulator(SafeRLSimulator):
    """
    Simulator for CWH Docking Task. Interfaces CWH platforms with underlying CWH entities in Docking simulation.
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (CWHSpacecraft, CWHPlatform),
            'cwh': (CWHSpacecraft, CWHPlatform),
        }


PluginLibrary.AddClassToGroup(CWHSimulator, "CWHSimulator", {})
