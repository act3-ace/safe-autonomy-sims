"""
This module contains the CWH Simulator for interacting with the CWH Docking task simulator
"""

from act3_rl_core.libraries.plugin_library import PluginLibrary

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator
from saferl_sim.cwh.cwh import CWHSpacecraft


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
