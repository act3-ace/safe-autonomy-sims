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
from safe_autonomy_dynamics.cwh import CWHSpacecraft

from saferl.platforms.cwh.cwh_platform import CWHPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator


class InspectionSimulator(SafeRLSimulator):
    """
    Simulator for CWH Inspection Task. Interfaces CWH platforms with underlying CWH entities in Inspection simulation.
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (CWHSpacecraft, CWHPlatform),
            'cwh': (CWHSpacecraft, CWHPlatform),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #TODO: concern, we just added a new variable to StateDict... do we need to extend it or does it not know there is a points variable until
        self._state.points = self._add_points()

    def reset(self, config):
        super().reset(config)
        #self._state.clear()
        self._state.points = self._add_points()
        return self._state

    def step(self):
        super().step()
        #update points
        for platform in self._state.sim_platforms:
            agent_id = platform.name
            #action = np.array(platform.get_applied_action(), dtype=np.float32)
            #entity = self.sim_entities[agent_id]
            #entity.step(action=action, step_size=self.config.step_size)
            #platform.sim_time = self.clock
        #return same as parent
        return self._stat

    def _add_points(self) -> dict:
        points_dict = {}
        points_dict[(1, 2, 3)]= False
        points_dict[(4, 5, 6)]= False
        return points_dict

PluginLibrary.AddClassToGroup(InspectionSimulator, "InspectionSimulator", {})
