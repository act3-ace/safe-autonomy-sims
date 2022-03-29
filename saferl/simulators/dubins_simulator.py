"""
Contains the implementations of classes that describe how the simulation is to proceed.
"""
import abc

from act3_rl_core.libraries.plugin_library import PluginLibrary

import saferl_sim.dubins.entities as bp
from saferl.platforms.dubins.dubins_platform import Dubins2dPlatform, Dubins3dPlatform
from saferl.simulators.saferl_simulator import SafeRLSimulator, SafeRLSimulatorValidator


class DubinsSimulatorValidator(SafeRLSimulatorValidator):
    """
    lead: the name of the lead aircraft
    """
    lead: str = ""


class DubinsSimulator(SafeRLSimulator):
    """
    Base simulator class for Dubins aircraft platforms.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.config.lead != "":
            self.register_lead(self.config.lead)

    def register_lead(self, lead):
        """
        Register the lead platform with all wingman platforms.

        Parameters
        ----------
        lead: str
            The id of the lead aircraft platform

        Returns
        -------
        None
        """
        lead_entity = self.sim_entities[lead]
        for agent_id, entity in self.sim_entities:
            if agent_id != lead:
                entity.register_partner(lead_entity)

    @abc.abstractmethod
    def _construct_platform_map(self) -> dict:
        raise NotImplementedError


class Dubins2dSimulator(DubinsSimulator):
    """
    A class that contains all essential components of a Dubins2D simulation
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (bp.Dubins2dAircraft, Dubins2dPlatform),
            'dubins2d': (bp.Dubins2dAircraft, Dubins2dPlatform),
        }


PluginLibrary.AddClassToGroup(Dubins2dSimulator, "Dubins2dSimulator", {})


class Dubins3dSimulator(DubinsSimulator):
    """
    A class that contains all essential components of a Dubins 3D simulation
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (bp.Dubins3dAircraft, Dubins3dPlatform),
            'dubins3d': (bp.Dubins3dAircraft, Dubins3dPlatform),
        }


PluginLibrary.AddClassToGroup(Dubins3dSimulator, "Dubins3dSimulator", {})
