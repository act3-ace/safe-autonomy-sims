"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

Contains the implementations of classes that describe how the simulation is to proceed.
"""
import abc

from corl.libraries.plugin_library import PluginLibrary

from saferl.backend.dubins import entities as bp
from saferl.core.platforms.dubins.dubins_platform import Dubins2dPlatform, Dubins3dPlatform
from saferl.core.simulators.saferl_simulator import SafeRLSimulator, SafeRLSimulatorValidator


class DubinsSimulatorValidator(SafeRLSimulatorValidator):
    """
    Validator for DubinsSimulator.

    lead: str
        The name of the lead aircraft.
    """
    lead: str = ""


class DubinsSimulator(SafeRLSimulator):
    """
    Base simulator class for Dubins aircraft platforms.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_lead()

    @property
    def get_simulator_validator(self):
        return DubinsSimulatorValidator

    def reset(self, config):
        config = self.get_reset_validator(**config)
        self._state.clear()
        self.clock = 0.0
        self.sim_entities = self.construct_sim_entities(config.platforms)
        self.register_lead()
        self._state.sim_platforms = self.construct_platforms()
        self.update_sensor_measurements()
        return self._state

    def register_lead(self):
        """
        Registers the lead platform with all wingman platforms.

        Returns
        -------
        None
        """
        lead = self.config.lead
        if lead != "":
            lead_entity = self.sim_entities[lead]
            for agent_id, entity in self.sim_entities.items():
                if agent_id != lead:
                    entity.register_partner(lead_entity)

    @abc.abstractmethod
    def _construct_platform_map(self) -> dict:
        raise NotImplementedError


class Dubins2dSimulator(DubinsSimulator):
    """
    A class that contains all essential components of a Dubins2D simulation.
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (bp.Dubins2dAircraft, Dubins2dPlatform),
            'dubins2d': (bp.Dubins2dAircraft, Dubins2dPlatform),
        }


PluginLibrary.AddClassToGroup(Dubins2dSimulator, "Dubins2dSimulator", {})


class Dubins3dSimulator(DubinsSimulator):
    """
    A class that contains all essential components of a Dubins 3D simulation.
    """

    def _construct_platform_map(self) -> dict:
        return {
            'default': (bp.Dubins3dAircraft, Dubins3dPlatform),
            'dubins3d': (bp.Dubins3dAircraft, Dubins3dPlatform),
        }


PluginLibrary.AddClassToGroup(Dubins3dSimulator, "Dubins3dSimulator", {})
