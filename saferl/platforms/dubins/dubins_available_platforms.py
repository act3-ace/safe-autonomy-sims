"""

AvailablePlatforms
"""
from __future__ import annotations

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_available_platforms import BaseAvailablePlatformTypes

from saferl.simulators.dubins.dubins_simulator import Dubins2dSimulator


class DubinsAvailablePlatformTypes(BaseAvailablePlatformTypes):
    """Enumeration that outlines the platform types that have been implemented
    """
    DUBINS2D = (1, )
    DUBINS3D = (2, )

    @classmethod
    def ParseFromNameModel(cls, config: dict) -> DubinsAvailablePlatformTypes:
        """Given a config with the keys "model" and "name" determine the PlatformType

        Raises:
            RuntimeError: if the given config doesn't have both "name" and "model" keys
            RuntimeError: if the "name" and "model" keys do not match a known model
        """
        if "name" not in config:
            raise RuntimeError("Attempting to parse a PlatformType from name/model config, but both are not given!")

        if config["name"] == "Dubins2d":
            return DubinsAvailablePlatformTypes.DUBINS2D
        if config["name"] == "Dubins3d":
            return DubinsAvailablePlatformTypes.DUBINS3D

        raise RuntimeError(f'name: {config["name"]} and model: {config["model"]} did not match a known platform type')


PluginLibrary.AddClassToGroup(DubinsAvailablePlatformTypes, "DubinsSimulator_Platforms", {"simulator": Dubins2dSimulator})
