"""

AvailablePlatforms
"""
from __future__ import annotations

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_available_platforms import BaseAvailablePlatformTypes

from saferl.core.simulators.dubins_simulator import Dubins2dSimulator, Dubins3dSimulator


class DubinsAvailablePlatformTypes(BaseAvailablePlatformTypes):
    """Enumeration that outlines the platform types that have been implemented"""

    DUBINS2D = (1, )
    DUBINS3D = (2, )

    # TODO: Figure out mypy typing error and re-annotate

    @classmethod
    def ParseFromNameModel(cls, config: dict):
        """Given a config with the keys "model" and "name" determine the PlatformType

        Raises:
            RuntimeError: if the given config doesn't have both "name" and "model" keys
            RuntimeError: if the "name" and "model" keys do not match a known model
        """
        if "name" not in config:
            raise RuntimeError("Attempting to parse a PlatformType from name/model config, but both are not given!")

        if config["name"] == "DUBINS2D":
            return DubinsAvailablePlatformTypes.DUBINS2D
        if config["name"] == "DUBINS3D":
            return DubinsAvailablePlatformTypes.DUBINS3D

        raise RuntimeError(f'name: {config["name"]} and model: {config["model"]} did not match a known platform type')


PluginLibrary.AddClassToGroup(DubinsAvailablePlatformTypes, "DubinsSimulator_Platforms", {"simulator": Dubins2dSimulator})
PluginLibrary.AddClassToGroup(DubinsAvailablePlatformTypes, "DubinsSimulator_Platforms", {"simulator": Dubins3dSimulator})
