"""

AvailablePlatforms
"""
from __future__ import annotations
from act3_rl_core.libraries.plugin_library import PluginLibrary

from simulators.cwh_simulator import CWHSimulator
from act3_rl_core.simulators.base_available_platforms import BaseAvailablePlatformTypes


class CWHAvailablePlatformTypes(BaseAvailablePlatformTypes):
    """Enumeration that outlines the platform types that have been implimented
    """
    CWH = (1, )

    @classmethod
    def ParseFromNameModel(cls, config: dict) -> CWHAvailablePlatformTypes:
        """Given a config with the keys "model" and "name" determine the PlatformType

        Raises:
            RuntimeError: if the given config doesn't have both "name" and "model" keys
            RuntimeError: if the "name" and "model" keys do not match a known model
        """
        if "name" not in config:
            raise RuntimeError("Attempting to parse a PlatformType from name/model config, but both are not given!")

        if config["name"] == "CWH":
            return CWHAvailablePlatformTypes.CWH

        raise RuntimeError(f'name: {config["name"]} and model: {config["model"]} did not match a known platform type')


PluginLibrary.AddClassToGroup(CWHAvailablePlatformTypes, "CWHSimulator_Platforms", {"simulator": CWHSimulator})
