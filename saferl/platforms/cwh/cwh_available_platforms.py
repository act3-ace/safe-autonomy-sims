"""

AvailablePlatforms
"""
from __future__ import annotations

from act3_rl_core.libraries.plugin_library import PluginLibrary
from act3_rl_core.simulators.base_available_platforms import BaseAvailablePlatformTypes

from saferl.simulators.cwh_simulator import CWHSimulator


class CWHAvailablePlatformTypes(BaseAvailablePlatformTypes):
    """Enumeration that outlines the platform types that have been implimented
    """
    CWH = (1, )

    # TODO: Figure out mypy typing error and re-annotate

    @classmethod
    def ParseFromNameModel(cls, config: dict):
        """Given a config with the keys "model" and "name" determine the PlatformType

        Parameters
        ----------
        config : dict
            Platform configuration dictionary

        Returns
        -------
        CWHAvailablePlatformTypes
            The platform type associated with the given configuration

        Raises
        ------
            RuntimeError
                if the given config doesn't have both "name" and "model" keys
                -or-
                if the "name" and "model" keys do not match a known model
        """
        print('AvailablePlatforms config=', config)

        if "name" not in config:
            raise RuntimeError("Attempting to parse a PlatformType from name/model config, but both are not given!")

        if config["name"] == "CWH":
            return CWHAvailablePlatformTypes.CWH

        raise RuntimeError(f'name: {config["name"]} did not match a known platform type')


PluginLibrary.AddClassToGroup(CWHAvailablePlatformTypes, "CWHSimulator_Platforms", {"simulator": CWHSimulator})
