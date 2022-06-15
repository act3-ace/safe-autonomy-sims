"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------
"""

from __future__ import annotations

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_available_platforms import BaseAvailablePlatformTypes

from saferl.simulators.cwh_simulator import CWHSimulator
from saferl.simulators.inspection_simulator import InspectionSimulator


class CWHAvailablePlatformTypes(BaseAvailablePlatformTypes):
    """
    Enumeration that outlines the platform types that have been implemented.
    """
    CWH = (1, )

    # TODO: Figure out mypy typing error and re-annotate

    @classmethod
    def ParseFromNameModel(cls, config: dict):
        """
        Given a config with the keys "model" and "name" determine the PlatformType.

        Parameters
        ----------
        config : dict
            Platform configuration dictionary.

        Returns
        -------
        CWHAvailablePlatformTypes
            The platform type associated with the given configuration.

        Raises
        ------
        RuntimeError
            If the given config doesn't have both "name" and "model" keys
            -or-
            If the "name" and "model" keys do not match a known model.
        """

        if "name" not in config:
            raise RuntimeError("Attempting to parse a PlatformType from name/model config, but both are not given!")

        if config["name"] == "CWH":
            return CWHAvailablePlatformTypes.CWH

        raise RuntimeError(f'name: {config["name"]} did not match a known platform type')

#TODO: figure out how to have both types of simulators here
PluginLibrary.AddClassToGroup(CWHAvailablePlatformTypes, "CWHSimulator_Platforms", {"simulator": CWHSimulator})
PluginLibrary.AddClassToGroup(CWHAvailablePlatformTypes, "CWHSimulator_Platforms", {"simulator": InspectionSimulator})
