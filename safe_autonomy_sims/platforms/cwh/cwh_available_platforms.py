"""
--------------------------------------------------------------------------
Air Force Research Laboratory (AFRL) Autonomous Capabilities Team (ACT3)
Reinforcement Learning (RL) Core  Extension.

This is a US Government Work not subject to copyright protection in the US.

The use, dissemination or disclosure of data in this file is subject to
limitation or restriction. See accompanying README and LICENSE for details.
---------------------------------------------------------------------------

This module defines and registers the available CWH platform types to the proper simulators.
"""

from __future__ import annotations

from corl.libraries.plugin_library import PluginLibrary
from corl.simulators.base_platform_type import BasePlatformType

from safe_autonomy_sims.simulators.cwh_simulator import CWHSimulator
from safe_autonomy_sims.simulators.inspection_simulator import InspectionSimulator


class CWHAvailablePlatformTypes(BasePlatformType):
    """
    """

    @classmethod
    def match_model(cls, config: dict) -> bool:
        if "name" not in config:
            raise RuntimeError("Attempting to parse a PlatformType from name/model config, but both are not given!")

        if config["name"] == "CWH":
            return True
        if config["name"] == "CWHSixDOF":
            return True
        return False


PluginLibrary.add_platform_to_sim(CWHAvailablePlatformTypes, CWHSimulator)
PluginLibrary.add_platform_to_sim(CWHAvailablePlatformTypes, InspectionSimulator)
